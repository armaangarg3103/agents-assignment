"""
Context-Aware Voice Agent with Intelligent Backchannel Handling.

This agent implements context-aware interruption filtering that:
1. Ignores backchannel words ("yeah", "ok", "hmm") when the agent is speaking
2. Allows real interruptions ("wait", "stop", "no") to stop the agent
3. Processes all input normally when the agent is silent
4. Handles mixed inputs ("yeah but wait") correctly

The solution works by:
1. Patching the AgentActivity's transcript handling methods
2. Filtering backchannels BEFORE they can trigger interruptions
3. Using the agent_state to determine if the agent is speaking
"""

import asyncio
import logging
import functools
from typing import Optional, Callable, Any

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AgentServer,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    UserInputTranscribedEvent,
    AgentStateChangedEvent,
    stt,
    vad,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.events import UserInputTranscribedEvent
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

try:
    from .interruption_filter import InterruptionFilter, FilterDecision
    from .config import InterruptionFilterConfig
except ImportError:
    from interruption_filter import InterruptionFilter, FilterDecision
    from config import InterruptionFilterConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("context-aware-agent")

load_dotenv()


def patch_agent_activity(activity: AgentActivity, filter_instance: InterruptionFilter) -> None:
    """
    Patch an AgentActivity instance to add backchannel filtering.
    
    This patches the on_interim_transcript and on_final_transcript methods
    to filter out backchannels when the agent is speaking.
    """
    
    # Store original methods
    original_on_interim = activity.on_interim_transcript
    original_on_final = activity.on_final_transcript
    original_on_vad_done = activity.on_vad_inference_done
    
    # Track the last filter decision for VAD coordination
    last_filter_decision = {"value": None}
    
    def is_agent_speaking() -> bool:
        """Check if the agent is currently speaking."""
        # Check agent state
        if activity._session.agent_state == "speaking":
            return True
        # Check for active speech handle
        if activity._current_speech is not None and not activity._current_speech.interrupted:
            return True
        # Check for paused speech
        if activity._paused_speech is not None:
            return True
        return False
    
    def should_filter(transcript: str) -> bool:
        """Determine if transcript should be filtered."""
        if not transcript or not transcript.strip():
            return True
        
        speaking = is_agent_speaking()
        result = filter_instance.filter(transcript, speaking)
        last_filter_decision["value"] = result.decision
        
        return result.decision == FilterDecision.IGNORE
    
    def patched_on_interim_transcript(ev: stt.SpeechEvent, *, speaking: bool | None) -> None:
        """Patched interim transcript handler with backchannel filtering."""
        transcript = ev.alternatives[0].text if ev.alternatives else ""
        
        if should_filter(transcript):
            # Still emit the transcription event for logging/display
            activity._session._user_input_transcribed(
                UserInputTranscribedEvent(
                    language=ev.alternatives[0].language if ev.alternatives else None,
                    transcript=transcript,
                    is_final=False,
                    speaker_id=ev.alternatives[0].speaker_id if ev.alternatives else None,
                ),
            )
            logger.info(f"[FILTERED-INTERIM] Ignored: '{transcript}' (agent speaking: {is_agent_speaking()})")
            # DON'T call original - this prevents _interrupt_by_audio_activity
            return
        
        # Not filtered - proceed with original behavior
        original_on_interim(ev, speaking=speaking)
    
    def patched_on_final_transcript(ev: stt.SpeechEvent, *, speaking: bool | None = None) -> None:
        """Patched final transcript handler with backchannel filtering."""
        transcript = ev.alternatives[0].text if ev.alternatives else ""
        
        if should_filter(transcript):
            # Emit the event for logging
            activity._session._user_input_transcribed(
                UserInputTranscribedEvent(
                    language=ev.alternatives[0].language if ev.alternatives else None,
                    transcript=transcript,
                    is_final=True,
                    speaker_id=ev.alternatives[0].speaker_id if ev.alternatives else None,
                ),
            )
            logger.info(f"[FILTERED-FINAL] Ignored: '{transcript}' (agent speaking: {is_agent_speaking()})")
            # DON'T call original - this prevents interruption
            return
        
        # Not filtered - proceed with original behavior
        original_on_final(ev, speaking=speaking)
    
    def patched_on_vad_inference_done(ev: vad.VADEvent) -> None:
        """Patched VAD handler that respects filter decisions."""
        # If turn detection is manual or realtime_llm, let original handle it
        if activity._turn_detection in ("manual", "realtime_llm"):
            original_on_vad_done(ev)
            return
        
        # If speech is too short, let original handle it (it will ignore anyway)
        if ev.speech_duration < activity._session.options.min_interruption_duration:
            original_on_vad_done(ev)
            return
        
        # If we recently filtered a transcript and agent is speaking, suppress VAD interrupt
        if last_filter_decision["value"] == FilterDecision.IGNORE and is_agent_speaking():
            logger.info(f"[FILTERED-VAD] Suppressing VAD interrupt (last transcript was backchannel)")
            return
        
        # Otherwise proceed normally
        original_on_vad_done(ev)
    
    # Apply patches
    activity.on_interim_transcript = patched_on_interim_transcript
    activity.on_final_transcript = patched_on_final_transcript
    activity.on_vad_inference_done = patched_on_vad_inference_done
    
    logger.info("[Patch] AgentActivity patched with backchannel filtering")


class ContextAwareAgent(Agent):
    """
    A voice agent that intelligently handles user backchannels.
    
    When the agent is speaking:
    - Backchannel words ("yeah", "ok", "hmm") are ignored
    - Command words ("wait", "stop", "no") trigger interruption
    
    When the agent is silent:
    - All input is processed normally
    """
    
    def __init__(self, filter_config: Optional[InterruptionFilterConfig] = None) -> None:
        super().__init__(
            instructions="""You are a helpful, friendly voice assistant. 
            Your name is Kelly. Keep responses clear and conversational.
            
            When explaining things, speak naturally and thoroughly.
            If the user acknowledges with "yeah", "ok", or "hmm" while you're speaking,
            that means they're listening - continue your explanation.
            
            If they say "wait", "stop", or ask a question, pause and address them.
            
            Do not use emojis, asterisks, markdown, or other special characters.
            Keep your responses suitable for voice output.""",
        )
        
        self._filter_config = filter_config or InterruptionFilterConfig()
        self._filter = InterruptionFilter(self._filter_config)
        self._patched = False
    
    async def on_enter(self) -> None:
        """Called when the agent becomes active."""
        
        # Patch the activity if not already done
        if not self._patched and self._activity is not None:
            patch_agent_activity(self._activity, self._filter)
            self._patched = True
        
        # Generate initial greeting
        self.session.generate_reply(
            instructions="Greet the user warmly and ask how you can help them today."
        )
    
    @function_tool
    async def tell_long_story(self, context: RunContext) -> str:
        """
        Tell a long story to test backchannel handling.
        This is useful for testing that "yeah" and "ok" don't interrupt.
        """
        logger.info("[Tool] Telling a long story...")
        return """Let me tell you an interesting story about the history of computing.
        
        In the 1940s, computers filled entire rooms and were operated by teams of specialists.
        The ENIAC, one of the first electronic computers, weighed over 27 tons and used 
        about 18,000 vacuum tubes. It could perform about 5,000 operations per second,
        which was revolutionary at the time.
        
        By the 1970s, the invention of the microprocessor changed everything.
        The Intel 4004, released in 1971, put an entire CPU on a single chip.
        This led to the personal computer revolution of the 1980s.
        
        Today, a smartphone has more computing power than all of NASA had in 1969
        when they sent astronauts to the moon. That's an incredible advancement
        in just a few decades of human innovation."""
    
    @function_tool
    async def count_to_ten(self, context: RunContext) -> str:
        """
        Count from one to ten slowly. Good for testing interruption handling.
        """
        logger.info("[Tool] Counting to ten...")
        return "I'll count slowly for you: One... Two... Three... Four... Five... Six... Seven... Eight... Nine... Ten. There you go!"
    
    @function_tool
    async def explain_something(self, context: RunContext, topic: str) -> str:
        """
        Explain a topic in detail. Good for testing that backchannels don't interrupt.
        
        Args:
            topic: The topic to explain
        """
        logger.info(f"[Tool] Explaining topic: {topic}")
        return f"""Let me explain {topic} in detail for you.
        
        This is a fascinating subject with many interesting aspects to explore.
        I'll break it down step by step so it's easy to understand.
        
        First, we need to understand the basic concepts and terminology.
        Then we can dive into the more complex aspects and nuances.
        Finally, I'll give you some practical examples and applications.
        
        This comprehensive approach will give you a solid understanding of {topic}.
        Feel free to ask questions if anything is unclear."""
    
    @function_tool
    async def get_weather(
        self, 
        context: RunContext, 
        location: str
    ) -> str:
        """
        Get the weather for a location.
        
        Args:
            location: The city or location to check weather for.
        """
        logger.info(f"[Tool] Getting weather for {location}")
        return f"The weather in {location} is currently sunny with a temperature of 72 degrees Fahrenheit. It's a beautiful day!"


# Create the agent server
server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    """Prewarm the VAD model during process startup."""
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Main entrypoint for the voice agent."""
    
    # Set up logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    logger.info(f"Starting context-aware agent in room: {ctx.room.name}")
    
    # Create custom configuration (optional - can customize word lists here)
    filter_config = InterruptionFilterConfig(
        debug_logging=True,  # Enable detailed logging
    )
    
    # Create a standard session
    session = AgentSession(
        # Speech-to-text
        stt="deepgram/nova-3",
        # Language model
        llm="openai/gpt-4.1-mini",
        # Text-to-speech
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        # Turn detection
        turn_detection=MultilingualModel(),
        # VAD
        vad=ctx.proc.userdata["vad"],
        # Enable preemptive generation for lower latency
        preemptive_generation=True,
        # Disable default false interruption handling - we handle it ourselves
        resume_false_interruption=False,
        false_interruption_timeout=None,
    )
    
    # Create our context-aware agent with the filter config
    agent = ContextAwareAgent(filter_config=filter_config)
    
    # Start the session with our agent
    await session.start(
        agent=agent,
        room=ctx.room,
    )
    
    logger.info("Context-aware agent started successfully")


if __name__ == "__main__":
    cli.run_app(server)
