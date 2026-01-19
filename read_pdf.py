import PyPDF2

pdf_path = r"C:\Users\garga\Saved Games\OneDrive\Desktop\GEN AI Assignment Campus (New).pdf"

with open(pdf_path, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n\n"
    
    print(text)
