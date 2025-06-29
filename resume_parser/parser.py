# resume_parser/parser.py

import fitz  # PyMuPDF

def parse_pdf_to_text(pdf_file) -> str:
    pdf_file.seek(0)  # Reset pointer to the beginning
    pdf_bytes = pdf_file.read()  # Read the binary contents
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")  # Open from stream
    text = ""
    for page in doc:
        text += page.get_text()
    return text
