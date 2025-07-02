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

def infer_domains_from_resume(resume_text: str) -> list[str]:
    domain_keywords = {
        "AI/ML": ["machine learning", "deep learning", "neural networks", "tensorflow", "keras", "generative ai"],
        "Finance": ["banking", "investment", "equity", "trading", "derivatives", "financial modeling"],
        "Consulting": ["strategy", "case study", "operations", "business model", "transformation"],
        "E-commerce": ["recommendation system", "e-commerce", "online shopping", "logistics", "flask", "checkout"],
        "Cloud/Infra": ["aws", "azure", "gcp", "cloud", "infrastructure", "devops"],
        "Data Analytics": ["data analysis", "pandas", "sql", "dashboard", "insights", "business intelligence"],
        # Add more domains as needed
    }

    resume_lower = resume_text.lower()
    inferred_domains = []
    for domain, keywords in domain_keywords.items():
        if any(keyword in resume_lower for keyword in keywords):
            inferred_domains.append(domain)

    return inferred_domains
