# resume_parser/extractor.py

import re
import os
import fitz

def extract_name(text: str) -> str:
    # naive approach: name is likely the first line or capitalized line
    lines = text.strip().split("\n")
    for line in lines:
        if line and line[0].isupper() and len(line.split()) <= 4:
            return line.strip()
    return "Name Not Found"

def load_skills_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        skills = {line.strip().lower() for line in f if line.strip()}
    return skills

def extract_skills_from_resume(pdf_path, skills_file):
    # Load keywords from the text file
    keyword_set = load_skills_from_file(skills_file)

    # Extract text from PDF
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    doc.close()

    # Lowercase the resume text for case-insensitive matching
    text_lower = text.lower()

    # Match skills
    found_skills = {skill for skill in keyword_set if skill in text_lower}

    return sorted(found_skills)

def extract_target_companies(text: str, known_companies: list) -> list:
    matches = []
    for company in known_companies:
        if company.lower() in text.lower():
            matches.append(company)
    return matches

def extract_resume_info(text: str, known_companies: list) -> dict:
    return {
        "name": extract_name(text),
        "skills": extract_skills_from_resume(text),
        "target_companies": []
    }
