# matcher.py

import os
import fitz  # PyMuPDF
import numpy as np
from utils.embedder import get_embedding
from utils.similarity import cosine_similarity
from resume_parser.parser import parse_pdf_to_text

RESUME_DATABASE = "resume database"

# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_company_folder(base_dir: str, target_company: str) -> str:
    company_folders = os.listdir(base_dir)
    for folder in company_folders:
        if target_company.lower() in folder.lower():
            return os.path.join(base_dir, folder)
    return ""

def read_resume_text(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def find_similar_resumes(user_resume_text: str, company: str, base_path="data/company resumes") -> list:
    company_dir = os.path.join(base_path, company)
    company_folder = find_company_folder(base_path, company)
    if not company_folder or not os.path.exists(company_folder):
        print(f"[WARN] No folder for company '{company}' found at: {company_folder}")
        return []

    user_embedding = get_embedding(user_resume_text)

    similar_resumes = []
    for file_name in os.listdir(company_dir):
        if not file_name.endswith(".pdf"):
            continue

        file_path = os.path.join(company_dir, file_name)
        try:
            text = parse_pdf_to_text(file_path)
            emb = get_embedding(text)
            score = cosine_similarity(user_embedding, emb)

            snippet = text[:400].strip().replace("\n", " ")
            similar_resumes.append((file_name, score, snippet))
            print(f"[MATCHER DEBUG] Compared with: {file_name}, Similarity = {score}")
        except Exception as e:
            print(f"[ERROR] Failed to process {file_name}: {e}")

    # Sort by similarity and return top 3
    similar_resumes.sort(key=lambda x: x[1], reverse=True)
    return similar_resumes[:3]