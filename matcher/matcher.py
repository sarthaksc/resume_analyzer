
import os
import json
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
from resume_parser.parser import parse_pdf_to_text
import torch


class ResumeMatcher:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.resume_embeddings = []
        self.resume_texts = []
        self.resume_metadata = []

    def build_index(self, resume_dir: str):
        """
        Parse all resumes under `resume_dir`, generate embeddings,
        and build an in-memory index.
        """
        temp_embeddings = []  # Use temporary list to collect embeddings

        for company in os.listdir(resume_dir):
            company_path = os.path.join(resume_dir, company)
            if not os.path.isdir(company_path):
                continue

            for fname in os.listdir(company_path):
                if fname.endswith(".pdf"):
                    path = os.path.join(company_path, fname)
                    try:
                        text = parse_pdf_to_text(open(path, "rb"))
                        embedding = self.model.encode(text, convert_to_tensor=True)
                        temp_embeddings.append(embedding)
                        self.resume_texts.append(text)
                        self.resume_metadata.append({"company": company, "filename": fname})
                    except Exception as e:
                        print(f"[WARN] Failed to process {path}: {e}")

        # Stack all embeddings into a single tensor (✅ critical fix)
        if temp_embeddings:
            self.resume_embeddings = torch.stack(temp_embeddings)
        else:
            self.resume_embeddings = torch.empty(0)

    def find_matches(self, query_text: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Given a resume string, return top_k similar resumes from the index.
        Returns list of (company, filename, score).
        """

        print(f"[DEBUG] self.resume_embeddings:{self.resume_embeddings}")
        if self.resume_embeddings is None or (isinstance(self.resume_embeddings, torch.Tensor) and self.resume_embeddings.numel() == 0):
            raise ValueError("Resume index is empty. Did you run build_index()?")

        query_embedding = self.model.encode(query_text, convert_to_tensor=True)
        if isinstance(self.resume_embeddings, list):
            resume_tensor = torch.stack(self.resume_embeddings)
        else:
            resume_tensor = self.resume_embeddings  # Already a tensor

        scores = util.cos_sim(query_embedding, resume_tensor)
        scores=scores[0]

        if scores.numel() == 0:
            raise ValueError("Similarity scores are empty.")

        top_indices = scores.argsort(descending=True)[:top_k]

        results = []
        for idx in top_indices:
            meta = self.resume_metadata[idx]
            score_tensor = scores[idx]
            try:
                score = float(score_tensor.item())
            except Exception as e:
                print(f"[ERROR] Failed to convert score tensor at index {idx}: {e}")
                raise
            results.append((meta["company"], meta["filename"], score))
        return results



