# # matcher/matcher.py
#
# import os
# import faiss
# import pickle
# import numpy as np
# from matcher.embedder import get_embedding
#
# class ResumeMatcher:
#     def __init__(self, index_path='matcher/faiss_index.index', metadata_path='matcher/index_metadata.pkl'):
#         self.index_path = index_path
#         self.metadata_path = metadata_path
#         self.index = None
#         self.metadata = []
#
#     def build_index(self, embeddings: list, metadata: list):
#         """
#         Builds and saves FAISS index from embeddings and associated metadata (e.g. file paths).
#         """
#         dim = len(embeddings[0])
#         self.index = faiss.IndexFlatL2(dim)
#         self.index.add(np.array(embeddings).astype('float32'))
#         self.metadata = metadata
#         self.save_index()
#
#     def save_index(self):
#         faiss.write_index(self.index, self.index_path)
#         with open(self.metadata_path, 'wb') as f:
#             pickle.dump(self.metadata, f)
#
#     def load_index(self):
#         if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
#             raise FileNotFoundError("FAISS index or metadata not found")
#         self.index = faiss.read_index(self.index_path)
#         with open(self.metadata_path, 'rb') as f:
#             self.metadata = pickle.load(f)
#
#     def match_resume(self, user_text: str, top_k: int = 5):
#         if self.index is None or self.metadata == []:
#             self.load_index()
#
#         user_embedding = get_embedding(user_text).astype('float32')
#         D, I = self.index.search(np.array([user_embedding]), top_k)
#         results = []
#         for idx, score in zip(I[0], D[0]):
#             results.append({
#                 "file_path": self.metadata[idx],
#                 "score": float(score)
#             })
#         return results

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

        print(f"[DEBUG] query_embedding:{query_embedding}")

        if isinstance(self.resume_embeddings, list):
            resume_tensor = torch.stack(self.resume_embeddings)
        else:
            resume_tensor = self.resume_embeddings  # Already a tensor

        scores = util.cos_sim(query_embedding, resume_tensor)
        print(f"[DEBUG] scores:{scores}")
        scores=scores[0]

        print(f"[DEBUG] scores shape: {scores.shape}")


        # DEBUG LOGS
        print(f"[DEBUG] query_embedding shape: {query_embedding.shape}")
        print(f"[DEBUG] resume_tensor shape: {resume_tensor.shape}")
        print(f"[DEBUG] scores shape: {scores.shape}")

        if scores.numel() == 0:
            raise ValueError("Similarity scores are empty.")

        top_indices = scores.argsort(descending=True)[:top_k]

        results = []
        for idx in top_indices:
            meta = self.resume_metadata[idx]
            score_tensor = scores[idx]
            print(
                f"[DEBUG] Score tensor at index {idx}: {score_tensor} (shape: {getattr(score_tensor, 'shape', 'N/A')})")
            try:
                score = float(score_tensor.item())
            except Exception as e:
                print(f"[ERROR] Failed to convert score tensor at index {idx}: {e}")
                raise
            results.append((meta["company"], meta["filename"], score))
        return results



