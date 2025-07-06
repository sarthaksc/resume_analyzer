# job_faiss_recommender.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3

class JobFaissRecommender:
    def __init__(self, index_path="data/job_index.faiss", meta_path="data/job_meta.npy", db_path="data/jobs.db", table_name="job_postings"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(index_path)
        self.metadata = np.load(meta_path, allow_pickle=True)
        self.db_path = db_path
        self.table_name = table_name

    def recommend_jobs(self, resume_text: str, top_k: int = 5):
        query_embedding = self.model.encode([resume_text], convert_to_numpy=True)
        scores, indices = self.index.search(query_embedding, top_k * 3)  # fetch more to allow de-duplication

        seen_descriptions = set()
        results = []

        # Connect to the database to fetch full job descriptions
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for idx, score in zip(indices[0], scores[0]):
            rowid = self.metadata[idx][0]  # rowid stored as first column in metadata
            cursor.execute(
                f"SELECT Title, Company, Location, Salary, JobDescription FROM {self.table_name} WHERE rowid = ?",
                (rowid,))
            row = cursor.fetchone()

            if not row:
                continue

            title, company, location, salary, description = row

            if description in seen_descriptions:
                continue  # skip duplicate job descriptions

            seen_descriptions.add(description)

            result = {
                "rowid": rowid,
                "title": title,
                "company": company,
                "location": location,
                "salary": salary,
                "description": description,
                "score": round(float(score), 2)
            }
            results.append(result)

            if len(results) >= top_k:
                break

        conn.close()
        return results
