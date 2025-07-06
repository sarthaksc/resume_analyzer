# job_index_builder.py
import pandas as pd
import faiss
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer


class JobIndexBuilder:
    def __init__(self, db_path: str, table_name: str, text_columns=None):
        self.db_path = db_path
        self.table_name = table_name
        self.text_columns = text_columns or ["JobDescription", "JobRequirment", "RequiredQual", "Title"]
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_jobs(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(f"SELECT rowid, * FROM {self.table_name}", conn)
        conn.close()
        return df

    def build_faiss_index(self, job_texts):
        print(f"[INFO] Generating embeddings for {len(job_texts)} job entries...")
        embeddings = self.model.encode(job_texts, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index, embeddings

    def build_and_save(self, index_path="job_index.faiss", meta_path="job_meta.npy"):
        df = self.load_jobs()
        job_texts = (df["Title"].fillna('') + " " +
                     df["JobDescription"].fillna('') + " " +
                     df["JobRequirment"].fillna('')).tolist()

        index, _ = self.build_faiss_index(job_texts)
        faiss.write_index(index, index_path)
        metadata = df[["rowid", "Title", "Company", "Location", "Salary"]].to_numpy()
        np.save(meta_path, metadata)
        # df[["rowid", "Title", "Company", "Location", "Salary"]].to_numpy().dump(meta_path)
        print(f"✅ Saved FAISS index to `{index_path}`")
        print(f"✅ Saved metadata to `{meta_path}`")