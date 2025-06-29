# matcher/matcher.py

import os
import faiss
import pickle
import numpy as np
from matcher.embedder import get_embedding

class ResumeMatcher:
    def __init__(self, index_path='matcher/faiss_index.index', metadata_path='matcher/index_metadata.pkl'):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []

    def build_index(self, embeddings: list, metadata: list):
        """
        Builds and saves FAISS index from embeddings and associated metadata (e.g. file paths).
        """
        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype('float32'))
        self.metadata = metadata
        self.save_index()

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def load_index(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError("FAISS index or metadata not found")
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

    def match_resume(self, user_text: str, top_k: int = 5):
        if self.index is None or self.metadata == []:
            self.load_index()

        user_embedding = get_embedding(user_text).astype('float32')
        D, I = self.index.search(np.array([user_embedding]), top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            results.append({
                "file_path": self.metadata[idx],
                "score": float(score)
            })
        return results
