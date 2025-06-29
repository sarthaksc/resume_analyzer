# matcher/embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np

class ResumeEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)


get_embedding = ResumeEmbedder().embed


