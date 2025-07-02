# matcher/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text: str) -> np.ndarray:
    return model.encode(text, convert_to_numpy=True)

def embed_bulk(texts: list[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True)
