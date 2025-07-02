# utils/embedder.py
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def get_embedding(text: str, model="text-embedding-3-small") -> list[float]:
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding
