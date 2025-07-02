# utils/llm_config.py

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

def get_llm(model="gpt-4o", temperature=0.3):
    """Return a configured OpenAI-compatible LLM."""
    openai_api_key = os.getenv('OPENAI_API_KEY')
    return ChatOpenAI(
        api_key=openai_api_key,
        model=model,   # You can change this to "gpt-3.5-turbo" or another model
        temperature=temperature,
        max_tokens=1024
    )
