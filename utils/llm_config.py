# utils/llm_config.py

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

def get_llm():
    """Return a configured OpenAI-compatible LLM."""
    openai_api_key = os.getenv('OPENAI_API_KEY')
    return ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-4o",   # You can change this to "gpt-3.5-turbo" or another model
        temperature=0.2,
        max_tokens=1024
    )
