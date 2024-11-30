import os
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "model_name": os.getenv("MODEL_NAME", "gpt-4-turbo-preview"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    }