# config.py
import os
import dotenv
import weaviate

dotenv.load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")

client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={
        "X-OpenAI-API-Key": API_KEY
    }
)

