import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    WEAVIATE_URL: str = os.getenv("WEAVIATE_URL")
    HOST: str = "192.168.137.236"
    PORT: int = 6789

settings = Settings()
