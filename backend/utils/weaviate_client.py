import weaviate
from utils.config import settings

client = weaviate.Client(
    url=settings.WEAVIATE_URL,
    additional_headers={
        "X-OpenAI-API-Key": settings.OPENAI_API_KEY
    }
)

def query_weaviate(concepts):
    class_name = "all_nov_jobs"
    results = client.query.get(
        class_name, ["title", "place", "description"]
    ).with_near_text(
        {"concepts": concepts}
    ).with_additional(
        ["distance", "id"]
    ).with_limit(1).do()
    return results
