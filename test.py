# Quick diagnostic script
from src.data.vector_store import VectorStore

vs = VectorStore()
collection = vs.client.get_collection("interview_questions")

# See what's actually stored
results = collection.get(limit=5, include=["metadatas"])
for m in results["metadatas"]:
    print(m)