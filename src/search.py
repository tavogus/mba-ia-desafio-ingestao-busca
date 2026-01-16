import os
import logging
from dotenv import load_dotenv
from langchain_postgres import PGVector
from src.ingest import get_embeddings # Reuse embedding logic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def search(query: str, k: int = 10):
    load_dotenv()
    
    db_url = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag")
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME", "document_vectors")

    embeddings = get_embeddings()

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=db_url,
        use_jsonb=True,
    )

    logging.info(f"Searching for: '{query}'")
    results = vector_store.similarity_search_with_score(query, k=k)
    
    return results

if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Qual o faturamento?"
    results = search(query)
    for doc, score in results:
        print(f"Score: {score}\nContent: {doc.page_content}\n---")