import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_embeddings():
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if openai_key:
        logging.info("Using OpenAI Embeddings")
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    elif google_key:
        logging.info("Using Google Generative AI Embeddings")
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)
    else:
        raise ValueError("Neither OPENAI_API_KEY nor GOOGLE_API_KEY set in environment variables.")

def main():
    load_dotenv()
    
    # Determine the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Get configuration from env
    pdf_path_env = os.getenv("PDF_PATH", "document.pdf")
    
    # If path is relative, resolve it against project root
    if not os.path.isabs(pdf_path_env):
        pdf_path = os.path.join(project_root, pdf_path_env)
    else:
        pdf_path = pdf_path_env
    
    db_url = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag")
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME", "document_vectors")

    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found at {pdf_path}")
        return

    # 1. Load PDF
    logging.info(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    logging.info(f"Loaded {len(docs)} pages.")

    # 2. Split Text
    logging.info("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)
    logging.info(f"Created {len(splits)} chunks.")

    # 3. Create Embeddings and Store in pgVector
    logging.info("Initializing Embeddings and Vector Store...")
    embeddings = get_embeddings()

    logging.info(f"Storing vectors in table '{collection_name}' in database...")
    
    # PGVector from langchain_postgres
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=db_url,
        use_jsonb=True,
    )

    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    from google.api_core.exceptions import ResourceExhausted

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=10, min=4, max=60),
        retry=retry_if_exception_type(ResourceExhausted),
        before_sleep=lambda retry_state: logging.warning(f"Quota exceeded. Retrying in {retry_state.next_action.sleep} seconds...")
    )
    def ingest_documents():
        vector_store.add_documents(splits)

    try:
        ingest_documents()
        logging.info("Ingestion complete!")
    except Exception as e:
        logging.error(f"Failed to ingest documents after retries: {e}")

if __name__ == "__main__":
    main()