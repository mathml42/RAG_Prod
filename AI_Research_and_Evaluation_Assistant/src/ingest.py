import os
import yaml
import shutil
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def load_config(config_path="config.yml"):
    """Loads configuration settings from a YAML file."""
    config_path = Path(__file__).parent / config_path
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise

def ingest_documents():
    config = load_config()
    data_dir = Path(config["paths"]["data_dir"])
    db_dir = config["paths"]["chroma_db_dir"]

    logger.info("--- Starting Ingestion & Embedding Pipeline ---")

    # Collect & Load PDFs
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in target directory: {data_dir}")
        return

    all_pages = []
    for pdf_path in pdf_files:
        logger.info(f"Loading document: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            logger.info(f"Successfully extracted {len(pages)} pages from {pdf_path.name}")
            all_pages.extend(pages)
        except Exception as e:
            logger.error(f"Failed to load document {pdf_path.name}: {str(e)}")
            continue

    logger.info(f"Total pages extracted across all documents: {len(all_pages)}")

    # Chunking Pipeline
    logger.info(f"Initializing text splitter (Size: {config['chunking']['chunk_size']}, Overlap: {config['chunking']['chunk_overlap']})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(all_pages)
    logger.info(f"Created {len(chunks)} text chunks out of raw documents.")

    # Initialize Local Embeddings via Ollama
    logger.info(f"Initializing Ollama Embeddings model: {config['models']['embeddings']}...")
    embeddings = OllamaEmbeddings(
        model=config["models"]["embeddings"],
    )

    # Create Vector DB and Save
    logger.info(f"Passing chunks to Ollama and saving to ChromaDB at '{db_dir}'...")
    logger.info("Processing embeddings locally... (this scale scales with hardware and file size)")
    
    try:
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_dir
        )
        logger.info("✅ Ingestion completely successful! Database is locked and ready.")
    except Exception as e:
        logger.critical(f"Failed to compile or write vectors to ChromaDB: {str(e)}")
        raise

if __name__ == "__main__":
    ingest_documents()