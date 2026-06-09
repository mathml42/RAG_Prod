import os
import yaml
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_config(config_path: str='config.yaml'):
    """Loads the configuration from a YAML file."""
    with open(config_path, 'r') as config:
        return yaml.safe_load(config)

def ingest_documents():
    # load configuration
    config = load_config()
    data_dir = config['paths']['data_dir']
    chunk_size = config['chunking']['chunk_size']
    chunk_overlap = config['chunking']['chunk_overlap']

    print("Starting Ingestion Pipeline...")
    print(f"Configuration Loaded. Target chunk size: {chunk_size}, chunk overlap: {chunk_overlap}.")

    # Collect all pdf from the data directory
    pdf_files = list(data_dir.glob('*.pdf'))
    if not pdf_files:
        print(f"No PDF files found in {data_dir}. Please check the path and try again.")
        return
    
    all_pages = []

    # load every pdf using document loader
    for pdf_path in pdf_files:
        print(f"Loading PDF: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        all_pages.extend(pages)

    print(f"\nTotal pages extracted from {len(pdf_files)} PDFs: {len(all_pages)}.")
    