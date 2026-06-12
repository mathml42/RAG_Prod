import yaml
import logging
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure logging to keep output clean
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path="config.yml"):
    config_path = Path(__file__).parent / config_path
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_vector_db():
    """Loads the existing Chroma database from disk."""
    config = load_config()
    db_dir = config["paths"]["chroma_db_dir"]
    
    # Initialize the EXACT same embedding model used during ingestion
    embeddings = OllamaEmbeddings(model=config["models"]["embeddings"])
    
    # Connect to the existing database (Notice we drop 'from_documents')
    db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    return db, config

def test_similarity_search(query: str):
    """Executes a raw vector search and prints the mathematical scores."""
    db, config = get_vector_db()
    
    logger.info(f"\n--- SIMILARITY SEARCH RESULTS FOR: '{query}' ---")
    
    # k=3 means we want the top 3 closest chunks
    results = db.similarity_search_with_score(query, k=config["retrieval"]["k"])
    
    if not results:
        logger.info("No results found. Is your database empty?")
        return

    for i, (doc, score) in enumerate(results):
        logger.info(f"\nResult #{i+1} | L2 Distance Score: {score:.4f} (Lower is closer/better)")
        logger.info(f"Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
        # Print the first 150 characters to see what it found
        logger.info(f"Preview: {doc.page_content[:150]}...")

def run_rag_chain(query: str):
    """The full RAG Pipeline: Retrieve -> Augment -> Generate"""
    db, config = get_vector_db()
    
    # 1. Initialize the LLM (e.g., llama3.2:1b)
    llm = ChatOllama(model=config["models"]["llm"])
    
    # 2. Retrieve the top 3 documents
    results = db.similarity_search(query, k=config["retrieval"]["k"])
    
    # 3. Augment: Combine the retrieved chunks into one big string of text
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    
    # 4. Augment: Build the Prompt Template
    # We explicitly tell the LLM to ONLY use the provided context to stop hallucinations.
    PROMPT_TEMPLATE = """
    You are an AI Research Assistant. Answer the question based ONLY on the following context:
    
    {context}
    
    ---
    
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # 5. Build the LangChain Runnable Pipeline (Prompt -> LLM -> String Output)
    chain = prompt | llm | StrOutputParser()
    
    logger.info(f"\n--- GENERATING RAG ANSWER FOR: '{query}' ---")
    logger.info(f"Using Model: {config['models']['llm']}")
    
    # 6. Generate the answer!
    response = chain.invoke({
        "context": context_text,
        "question": query
    })
    
    logger.info("\nFINAL ANSWER:")
    logger.info(response)

if __name__ == "__main__":
    # Feel free to change this question to something relevant to your PDFs!
    sample_question = "What is the main topic of the documents provided?"
    
    # Step 1: See the raw mathematical search results
    test_similarity_search(sample_question)
    
    # Step 2: See the final AI generated answer using those results
    run_rag_chain(sample_question)