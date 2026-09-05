import yaml
import logging
import warnings
from pathlib import Path

# Suppress the LangChain community deprecation warnings to keep our terminal clean
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_core.globals import set_debug

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path="config.yml"):
    config_path = Path(__file__).parents[1] / config_path
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_hybrid_retriever():
    """Sets up retrievers based on the selected method in config."""
    config = load_config()
    db_dir = config["paths"]["chroma_db_dir"]
    k = config.get("retrieval", {}).get("k", 4) # Let's pull 4 chunks total
    method = config.get("retrieval", {}).get("method", "hybrid").lower()
    
    # --- 1. The Vector Retriever (Dense) ---
    logger.info("Initializing Vector Retriever (ChromaDB)...")
    embeddings = OllamaEmbeddings(model=config["models"]["embeddings"])
    vector_db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": k})
    
    keyword_retriever = None
    
    # --- 2. The Keyword Retriever (Sparse/BM25) - ONLY IF HYBRID ---
    if method == "hybrid":
        logger.info("Initializing Keyword Retriever (BM25) for Hybrid Search...")
        # BM25 needs to build an index of the raw text. We pull all documents from Chroma to build it.
        all_docs = vector_db.get() # Fetches everything
        
        # Reconstruct the LangChain Document objects from the raw Chroma data
        from langchain_core.documents import Document
        reconstructed_docs = [
            Document(page_content=text, metadata=meta) 
            for text, meta in zip(all_docs['documents'], all_docs['metadatas'])
        ]
        
        keyword_retriever = BM25Retriever.from_documents(reconstructed_docs)
        keyword_retriever.k = k
    else:
        logger.info("Skipping Keyword Retriever setup (Semantic Search only).")
    
    return vector_retriever, keyword_retriever, config

def reciprocal_rank_fusion(vector_results, keyword_results, rrf_k=60):
    """Combines results using the Reciprocal Rank Fusion algorithm."""
    fused_scores = {}
    doc_map = {}

    # Score Vector Results
    for rank, doc in enumerate(vector_results):
        content = doc.page_content
        doc_map[content] = doc
        fused_scores[content] = fused_scores.get(content, 0) + 1 / (rank + rrf_k)

    # Score Keyword Results
    for rank, doc in enumerate(keyword_results):
        content = doc.page_content
        doc_map[content] = doc
        fused_scores[content] = fused_scores.get(content, 0) + 1 / (rank + rrf_k)

    # Sort documents by their combined RRF score
    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return sorted document objects
    return [doc_map[content] for content, score in reranked]

def run_hybrid_rag(query: str):
    vector_retriever, keyword_retriever, config = setup_hybrid_retriever()
    method = config.get("retrieval", {}).get("method", "hybrid").lower()
    
    if config.get("system", {}).get("debug", False):
        set_debug(True)
        
    llm = ChatOllama(
        model=config["models"]["llm"],
        temperature=config.get("generation", {}).get("temperature", 0.0),
        num_ctx=config.get("generation", {}).get("num_ctx", 4096)
    )
    
    # 1. Retrieval happens here!
    logger.info(f"\n--- EXECUTING {method.upper()} SEARCH FOR: '{query}' ---")
    k = config.get("retrieval", {}).get("k", 4)
    
    if method == "hybrid":
        # Query both systems independently
        vector_docs = vector_retriever.invoke(query)
        keyword_docs = keyword_retriever.invoke(query)
        
        # Combine using our custom RRF algorithm
        results = reciprocal_rank_fusion(vector_docs, keyword_docs)[:k]
        logger.info(f"\nFound {len(results)} optimal chunks using Reciprocal Rank Fusion.")
    else:
        # Query only the vector database
        results = vector_retriever.invoke(query)
        logger.info(f"\nFound {len(results)} optimal chunks using pure Semantic Search.")
    
    # 2. Build the context
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    
    PROMPT_TEMPLATE = """
    You are an AI Research Assistant. Answer the question based ONLY on the following context and if context is insufficient, respond with 'Insufficient information to answer the question.' and also if hybrid search doesn't have any relevant information, respond with 'No relevant information found in the documents.':
    
    {context}
    
    ---
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    
    logger.info("\n--- GENERATING ANSWER ---")
    response = chain.invoke({
        "context": context_text,
        "question": query
    })
    
    logger.info("\nFINAL ANSWER:")
    logger.info(response)

if __name__ == "__main__":
    # A query that requires BOTH exact acronym matching and understanding meaning
    sample_question = "What does the TEAL acronym stand for in the context of the AI RMF?"
    run_hybrid_rag(sample_question)