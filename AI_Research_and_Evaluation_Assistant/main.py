import streamlit as st
import yaml
from pathlib import Path
import logging
import os
from dotenv import load_dotenv

# 1. Load environment variables from your .env file
load_dotenv()

# 2. Enable LangSmith Tracing automatically if the key is found
if os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    # Map your LANGSMITH_API_KEY to the official LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY"))
    os.environ["LANGCHAIN_PROJECT"] = "AI_Research_Assistant"
    logging.info("✅ LangSmith Tracing is ENABLED!")

# AI & LangChain Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Configure logging to keep the terminal output clean and structured
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --- DYNAMIC IMPORTS FROM SRC DIRECTORY ---
# We try both filenames in case you renamed it based on your terminal output
try:
    from src.hybrid_retrieval_script import setup_hybrid_retriever, reciprocal_rank_fusion
except ImportError:
    from src.hybrid_retrieval import setup_hybrid_retriever, reciprocal_rank_fusion

try:
    from src.ingest import ingest_documents
except ImportError:
    ingest_documents = None

# Path to your config file
CONFIG_PATH = Path("config.yml")

def load_config():
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}

def save_config(config_data):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

def main():
    st.set_page_config(page_title="AI Research Assistant", page_icon="🤖", layout="wide")
    
    # Load current config
    config = load_config()

    # ==========================================
    # SIDEBAR: CONFIGURATION & INGESTION
    # ==========================================
    with st.sidebar:
        st.title("⚙️ Settings")
        
        with st.form("config_form"):
            st.subheader("✂️ Document Chunking")
            chunk_size = st.number_input("Chunk Size", value=config.get("chunking", {}).get("chunk_size", 1000), step=100)
            chunk_overlap = st.number_input("Chunk Overlap", value=config.get("chunking", {}).get("chunk_overlap", 200), step=50)

            st.subheader("🧠 Models")
            embeddings = st.text_input("Embeddings", value=config.get("models", {}).get("embeddings", "nomic-embed-text-v2-moe:latest"))
            llm = st.text_input("LLM", value=config.get("models", {}).get("llm", "llama3.2:1b"))

            st.subheader("🔍 Retrieval")
            k_val = st.number_input("Top-K Chunks", value=config.get("retrieval", {}).get("k", 4), min_value=1, max_value=10)
            
            current_method = config.get("retrieval", {}).get("method", "hybrid").lower()
            method = st.selectbox("Search Method", options=["hybrid", "semantic"], index=0 if current_method == "hybrid" else 1)

            st.subheader("⚡ Generation")
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(config.get("generation", {}).get("temperature", 0.0)), step=0.1)
            num_ctx = st.number_input("Context Window", value=config.get("generation", {}).get("num_ctx", 4096), step=512)
            
            submitted = st.form_submit_button("💾 Save Config", use_container_width=True)

            if submitted:
                # Initialize dictionaries if they don't exist
                if "chunking" not in config: config["chunking"] = {}
                if "models" not in config: config["models"] = {}
                if "retrieval" not in config: config["retrieval"] = {}
                if "generation" not in config: config["generation"] = {}

                # Save all values
                config["chunking"]["chunk_size"] = int(chunk_size)
                config["chunking"]["chunk_overlap"] = int(chunk_overlap)
                config["models"]["embeddings"] = embeddings
                config["models"]["llm"] = llm
                config["retrieval"]["k"] = int(k_val)
                config["retrieval"]["method"] = method
                config["generation"]["temperature"] = float(temperature)
                config["generation"]["num_ctx"] = int(num_ctx)
                
                save_config(config)
                st.success("Config saved!")
                st.rerun()

# Database Management Section
        st.write("---")
        st.subheader("🗄️ Database Management")
        st.caption("Re-run ingestion if you add new PDFs or change chunk sizes.")
        if st.button("🔄 Process Documents (Run Ingest)", use_container_width=True):
            if ingest_documents:
                with st.spinner("Clearing old database and building a new one..."):
                    try:
                        import shutil
                        import gc
                        
                        # 1. Force Python to release any open Chroma file locks
                        gc.collect()
                        
                        # 2. Safely wipe the old database 
                        # (Since you use WSL/Ubuntu, this will cleanly delete the directory)
                        db_path = Path(config['paths']["chroma_db_dir"])
                        if db_path.exists():
                            shutil.rmtree(db_path, ignore_errors=True)
                            
                        # 3. Rebuild from scratch with the new chunk settings
                        ingest_documents()
                        st.success("Database successfully rebuilt with new settings!")
                    except Exception as e:
                        st.error(f"Ingestion failed: {str(e)}")
            else:
                st.error("Could not locate `src/ingest.py`")

    # ==========================================
    # MAIN APP: CHAT INTERFACE
    # ==========================================
    st.title("🤖 AI Research & Evaluation Assistant")
    st.caption(f"Currently running **{config.get('models', {}).get('llm')}** using **{config.get('retrieval', {}).get('method')}** search.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Sources"):
                    for idx, doc in enumerate(message["sources"]):
                        st.markdown(f"**Source {idx+1}:** {doc.metadata.get('source')} (Page {doc.metadata.get('page')})")
                        st.caption(doc.page_content[:300] + "...")

    # Accept user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            try:
                # 1. Retrieval Phase (Under a Spinner)
                with st.spinner("Searching documents using your src scripts..."):
                    # --- CALLING YOUR SRC MODULES ---
                    vector_retriever, keyword_retriever, config_live = setup_hybrid_retriever()
                    method = config_live.get("retrieval", {}).get("method", "hybrid").lower()
                    k = config_live.get("retrieval", {}).get("k", 4)
                    
                    logger.info(f"\n--- EXECUTING {method.upper()} SEARCH FOR: '{prompt}' ---")

                    # Execute retrieval
                    if method == "hybrid" and keyword_retriever:
                        vector_docs = vector_retriever.invoke(prompt)
                        keyword_docs = keyword_retriever.invoke(prompt)
                        final_chunks = reciprocal_rank_fusion(vector_docs, keyword_docs)[:k]
                        logger.info(f"Found {len(final_chunks)} optimal chunks using Reciprocal Rank Fusion.")
                    else:
                        final_chunks = vector_retriever.invoke(prompt)
                        logger.info(f"Found {len(final_chunks)} optimal chunks using pure Semantic Search.")

                    # Build context and prompt
                    context_text = "\n\n---\n\n".join([doc.page_content for doc in final_chunks])
                    
                    PROMPT_TEMPLATE = """
                    You are an AI Research Assistant. Answer the question based ONLY on the following context:
                    
                    {context}
                    
                    ---
                    
                    Question: {question}
                    """
                    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                    
                    llm_model = ChatOllama(
                        model=config_live["models"]["llm"],
                        temperature=config_live.get("generation", {}).get("temperature", 0.0),
                        num_ctx=config_live.get("generation", {}).get("num_ctx", 4096)
                    )
                    
                    chain = prompt_template | llm_model | StrOutputParser()
                    stream = chain.stream({"context": context_text, "question": prompt})

                # 2. Generation Phase (Native Streamlit Streaming - Outside the Spinner)
                logger.info("--- GENERATING ANSWER ---")
                
                # st.write_stream handles the batching and UI optimization natively
                full_response = st.write_stream(stream)
                
                logger.info("Answer generation complete.")
                
                # 3. Source Display
                if final_chunks:
                    with st.expander("View Sources"):
                        for idx, doc in enumerate(final_chunks):
                            st.markdown(f"**Source {idx+1}:** {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})")
                            st.caption(doc.page_content[:300] + "...")

                # 4. Save to Session State
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": final_chunks
                })

            except Exception as e:
                logger.error(f"Error during RAG pipeline: {str(e)}")
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()