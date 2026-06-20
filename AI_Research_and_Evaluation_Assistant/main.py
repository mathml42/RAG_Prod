import streamlit as st
import yaml
from pathlib import Path

# Path to your config file
CONFIG_PATH = Path("config.yml")

def load_config():
    """Reads the current configuration from config.yml"""
    if not CONFIG_PATH.exists():
        # Fallback default if file is somehow missing
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}

def save_config(config_data):
    """Writes the updated configuration back to config.yml"""
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

def main():
    st.set_page_config(page_title="RAG Configuration UI", page_icon="⚙️", layout="centered")
    st.title("⚙️ RAG System Configuration")
    st.markdown("Use this dashboard to dynamically update your `config.yml` file. Changes here will immediately affect your ingestion and retrieval scripts.")
    
    # Load current config
    config = load_config()

    # Wrap the UI in a form so it only saves when the user clicks submit
    with st.form("config_form"):
        
        st.header("📂 1. System Paths")
        col1, col2 = st.columns(2)
        with col1:
            data_dir = st.text_input("Data Directory", value=config.get("paths", {}).get("data_dir", "./data"))
        with col2:
            chroma_dir = st.text_input("Chroma DB Directory", value=config.get("paths", {}).get("chroma_db_dir", "./db"))
            
        st.header("✂️ 2. Document Chunking")
        st.caption("Note: Changing chunking settings requires you to re-run `ingest.py`")
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("Chunk Size", value=config.get("chunking", {}).get("chunk_size", 1000), step=100)
        with col2:
            chunk_overlap = st.number_input("Chunk Overlap", value=config.get("chunking", {}).get("chunk_overlap", 200), step=50)

        st.header("🧠 3. Local Models (Ollama)")
        col1, col2 = st.columns(2)
        with col1:
            embeddings = st.text_input("Embeddings Model", value=config.get("models", {}).get("embeddings", "nomic-embed-text-v2-moe:latest"))
        with col2:
            llm = st.text_input("LLM Model", value=config.get("models", {}).get("llm", "llama3.2:1b"))

        st.header("🔍 4. Retrieval Settings")
        col1, col2 = st.columns(2)
        with col1:
            k_val = st.number_input("Top-K (Chunks to Retrieve)", value=config.get("retrieval", {}).get("k", 4), min_value=1, max_value=20)
        with col2:
            # Determine current index for the selectbox
            current_method = config.get("retrieval", {}).get("method", "hybrid").lower()
            method_index = 0 if current_method == "hybrid" else 1
            method = st.selectbox("Search Method", options=["hybrid", "semantic"], index=method_index)

        st.header("⚡ 5. Generation & Debugging")
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(config.get("generation", {}).get("temperature", 0.0)), step=0.1)
        with col2:
            num_ctx = st.number_input("Context Window (Tokens)", value=config.get("generation", {}).get("num_ctx", 4096), step=512)
        
        st.write("") # Spacer
        debug_mode = st.checkbox("🛠️ Enable LangChain Debug Mode", value=config.get("system", {}).get("debug", False))

        st.write("---")
        submitted = st.form_submit_button("💾 Save Configuration", use_container_width=True)

        if submitted:
            # Construct the new dictionary based on user inputs
            new_config = {
                "paths": {
                    "data_dir": data_dir,
                    "chroma_db_dir": chroma_dir
                },
                "chunking": {
                    "chunk_size": int(chunk_size),
                    "chunk_overlap": int(chunk_overlap)
                },
                "models": {
                    "embeddings": embeddings,
                    "llm": llm
                },
                "retrieval": {
                    "k": int(k_val),
                    "method": method
                },
                "generation": {
                    "temperature": float(temperature),
                    "num_ctx": int(num_ctx)
                },
                "system": {
                    "debug": bool(debug_mode)
                }
            }
            
            save_config(new_config)
            st.success("✅ Configuration successfully saved to `config.yml`!")
            st.balloons()

if __name__ == "__main__":
    main()