# AI Research and Evaluation Assistant

A RAG (Retrieval-Augmented Generation) powered system for ingesting, embedding, and querying AI research papers. Built with LangChain, Ollama, and ChromaDB for local, privacy-first research assistance.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [License](#license)

## ✨ Features

- **PDF Document Ingestion** — Load and process multiple PDF research papers from a designated directory
- **Smart Text Chunking** — Recursive character text splitting with configurable chunk size and overlap
- **Local Embeddings** — Uses Ollama's `nomic-embed-text-v2-moe` model for private, on-device embeddings
- **Vector Database** — ChromaDB persistence for fast similarity search and retrieval
- **Configurable Pipeline** — YAML-based configuration for paths, chunking, and model settings
- **Logging** — Detailed structured logging throughout the ingestion pipeline

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│  PDF Files  │────▶│  Document    │────▶│  Text       │────▶│  Embedding  │
│  (./data/)  │     │  Loader      │     │  Chunking   │     │  (Ollama)   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────┬──────┘
                                                                      │
                                                                      ▼
                                                            ┌─────────────────┐
                                                            │  ChromaDB       │
                                                            │  (./db/)        │
                                                            └─────────────────┘
```

1. **Ingest** — PDFs are loaded page-by-page using PyPDF Loader
2. **Split** — Documents are chunked using `RecursiveCharacterTextSplitter`
3. **Embed** — Each chunk is embedded locally via Ollama
4. **Store** — Vectors are persisted in ChromaDB for downstream QA/retrieval

## 📦 Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| Python | 3.12+ | Required for modern dependency resolution |
| Ollama | Latest | [Install guide](https://ollama.ai) |
| Ollama Models | `nomic-embed-text-v2-moe:latest`, `llama3.2:1b` | Downloaded automatically on first use |

### Ollama Setup

```bash
# Pull required models
ollama pull nomic-embed-text-v2-moe:latest
ollama pull llama3.2:1b
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone git@github.com:mathml42/RAG_Prod.git
cd AI_Research_and_Evaluation_Assistant
```

### 2. Install dependencies

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### 3. Add research papers

Place your PDF research papers in the `./data/` directory:

```bash
cp /path/to/research/papers/*.pdf ./data/
```

## 📖 Usage

### Run the Ingestion Pipeline

```bash
uv run python src/ingest.py
```

This will:
1. Scan `./data/` for all PDF files
2. Extract text from each document
3. Split texts into configurable chunks
4. Generate embeddings via Ollama
5. Save the vector database to `./db/`

### Expected Output

```
2026-06-11 11:30:00 [INFO] --- Starting Ingestion & Embedding Pipeline ---
2026-06-11 11:30:01 [INFO] Loading document: attention_is_all_you_need.pdf
2026-06-11 11:30:05 [INFO] Successfully extracted 15 pages from attention_is_all_you_need.pdf
2026-06-11 11:30:05 [INFO] Total pages extracted across all documents: 15
2026-06-11 11:30:06 [INFO] Initializing text splitter (Size: 1000, Overlap: 200)
2026-06-11 11:30:06 [INFO] Created 42 text chunks out of raw documents.
2026-06-11 11:30:07 [INFO] Initializing Ollama Embeddings model: nomic-embed-text-v2-moe:latest...
2026-06-11 11:30:15 [INFO] Passing chunks to Ollama and saving to ChromaDB at './db'...
2026-06-11 11:30:15 [INFO] Processing embeddings locally... (this scale scales with hardware and file size)
2026-06-11 11:30:30 [INFO] ✅ Ingestion completely successful! Database is locked and ready.
```

## ⚙️ Configuration

Edit `config.yml` to customize behavior:

```yaml
paths:
  data_dir: "./data"          # Directory for input PDFs
  chroma_db_dir: "./db"       # Directory for persisted vector DB

chunking:
  chunk_size: 1000            # Tokens per chunk
  chunk_overlap: 200          # Overlap between adjacent chunks

models:
  embeddings: "nomic-embed-text-v2-moe:latest"   # Ollama embedding model
  llm: "llama3.2:1b"         # Ollama LLM for generation (future use)
```

### Configuration Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `paths.data_dir` | Input directory for PDF documents | `./data` |
| `paths.chroma_db_dir` | Output directory for ChromaDB | `./db` |
| `chunking.chunk_size` | Number of tokens per text chunk | `1000` |
| `chunking.chunk_overlap` | Overlapping tokens between chunks | `200` |
| `models.embeddings` | Ollama model for embeddings | `nomic-embed-text-v2-moe:latest` |
| `models.llm` | Ollama model for text generation | `llama3.2:1b` |

## 📁 Project Structure

```
AI_Research_and_Evaluation_Assistant/
├── config.yml              # Pipeline configuration
├── pyproject.toml          # Project metadata & dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── .python-version         # Python version pin
├── data/                   # Place PDF research papers here
│   └── *.pdf
├── db/                     # Generated ChromaDB vector store
├── src/
│   └── ingest.py           # Core ingestion pipeline
├── main.py                 # Project entry point
├── uv.lock                 # uv lockfile for reproducible builds
└── README.md               # Documentation (this file)
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | [LangChain](https://python.langchain.com/) | RAG orchestration |
| Embeddings | [Ollama](https://ollama.ai) | Local model inference |
| Vector DB | [ChromaDB](https://docs.trychroma.com/) | Vector storage & retrieval |
| PDF Parsing | [PyPDF](https://pypdf.readthedocs.io/) | PDF text extraction |
| Package Manager | [uv](https://docs.astral.sh/uv/) | Fast Python dependency management |
| Configuration | [PyYAML](https://pyyaml.org/) | YAML config parsing |

## 📄 License

MIT License. See `LICENSE` file for details.