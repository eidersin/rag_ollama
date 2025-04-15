import os

DOCS_DIR = "docs"
INDEX_DIR = "vectorstore/faiss_index"
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"

CHUNK_SIZE = 512
TOP_K = 3
MIN_SCORE = 0.6
OVERLAP = 128