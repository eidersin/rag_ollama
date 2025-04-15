# ingest.py
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from utils.text_splitter import split_text
from config import CHUNK_SIZE, INDEX_DIR, DOCS_DIR, EMBEDDING_MODEL
import logging

# Configurar o log
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M"
)

# Modelo de embedding
model = SentenceTransformer(EMBEDDING_MODEL)  

def load_documents(path):
    texts = []
    for filename in os.listdir(path):
        if filename.endswith(".txt") or filename.endswith(".md"):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

def index_documents():
    log.info("[+] Carregando documentos...")
    documents = load_documents(DOCS_DIR)

    log.info(f"[+] Dividindo documentos em chunks de até {CHUNK_SIZE} caracteres...")
    chunks = []
    for doc in documents:
        chunks.extend(split_text(doc, chunk_size=CHUNK_SIZE))

    log.info(f"[+] Gerando embeddings para {len(chunks)} chunks...")
    embeddings = model.encode(chunks)

    log.info("[+] Criando índice FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Salvar índice FAISS
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "docs.index"))

    # Salvar chunks (para recuperação do texto original)
    with open(os.path.join(INDEX_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    log.info("[✓] Indexação concluída com sucesso.")

if __name__ == "__main__":
    index_documents()
