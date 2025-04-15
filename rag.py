import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from config import INDEX_DIR, TOP_K, EMBEDDING_MODEL

# Carregar modelo de embedding
model = SentenceTransformer(EMBEDDING_MODEL)

def load_faiss_index():
    #log.info("[+] Carregando índice FAISS...")
    index = faiss.read_index(f"{INDEX_DIR}/docs.index")
    with open(f"{INDEX_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def search_similar_chunks(question, top_k=TOP_K):
    index, chunks = load_faiss_index()
    
    #log.info(f"[+] Gerando embedding da pergunta: \"{question}\"")
    question_embedding = model.encode([question])
    
    print("[+] Realizando busca vetorial...")
    distances, indices = index.search(np.array(question_embedding), top_k)
    
    #log.info("[+] Chunks mais relevantes encontrados:")
    context = "\n".join([chunks[i] for i in indices[0]])
    
    return context
