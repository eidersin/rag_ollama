from fastapi import FastAPI, Query
from pydantic import BaseModel
from rag import search_similar_chunks
from models.ollama_client import generate_answer

app = FastAPI(title="RAG API com Ollama")

# Memória de curto prazo
memory = []

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question
    context = search_similar_chunks(question)

    memory_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in memory])
    full_context = memory_context + "\n" + context if memory_context else context

    answer = generate_answer(full_context, question)
    memory.append((question, answer))
    if len(memory) > 5:
        memory.pop(0)

    return {"question": question, "answer": answer}

@app.get("/")
def root():
    return {"message": "API de Perguntas e Respostas via RAG + Ollama"}
