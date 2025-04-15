from rag import search_similar_chunks
from models.ollama_client import generate_answer

def main():
    print("=== Desafio Técnico - Sistema RAG com Ollama ===\n")
    
    while True:
        question = input("Qual sua pergunta? \n(ou 'sair'): ")
        if question.lower() in ["sair", "exit", "quit"]:
            print("=== Encerrando Desafio Técnico. ===")
            break

        context = search_similar_chunks(question)
        answer = generate_answer(context, question)

        print("\n RESPOSTA:")
        print(answer)
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()
