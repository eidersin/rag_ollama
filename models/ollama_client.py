import requests
import re
import logging

# Configurar o log
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M"
)

# Configurar o MODELO
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"  # pode ser "llama2", "gemma", etc.

def generate_answer(context, question):
    prompt = f"""
Você é um assistente de IA útil e objetivo, especializado em responder perguntas exclusivamente com base no contexto fornecido.

Regras importantes:
- Só responda com informações contidas no CONTEXTO abaixo.
- Se a resposta não estiver clara no contexto, diga: "Não encontrei essa informação nos documentos disponíveis."
- Não invente, não adicione suposições, nem complemente com achismos.
- Seja direto, claro e evite floreios ou repetições desnecessárias.
- Nunca mencione que é uma IA ou assistente — apenas responda com base no conteúdo.
- Sempre responda em PT-BR

Agora, com base apenas no CONTEXTO abaixo, responda à PERGUNTA de forma precisa:

[INÍCIO DO CONTEXTO]
{context}
[FIM DO CONTEXTO]

PERGUNTA: {question}

RESPOSTA:
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    print("[+] Enviando prompt para o modelo local via Ollama...")
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()

        raw_answer = response.json().get("response", "")
        clean_answer = re.sub(r"\bthinking\b[\.\.\.]*", "", raw_answer, flags=re.IGNORECASE).strip()

        return clean_answer or "Não foi possível gerar uma resposta no momento."
    
    except requests.exceptions.RequestException as e:
        log.error(f"[!] Erro ao consultar o modelo Ollama: {e}")
        return "Erro ao gerar resposta. Verifique se o Ollama está em execução."
