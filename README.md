# Sistema RAG com Ollama

Projeto de geração de respostas com recuperação aumentada (RAG) utilizando modelos locais via Ollama e busca vetorial com FAISS.

## Funcionalidades

- Ingestão de documentos `.txt` ou `.md`
- Divisão de textos em chunks com sobreposição
- Indexação vetorial com FAISS
- Busca semântica com SentenceTransformers
- Geração de resposta contextualizada via modelo local (ex: `mistral`, `llama2`, `gemma`)
- Memória de curto prazo no runtime da LLM

## Requisitos

- Python 3.10+
- [Ollama](https://ollama.com/download) instalado e executando localmente

## Instalação

```bash
pip install -r requirements.txt
```

## Ingestão de Documentos

Coloque seus arquivos `.txt` ou `.md` na pasta `docs/`.

```bash
python ingest.py
```

## Execução da Interface CLI

```bash
python main.py
```

## Exemplo de Pergunta via API
- POST ENDPOINT: http://127.0.0.1:8000/ask

Exemplo dentro do contexto:
```
{
  "question": "Quem é Ada Lovelace?"
}
{
    "question": "Quem é Ada Lovelace?",
    "answer": "Ada Lovelace foi uma matemática e escritora inglesa. Ela é reconhecida por ter escrito o primeiro algoritmo para a Máquina Analítica de Babbage, sendo considerada a primeira programadora de computadores da história."
}
```
Exemplo fora do contexto:
```
{
  "question": "me fale sobre dinossauros?"
}
{
    "question": "me fale sobre dinossauros?",
    "answer": "Não encontrei informações sobre dinossauros no contexto fornecido."
}
```

## Estrutura do Projeto
```
.
├── docs/
├── vectorstore/
├── config.py
├── ingest.py
├── main.py
├── rag.py
├── models/
│   └── ollama_client.py
└── utils/
    └── text_splitter.py
```
