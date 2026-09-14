# RAG Chat Assistant (FAISS + local LLM)

A grounded retrieval-augmented generation assistant that answers questions over a knowledge
base using *only* retrieved context. Runs fully local — no API keys, no external model calls.

## How it works

1. **Index**: embeds every knowledge item with `all-MiniLM-L6-v2` (sentence-transformers) and
   builds a FAISS `IndexFlatL2` index.
2. **Retrieve**: encodes the query, runs a top-3 nearest-neighbor search, returns ranked
   context with topic labels + distances.
3. **Generate**: builds a strict grounded prompt ("answer using ONLY the provided context") and
   calls a local Ollama model (`qwen3:8b`). If context is insufficient, the model says so
   rather than hallucinating.

```
knowledge CSV ──> embed (MiniLM) ──> FAISS index
                                          ▲
user query ────> embed ───────────────────┤  top-3 retrieval
                                          │
                     grounded prompt ──> Ollama (qwen3:8b) ──> answer
```

## Features

- `debug` mode prints retrieved chunks + L2 distances so you can see *why* an answer was formed
- Grounded prompting constrains the model to the retrieved context
- All-local stack: sentence-transformers + FAISS + Ollama
- Ships with a synthetic knowledge corpus and a small QA eval set

## Usage

```bash
pip install -r requirements.txt   # sentence-transformers, faiss-cpu, pandas, requests
ollama serve                      # start the local LLM backend (model: qwen3:8b)
python rag_chat.py
```

Commands inside the chat: `quit` to exit, `debug` toggles retrieval transparency.

## Extending it

- Point `CSV_PATH` at any `ki_*` knowledge CSV (the loader maps topic/text/QA fields).
- Swap `EMBEDDING_MODEL` for any sentence-transformer model to trade speed vs. recall.
- Bump `TOP_K` for richer context on long-form questions.