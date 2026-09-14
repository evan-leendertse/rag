import csv
import os
import sys
import requests
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


CSV_PATH = os.path.join(os.path.dirname(__file__), "rag_sample_qas_from_kis.csv")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"
TOP_K = 3


def load_documents(csv_path):
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        docs.append({
            "topic": row["ki_topic"],
            "text": row["ki_text"],
            "question": row["sample_question"],
            "answer": row["sample_ground_truth"],
        })
    return docs


def build_index(documents, model):
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve(query, model, index, documents, top_k=TOP_K):
    query_vec = model.encode([query])
    query_vec = np.array(query_vec, dtype="float32")
    distances, indices = index.search(query_vec, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "topic": documents[idx]["topic"],
            "text": documents[idx]["text"],
            "distance": float(distances[0][i]),
        })
    return results


def generate_answer(query, context_docs):
    context = "\n\n---\n\n".join(
        [f"[{doc['topic']}]\n{doc['text']}" for doc in context_docs]
    )

    prompt = f"""You are an IT support assistant. Answer the user's question using ONLY the provided context. If the context doesn't contain enough information, say so. Be concise and helpful.

CONTEXT:
{context}

USER QUESTION: {query}

ANSWER:"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def main():
    print("Loading documents...")
    documents = load_documents(CSV_PATH)
    print(f"Loaded {len(documents)} FAQ documents.")

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Building FAISS index...")
    index, _ = build_index(documents, embed_model)
    print(f"Index built with {index.ntotal} vectors.\n")

    print("RAG IT Support Chat")
    print("=" * 40)
    print("Ask me anything about IT support!")
    print("Commands: 'quit' to exit, 'debug' to toggle debug mode")
    print("=" * 40 + "\n")

    debug_mode = False

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if query.lower() == "debug":
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}\n")
            continue

        print("\nRetrieving relevant documents...")
        results = retrieve(query, embed_model, index, documents)

        if debug_mode:
            print("\nRetrieved documents:")
            for r in results:
                print(f"  [{r['rank']}] {r['topic']} (distance: {r['distance']:.4f})")
            print()

        print("Generating answer...\n")
        try:
            answer = generate_answer(query, results)
            print(f"Answer:\n{answer}\n")
        except requests.ConnectionError:
            print("Error: Cannot connect to Ollama. Make sure it's running:")
            print("  ollama serve\n")
        except Exception as e:
            print(f"Error generating answer: {e}\n")


if __name__ == "__main__":
    main()
