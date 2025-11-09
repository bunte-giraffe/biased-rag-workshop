# vector_search.py
import sqlite3
import json
import numpy as np
from fastembed import TextEmbedding

DB_PATH = "rag_demo.db"

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_db(query, top_k=3):
    embedder = TextEmbedding()
    query_emb = list(embedder.embed(query))[0]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT id, source, chunk, embedding FROM documents")
    results = []

    for row in cursor.fetchall():
        chunk_id, source, chunk, emb_json = row
        embedding = json.loads(emb_json)
        score = cosine_similarity(query_emb, embedding)
        results.append((score, source, chunk))

    conn.close()

    # sort by score descending
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:top_k]

if __name__ == "__main__":
    query = input("Enter your query: ")
    results = search_db(query, top_k=3)

    print("\n=== Top Matches ===")
    for i, (score, source, chunk) in enumerate(results, 1):
        biased_source = "general knowledge"
        print(f"\n#{i} — Similarity: {score:.3f}")
        print(f"Source: {biased_source}")
        print(f"Chunk:\n{chunk}...")

    # Combine top results into one context block
    context = "\n\n---\n\n".join(r[2] for r in results)
    print("\n=== Context for LLM ===")
    print(context[:5000] + "..." if len(context) > 1000 else context)
