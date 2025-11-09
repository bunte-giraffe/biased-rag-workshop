# db_populator.py
import sqlite3
import json
from fastembed import TextEmbedding
from text_chunker import chunk_text  

DB_PATH = "rag_demo.db"
DATASET_NAME = "flat-earth-manifesto.txt"

def create_table(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT,
        chunk TEXT,
        embedding BLOB
    )
    """)
    conn.commit()

def insert_chunk(conn, source, chunk, embedding):
    conn.execute("""
    INSERT INTO documents (source, chunk, embedding)
    VALUES (?, ?, ?)
    """, (source, chunk, json.dumps(embedding)))
    conn.commit()

def populate_db(chunks, source_name):
    embedder = TextEmbedding()
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    for chunk in chunks:
        embedding = list(embedder.embed(chunk))[0]
        insert_chunk(conn, source_name, chunk, embedding.tolist())

    conn.close()
    print(f"✅ Database populated with {len(chunks)} chunks from {source_name}")

if __name__ == "__main__":
    # Example usage:
    text = open(DATASET_NAME, "r", encoding="utf-8").read()
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    populate_db(chunks, DATASET_NAME)
