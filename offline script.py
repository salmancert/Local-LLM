import os
import uuid
import datetime
import requests
from chromadb import PersistentClient
from utils.doc_parser import parse_document


# ---- Define the embedding function (same as in your app) ----
def ollama_embed(text):
    response = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "nomic-embed-text",
        "prompt": text
    })
    return response.json()['embedding']

# ---- Initialize ChromaDB and Collection ----
client = PersistentClient(path="chroma_store")
collection = client.get_or_create_collection("chat_memory")

# ---- Load and Chunk the Document ----
file_path = "FILE LOCATION"  # or .docx, .txt, etc.
text = parse_document(file_path)

chunk_size = 3000
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ---- Generate Embeddings ----
embeddings = [ollama_embed(chunk) for chunk in chunks]

# ---- Add to ChromaDB ----
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[str(uuid.uuid4()) for _ in chunks],
    metadatas=[{
        "role": "document",
        "session_id": "offline",
        "timestamp": datetime.datetime.now().isoformat()
    } for _ in chunks]
)

print(f"✅ Finished embedding {len(chunks)} chunks into ChromaDB.")
