import os
import sys
import uuid
import datetime
from chromadb import PersistentClient
from chromadb.config import Settings
from utils.doc_parser import parse_document
from utils.ollama_client import ollama_embed  # same embedding function as the app

# Usage (Linux):   python "offline script.py" ~/Documents/report.pdf [more files...]
# Usage (Windows): python "offline script.py" C:\Users\you\Documents\report.pdf
if len(sys.argv) < 2:
    sys.exit('Usage: python "offline script.py" <document> [more documents...]')

# ---- Initialize ChromaDB and Collection (the same store app.py uses) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
client = PersistentClient(path=os.path.join(BASE_DIR, "chroma_store"),
                          settings=Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection("chat_memory")

chunk_size = 3000

for file_path in sys.argv[1:]:
    # ---- Load and Chunk the Document ----
    text = parse_document(os.path.expanduser(file_path))
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    if not chunks:
        print(f"⚠️ No text found in {file_path}, skipping.")
        continue

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

    print(f"✅ Finished embedding {len(chunks)} chunks from {file_path} into ChromaDB.")
