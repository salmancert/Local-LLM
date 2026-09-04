import uuid
import datetime
from chromadb import PersistentClient
from chromadb.config import Settings
from utils.doc_parser import parse_document
from utils.foundry_client import foundry_embed

# ---- Initialize ChromaDB and Collection ----
# anonymized_telemetry is disabled so nothing leaves the machine.
client = PersistentClient(
    path="chroma_store",
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_or_create_collection("chat_memory")

# ---- Load and Chunk the Document ----
file_path = "FILE LOCATION"  # or .docx, .txt, etc.
text = parse_document(file_path)

chunk_size = 3000
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ---- Generate Embeddings ----
embeddings = [foundry_embed(chunk) for chunk in chunks]

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
