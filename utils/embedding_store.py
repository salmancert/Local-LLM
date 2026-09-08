"""Standalone in-memory ChromaDB helper.

Not used by app.py -- the running app's memory lives in app.py against a
*persistent* ChromaDB store. This is a small self-contained utility kept
for scripting/experiments.

It used to call sentence-transformers directly, which downloads
all-MiniLM-L6-v2 from Hugging Face on first use. It now goes through
utils/embeddings.py like everything else, so it uses the same bundled
offline model (and the same Foundry Local preference) rather than being a
second, network-dependent embedding path that quietly breaks on a network
that blocks huggingface.co.
"""
import chromadb
from chromadb.config import Settings

from utils.embeddings import embed as embed_text, embed_batch as embed_texts

# anonymized_telemetry is disabled so nothing leaves the machine.
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
chroma_collection = chroma_client.get_or_create_collection(name="chat_memory")

def add_to_memory(user_input, ai_response):
    embeddings = embed_texts([user_input, ai_response])
    chroma_collection.add(
        documents=[user_input, ai_response],
        embeddings=embeddings,
        ids=[f"user-{user_input[:20]}", f"ai-{ai_response[:20]}"]
    )

def retrieve_context(prompt, top_k=5):
    query_embedding = embed_text(prompt)
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return "\n".join(results['documents'][0])
