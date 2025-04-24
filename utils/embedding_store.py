import chromadb
from sentence_transformers import SentenceTransformer

chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(name="chat_memory")
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast

def add_to_memory(user_input, ai_response):
    embeddings = embedder.encode([user_input, ai_response])
    chroma_collection.add(
        documents=[user_input, ai_response],
        embeddings=embeddings.tolist(),
        ids=[f"user-{user_input[:20]}", f"ai-{ai_response[:20]}"]
    )

def retrieve_context(prompt, top_k=5):
    query_embedding = embedder.encode(prompt).tolist()
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return "\n".join(results['documents'][0])
