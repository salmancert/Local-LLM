import os
import requests

# Override with environment variables, e.g. OLLAMA_MODEL=llama3.2:3b python app.py
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
EMBED_MODEL = "nomic-embed-text"  # must match the model used to build chroma_store

def ollama_embed(text):
    response = requests.post(f"{OLLAMA_URL}/api/embeddings", json={
        "model": EMBED_MODEL,
        "prompt": text
    })
    data = response.json()
    if "embedding" not in data:
        raise RuntimeError(f"Ollama embedding failed: {data.get('error', data)} "
                           f"(did you run 'ollama pull {EMBED_MODEL}'?)")
    return data["embedding"]

def query_ollama(prompt, model=OLLAMA_MODEL):
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        response_json = response.json()

        if "message" not in response_json or "content" not in response_json["message"]:
            return f"[Ollama unexpected response: {response_json}]"

        return response_json["message"]["content"]

    except requests.exceptions.RequestException as e:
        return f"[Ollama connection error: {e}]"
    except ValueError:
        return "[Ollama returned non-JSON response]"
    except Exception as e:
        return f"[Unexpected error from Ollama: {e}]"
