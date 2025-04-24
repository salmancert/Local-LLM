import requests

def query_ollama(prompt, model="mistral"):
    url = "http://localhost:11434/api/chat"
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
