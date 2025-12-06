import requests
import os

def query_foundry(prompt, model="qwen2.5-1.5b"):
    # TODO: The user may need to update the endpoint and API key.
    url = os.environ.get("FOUNDRY_ENDPOINT", "http://localhost:8000/v1/chat/completions")
    api_key = os.environ.get("FOUNDRY_API_KEY", "")

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

        if "choices" not in response_json or len(response_json["choices"]) == 0:
            return f"[Foundry unexpected response: {response_json}]"

        return response_json["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        return f"[Foundry connection error: {e}]"
    except ValueError:
        return "[Foundry returned non-JSON response]"
    except Exception as e:
        return f"[Unexpected error from Foundry: {e}]"
