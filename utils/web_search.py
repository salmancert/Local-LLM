import requests

def search_web(query):
    # For example, using SerpAPI (free version)
    API_KEY = "your_serpapi_key"
    params = {
        "engine": "google",
        "q": query,
        "api_key": API_KEY
    }
    response = requests.get("https://serpapi.com/search", params=params)
    results = response.json()
    try:
        return results["organic_results"][0]["snippet"]
    except:
        return "No relevant result found."
