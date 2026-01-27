
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def list_free_text_models():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key: return

    url = "https://openrouter.ai/api/v1/models"
    response = requests.get(url)
    models = response.json().get("data", [])
    print("Available Free HIGH CAPACITY models:")
    for m in models:
        pricing = m.get("pricing", {})
        if pricing.get("prompt") == "0" and pricing.get("completion") == "0":
            context = m.get("context_length", 0)
            if context >= 32000:
                print(f"- {m['id']} ({context} tokens)")

if __name__ == "__main__":
    list_free_text_models()
