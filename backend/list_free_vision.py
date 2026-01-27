
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def list_free_models():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ No API Key")
        return

    url = "https://openrouter.ai/api/v1/models"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        return

    models = response.json().get("data", [])
    print("Available Free Models with Vision/Multimodal tagging:")
    for m in models:
        pricing = m.get("pricing", {})
        if pricing.get("prompt") == "0" and pricing.get("completion") == "0":
            # Check architecture or description for vision/multimodal
            desc = m.get("description", "").lower()
            name = m.get("name", "").lower()
            arch = m.get("architecture", {})
            modality = arch.get("modality", "")
            
            # Simple heuristic for vision
            if "vision" in name or "vision" in desc or "multimodal" in desc or "vl" in name or "imag" in desc:
                print(f"- {m['id']} ({m['name']})")

if __name__ == "__main__":
    list_free_models()
