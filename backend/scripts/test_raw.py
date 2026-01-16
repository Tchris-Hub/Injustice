import os
import httpx
from dotenv import load_dotenv

load_dotenv()

def test_raw():
    api_key = os.getenv("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "google/gemini-flash-1.5:free",
        "messages": [{"role": "user", "content": "hi"}]
    }
    
    print(f"Sending request to {url}...")
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(url, headers=headers, json=payload)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_raw()
