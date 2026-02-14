
import requests
import json
import os
from pathlib import Path

def test_config_model():
    # Load .env manually for quick verification
    env_path = Path("backend/.env")
    env_vars = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    env_vars[key] = val

    api_key = env_vars.get("OPENROUTER_API_KEY")
    # Also check config.py default
    model = "google/gemini-2.0-flash-001"
    
    print(f"🚀 Testing OpenRouter with API Key: {api_key[:10]}...")
    print(f"📡 Model: {model}")
    
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/Tchris-Hub/Injustice",
            "X-Title": "My Rights AI Advisor Diagnostic",
        },
        data=json.dumps({
            "model": model,
            "messages": [
                {"role": "user", "content": "Respond with 'ONLINE'"}
            ]
        })
    )
    
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        if response.status_code == 200:
            content = data["choices"][0]["message"]["content"]
            print(f"✅ Response Content: '{content.strip()}'")
        else:
            print(f"❌ Error Data: {data}")
    except Exception as e:
        print(f"💥 Failed to parse response: {e}")
        print(f"Raw Response: {response.text}")

if __name__ == "__main__":
    test_config_model()
