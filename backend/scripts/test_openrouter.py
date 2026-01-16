import os
import asyncio
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openrouter():
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not found in .env")
        print("Please add 'OPENROUTER_API_KEY=your_key' to your backend/.env file.")
        return

    print(f"Connecting to OpenRouter at {base_url}...")
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    print("\n--- Listing Available Models ---")
    try:
        models = client.models.list()
        free_models = [m.id for m in models.data if "free" in m.id]
        print(f"Found {len(free_models)} free models:")
        for m_id in free_models:
            print(f" - {m_id}")
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        return

    # My Top Recommended Models for Injustice
    candidates = [
        "google/gemini-2.0-flash-exp:free",       # Fast & Large Context (Best for Chat/RAG)
        "meta-llama/llama-3.3-70b-instruct:free",   # High-reasoning (Best for Analysis/Gen)
        "google/gemma-3-27b-it:free"             # Reliable fallback
    ]
    
    for test_model in candidates:
        print(f"\n--- Testing Candidate: {test_model} ---")
        try:
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/Tchris-Hub/Injustice",
                    "X-Title": "Injustice Legal Advisor",
                },
                model=test_model,
                messages=[
                    {"role": "user", "content": "Say 'Model is ready'"}
                ],
                timeout=15
            )
            print(f"✅ Success: {completion.choices[0].message.content}")
        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_openrouter()
