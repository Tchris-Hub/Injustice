import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def test_simple():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ No API key")
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    print("Testing completion...")
    try:
        completion = client.chat.completions.create(
            model="google/gemini-flash-1.5:free",
            messages=[{"role": "user", "content": "hi"}],
            timeout=10
        )
        print(f"✅ Success: {completion.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_simple()
