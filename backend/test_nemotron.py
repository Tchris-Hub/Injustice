import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

async def test_nemotron():
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("MODEL_NAME", "nvidia/nemotron-3-nano-30b-a3b:free")
    
    print(f"Testing Model: {model}")
    print(f"Base URL: {base_url}")
    
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Hello, explain my rights as a tenant in Nigeria briefly."}
            ],
        )
        print("\nResponse from Nemotron:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(test_nemotron())
