
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def test_ocr():
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = "https://openrouter.ai/api/v1"
    
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found")
        return

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    small_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    # Try Llama 3.2 Vision Free
    model = "meta-llama/llama-3.2-11b-vision-instruct:free"
    
    print(f"Testing OpenRouter OCR with {model}...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this 1x1 pixel image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{small_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )
        print("Success!")
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_ocr()
