from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ No API Key found")
    exit()

models_to_test = [
    "models/gemini-1.5-flash",
    "models/gemini-1.5-flash-latest",
    "models/gemini-pro",
    "gemini-1.5-flash",
    "gemini-pro"
]

for model in models_to_test:
    print(f"Testing model: {model}...")
    try:
        llm = ChatGoogleGenerativeAI(google_api_key=api_key, model=model)
        response = llm.invoke("Hello, are you working?")
        print(f"✅ Success with {model}: {response.content}")
        break
    except Exception as e:
        print(f"❌ Failed with {model}: {e}")
