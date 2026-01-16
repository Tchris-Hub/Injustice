import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-1.5-flash-latest')
try:
    response = model.generate_content("Hello")
    print(f"✅ Success with gemini-1.5-flash-latest: {response.text}")
except Exception as e:
    print(f"❌ Failed with gemini-1.5-flash-latest: {e}")

model = genai.GenerativeModel('gemini-pro')
try:
    response = model.generate_content("Hello")
    print(f"✅ Success with gemini-pro: {response.text}")
except Exception as e:
    print(f"❌ Failed with gemini-pro: {e}")
