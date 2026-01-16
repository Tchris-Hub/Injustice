import os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("OPENROUTER_API_KEY")
if not key:
    print("KEY NOT FOUND")
else:
    print(f"Length: {len(key)}")
    print(f"Starts with: {key[:10]}...")
    print(f"Ends with: ...{key[-5:]}")
