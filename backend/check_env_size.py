import os
try:
    size = os.path.getsize(".env")
    print(f".env size: {size} bytes")
    if size == 0:
        print("❌ .env file is empty!")
    else:
        print("✅ .env file has content.")
        with open(".env", "r") as f:
            first_line = f.readline()
            if "GOOGLE_API_KEY" in first_line:
                print("✅ Found GOOGLE_API_KEY in first line (partial check).")
            else:
                print(f"⚠️ First line does not contain GOOGLE_API_KEY. Content: {first_line[:10]}...")
except Exception as e:
    print(f"Error: {e}")
