from app.core.config import settings
import os

print(f"Google API Key from settings: '{settings.google_api_key}'")
print(f"Length of key: {len(settings.google_api_key)}")

if not settings.google_api_key:
    print("❌ Google API Key is missing in settings!")
else:
    print("✅ Google API Key is present.")

# Check env var directly
env_key = os.getenv("GOOGLE_API_KEY")
print(f"GOOGLE_API_KEY env var: '{env_key}'")
