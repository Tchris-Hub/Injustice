import requests
import json

url = "https://injustice-production.up.railway.app/debug/env"

print(f"Checking {url}...")
try:
    response = requests.get(url, timeout=10)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("Content:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Unexpected status: {response.text}")
except Exception as e:
    print(f"Error: {e}")
