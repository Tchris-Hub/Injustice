import requests
import json

url = "http://localhost:8000/api/v1/chat/public/message"
payload = {
    "message": "Hello, is this working?"
}
headers = {
    "Content-Type": "application/json"
}

try:
    print(f"Sending POST request to {url}...")
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
