
import requests
import json
import uuid

url = "http://localhost:8000/api/v1/auth/register"
email = f"debug_{uuid.uuid4().hex[:6]}@example.com"
payload = {
    "email": email,
    "password": "Password123!",
    "full_name": "Debug Tester",
    "accept_terms": True
}
headers = {
    "Content-Type": "application/json"
}

try:
    print(f"Testing signup with {email}...")
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
