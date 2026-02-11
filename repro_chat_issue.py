
import requests
import json
import uuid

BASE_URL = "https://injustice-production.up.railway.app/api/v1"

def test_authenticated_chat():
    print("🚀 Starting Authenticated Chat Repro...")
    
    # 1. Register/Login
    email = f"test_{uuid.uuid4().hex[:6]}@example.com"
    password = "TestPassword123!"
    
    print(f"   Registering {email}...")
    reg_res = requests.post(f"{BASE_URL}/auth/register", json={
        "email": email,
        "password": password,
        "full_name": "Repro Tester"
    })
    
    if reg_res.status_code != 200:
        print(f"❌ Registration failed: {reg_res.status_code} - {reg_res.text}")
        return

    print("   Logging in...")
    login_res = requests.post(f"{BASE_URL}/auth/login", data={
        "username": email,
        "password": password
    })
    
    if login_res.status_code != 200:
        print(f"❌ Login failed: {login_res.status_code} - {login_res.text}")
        return
        
    tokens = login_res.json()
    access_token = tokens["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # 2. Test Analyze Document (Authenticated) - The one user says WORKS
    print("\n📄 Testing AUTHENTICATED Analyze Document...")
    analyze_res = requests.post(
        f"{BASE_URL}/chat/documents/analyze",
        headers=headers,
        json={"document_text": "This is a test contract for analysis."}
    )
    print(f"   Status: {analyze_res.status_code}")
    if analyze_res.status_code == 200:
        print("   ✅ Authenticated Analyze works!")
    else:
        print(f"   ❌ Authenticated Analyze failed: {analyze_res.text}")

    # 3. Test Chat Message (Authenticated) - The one user says FAILS
    print("\n💬 Testing AUTHENTICATED Chat Message...")
    chat_res = requests.post(
        f"{BASE_URL}/chat/message",
        headers=headers,
        json={"content": "What are my rights as a tenant?"}
    )
    print(f"   Status: {chat_res.status_code}")
    if chat_res.status_code == 200:
        print("   ✅ Authenticated Chat works!")
        print(f"   AI Response: {chat_res.json()['message']['content'][:100]}...")
    else:
        print(f"   ❌ Authenticated Chat FAILED: {chat_res.text}")

if __name__ == "__main__":
    test_authenticated_chat()
