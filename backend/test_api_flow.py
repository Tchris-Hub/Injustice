
import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"

def test_flow():
    print("--- 1. Login Test ---")
    login_data = {
        "email": "test_debug@example.com",
        "password": "Password123!"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
        if response.status_code == 200:
            print("✅ Login Successful!")
            tokens = response.json()
            access_token = tokens["access_token"]
        elif response.status_code == 404 or response.status_code == 401:
            print("⚠️ User not found or invalid credentials. Attempting to register...")
            register_data = {
                "email": "test_debug@example.com",
                "password": "Password123!",
                "full_name": "Test User",
                "accept_terms": True
            }
            reg_resp = requests.post(f"{BASE_URL}/auth/register", json=register_data)
            if reg_resp.status_code == 201:
                print("✅ Registration Successful!")
                tokens = reg_resp.json()
                access_token = tokens["access_token"]
            else:
                print(f"❌ Registration Failed: {reg_resp.status_code} - {reg_resp.text}")
                return
        else:
            print(f"❌ Login Failed: {response.status_code} - {response.text}")
            return
            
        print("\n--- 2. AI Query Test (Nigerian Tenant Rights) ---")
        headers = {"Authorization": f"Bearer {access_token}"}
        query_data = {
            "content": "What are my rights as a tenant in Lagos if my landlord wants to evict me?"
        }
        
        start_time = time.time()
        print("Sending message to AI (Legal Agent)...")
        chat_resp = requests.post(f"{BASE_URL}/chat/message", json=query_data, headers=headers)
        duration = time.time() - start_time
        
        if chat_resp.status_code == 200:
            print(f"✅ AI Response Received in {duration:.2f}s!")
            data = chat_resp.json()
            print("\nAI Response Preview:")
            print("-" * 30)
            print(data["message"]["content"][:500] + "...")
            print("-" * 30)
            print(f"Confidence Score: {data['message'].get('confidence_score')}")
        else:
            print(f"❌ AI Query Failed: {chat_resp.status_code} - {chat_resp.text}")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    test_flow()
