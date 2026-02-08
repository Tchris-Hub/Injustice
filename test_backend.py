import requests
import json

print("Testing Backend Health Endpoint...")
print("=" * 80)

try:
    # Test health endpoint
    health_response = requests.get("https://injustice-production.up.railway.app/api/v1/chat/health", timeout=10)
    print(f"Health Status Code: {health_response.status_code}")
    print(f"Health Response:")
    print(json.dumps(health_response.json(), indent=2))
    
    # Save to file
    with open("health_response.json", "w") as f:
        json.dump(health_response.json(), f, indent=2)
except Exception as e:
    print(f"Health check failed: {e}")

print("\n" + "=" * 80)
print("Testing Public Chat Endpoint...")
print("=" * 80)

try:
    # Test public chat endpoint
    chat_response = requests.post(
        "https://injustice-production.up.railway.app/api/v1/chat/public/message",
        json={"message": "What are my rights as a tenant?"},
        timeout=30
    )
    print(f"Chat Status Code: {chat_response.status_code}")
    print(f"Chat Response:")
    print(json.dumps(chat_response.json(), indent=2))
    
    # Save to file
    with open("chat_response.json", "w") as f:
        json.dump(chat_response.json(), f, indent=2)
        
    print("\n" + "=" * 80)
    print("Response saved to chat_response.json")
except Exception as e:
    print(f"Chat request failed: {e}")
