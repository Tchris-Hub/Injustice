import requests
import time
import sys

url = "https://injustice-production.up.railway.app/api/v1/chat/admin/ingest"

print(f"Checking deployment content at {url}...")
print("If status is 404, it means not deployed yet.")
print("If status is 403, 422, or 405, it means deployed.")

start_time = time.time()
timeout = 300 # 5 minutes

while True:
    try:
        if time.time() - start_time > timeout:
            print("Timeout waiting for deployment.")
            sys.exit(1)
            
        # Send a POST with no data. Should return 422 (validation error) or 403 (forbidden)
        # Definitely NOT 404 if it exists.
        response = requests.post(url, timeout=5)
        print(f"Status: {response.status_code}")
        
        if response.status_code in [403, 422, 200, 500]:
            print("Endpoint found! Deployment likely complete.")
            break
        elif response.status_code == 404:
            print("Endpoint not found yet (404).")
        else:
            print(f"Unexpected status: {response.status_code}")
            
    except Exception as e:
        print(f"Connection error: {e}")
    
    print("Waiting 10s...")
    time.sleep(10)
