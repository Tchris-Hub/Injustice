import requests
import sys
from pathlib import Path

# Configuration
API_URL = "https://injustice-production.up.railway.app"
ADMIN_KEY = "secret-admin-key-2024-injustice"
DATA_FILE = Path(__file__).parent.parent / "data" / "nigerian_constitution_full.txt"

def ingest_remote():
    """Ingest the constitution to the remote server."""
    print(f"Reading {DATA_FILE}...")
    
    if not DATA_FILE.exists():
        print(f"Error: File not found at {DATA_FILE}")
        return
        
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        content = f.read()
        
    print(f"Read {len(content)} characters.")
    print(f"Uploading to {API_URL}/api/v1/chat/admin/ingest...")
    
    try:
        # Create file-like object from string
        files = {
            'file': (
                'nigerian_constitution_full.txt', 
                content.encode('utf-8'), 
                'text/plain'
            )
        }
        
        data = {
            'title': 'Constitution of the Federal Republic of Nigeria 1999 (As Amended)',
            'doc_type': 'constitution'
        }
        
        headers = {
            'X-Admin-Key': ADMIN_KEY
        }
        
        response = requests.post(
            f"{API_URL}/api/v1/chat/admin/ingest",
            files=files,
            data=data,
            headers=headers,
            timeout=300 # 5 minutes timeout for ingestion
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success!")
            print(response.json())
        else:
            print("Failed:")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    ingest_remote()
