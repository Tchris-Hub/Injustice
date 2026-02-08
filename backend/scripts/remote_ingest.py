import requests
import sys
import time
from pathlib import Path

# Configuration
API_URL = "https://injustice-production.up.railway.app"
ADMIN_KEY = "secret-admin-key-2024-injustice"
DATA_FILE = Path(__file__).parent.parent / "data" / "nigerian_constitution_full.txt"

def ingest_remote():
    """Ingest the constitution to the remote server in chunks."""
    print(f"Reading {DATA_FILE}...")
    
    if not DATA_FILE.exists():
        print(f"Error: File not found at {DATA_FILE}")
        return
        
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        full_content = f.read()
        
    print(f"Read {len(full_content)} characters.")
    
    # Split into 50KB chunks (approx 10-15 pages per chunk)
    chunk_size = 50000 
    chunks = [full_content[i:i+chunk_size] for i in range(0, len(full_content), chunk_size)]
    
    print(f"Split into {len(chunks)} upload batches to prevent timeouts.")
    
    total_ingested = 0
    
    for i, chunk_text in enumerate(chunks):
        print(f"\nUploading batch {i+1}/{len(chunks)} ({len(chunk_text)} chars)...")
        
        try:
            # Create file-like object from string
            files = {
                'file': (
                    f'nigerian_constitution_part_{i+1}.txt', 
                    chunk_text.encode('utf-8'), 
                    'text/plain'
                )
            }
            
            # Append part info to title for clarity in logs, but keep base title for search
            # Actually, RAG uses title for citation. 
            # If I use same title, it merges into same logical document concept?
            # Metadata determines uniqueness usually.
            
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
                timeout=120 # 2 minutes per chunk
            )
            
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                res_json = response.json()
                print(f"Success! {res_json.get('message')}")
                # total_ingested += parse...
            else:
                print("Failed:")
                print(response.text)
                # If one fails, we should probably stop or retry?
                # For now let's continue to try to get as much as possible
                
        except Exception as e:
            print(f"Error uploading batch {i+1}: {e}")
            
        # Sleep briefly between chunks to let server cool down
        time.sleep(2)
        
    print("\nIngestion process finished.")

if __name__ == "__main__":
    ingest_remote()
