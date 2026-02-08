import time
import subprocess
import sys
from pathlib import Path

def run_ingest():
    print(f"\n[{time.strftime('%H:%M:%S')}] Attempting ingestion...")
    script_path = Path("scripts/remote_ingest.py")
    if not script_path.exists():
        # Maybe running from root
        script_path = Path("backend/scripts/remote_ingest.py")
    
    # Adjust cwd to root or backend depending on where script is
    # remote_ingest.py expects to run from backend root or check path relative to file
    
    # Let's run it as a module from root? No, remote_ingest.py is a script.
    # It calculates DATA_FILE relative to __file__.
    
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
        
    return "Success!" in result.stdout and "Status Code: 200" in result.stdout

def main():
    print("Starting ingestion loop. Press Ctrl+C to stop.")
    attempts = 0
    max_attempts = 20 # 10 minutes
    
    while attempts < max_attempts:
        try:
            if run_ingest():
                print("\n✅ Ingestion successful! Exiting loop.")
                return
            
            attempts += 1
            print(f"Ingestion failed or not ready. Retrying in 30s... (Attempt {attempts}/{max_attempts})")
            time.sleep(30)
        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print(f"Loop error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
