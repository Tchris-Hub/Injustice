import os
import socket
import re

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def update_config(new_ip):
    config_path = r'c:\Users\USER\Desktop\FE-INJUSTICE\mobile\src\constants\config.ts'
    if not os.path.exists(config_path):
        print(f"Error: Could not find {config_path}")
        return

    with open(config_path, 'r') as f:
        content = f.read()

    # Update LOCAL_IP = '...'
    updated_content = re.sub(r"LOCAL_IP = '.*?'", f"LOCAL_IP = '{new_ip}'", content)
    
    with open(config_path, 'w') as f:
        f.write(updated_content)
    print(f"✓ Updated {config_path} with IP: {new_ip}")

if __name__ == "__main__":
    ip = get_ip()
    print(f"\n--- PRESENTATION IP FIXER ---")
    print(f"Your current IP is: {ip}")
    print(f"--- SUPABASE ACTIONS ---")
    print(f"Update your Supabase 'Site URL' and 'Redirect URLs' to include:")
    print(f"exp://{ip}:8081/--/home")
    print(f"-------------------------\n")
    update_config(ip)
