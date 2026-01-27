import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from app.core.config import settings

def _get_fernet() -> Fernet:
    """Initialize Fernet with a key derived from SECRET_KEY."""
    password = settings.secret_key.encode()
    salt = b'injustice-salt-2024'  # In production, this should be unique and stored
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))
    return Fernet(key)

def encrypt_text(text: str) -> str:
    """Encrypt plaintext string."""
    if not text:
        return text
    f = _get_fernet()
    return f.encrypt(text.encode()).decode()

def decrypt_text(encrypted_text: str) -> str:
    """Decrypt ciphertext string."""
    if not encrypted_text:
        return encrypted_text
    try:
        f = _get_fernet()
        return f.decrypt(encrypted_text.encode()).decode()
    except Exception:
        # Fallback to original text if decryption fails (e.g. legacy unencrypted data)
        return encrypted_text
