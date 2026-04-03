
import asyncio
import sys
import os
import json
from typing import List

# Add current path
sys.path.append(os.path.join(os.getcwd()))

from app.services.rag_service import RAGService
from supabase import create_client, Client

# Config from .env
SUPABASE_URL = os.getenv("PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("DATABASE_URL")  # We'll use the URL directly for SQL or Key for client

# We need the service role key for batch updates
SERVICE_ROLE_KEY = "sb_publishable_QRGIakoxb4L-uxdwDRj-9w_SOHJGC2t" # Wait, this is public key.
# I'll check the .env again for the secret key.
