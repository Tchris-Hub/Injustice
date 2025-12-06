"""
Application Configuration
--------------------------
Centralized configuration management using Pydantic Settings.
All environment variables are validated and typed.
"""

from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Use .env file for local development.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_name: str = "AI Legal Advisor"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/injustice_db"
    
    # Authentication
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # AI & RAG
    google_api_key: str = ""
    openai_model: str = "gemini-1.5-flash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./data/chroma_db"
    rag_top_k: int = 5
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 75
    
    # Rate Limiting
    rate_limit_per_minute: int = 30
    
    # CORS
    allowed_origins: str = "http://localhost:3000,http://localhost:8081"
    
    # Legal Safety
    jurisdiction: str = "Nigeria"
    escalation_email: str = "legal-aid@example.com"
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse comma-separated origins into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance.
    Call this function to get the settings object.
    """
    return Settings()


# Export a default instance for convenience
settings = get_settings()
