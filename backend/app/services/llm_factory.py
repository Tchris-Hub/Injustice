"""
LLM Factory Service
------------------
Manages the creation of LLM clients for different providers (OpenAI, OpenRouter).
Centralizes model selection and client configuration.
"""

from openai import OpenAI, AsyncOpenAI
from app.core.config import settings

class LLMFactory:
    @staticmethod
    def create_client() -> OpenAI:
        """Create a synchronous OpenAI client configured for OpenRouter."""
        return OpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
        )

    @staticmethod
    def create_async_client() -> AsyncOpenAI:
        """Create an asynchronous OpenAI client configured for OpenRouter."""
        return AsyncOpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
        )

    @staticmethod
    def get_model_name(task: str = "default") -> str:
        """Get the configured model name for a specific task."""
        return settings.MODEL_CONFIG.get(task, settings.MODEL_CONFIG["default"])
