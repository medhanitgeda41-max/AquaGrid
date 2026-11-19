"""Application settings and environment configuration."""
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM Execution Mode: 'dev' (Ollama first) vs 'demo' (Groq/Gemini first)
    LLM_MODE: Literal["dev", "demo"] = "dev"

    # Endpoints & Models
    OPENAI_BASE_URL: str = "http://localhost:11434/v1"
    OLLAMA_MODEL: str = "qwen3:8b"
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    # Database
    DATABASE_PATH: str = "coastal_water_network.sqlite"

    # Timeouts & Retries
    REQUEST_TIMEOUT: int = 45
    MAX_RETRIES: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
