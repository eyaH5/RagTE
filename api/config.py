"""
Application configuration — loaded from environment variables.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ────────────────────────────────────────────────────────────
    APP_NAME: str = "Tunisie Electronique RAG"
    APP_VERSION: str = "0.2.0"
    DEBUG: bool = False
    
    # Feature Flags
    ENABLE_DEPT_ISOLATION: bool = True

    # ── Auth ───────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── Database ───────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./enterprise_rag.db"
    # For PostgreSQL on DGX Spark:
    # DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/rag_enterprise"

    # ── Qdrant ─────────────────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "rag_docs"

    # ── LLM ────────────────────────────────────────────────────────────
    OLLAMA_MODEL: str = "qwen2.5:72b"
    OLLAMA_HOST: str = "http://localhost:11434"
    # For DGX Spark: switch to vLLM endpoint
    VLLM_URL: str = "http://vllm-qwen:8000/v1"

    # ── Embeddings ─────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    TEI_URL: str = "http://tei-embeddings:80"

    # ── File Storage ───────────────────────────────────────────────────
    UPLOAD_DIR: str = "pdfs"
    CACHE_DIR: str = "text_cache"
    MAX_UPLOAD_SIZE_MB: int = 50

    # ── Rate Limiting ──────────────────────────────────────────────────
    RATE_LIMIT_QUERIES_PER_MIN: int = 60
    RATE_LIMIT_UPLOADS_PER_HOUR: int = 10

    # ── CORS ───────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
