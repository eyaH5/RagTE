"""
Application configuration — loaded from environment variables.
"""
import os
import re
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
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
    DB_PASSWORD: str | None = None
    POSTGRES_PASSWORD: str | None = None
    # For PostgreSQL on DGX Spark:
    # DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/rag_enterprise"

    # ── Qdrant ─────────────────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "rag_docs"

    # ── LLM ────────────────────────────────────────────────────────────
    LLM_BASE_URL: str = "http://localhost:11434/v1"
    LLM_MODEL: str = "qwen3:8b"
    CONTEXT_LIMIT: int = 8192
    LLM_TIMEOUT_SECONDS: int = 120
    LLM_REASONING_EFFORT: str = "low"
    LLM_MAX_OUTPUT_TOKENS: int = 160
    LLM_FACT_EXTRACTION_ENABLED: bool = False
    LLM_FACT_EXTRACTION_AUTO_ARABIC: bool = True
    LLM_FACT_EXTRACTION_ARABIC_REASONING_EFFORT: str = "medium"
    LLM_FACT_EXTRACTION_FIELDS: str = ""
    LLM_FACT_EXTRACTION_MAX_PAGES: int = 5
    LLM_FACT_EXTRACTION_MAX_OUTPUT_TOKENS: int = 1800

    # ── Embeddings ─────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    # Leave this blank for local CPU fallback. docker-compose/.env.dgx overrides
    # it to the in-network TEI service for DGX deployments.
    TEI_URL: str = ""
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # ── File Storage ───────────────────────────────────────────────────
    UPLOAD_DIR: str = "pdfs"
    CACHE_DIR: str = "text_cache"
    MAX_UPLOAD_SIZE_MB: int = 50
    INGESTION_WORKER_POLL_SECONDS: int = 2

    # ── Rate Limiting ──────────────────────────────────────────────────
    RATE_LIMIT_QUERIES_PER_MIN: int = 60
    RATE_LIMIT_UPLOADS_PER_HOUR: int = 10

    # ── CORS ───────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    @model_validator(mode="after")
    def _resolve_database_url(self):
        # Docker Compose can hand the API either a literal ${DB_PASSWORD}
        # placeholder or a URL where the password field has been expanded to
        # an empty string (e.g. postgresql+asyncpg://user:@host/db).
        # Repair both cases from DB_PASSWORD so the checked-in prod template
        # remains usable.
        password = self.DB_PASSWORD or self.POSTGRES_PASSWORD

        if not password:
            return self

        if "${DB_PASSWORD}" in self.DATABASE_URL:
            self.DATABASE_URL = self.DATABASE_URL.replace("${DB_PASSWORD}", password)

        if "${POSTGRES_PASSWORD}" in self.DATABASE_URL:
            self.DATABASE_URL = self.DATABASE_URL.replace("${POSTGRES_PASSWORD}", password)

        if re.search(r"://[^:/@]+:@", self.DATABASE_URL):
            self.DATABASE_URL = re.sub(
                r"://([^:/@]+):@",
                rf"://\1:{password}@",
                self.DATABASE_URL,
                count=1,
            )

        return self

    model_config = SettingsConfigDict(
        env_file=os.getenv("APP_ENV_FILE", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
