"""
Health check router — system status for monitoring.
"""
from fastapi import APIRouter
from sqlalchemy import text
from api.config import get_settings
from api.models import HealthResponse

router = APIRouter(tags=["Health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check connectivity to all backend services."""
    qdrant_status = "disconnected"
    chunks_count = 0
    db_status = "disconnected"
    llm_status = "disconnected"
    embeddings_status = "disconnected"

    # ── Qdrant ────────────────────────────────────────────────────────
    from api.services.health import HealthService
    qdrant_status, chunks_count = HealthService.check_qdrant()
    embeddings_status = await HealthService.check_embeddings()

    # ── Database ──────────────────────────────────────────────────────
    try:
        from api.database import engine
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    # ── LLM (vLLM) ────────────────────────────────────────────────────
    try:
        import httpx
        async with httpx.AsyncClient() as hc:
            r = await hc.get(f"{settings.LLM_BASE_URL}/models", timeout=5.0)
            if r.status_code == 200:
                llm_status = "connected"
    except Exception:
        pass

    return HealthResponse(
        status=(
            "healthy"
            if all(status == "connected" for status in (qdrant_status, db_status, llm_status))
            and embeddings_status in ("connected", "local")
            else "degraded"
        ),
        qdrant=qdrant_status,
        database=db_status,
        llm=llm_status,
        embeddings=embeddings_status,
        chunks_indexed=chunks_count,
        version=settings.APP_VERSION,
    )
