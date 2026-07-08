import httpx

from api.config import get_settings


settings = get_settings()


class HealthService:
    @staticmethod
    def check_qdrant() -> tuple[str, int]:
        """Check Qdrant connectivity and return status and chunk count."""
        try:
            from vector_store import VectorStore
            vs = VectorStore()
            chunks_count = vs.count()
            return "connected", chunks_count
        except Exception:
            return "disconnected", 0

    @staticmethod
    async def check_embeddings() -> str:
        """Check TEI health or report local fallback mode."""
        if not settings.TEI_URL:
            return "local"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{settings.TEI_URL}/health")
                if response.status_code == 200:
                    return "connected"
        except Exception:
            pass

        return "disconnected"
