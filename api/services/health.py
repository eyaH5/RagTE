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
