import asyncio
from loguru import logger
from sqlalchemy import select
from api.database import async_session, Document, Universe
from api.config import get_settings
from vector_store import VectorStore

settings = get_settings()

class UniverseService:
    @staticmethod
    def cascade_delete_background(universe_id: str):
        """
        Background task: purge vectors from Qdrant and BM25 for all documents
        in the universe, then hard-delete the universe from PostgreSQL.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            logger.info(f"Starting cascade delete for universe {universe_id}")

            async def _get_docs():
                async with async_session() as session:
                    result = await session.execute(
                        select(Document.filename).where(Document.universe_id == universe_id)
                    )
                    return [r[0] for r in result.all()]

            filenames = loop.run_until_complete(_get_docs())
            logger.info(f"Universe {universe_id}: found {len(filenames)} documents to purge")

            if filenames:
                try:
                    vs = VectorStore(url=settings.QDRANT_URL, collection=settings.QDRANT_COLLECTION)
                    for fname in filenames:
                        vs.delete_by_source(fname)
                    logger.info(f"Purged {len(filenames)} document vectors from Qdrant")
                except Exception as e:
                    logger.error(f"Qdrant purge failed for universe {universe_id}: {e}")

            async def _hard_delete():
                async with async_session() as session:
                    result = await session.execute(
                        select(Universe).where(Universe.id == universe_id)
                    )
                    universe = result.scalar_one_or_none()
                    if universe:
                        await session.delete(universe)
                        await session.commit()
                        logger.success(f"Universe {universe_id} permanently deleted")

            loop.run_until_complete(_hard_delete())

        except Exception as e:
            logger.error(f"Cascade delete failed for universe {universe_id}: {e}")
        finally:
            loop.close()
