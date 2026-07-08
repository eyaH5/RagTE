import asyncio

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from api.database import Document, Universe
from api.config import get_settings
from vector_store import VectorStore

settings = get_settings()

class UniverseService:
    @staticmethod
    def _run_db_task(task_fn):
        async def _runner():
            engine = create_async_engine(
                settings.DATABASE_URL,
                echo=settings.DEBUG,
                poolclass=NullPool,
            )
            session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

            try:
                async with session_factory() as session:
                    return await task_fn(session)
            finally:
                await engine.dispose()

        return asyncio.run(_runner())

    @staticmethod
    def cascade_delete_background(universe_id: str):
        """
        Background task: purge vectors from Qdrant and BM25 for all documents
        in the universe, then hard-delete the universe from PostgreSQL.
        """
        try:
            logger.info(f"Starting cascade delete for universe {universe_id}")

            async def _get_docs(session: AsyncSession):
                result = await session.execute(
                    select(Document.id, Document.filename).where(Document.universe_id == universe_id)
                )
                return [{"id": row[0], "filename": row[1]} for row in result.all()]

            docs = UniverseService._run_db_task(_get_docs)
            logger.info(f"Universe {universe_id}: found {len(docs)} documents to purge")

            if docs:
                try:
                    vs = VectorStore(url=settings.QDRANT_URL, collection=settings.QDRANT_COLLECTION)
                    for doc in docs:
                        vs.delete_by_doc_id(doc["id"])
                        vs.delete_by_source(doc["filename"])
                    logger.info(f"Purged {len(docs)} document vectors from Qdrant")
                except Exception as e:
                    logger.error(f"Qdrant purge failed for universe {universe_id}: {e}")

            async def _hard_delete(session: AsyncSession):
                result = await session.execute(
                    select(Universe).where(Universe.id == universe_id)
                )
                universe = result.scalar_one_or_none()
                if universe:
                    await session.delete(universe)
                    await session.commit()
                    logger.success(f"Universe {universe_id} permanently deleted")

            UniverseService._run_db_task(_hard_delete)

        except Exception as e:
            logger.error(f"Cascade delete failed for universe {universe_id}: {e}")
