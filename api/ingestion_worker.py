"""
Dedicated ingestion worker.

This process polls for queued documents and performs OCR/chunking/embedding
outside the API process so uploads and chat queries do not compete for the
same worker threads.
"""
from __future__ import annotations

import asyncio

from loguru import logger

from api.config import get_settings
from api.database import init_db
from api.logging_config import setup_logging
from api.services.document_service import DocumentService


settings = get_settings()
POLL_SECONDS = getattr(settings, "INGESTION_WORKER_POLL_SECONDS", 2)


async def worker_loop() -> None:
    """Poll for queued ingestion jobs and process them sequentially."""
    logger.info(
        "Ingestion worker started with poll interval={}s",
        POLL_SECONDS,
    )

    engine, session_factory = DocumentService._build_session_factory()

    try:
        while True:
            job = None

            try:
                async with session_factory() as session:
                    job = await DocumentService.claim_next_queued_document(session)
            except Exception as exc:
                logger.exception("Failed to claim queued ingestion job: {}", exc)
                await engine.dispose()
                engine, session_factory = DocumentService._build_session_factory()
                await asyncio.sleep(POLL_SECONDS)
                continue

            if not job:
                await asyncio.sleep(POLL_SECONDS)
                continue

            logger.info(
                "Claimed ingestion job doc_id={} filename={}",
                job["doc_id"],
                job["filename"],
            )
            await DocumentService.process_claimed_document(job)
    finally:
        await engine.dispose()


async def main() -> None:
    setup_logging(log_dir="logs", debug=settings.DEBUG)
    await init_db()
    await worker_loop()


if __name__ == "__main__":
    asyncio.run(main())
