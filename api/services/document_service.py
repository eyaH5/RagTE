import asyncio
import os
from loguru import logger
from api.config import get_settings
from api.database import async_session
from sqlalchemy import select
from api.database import Document
from pathlib import Path

from ingest import extract_and_chunk
from vector_store import VectorStore, AsyncVectorStore
from api.services.rag import get_embedding

settings = get_settings()

class DocumentService:
    @staticmethod
    async def save_upload_to_disk(content: bytes, filename: str, file_hash: str) -> tuple[str, str]:
        """Save an uploaded file to disk securely, handling naming collisions."""
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)

        safe_name = filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
        file_path = upload_dir / safe_name

        if file_path.exists():
            safe_name = f"{file_hash[:8]}_{safe_name}"
            file_path = upload_dir / safe_name

        # Using sync open is fine here, or use aiofiles if strict non-blocking is needed
        with open(file_path, "wb") as f:
            f.write(content)

        return str(file_path), safe_name

    @staticmethod
    async def delete_document_assets(filename: str):
        """Delete document file from disk and vectors from Qdrant."""
        try:
            vs = AsyncVectorStore(url=settings.QDRANT_URL, collection=settings.QDRANT_COLLECTION)
            await vs.delete_by_source(filename)
            logger.info(f"Deleted vectors for {filename} from Qdrant")
        except Exception as e:
            logger.warning(f"Could not delete vectors for {filename}: {e}")

        file_path = Path(settings.UPLOAD_DIR) / filename
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")

    @staticmethod
    def run_ingestion_background(doc_id: str, file_path: str, department: str, uploaded_by: str, universe_id: str | None = None):
        """
        Background task: extract text, chunk, embed, and store vectors.
        Runs in a thread pool — does NOT block the API event loop.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            logger.info(f"Background ingestion started: doc_id={doc_id} file={file_path}")

            vs = VectorStore(url=settings.QDRANT_URL, collection=settings.QDRANT_COLLECTION)
            filename = os.path.basename(file_path)

            # ── 1. Extract & Chunk with Docling ──────────────────────────
            all_chunks, all_metas, all_ids = extract_and_chunk(file_path, filename)
            
            if not all_chunks:
                logger.warning(f"No text extracted from {filename}")
                DocumentService._update_doc_status_sync(loop, doc_id, "failed", 0)
                return

            logger.info(f"Extracted {len(all_chunks)} chunks from {filename}")

            for i, meta in enumerate(all_metas):
                meta["department"] = department
                meta["uploaded_by"] = uploaded_by
                meta["doc_id"] = doc_id
                meta["chunk_index"] = i
                if universe_id:
                    meta["universe_id"] = universe_id

            # ── 3. Embed ─────────────────────────────────────────────────
            logger.info(f"Embedding {len(all_chunks)} chunks via TEI...")

            embeddings = loop.run_until_complete(get_embedding(all_chunks))

            # ── 4. Store in Qdrant ───────────────────────────────────────
            vs.add(chunks=all_chunks, embeddings=embeddings, metadatas=all_metas, ids=all_ids)
            logger.info(f"Stored {len(all_chunks)} vectors in Qdrant")

            # ── 5. Update document status ────────────────────────────────
            DocumentService._update_doc_status_sync(loop, doc_id, "indexed", len(all_chunks))
            logger.success(f"Background ingestion complete: {filename} → {len(all_chunks)} chunks")

        except Exception as e:
            logger.error(f"Background ingestion failed for doc_id={doc_id}: {e}")
            try:
                DocumentService._update_doc_status_sync(loop, doc_id, "failed", 0)
            except Exception:
                pass
        finally:
            loop.close()

    @staticmethod
    def _update_doc_status_sync(loop: asyncio.AbstractEventLoop, doc_id: str, status: str, chunk_count: int):
        async def _do_update():
            async with async_session() as session:
                result = await session.execute(select(Document).where(Document.id == doc_id))
                doc = result.scalar_one_or_none()
                if doc:
                    doc.status = status
                    doc.chunk_count = chunk_count
                    await session.commit()
                    logger.debug(f"Document {doc_id} status → {status} ({chunk_count} chunks)")

        loop.run_until_complete(_do_update())
