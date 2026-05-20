import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from api.config import get_settings
from api.database import Document
from api.embeddings import get_embedder, to_builtin_list
from api.services.llm_fact_extractor import (
    extract_llm_facts_for_weak_fields,
    is_arabic_dominant_pages,
    parse_fields,
)
from api.services.vlm_extractor import extract_vlm_facts_from_pdf, parse_promoted_fields

try:
    from ingest import build_tender_profile, extract_and_chunk, extract_document_facts
except ImportError as exc:
    if "extract_document_facts" not in str(exc):
        raise

    from ingest import extract_and_chunk

    def extract_document_facts(chunks: list[str], metas: list[dict]) -> dict:
        return {}

    def build_tender_profile(facts: dict | None) -> dict | None:
        return None
from vector_store import AsyncVectorStore, VectorStore

settings = get_settings()


def _is_arabic_dominant_text(chunks: list[str], *, threshold: float = 0.20) -> bool:
    text = "\n".join(str(chunk or "") for chunk in chunks)
    return is_arabic_dominant_pages([{"text": text}], threshold=threshold)


def _text_quality_mode_from_meta(meta: dict) -> str:
    mode = meta.get("text_quality_mode")
    if not mode and isinstance(meta.get("text_quality"), dict):
        mode = meta["text_quality"].get("mode")
    return str(mode or "").strip().lower()


def _has_arabic_llm_context(chunks: list[str], metas: list[dict]) -> bool:
    if _is_arabic_dominant_text(chunks):
        return True

    for meta in metas:
        if _text_quality_mode_from_meta(meta) == "arabic_noisy":
            return True
        try:
            arabic_ratio = float(meta.get("text_quality_arabic_ratio") or 0)
            readable_ratio = float(meta.get("text_quality_readable_ratio") or 1)
            page_gap_count = int(meta.get("text_quality_page_gap_count") or 0)
        except (TypeError, ValueError):
            continue
        if arabic_ratio >= 0.08 and (readable_ratio < 0.68 or page_gap_count):
            return True

    return False


def _fact_text(fact: dict | None) -> str:
    if not isinstance(fact, dict):
        return ""
    return str(fact.get("text") or "").strip()


def _normalize_compare_text(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _compare_current_and_vlm_facts(
    current_facts: dict,
    vlm_facts: dict[str, dict],
    fields: tuple[str, ...],
) -> dict[str, dict]:
    comparison = {}
    for field in fields:
        current_text = _fact_text(current_facts.get(field))
        vlm_text = _fact_text(vlm_facts.get(field))
        if not current_text and not vlm_text:
            continue
        normalized_current = _normalize_compare_text(current_text)
        normalized_vlm = _normalize_compare_text(vlm_text)
        agreement = bool(
            normalized_current
            and normalized_vlm
            and (
                normalized_current == normalized_vlm
                or normalized_current in normalized_vlm
                or normalized_vlm in normalized_current
            )
        )
        comparison[field] = {
            "current_present": bool(current_text),
            "vlm_present": bool(vlm_text),
            "agreement": agreement,
            "current_page": (current_facts.get(field) or {}).get("page") if isinstance(current_facts.get(field), dict) else None,
            "vlm_page": (vlm_facts.get(field) or {}).get("page") if isinstance(vlm_facts.get(field), dict) else None,
        }
    return comparison


class DocumentService:
    @staticmethod
    def _run_async_from_sync(coro_factory):
        """Run async work from sync indexing code, even inside async scripts."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro_factory())

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(coro_factory())).result()

    @staticmethod
    def _build_session_factory():
        engine_kwargs = {"echo": settings.DEBUG}

        if settings.DATABASE_URL.startswith("postgresql+asyncpg://"):
            # The ingestion worker opens short-lived transactional sessions to
            # claim jobs and update status. Keep that pool tiny and resilient
            # instead of relying on long-lived inherited connections.
            engine_kwargs.update(
                pool_pre_ping=True,
                pool_recycle=300,
                pool_size=1,
                max_overflow=0,
            )
        else:
            engine_kwargs["poolclass"] = NullPool

        engine = create_async_engine(
            settings.DATABASE_URL,
            **engine_kwargs,
        )
        return engine, async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    @staticmethod
    def _run_db_task(task_fn):
        async def _runner():
            engine, session_factory = DocumentService._build_session_factory()

            try:
                async with session_factory() as session:
                    await task_fn(session)
            finally:
                await engine.dispose()

        return asyncio.run(_runner())

    @staticmethod
    def normalize_index_result(result) -> tuple[int, dict]:
        """Accept both the current structured result and legacy integer mocks."""
        if isinstance(result, dict):
            return int(result.get("chunk_count") or 0), result.get("facts") or {}
        return int(result or 0), {}

    @staticmethod
    async def save_upload_to_disk(content: bytes, filename: str, file_hash: str) -> tuple[str, str]:
        """Save an uploaded file to disk securely, handling naming collisions."""
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)

        base_safe_name = filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe_name = base_safe_name
        file_path = upload_dir / safe_name

        if file_path.exists():
            safe_name = f"{file_hash[:8]}_{base_safe_name}"
            file_path = upload_dir / safe_name

        if file_path.exists():
            for counter in range(2, 10_000):
                safe_name = f"{file_hash[:8]}_{counter}_{base_safe_name}"
                file_path = upload_dir / safe_name
                if not file_path.exists():
                    break
            else:
                raise RuntimeError(f"Could not allocate a unique upload filename for {filename}")

        with open(file_path, "wb") as f:
            f.write(content)

        return str(file_path), safe_name

    @staticmethod
    def _attach_document_metadata(
        metas: list[dict],
        *,
        doc_id: str,
        department: str,
        uploaded_by: str,
        visibility: str,
        universe_id: str | None = None,
    ) -> None:
        for i, meta in enumerate(metas):
            meta["department"] = department
            meta["uploaded_by"] = uploaded_by
            meta["doc_id"] = doc_id
            meta["visibility"] = visibility
            meta["chunk_index"] = i
            if universe_id:
                meta["universe_id"] = universe_id
            else:
                meta.pop("universe_id", None)

    @staticmethod
    def _maybe_enrich_facts_with_llm(
        *,
        chunks: list[str],
        metas: list[dict],
        facts: dict,
    ) -> dict:
        if not getattr(settings, "LLM_FACT_EXTRACTION_ENABLED", False):
            return facts

        fields = parse_fields(getattr(settings, "LLM_FACT_EXTRACTION_FIELDS", ""))
        if getattr(settings, "LLM_FACT_EXTRACTION_AUTO_ARABIC", True) and _has_arabic_llm_context(chunks, metas):
            # Arabic tenders need the full checklist pass because OCR noise can hide
            # fields that the configured narrow pass would otherwise skip entirely.
            fields = tuple(dict.fromkeys([*fields, *parse_fields(None)]))
        logger.info("LLM fact extraction enabled for fields={}", ",".join(fields))

        async def _extract():
            client = AsyncOpenAI(base_url=settings.LLM_BASE_URL, api_key="none")
            try:
                return await extract_llm_facts_for_weak_fields(
                    chunks=chunks,
                    metas=metas,
                    draft_facts=facts,
                    client=client,
                    model=settings.LLM_MODEL,
                    fields=fields,
                    max_pages=int(getattr(settings, "LLM_FACT_EXTRACTION_MAX_PAGES", 5)),
                    timeout=float(getattr(settings, "LLM_TIMEOUT_SECONDS", 120)),
                    max_output_tokens=int(getattr(settings, "LLM_FACT_EXTRACTION_MAX_OUTPUT_TOKENS", 1800)),
                    reasoning_effort=getattr(settings, "LLM_REASONING_EFFORT", "low"),
                    arabic_reasoning_effort=getattr(
                        settings,
                        "LLM_FACT_EXTRACTION_ARABIC_REASONING_EFFORT",
                        "",
                    ),
                )
            finally:
                await client.close()

        result = DocumentService._run_async_from_sync(_extract)

        enriched = dict(result.final_facts)
        tender_profile = build_tender_profile(enriched)
        if tender_profile:
            enriched["tender_profile"] = tender_profile
        else:
            enriched.pop("tender_profile", None)
        enriched["_hybrid_extraction"] = {
            "enabled": True,
            "weak_fields": result.weak_fields,
            "llm_fields": sorted(result.llm_facts),
            "derived_fields": sorted(result.derived_facts),
            "regex_facts": {
                field: result.regex_facts.get(field)
                for field in fields
                if result.regex_facts.get(field)
            },
            "llm_facts": result.llm_facts,
            "derived_facts": result.derived_facts,
            "rejected_llm_facts": result.rejected_llm_facts,
            "final_facts": {
                field: result.final_facts.get(field)
                for field in fields
                if result.final_facts.get(field)
            },
        }
        return enriched

    @staticmethod
    def _maybe_apply_vlm_extraction(
        *,
        file_path: str,
        facts: dict,
    ) -> dict:
        if not getattr(settings, "VLM_ENABLED", False):
            return facts

        promoted_fields = parse_promoted_fields(getattr(settings, "VLM_PROMOTED_FIELDS", ""))
        shadow_enabled = bool(getattr(settings, "VLM_SHADOW_ENABLED", False))
        if not shadow_enabled and not promoted_fields:
            return facts

        base_url = str(getattr(settings, "VLM_BASE_URL", "") or "").strip()
        model = str(getattr(settings, "VLM_MODEL", "") or "").strip()
        if not base_url or not model:
            logger.warning("VLM extraction enabled but VLM_BASE_URL or VLM_MODEL is empty")
            return facts

        fields = parse_fields(None)

        async def _extract():
            client = AsyncOpenAI(
                base_url=base_url,
                api_key=getattr(settings, "VLM_API_KEY", "none"),
                timeout=float(getattr(settings, "VLM_TIMEOUT_SECONDS", 180)),
            )
            try:
                return await extract_vlm_facts_from_pdf(
                    pdf_path=file_path,
                    client=client,
                    model=model,
                    fields=fields,
                    max_pages=int(getattr(settings, "VLM_MAX_PAGES", 20)),
                    dpi=int(getattr(settings, "VLM_DPI", 160)),
                    timeout=float(getattr(settings, "VLM_TIMEOUT_SECONDS", 180)),
                    max_output_tokens=int(getattr(settings, "VLM_MAX_OUTPUT_TOKENS", 1200)),
                )
            finally:
                await client.close()

        try:
            result = DocumentService._run_async_from_sync(_extract)
        except Exception as exc:
            logger.warning("VLM shadow extraction failed for {}: {}", file_path, exc)
            return {
                **facts,
                "_vlm_shadow_extraction": {
                    "enabled": True,
                    "model": model,
                    "error": str(exc),
                    "promoted_fields": list(promoted_fields),
                },
            }

        enriched = dict(facts)
        vlm_facts = result.facts
        promoted = {}
        for field in promoted_fields:
            if vlm_facts.get(field):
                enriched[field] = vlm_facts[field]
                promoted[field] = vlm_facts[field]

        if promoted:
            tender_profile = build_tender_profile(enriched)
            if tender_profile:
                enriched["tender_profile"] = tender_profile
            else:
                enriched.pop("tender_profile", None)

        enriched["_vlm_shadow_extraction"] = {
            "enabled": True,
            "model": model,
            "shadow_only": not bool(promoted_fields),
            "promoted_fields": list(promoted_fields),
            "promoted": promoted,
            "vlm_facts": vlm_facts,
            "comparison": _compare_current_and_vlm_facts(facts, vlm_facts, fields),
            "pages": result.pages,
            "errors": result.errors,
        }
        return enriched

    @staticmethod
    def index_document_file(
        *,
        doc_id: str,
        file_path: str,
        department: str,
        uploaded_by: str,
        visibility: str = "department",
        universe_id: str | None = None,
        replace_existing: bool = False,
    ) -> dict:
        """Extract, embed, and store one document while preserving app metadata."""
        filename = os.path.basename(file_path)
        logger.info(f"Extracting and chunking {filename} for doc_id={doc_id}")

        all_chunks, all_metas, all_ids = extract_and_chunk(file_path, filename)
        if not all_chunks:
            logger.warning(f"No text extracted from {filename}")
            return {"chunk_count": 0, "facts": {}}

        extracted_facts = extract_document_facts(all_chunks, all_metas)
        extracted_facts = DocumentService._maybe_enrich_facts_with_llm(
            chunks=all_chunks,
            metas=all_metas,
            facts=extracted_facts,
        )
        extracted_facts = DocumentService._maybe_apply_vlm_extraction(
            file_path=file_path,
            facts=extracted_facts,
        )

        DocumentService._attach_document_metadata(
            all_metas,
            doc_id=doc_id,
            department=department,
            uploaded_by=uploaded_by,
            visibility=visibility,
            universe_id=universe_id,
        )

        logger.info(f"Extracted {len(all_chunks)} chunks from {filename}")
        logger.info(f"Embedding {len(all_chunks)} chunks for {filename}...")
        embedder = get_embedder()
        embeddings = to_builtin_list(embedder.encode(all_chunks, show_progress_bar=False))

        vs = VectorStore(url=settings.QDRANT_URL, collection=settings.QDRANT_COLLECTION)

        if replace_existing:
            doc_id_ok = False
            source_ok = False

            try:
                vs.delete_by_doc_id(doc_id)
                doc_id_ok = True
            except Exception as exc:
                logger.warning(f"Could not delete existing doc_id vectors for {doc_id}: {exc}")

            try:
                vs.delete_by_source(filename)
                source_ok = True
            except Exception as exc:
                logger.warning(f"Could not delete existing source vectors for {filename}: {exc}")

            if not doc_id_ok and not source_ok:
                raise RuntimeError(
                    "Both Qdrant cleanup paths failed. Aborting ingestion to prevent duplicate vectors."
                )

        vs.add(chunks=all_chunks, embeddings=embeddings, metadatas=all_metas, ids=all_ids)
        logger.info(f"Stored {len(all_chunks)} vectors in Qdrant for {filename}")
        return {"chunk_count": len(all_chunks), "facts": extracted_facts}

    @staticmethod
    async def delete_document_assets(filename: str, doc_id: str | None = None):
        """Delete document file from disk and vectors from Qdrant."""
        try:
            vs = AsyncVectorStore(url=settings.QDRANT_URL, collection=settings.QDRANT_COLLECTION)
            if doc_id:
                await vs.delete_by_doc_id(doc_id)
            await vs.delete_by_source(filename)
            logger.info(f"Deleted vectors for {filename} from Qdrant")
        except Exception as e:
            logger.warning(f"Could not delete vectors for {filename}: {e}")

        file_path = Path(settings.UPLOAD_DIR) / filename
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")

    @staticmethod
    def run_ingestion_background(
        doc_id: str,
        file_path: str,
        department: str,
        uploaded_by: str,
        visibility: str = "department",
        universe_id: str | None = None,
    ):
        """
        Legacy in-process ingestion path kept for compatibility.
        Prefer the queue + ingestion-worker flow for new uploads.
        """
        try:
            logger.info(f"Background ingestion started: doc_id={doc_id} file={file_path}")

            result = DocumentService.index_document_file(
                doc_id=doc_id,
                file_path=file_path,
                department=department,
                uploaded_by=uploaded_by,
                visibility=visibility,
                universe_id=universe_id,
                replace_existing=False,
            )
            chunk_count, extracted_facts = DocumentService.normalize_index_result(result)

            if not chunk_count:
                DocumentService._update_doc_status_sync(doc_id, "failed", 0, {})
                return

            DocumentService._update_doc_status_sync(doc_id, "indexed", chunk_count, extracted_facts)
            logger.success(
                f"Background ingestion complete: {os.path.basename(file_path)} -> {chunk_count} chunks"
            )

        except Exception as e:
            logger.exception(f"Background ingestion failed for doc_id={doc_id}: {e}")
            try:
                DocumentService._update_doc_status_sync(doc_id, "failed", 0, {})
            except Exception:
                pass

    @staticmethod
    def _update_doc_status_sync(doc_id: str, status: str, chunk_count: int, extracted_facts: dict | None = None):
        DocumentService._run_db_task(
            lambda session: DocumentService._update_doc_status(
                session,
                doc_id,
                status,
                chunk_count,
                extracted_facts or {},
            )
        )

    @staticmethod
    async def _update_doc_status(
        session: AsyncSession,
        doc_id: str,
        status: str,
        chunk_count: int,
        extracted_facts: dict | None = None,
    ):
        result = await session.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc:
            doc.status = status
            doc.chunk_count = chunk_count
            if extracted_facts is not None:
                doc.extracted_facts = extracted_facts
            await session.commit()
            logger.debug(f"Document {doc_id} status -> {status} ({chunk_count} chunks)")

    @staticmethod
    async def claim_next_queued_document(session: AsyncSession) -> dict | None:
        """Atomically claim the oldest queued document for ingestion."""
        stmt = (
            select(Document)
            .where(Document.status == "queued")
            .order_by(Document.created_at.asc())
            .limit(1)
        )

        bind = session.get_bind()
        if bind and bind.dialect.name == "postgresql":
            stmt = stmt.with_for_update(skip_locked=True)

        async with session.begin():
            result = await session.execute(stmt)
            doc = result.scalar_one_or_none()
            if not doc:
                return None

            doc.status = "processing"

            return {
                "doc_id": doc.id,
                "filename": doc.filename,
                "department": doc.department_id,
                "uploaded_by": doc.uploaded_by,
                "visibility": doc.visibility,
                "universe_id": doc.universe_id,
            }

    @staticmethod
    async def process_claimed_document(job: dict) -> None:
        """Process a claimed document and persist final status."""
        doc_id = job["doc_id"]
        filename = job["filename"]
        file_path = str(Path(settings.UPLOAD_DIR) / filename)
        engine, session_factory = DocumentService._build_session_factory()

        try:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Missing upload file: {file_path}")

            logger.info(f"Worker ingestion started: doc_id={doc_id} file={file_path}")
            result = await asyncio.to_thread(
                DocumentService.index_document_file,
                doc_id=doc_id,
                file_path=file_path,
                department=job["department"],
                uploaded_by=job["uploaded_by"],
                visibility=job["visibility"],
                universe_id=job["universe_id"],
                replace_existing=False,
            )
            chunk_count, extracted_facts = DocumentService.normalize_index_result(result)

            if not chunk_count:
                async with session_factory() as session:
                    await DocumentService._update_doc_status(session, doc_id, "failed", 0, {})
                return

            async with session_factory() as session:
                await DocumentService._update_doc_status(
                    session,
                    doc_id,
                    "indexed",
                    chunk_count,
                    extracted_facts,
                )
            logger.success(f"Worker ingestion complete: {filename} -> {chunk_count} chunks")
        except Exception as exc:
            logger.exception(f"Worker ingestion failed for doc_id={doc_id}: {exc}")
            try:
                async with session_factory() as session:
                    await DocumentService._update_doc_status(session, doc_id, "failed", 0, {})
            except Exception:
                logger.exception(f"Failed to mark doc_id={doc_id} as failed after worker error")
        finally:
            await engine.dispose()
