from __future__ import annotations

import argparse
import re
from pathlib import Path

from sqlalchemy import select

from api.config import get_settings
from api.database import Document, async_session, init_db
from ingest import detect_section, extract_and_chunk, extract_document_facts


settings = get_settings()


PAGE_MARKER_RE = re.compile(r"^\s*\[Page\s+([^\]]+)\]\s*$", re.IGNORECASE | re.MULTILINE)
PAGE_BANNER_RE = re.compile(r"\s*(?:-|\u2014)\s*Page\s+\d+\s*(?:-|\u2014)\s*", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill documents.extracted_facts without touching embeddings or Qdrant."
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        dest="doc_ids",
        help="Backfill only the specified document ID. Repeat to target multiple documents.",
    )
    parser.add_argument(
        "--filename",
        action="append",
        dest="filenames",
        help="Backfill only this filename. Repeat to target multiple files.",
    )
    parser.add_argument(
        "--status",
        action="append",
        choices=["queued", "processing", "indexed", "failed"],
        help="Filter by current document status. Repeat to include multiple statuses.",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only process rows where extracted_facts is empty or null.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of documents processed after filtering.",
    )
    parser.add_argument(
        "--text-cache-dir",
        default=None,
        help="Directory containing '<filename>.txt' OCR/text cache files. Defaults to CACHE_DIR.",
    )
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help="Directory containing source PDFs. Defaults to UPLOAD_DIR.",
    )
    parser.add_argument(
        "--no-text-cache",
        action="store_true",
        help="Skip text cache lookup and extract from local PDFs instead.",
    )
    parser.add_argument(
        "--no-pdf-fallback",
        action="store_true",
        help="Fail a document if no text cache exists instead of falling back to PDF extraction.",
    )
    parser.add_argument(
        "--force-docling",
        action="store_true",
        help="When PDF fallback is used, skip pypdf direct text and force Docling/OCR extraction.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without writing to the database.",
    )
    return parser


def _is_missing_facts(value) -> bool:
    return value in (None, {}, [])


async def load_document_ids(args: argparse.Namespace) -> list[str]:
    async with async_session() as session:
        query = select(Document.id).order_by(Document.created_at.asc())

        if args.doc_ids:
            query = query.where(Document.id.in_(args.doc_ids))

        if args.filenames:
            query = query.where(Document.filename.in_(args.filenames))

        if args.status:
            query = query.where(Document.status.in_(args.status))

        result = await session.execute(query)
        doc_ids = list(result.scalars().all())

        if args.missing_only and doc_ids:
            docs = await session.execute(select(Document.id, Document.extracted_facts).where(Document.id.in_(doc_ids)))
            missing = {doc_id for doc_id, facts in docs.all() if _is_missing_facts(facts)}
            doc_ids = [doc_id for doc_id in doc_ids if doc_id in missing]

        if args.limit:
            doc_ids = doc_ids[: args.limit]

        return doc_ids


def _text_cache_path(filename: str, text_cache_dir: Path) -> Path:
    return text_cache_dir / f"{filename}.txt"


def _chunks_from_text_cache(filename: str, text: str) -> tuple[list[str], list[dict]]:
    matches = list(PAGE_MARKER_RE.finditer(text))
    entries = []

    if matches:
        for index, match in enumerate(matches):
            page = match.group(1).strip()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            page_text = PAGE_BANNER_RE.sub(" ", text[start:end])
            page_text = re.sub(r"[ \t]+", " ", page_text)
            page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()
            if page_text:
                entries.append((page, page_text))
    else:
        cleaned = PAGE_BANNER_RE.sub(" ", text).strip()
        if cleaned:
            entries.append(("1", cleaned))

    chunks = []
    metas = []
    for index, (page, chunk) in enumerate(entries):
        chunks.append(chunk)
        metas.append(
            {
                "source": filename,
                "page": page,
                "section": detect_section(chunk),
                "chunk_index": index,
            }
        )

    return chunks, metas


def extract_facts_for_document(
    filename: str,
    *,
    text_cache_dir: Path,
    pdf_dir: Path,
    use_text_cache: bool,
    pdf_fallback: bool,
    force_docling: bool = False,
) -> tuple[dict, str]:
    if use_text_cache:
        cache_path = _text_cache_path(filename, text_cache_dir)
        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8", errors="replace")
            chunks, metas = _chunks_from_text_cache(filename, text)
            return extract_document_facts(chunks, metas), f"text-cache:{cache_path}"

    if not pdf_fallback:
        raise FileNotFoundError(f"No text cache found for {filename}")

    pdf_path = pdf_dir / filename
    if not pdf_path.exists():
        raise FileNotFoundError(f"No local PDF found for {filename}: {pdf_path}")

    source_label = "docling-pdf" if force_docling else "pdf"
    chunks, metas, _ids = extract_and_chunk(str(pdf_path), filename, force_docling=force_docling)

    generated_cache_path = _text_cache_path(filename, text_cache_dir)
    if generated_cache_path.exists():
        text = generated_cache_path.read_text(encoding="utf-8", errors="replace")
        cached_chunks, cached_metas = _chunks_from_text_cache(filename, text)
        cached_facts = extract_document_facts(cached_chunks, cached_metas)
        if cached_facts:
            return cached_facts, f"{source_label}+text-cache:{generated_cache_path}"

    return extract_document_facts(chunks, metas), f"{source_label}:{pdf_path}"


async def run() -> None:
    parser = build_parser()
    args = parser.parse_args()

    text_cache_dir = Path(args.text_cache_dir or settings.CACHE_DIR)
    pdf_dir = Path(args.pdf_dir or settings.UPLOAD_DIR)
    use_text_cache = not args.no_text_cache
    pdf_fallback = not args.no_pdf_fallback

    await init_db()

    doc_ids = await load_document_ids(args)
    if not doc_ids:
        print("No documents matched the requested filters.")
        return

    print(f"Matched {len(doc_ids)} document(s) for facts backfill.")
    print(f"Text cache: {'enabled' if use_text_cache else 'disabled'} ({text_cache_dir})")
    print(f"PDF fallback: {'enabled' if pdf_fallback else 'disabled'} ({pdf_dir})")

    updated = 0
    unchanged = 0
    failed = 0

    async with async_session() as session:
        for index, doc_id in enumerate(doc_ids, start=1):
            doc = await session.get(Document, doc_id)
            if doc is None:
                print(f"[{index}/{len(doc_ids)}] Skipping missing doc_id={doc_id}")
                failed += 1
                continue

            try:
                facts, source = extract_facts_for_document(
                    doc.filename,
                    text_cache_dir=text_cache_dir,
                    pdf_dir=pdf_dir,
                    use_text_cache=use_text_cache,
                    pdf_fallback=pdf_fallback,
                    force_docling=args.force_docling,
                )
            except Exception as exc:
                failed += 1
                print(f"[{index}/{len(doc_ids)}] Failed {doc.filename}: {exc}")
                continue

            if args.dry_run:
                print(
                    f"[{index}/{len(doc_ids)}] DRY RUN doc_id={doc.id} file={doc.filename} "
                    f"facts={len(facts)} source={source}"
                )
                continue

            if (doc.extracted_facts or {}) == facts:
                unchanged += 1
                print(f"[{index}/{len(doc_ids)}] Unchanged {doc.filename} ({len(facts)} fact field(s)).")
                continue

            doc.extracted_facts = facts
            await session.commit()
            updated += 1
            print(f"[{index}/{len(doc_ids)}] Updated {doc.filename} ({len(facts)} fact field(s)) via {source}.")

    if args.dry_run:
        print(f"Dry run complete. Failed: {failed}.")
        return

    print(f"Facts backfill complete. Updated: {updated}, Unchanged: {unchanged}, Failed: {failed}.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
