import argparse
from pathlib import Path

from sqlalchemy import select

from api.config import get_settings
from api.database import Document, async_session, init_db
from api.services.document_service import DocumentService


settings = get_settings()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reindex app-managed documents while preserving DB metadata."
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        dest="doc_ids",
        help="Reindex only the specified document ID. Repeat to target multiple documents.",
    )
    parser.add_argument(
        "--status",
        action="append",
        choices=["processing", "indexed", "failed"],
        help="Filter by current document status. Repeat to include multiple statuses.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of documents processed after filtering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be reindexed without touching Qdrant or DB status.",
    )
    return parser


def find_untracked_pdfs(tracked_filenames: set[str]) -> list[str]:
    pdf_dir = Path(settings.UPLOAD_DIR)
    if not pdf_dir.exists():
        return []

    return sorted(
        path.name
        for path in pdf_dir.glob("*.pdf")
        if path.name not in tracked_filenames
    )


async def load_document_ids(args: argparse.Namespace) -> list[str]:
    async with async_session() as session:
        query = select(Document.id).order_by(Document.created_at.asc())

        if args.doc_ids:
            query = query.where(Document.id.in_(args.doc_ids))

        if args.status:
            query = query.where(Document.status.in_(args.status))

        if args.limit:
            query = query.limit(args.limit)

        result = await session.execute(query)
        return list(result.scalars().all())


async def load_tracked_filenames() -> set[str]:
    async with async_session() as session:
        result = await session.execute(select(Document.filename))
        return {row[0] for row in result.all()}


async def run() -> None:
    parser = build_parser()
    args = parser.parse_args()

    await init_db()

    doc_ids = await load_document_ids(args)
    if not doc_ids:
        print("No documents matched the requested filters.")
        return

    tracked_filenames = await load_tracked_filenames()
    untracked = find_untracked_pdfs(tracked_filenames)

    print(f"Matched {len(doc_ids)} document(s) for reindexing.")
    if untracked:
        print(
            f"Warning: {len(untracked)} PDF(s) exist in '{settings.UPLOAD_DIR}' without a matching Document row. "
            "They will not be touched by this script."
        )
        for name in untracked[:10]:
            print(f"  - {name}")
        if len(untracked) > 10:
            print(f"  ... and {len(untracked) - 10} more")

    success = 0
    failed = 0
    skipped = 0

    async with async_session() as session:
        for index, doc_id in enumerate(doc_ids, start=1):
            managed_doc = await session.get(Document, doc_id)
            if managed_doc is None:
                print(f"[{index}/{len(doc_ids)}] Skipping missing doc_id={doc_id}")
                skipped += 1
                continue

            file_path = Path(settings.UPLOAD_DIR) / managed_doc.filename
            exists = file_path.exists()

            if args.dry_run:
                print(
                    f"[{index}/{len(doc_ids)}] DRY RUN doc_id={managed_doc.id} file={managed_doc.filename} "
                    f"status={managed_doc.status} dept={managed_doc.department_id} universe={managed_doc.universe_id or '-'} exists={exists}"
                )
                if not exists:
                    skipped += 1
                continue

            print(
                f"[{index}/{len(doc_ids)}] Reindexing doc_id={managed_doc.id} file={managed_doc.filename} "
                f"dept={managed_doc.department_id}"
            )

            if not exists:
                managed_doc.status = "failed"
                managed_doc.chunk_count = 0
                await session.commit()
                print(f"  Missing file on disk: {file_path}")
                failed += 1
                continue

            managed_doc.status = "processing"
            await session.commit()

            try:
                result = DocumentService.index_document_file(
                    doc_id=managed_doc.id,
                    file_path=str(file_path),
                    department=managed_doc.department_id,
                    uploaded_by=managed_doc.uploaded_by,
                    visibility=managed_doc.visibility,
                    universe_id=managed_doc.universe_id,
                    replace_existing=True,
                )
                chunk_count, extracted_facts = DocumentService.normalize_index_result(result)

                if chunk_count <= 0:
                    managed_doc.status = "failed"
                    managed_doc.chunk_count = 0
                    managed_doc.extracted_facts = {}
                    failed += 1
                    print("  No chunks extracted; marked as failed.")
                else:
                    managed_doc.status = "indexed"
                    managed_doc.chunk_count = chunk_count
                    managed_doc.extracted_facts = extracted_facts
                    success += 1
                    print(f"  Indexed {chunk_count} chunk(s).")
            except Exception as exc:
                managed_doc.status = "failed"
                managed_doc.chunk_count = 0
                managed_doc.extracted_facts = {}
                failed += 1
                print(f"  Reindex failed: {exc}")

            await session.commit()

    if args.dry_run:
        print(f"Dry run complete. Skipped due to missing files: {skipped}.")
        return

    print(
        f"Reindex complete. Success: {success}, Failed: {failed}, "
        f"Untracked PDFs left untouched: {len(untracked)}."
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
