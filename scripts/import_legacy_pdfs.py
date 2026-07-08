import argparse
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from api.config import get_settings
from api.database import Document, Universe, User, async_session, init_db
from api.services.document_service import DocumentService


settings = get_settings()


@dataclass
class ImportContext:
    department_id: str
    uploaded_by: User
    universe: Universe | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import legacy PDFs from the upload directory into the app database."
    )
    parser.add_argument("--department-id", help="Assign imported PDFs to this department.")
    parser.add_argument("--universe-id", help="Attach imported PDFs to this universe.")
    parser.add_argument("--uploaded-by-id", help="User ID recorded as the uploader.")
    parser.add_argument("--uploaded-by-email", help="User email recorded as the uploader.")
    parser.add_argument(
        "--visibility",
        default="department",
        choices=["private", "department", "shared", "restricted"],
        help="Visibility assigned to imported documents.",
    )
    parser.add_argument(
        "--doc-type",
        default="cahier_charges",
        help="Document type stored in the database for imported PDFs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit how many untracked PDFs are imported.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Immediately rebuild vectors for each imported PDF after creating its DB row.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without creating DB rows or touching Qdrant.",
    )
    return parser


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def resolve_context(args: argparse.Namespace) -> ImportContext:
    async with async_session() as session:
        universe = None
        if args.universe_id:
            result = await session.execute(select(Universe).where(Universe.id == args.universe_id))
            universe = result.scalar_one_or_none()
            if not universe:
                raise ValueError(f"Universe '{args.universe_id}' not found.")

        uploaded_by = None
        if args.uploaded_by_id:
            result = await session.execute(select(User).where(User.id == args.uploaded_by_id))
            uploaded_by = result.scalar_one_or_none()
            if not uploaded_by:
                raise ValueError(f"User '{args.uploaded_by_id}' not found.")
        elif args.uploaded_by_email:
            result = await session.execute(select(User).where(User.email == args.uploaded_by_email))
            uploaded_by = result.scalar_one_or_none()
            if not uploaded_by:
                raise ValueError(f"User '{args.uploaded_by_email}' not found.")
        elif universe is not None:
            result = await session.execute(select(User).where(User.id == universe.created_by))
            uploaded_by = result.scalar_one_or_none()
            if uploaded_by is None:
                raise ValueError(
                    f"Universe '{args.universe_id}' has no valid creator user to record as uploader."
                )

        if uploaded_by is None:
            raise ValueError(
                "Could not infer the uploader. Pass --uploaded-by-email or --uploaded-by-id."
            )

        department_id = args.department_id
        if universe is not None:
            if department_id and department_id != universe.department_id:
                raise ValueError(
                    f"Department '{department_id}' conflicts with universe department '{universe.department_id}'."
                )
            department_id = universe.department_id

        if not department_id:
            if uploaded_by.role != "admin":
                department_id = uploaded_by.department_id
            else:
                raise ValueError(
                    "Could not infer the department. Pass --department-id or --universe-id."
                )

        if uploaded_by.role != "admin" and uploaded_by.department_id != department_id:
            raise ValueError(
                f"Uploader '{uploaded_by.email}' belongs to department '{uploaded_by.department_id}', "
                f"but import target department is '{department_id}'."
            )

        return ImportContext(
            department_id=department_id,
            uploaded_by=uploaded_by,
            universe=universe,
        )


async def load_existing_documents() -> tuple[dict[str, Document], dict[str, Document]]:
    async with async_session() as session:
        result = await session.execute(select(Document))
        docs = list(result.scalars().all())
        by_filename = {doc.filename: doc for doc in docs}
        by_hash = {doc.file_hash: doc for doc in docs}
        return by_filename, by_hash


def find_untracked_pdfs(tracked_filenames: set[str]) -> list[Path]:
    upload_dir = Path(settings.UPLOAD_DIR)
    if not upload_dir.exists():
        return []

    return sorted(
        path for path in upload_dir.glob("*.pdf")
        if path.name not in tracked_filenames
    )


async def run() -> None:
    parser = build_parser()
    args = parser.parse_args()

    await init_db()
    context = await resolve_context(args)
    by_filename, by_hash = await load_existing_documents()

    candidates = find_untracked_pdfs(set(by_filename.keys()))
    if args.limit:
        candidates = candidates[:args.limit]

    if not candidates:
        print("No untracked PDFs found in the upload directory.")
        return

    print("Import context:")
    print(f"  department: {context.department_id}")
    print(f"  uploaded_by: {context.uploaded_by.email} ({context.uploaded_by.id})")
    print(f"  universe: {context.universe.id if context.universe else '-'}")
    print(f"  reindex: {args.reindex}")
    print(f"  dry_run: {args.dry_run}")
    print(f"Found {len(candidates)} untracked PDF(s).")

    imported = 0
    skipped_duplicate_hash = 0
    failed = 0

    async with async_session() as session:
        for index, file_path in enumerate(candidates, start=1):
            file_hash = sha256_file(file_path)
            existing_same_hash = by_hash.get(file_hash)

            if existing_same_hash:
                print(
                    f"[{index}/{len(candidates)}] Skipping {file_path.name} because the same file hash already "
                    f"exists as {existing_same_hash.filename} (doc_id={existing_same_hash.id})."
                )
                skipped_duplicate_hash += 1
                continue

            if args.dry_run:
                print(
                    f"[{index}/{len(candidates)}] DRY RUN import {file_path.name} -> "
                    f"department={context.department_id} universe={context.universe.id if context.universe else '-'}"
                )
                continue

            doc = Document(
                id=str(uuid.uuid4()),
                filename=file_path.name,
                file_hash=file_hash,
                department_id=context.department_id,
                uploaded_by=context.uploaded_by.id,
                universe_id=context.universe.id if context.universe else None,
                visibility=args.visibility,
                doc_type=args.doc_type,
                chunk_count=0,
                status="processing",
            )
            session.add(doc)
            await session.commit()
            await session.refresh(doc)

            try:
                if args.reindex:
                    result = DocumentService.index_document_file(
                        doc_id=doc.id,
                        file_path=str(file_path),
                        department=doc.department_id,
                        uploaded_by=doc.uploaded_by,
                        visibility=doc.visibility,
                        universe_id=doc.universe_id,
                        replace_existing=True,
                    )
                    chunk_count, extracted_facts = DocumentService.normalize_index_result(result)
                    if chunk_count <= 0:
                        doc.status = "failed"
                        doc.chunk_count = 0
                        doc.extracted_facts = {}
                        failed += 1
                        print(f"[{index}/{len(candidates)}] Imported {file_path.name} but reindex extracted no chunks.")
                    else:
                        doc.status = "indexed"
                        doc.chunk_count = chunk_count
                        doc.extracted_facts = extracted_facts
                        imported += 1
                        print(f"[{index}/{len(candidates)}] Imported and indexed {file_path.name} ({chunk_count} chunks).")
                else:
                    imported += 1
                    print(f"[{index}/{len(candidates)}] Imported {file_path.name}; pending reindex.")

                await session.commit()
                by_hash[file_hash] = doc
                by_filename[file_path.name] = doc
            except Exception as exc:
                doc.status = "failed"
                doc.chunk_count = 0
                doc.extracted_facts = {}
                await session.commit()
                failed += 1
                print(f"[{index}/{len(candidates)}] Failed to import {file_path.name}: {exc}")

    if args.dry_run:
        print("Dry run complete.")
        return

    print(
        f"Import complete. Imported: {imported}, Duplicate-hash skips: {skipped_duplicate_hash}, Failed: {failed}."
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
