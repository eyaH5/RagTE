import argparse
from collections import Counter
from dataclasses import dataclass

from qdrant_client import QdrantClient
from sqlalchemy import select

from api.config import get_settings
from api.database import Document, async_session, init_db


settings = get_settings()


@dataclass(frozen=True)
class DocumentState:
    id: str
    filename: str
    status: str
    chunk_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Document rows against Qdrant doc_id payloads and optionally repair status drift."
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        dest="doc_ids",
        help="Restrict reconciliation to the specified document ID. Repeat to target multiple docs.",
    )
    parser.add_argument(
        "--repair-status",
        action="store_true",
        help="Set docs with vectors to indexed and align chunk_count with Qdrant.",
    )
    parser.add_argument(
        "--mark-missing-failed",
        action="store_true",
        help="When used with --repair-status, mark docs with no vectors as failed and set chunk_count to 0.",
    )
    return parser


async def load_documents(doc_ids: list[str] | None = None) -> list[DocumentState]:
    async with async_session() as session:
        query = select(Document.id, Document.filename, Document.status, Document.chunk_count).order_by(Document.created_at.asc())
        if doc_ids:
            query = query.where(Document.id.in_(doc_ids))

        result = await session.execute(query)
        return [
            DocumentState(
                id=row[0],
                filename=row[1],
                status=row[2],
                chunk_count=row[3],
            )
            for row in result.all()
        ]


def collect_qdrant_doc_counts(doc_ids: set[str] | None = None) -> tuple[Counter[str], int]:
    client = QdrantClient(url=settings.QDRANT_URL, timeout=60)
    counts: Counter[str] = Counter()
    points_without_doc_id = 0
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=1000,
            with_payload=["doc_id"],
            offset=offset,
        )
        for point in points:
            doc_id = point.payload.get("doc_id")
            if not doc_id:
                if doc_ids is None:
                    points_without_doc_id += 1
                continue
            if doc_ids is not None and doc_id not in doc_ids:
                continue
            counts[doc_id] += 1

        if offset is None:
            break

    return counts, points_without_doc_id


def build_report(documents: list[DocumentState], qdrant_counts: Counter[str], points_without_doc_id: int) -> dict:
    docs_by_id = {doc.id: doc for doc in documents}
    missing_vectors = [doc for doc in documents if qdrant_counts.get(doc.id, 0) == 0]
    status_mismatches = [
        doc for doc in documents
        if qdrant_counts.get(doc.id, 0) > 0 and doc.status != "indexed"
    ]
    chunk_mismatches = [
        (doc, qdrant_counts[doc.id]) for doc in documents
        if qdrant_counts.get(doc.id, 0) > 0 and doc.chunk_count != qdrant_counts[doc.id]
    ]
    orphan_doc_ids = {
        doc_id: count for doc_id, count in qdrant_counts.items()
        if doc_id not in docs_by_id
    }

    return {
        "document_count": len(documents),
        "indexed_in_qdrant": sum(1 for doc in documents if qdrant_counts.get(doc.id, 0) > 0),
        "missing_vectors": missing_vectors,
        "status_mismatches": status_mismatches,
        "chunk_mismatches": chunk_mismatches,
        "orphan_doc_ids": orphan_doc_ids,
        "points_without_doc_id": points_without_doc_id,
    }


def compute_repair_updates(
    documents: list[DocumentState],
    qdrant_counts: Counter[str],
    *,
    mark_missing_failed: bool,
) -> dict[str, tuple[str, int]]:
    updates: dict[str, tuple[str, int]] = {}

    for doc in documents:
        vector_count = qdrant_counts.get(doc.id, 0)
        if vector_count > 0:
            if doc.status != "indexed" or doc.chunk_count != vector_count:
                updates[doc.id] = ("indexed", vector_count)
        elif mark_missing_failed and (doc.status != "failed" or doc.chunk_count != 0):
            updates[doc.id] = ("failed", 0)

    return updates


async def apply_repair_updates(updates: dict[str, tuple[str, int]]) -> int:
    if not updates:
        return 0

    changed = 0
    async with async_session() as session:
        for doc_id, (status, chunk_count) in updates.items():
            doc = await session.get(Document, doc_id)
            if doc is None:
                continue
            doc.status = status
            doc.chunk_count = chunk_count
            changed += 1
        await session.commit()

    return changed


def print_report(report: dict) -> None:
    print(f"Documents inspected: {report['document_count']}")
    print(f"Documents with vectors: {report['indexed_in_qdrant']}")
    print(f"Documents missing vectors: {len(report['missing_vectors'])}")
    print(f"Status mismatches: {len(report['status_mismatches'])}")
    print(f"Chunk-count mismatches: {len(report['chunk_mismatches'])}")
    print(f"Orphan Qdrant doc_ids: {len(report['orphan_doc_ids'])}")
    print(f"Points without doc_id payload: {report['points_without_doc_id']}")

    if report["missing_vectors"]:
        print("Missing vectors:")
        for doc in report["missing_vectors"][:10]:
            print(f"  - {doc.id} | {doc.filename} | status={doc.status} chunk_count={doc.chunk_count}")
        if len(report["missing_vectors"]) > 10:
            print(f"  ... and {len(report['missing_vectors']) - 10} more")

    if report["status_mismatches"]:
        print("Status mismatches:")
        for doc in report["status_mismatches"][:10]:
            print(f"  - {doc.id} | {doc.filename} | status={doc.status}")
        if len(report["status_mismatches"]) > 10:
            print(f"  ... and {len(report['status_mismatches']) - 10} more")

    if report["chunk_mismatches"]:
        print("Chunk-count mismatches:")
        for doc, expected in report["chunk_mismatches"][:10]:
            print(f"  - {doc.id} | {doc.filename} | db={doc.chunk_count} qdrant={expected}")
        if len(report["chunk_mismatches"]) > 10:
            print(f"  ... and {len(report['chunk_mismatches']) - 10} more")

    if report["orphan_doc_ids"]:
        print("Orphan Qdrant doc_ids:")
        for doc_id, count in list(report["orphan_doc_ids"].items())[:10]:
            print(f"  - {doc_id} | points={count}")
        if len(report["orphan_doc_ids"]) > 10:
            print(f"  ... and {len(report['orphan_doc_ids']) - 10} more")


async def run() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mark_missing_failed and not args.repair_status:
        raise ValueError("--mark-missing-failed requires --repair-status.")

    await init_db()
    documents = await load_documents(args.doc_ids)
    if not documents:
        print("No documents matched the requested filters.")
        return

    scoped_doc_ids = {doc.id for doc in documents} if args.doc_ids else None
    qdrant_counts, points_without_doc_id = collect_qdrant_doc_counts(scoped_doc_ids)
    report = build_report(documents, qdrant_counts, points_without_doc_id if args.doc_ids is None else 0)
    print_report(report)

    if not args.repair_status:
        return

    updates = compute_repair_updates(
        documents,
        qdrant_counts,
        mark_missing_failed=args.mark_missing_failed,
    )
    changed = await apply_repair_updates(updates)
    print(f"Applied {changed} document status repair(s).")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
