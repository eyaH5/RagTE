from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("APP_ENV_FILE", os.environ.get("APP_ENV_FILE", ".env.local"))
os.environ.setdefault("JWT_SECRET_KEY", "diagnostic-only")

from qdrant_client import QdrantClient

from api.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="List Qdrant doc ids matching a filename/source substring.")
    parser.add_argument("needle")
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    settings = get_settings()
    client = QdrantClient(url=settings.QDRANT_URL, timeout=60)
    offset = None
    docs: dict[str, dict] = {}
    needle = args.needle.casefold()

    seen = 0
    while True:
        points, offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=min(1000, args.limit - seen),
            with_payload=True,
            offset=offset,
        )
        for point in points:
            seen += 1
            payload = point.payload or {}
            haystack = " ".join(
                str(payload.get(key) or "")
                for key in ("source", "filename", "doc_title", "document_name")
            ).casefold()
            if needle in haystack:
                doc_id = str(payload.get("doc_id") or "")
                if doc_id:
                    docs.setdefault(
                        doc_id,
                        {
                            "doc_id": doc_id,
                            "source": payload.get("source"),
                            "filename": payload.get("filename"),
                            "chunk_count": 0,
                            "first_page": payload.get("page"),
                        },
                    )["chunk_count"] += 1
        if offset is None or seen >= args.limit:
            break

    for doc in docs.values():
        print(doc)
    if not docs:
        print("No matching Qdrant docs found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
