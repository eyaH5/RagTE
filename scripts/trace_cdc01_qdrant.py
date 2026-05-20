from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("APP_ENV_FILE", os.environ.get("APP_ENV_FILE", ".env.local"))
os.environ.setdefault("JWT_SECRET_KEY", "diagnostic-only")

from api.config import get_settings
from ingest import extract_document_facts
from scripts.diagnose_arabic_extraction import _print_report, _run_llm_extraction


DEFAULT_CDC01_DOC_ID = "45e142a3-7bf4-4260-8ebb-744ba9075356"


def _load_qdrant_chunks(doc_id: str) -> tuple[list[str], list[dict]]:
    settings = get_settings()
    client = QdrantClient(url=settings.QDRANT_URL, timeout=60)
    points = []
    offset = None

    while True:
        batch, offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            limit=1000,
            with_payload=True,
            offset=offset,
        )
        points.extend(batch)
        if offset is None:
            break

    points.sort(key=lambda point: point.payload.get("chunk_index", 0))
    chunks = [point.payload.get("text", "") for point in points]
    metas = [
        {key: value for key, value in point.payload.items() if key not in ("text", "original_id")}
        for point in points
    ]
    return chunks, metas


def _fact_field_names(facts: dict) -> list[str]:
    ignored = {"summary", "tender_profile", "extraction_warning", "_hybrid_extraction"}
    return sorted(key for key in facts if key not in ignored)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Arabic Phase 0 LLM trace against production Qdrant chunks."
    )
    parser.add_argument("--doc-id", default=DEFAULT_CDC01_DOC_ID)
    parser.add_argument("--reasoning-effort", default="low", choices=("low", "medium", "high"))
    parser.add_argument(
        "--force-quality-mode",
        default="",
        help="Diagnostic-only override for text_quality_mode on all chunk metadata.",
    )
    parser.add_argument(
        "--output",
        default="/data/text_cache/CDC_01-2026.qdrant.diagnostic.json",
        help="Path for the machine-readable JSON trace.",
    )
    args = parser.parse_args()

    chunks, metas = _load_qdrant_chunks(args.doc_id)
    if not chunks:
        raise SystemExit(f"No Qdrant chunks found for doc_id={args.doc_id}")

    if args.force_quality_mode:
        for meta in metas:
            meta["text_quality_mode"] = args.force_quality_mode
            quality = meta.get("text_quality")
            if isinstance(quality, dict):
                quality["mode"] = args.force_quality_mode

    quality_modes = sorted(
        {
            str(
                meta.get("text_quality_mode")
                or (meta.get("text_quality") or {}).get("mode")
                or "None"
            )
            for meta in metas
        }
    )

    print(f"Loaded {len(chunks)} production chunks from Qdrant for {args.doc_id}")
    print(f"Quality modes: {quality_modes}")

    regex_facts = extract_document_facts(chunks, metas)
    print(f"Regex fields: {_fact_field_names(regex_facts)}")

    results = await _run_llm_extraction(
        chunks,
        metas,
        regex_facts,
        reasoning_effort=args.reasoning_effort,
    )
    results["source"] = {
        "kind": "qdrant",
        "doc_id": args.doc_id,
        "chunk_count": len(chunks),
        "quality_modes": quality_modes,
        "reasoning_effort": args.reasoning_effort,
    }

    _print_report(results, regex_facts)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nFull production-chunk trace saved to {output}")


if __name__ == "__main__":
    asyncio.run(main())
