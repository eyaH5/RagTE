from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backfill_document_facts import _chunks_from_text_cache
from ingest import extract_document_facts


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect extracted tender facts from a text cache file.")
    parser.add_argument("filename")
    parser.add_argument("text_cache", type=Path)
    parser.add_argument("--fields", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    text = args.text_cache.read_text(encoding="utf-8", errors="replace")
    chunks, metas = _chunks_from_text_cache(args.filename, text)
    facts = extract_document_facts(chunks, metas)
    ignored = {"summary", "tender_profile", "extraction_warning", "_hybrid_extraction"}
    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    if not fields:
        fields = sorted(key for key in facts if key not in ignored)

    result = {}
    for field in fields:
        fact = facts.get(field)
        if isinstance(fact, dict):
            result[field] = {
                "text": fact.get("text"),
                "page": fact.get("page"),
                "confidence": fact.get("confidence"),
            }
        else:
            result[field] = None

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for field, fact in result.items():
            print(f"\n[{field}]")
            if fact is None:
                print("None")
            else:
                print(f"page={fact.get('page')} confidence={fact.get('confidence')}")
                print(fact.get("text") or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
