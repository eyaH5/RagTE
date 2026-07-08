from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backfill_document_facts import _chunks_from_text_cache
from ingest import extract_document_facts


DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "tender_regression_expected.json"
DEFAULT_TEXT_CACHE_DIR = ROOT / "text_cache"


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()


def _resolve_text_cache_path(filename: str, text_cache_dir: Path) -> Path | None:
    candidates = [
        text_cache_dir / f"{filename}.txt",
        text_cache_dir / f"{filename.replace('_', ' ')}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    wanted = _normalize(Path(filename).stem)
    for candidate in text_cache_dir.glob("*.txt"):
        if _normalize(candidate.stem.removesuffix(".pdf")) == wanted:
            return candidate
    return None


def _fact_text(facts: dict, field: str) -> str:
    value = facts.get(field)
    if isinstance(value, dict):
        return str(value.get("text") or "")
    return ""


def _matches_expected_contains(actual_text: str, expected_contains: list[str]) -> bool:
    normalized_actual = _normalize(actual_text)
    return all(_normalize(part) in normalized_actual for part in expected_contains)


def evaluate(fixture_path: Path, text_cache_dir: Path, trusted_only: bool) -> dict:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    documents = []

    for document in fixture.get("documents", []):
        if trusted_only and document.get("review_status") != "trusted":
            continue

        cache_path = _resolve_text_cache_path(document["filename"], text_cache_dir)
        if cache_path is None:
            documents.append(
                {
                    "filename": document["filename"],
                    "status": "missing_text_cache",
                    "matched": 0,
                    "present_total": 0,
                }
            )
            continue

        chunks, metas = _chunks_from_text_cache(
            document["filename"],
            cache_path.read_text(encoding="utf-8", errors="replace"),
        )
        facts = extract_document_facts(chunks, metas)
        profile = facts.get("tender_profile") or {}
        coverage = (profile.get("coverage") or {}).get("core_ratio", 0)

        matched = []
        missing = []
        mismatched = []
        present_fields = [
            (field, expected)
            for field, expected in document.get("fields", {}).items()
            if expected.get("expected_status") == "present"
        ]

        for field, expected in present_fields:
            actual_text = _fact_text(facts, field)
            if not actual_text:
                missing.append(field)
                continue
            if _matches_expected_contains(actual_text, expected.get("expected_contains", [])):
                matched.append(field)
            else:
                mismatched.append(field)

        minimum = document.get("extractor_baseline", {}).get("min_present_matches", 0)
        documents.append(
            {
                "filename": document["filename"],
                "status": "scored",
                "matched": len(matched),
                "present_total": len(present_fields),
                "baseline_min": minimum,
                "passes_baseline": len(matched) >= minimum,
                "coverage": coverage,
                "matched_fields": matched,
                "missing_fields": missing,
                "mismatched_fields": mismatched,
            }
        )

    scored = [doc for doc in documents if doc["status"] == "scored"]
    failed = [doc for doc in scored if not doc["passes_baseline"]]
    return {
        "schema": "tender_eval.v1",
        "fixture": str(fixture_path),
        "text_cache_dir": str(text_cache_dir),
        "documents": documents,
        "summary": {
            "scored_documents": len(scored),
            "failed_documents": len(failed),
            "matched_fields": sum(doc["matched"] for doc in scored),
            "present_fields": sum(doc["present_total"] for doc in scored),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate tender extraction/profile quality against fixtures.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--text-cache-dir", type=Path, default=DEFAULT_TEXT_CACHE_DIR)
    parser.add_argument("--all", action="store_true", help="Evaluate documents even if not marked trusted.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0.")
    args = parser.parse_args()

    report = evaluate(args.fixture, args.text_cache_dir, trusted_only=not args.all)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        summary = report["summary"]
        print(
            f"Scored {summary['scored_documents']} document(s): "
            f"{summary['matched_fields']}/{summary['present_fields']} expected present fields matched."
        )
        for document in report["documents"]:
            if document["status"] != "scored":
                print(f"- {document['filename']}: {document['status']}")
                continue
            marker = "PASS" if document["passes_baseline"] else "FAIL"
            print(
                f"- {marker} {document['filename']}: "
                f"{document['matched']}/{document['present_total']} matched, "
                f"coverage={document['coverage']}"
            )
            if not document["passes_baseline"]:
                print(f"  missing={document['missing_fields']}")
                print(f"  mismatched={document['mismatched_fields']}")

    if args.no_fail:
        return 0
    return 1 if report["summary"]["failed_documents"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
