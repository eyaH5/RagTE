from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest import (  # noqa: E402
    ARABIC_SCALAR_FALLBACK_EXTRACTORS,
    FACT_SCALAR_PATTERNS,
    FACT_SUBJECT_PATTERNS,
    SCALAR_FALLBACK_EXTRACTORS,
    _extract_arabic_subject_fallback,
    _extract_article_facts,
    _extract_fact_candidates_from_pages,
    _extract_fact_from_pages,
    _extract_instruction_marker_facts,
    _extract_placeholder_values,
    _extract_subject_from_pages_fallback,
    _extract_submission_method_from_pages,
    _fact_quality_score,
    _group_chunks_by_page,
    _is_reliable_scalar_fact,
    _score_subject_fact,
    _subject_formatter,
    extract_and_chunk,
    extract_document_facts,
)


BIG3_FIELDS = ("subject", "submission_method", "validity")
DEFAULT_DEMO_FILES = (
    "TUNISIAN_SAUDI_BANK.pdf",
    "Societe Tunisienne de Banque.pdf",
    "Societe Tunisienne de l'electricite et du gaz (STEG).pdf",
    "Rose Blanche.pdf",
    "YESNETWORKS TECHNOLOGIES.pdf",
)


def _snippet(text: Any, limit: int = 220) -> str:
    value = " ".join(str(text or "").split())
    return value[:limit] + ("..." if len(value) > limit else "")


def _find_pdf(filename: str, search_dirs: list[Path]) -> Path:
    wanted = filename.lower()
    normalized_wanted = wanted.replace("_", " ").replace("-", " ")
    for directory in search_dirs:
        if not directory.exists():
            continue
        direct = directory / filename
        if direct.exists():
            return direct
        for path in directory.glob("*.pdf"):
            name = path.name.lower()
            normalized_name = name.replace("_", " ").replace("-", " ")
            if name == wanted or normalized_name == normalized_wanted:
                return path
    raise FileNotFoundError(f"Could not find {filename!r} in: {', '.join(str(d) for d in search_dirs)}")


def _subject_candidates(pages: list[dict]) -> list[dict]:
    candidates = _extract_fact_candidates_from_pages(
        pages,
        FACT_SUBJECT_PATTERNS,
        formatter=_subject_formatter(pages),
    )
    article_facts = _extract_article_facts(pages)
    if article_facts.get("subject"):
        article_subject = dict(article_facts["subject"])
        article_subject["_source"] = "article"
        article_subject["_pattern_index"] = -1
        candidates.append(article_subject)
    page_subject = _extract_subject_from_pages_fallback(pages)
    if page_subject:
        page_subject = dict(page_subject)
        page_subject["_source"] = "page_fallback"
        page_subject["_pattern_index"] = -2
        candidates.append(page_subject)
    arabic_subject = _extract_arabic_subject_fallback(pages)
    if arabic_subject:
        arabic_subject = dict(arabic_subject)
        arabic_subject["_source"] = "arabic_fallback"
        arabic_subject["_pattern_index"] = -3
        candidates.append(arabic_subject)
    return candidates


def _scalar_candidates(field: str, pages: list[dict]) -> list[tuple[str, dict]]:
    placeholder_values = _extract_placeholder_values(pages)
    marker_facts = _extract_instruction_marker_facts(placeholder_values)
    article_facts = _extract_article_facts(pages)
    candidates: list[tuple[str, dict]] = []

    if article_facts.get(field):
        candidates.append(("article", article_facts[field]))
    pattern_fact = _extract_fact_from_pages(pages, FACT_SCALAR_PATTERNS.get(field, ()))
    if pattern_fact:
        candidates.append(("pattern", pattern_fact))
    if field == "submission_method":
        page_fact = _extract_submission_method_from_pages(pages)
        if page_fact:
            candidates.append(("page_fallback", page_fact))
    fallback_extractor = SCALAR_FALLBACK_EXTRACTORS.get(field)
    if fallback_extractor:
        fallback_fact = fallback_extractor(pages)
        if fallback_fact:
            candidates.append(("fallback", fallback_fact))
    arabic_fallback_extractor = ARABIC_SCALAR_FALLBACK_EXTRACTORS.get(field)
    if arabic_fallback_extractor:
        arabic_fact = arabic_fallback_extractor(pages)
        if arabic_fact:
            candidates.append(("arabic_fallback", arabic_fact))
    if marker_facts.get(field):
        candidates.append(("marker", marker_facts[field]))

    return candidates


def trace_file(pdf_path: Path, *, force_docling: bool = False) -> dict[str, Any]:
    chunks, metas, _ = extract_and_chunk(str(pdf_path), pdf_path.name, force_docling=force_docling)
    pages = _group_chunks_by_page(chunks, metas)
    facts = extract_document_facts(chunks, metas)

    trace: dict[str, Any] = {
        "file": pdf_path.name,
        "chunks": len(chunks),
        "fields": {},
    }

    selected_subject = (facts.get("subject") or {}).get("text")
    subject_rows = []
    for candidate in _subject_candidates(pages):
        subject_rows.append(
            {
                "source": candidate.get("_source", "pattern"),
                "page": candidate.get("page"),
                "score": _score_subject_fact(candidate),
                "selected": bool(selected_subject and candidate.get("text") == selected_subject),
                "text": _snippet(candidate.get("text")),
            }
        )
    subject_rows.sort(key=lambda row: row["score"], reverse=True)
    trace["fields"]["subject"] = {
        "selected": _snippet(selected_subject),
        "selected_page": (facts.get("subject") or {}).get("page"),
        "candidates": subject_rows,
    }

    for field in ("submission_method", "validity"):
        selected = facts.get(field) or {}
        rows = []
        for source, candidate in _scalar_candidates(field, pages):
            rows.append(
                {
                    "source": source,
                    "page": candidate.get("page"),
                    "quality": _fact_quality_score(field, candidate),
                    "reliable": _is_reliable_scalar_fact(field, candidate),
                    "selected": bool(selected.get("text") and candidate.get("text") == selected.get("text")),
                    "text": _snippet(candidate.get("text")),
                }
            )
        rows.sort(key=lambda row: (row["reliable"], row["quality"]), reverse=True)
        trace["fields"][field] = {
            "selected": _snippet(selected.get("text")),
            "selected_page": selected.get("page"),
            "candidates": rows,
        }

    return trace


def print_trace(trace: dict[str, Any]) -> None:
    print(f"\n=== {trace['file']} ({trace['chunks']} chunks) ===")
    for field in BIG3_FIELDS:
        data = trace["fields"].get(field, {})
        print(f"\n[{field}] selected page={data.get('selected_page')}: {data.get('selected') or 'None'}")
        for row in data.get("candidates", [])[:8]:
            score = row.get("score", row.get("quality"))
            reliable = row.get("reliable")
            reliable_text = "" if reliable is None else f" reliable={reliable}"
            marker = "*" if row.get("selected") else "-"
            print(
                f"  {marker} source={row.get('source')} page={row.get('page')} "
                f"score={score}{reliable_text} :: {row.get('text')}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Big 3 extraction trace for tender PDFs.")
    parser.add_argument("--filename", action="append", help="PDF filename to trace. Can be repeated.")
    parser.add_argument("--pdf-dir", action="append", default=[], help="Directory to search for PDFs.")
    parser.add_argument("--force-docling", action="store_true", help="Skip direct PDF text and force Docling.")
    parser.add_argument("--json", dest="json_path", help="Write trace JSON to this path.")
    args = parser.parse_args()

    search_dirs = [Path(value) for value in args.pdf_dir]
    search_dirs.extend([ROOT / "raw_docs", ROOT / "pdfs"])
    filenames = args.filename or list(DEFAULT_DEMO_FILES)

    traces = []
    for filename in filenames:
        pdf_path = _find_pdf(filename, search_dirs)
        trace = trace_file(pdf_path, force_docling=args.force_docling)
        traces.append(trace)
        print_trace(trace)

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
