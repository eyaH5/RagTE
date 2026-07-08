from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib import request


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.services.llm_fact_extractor import FIELD_LABELS, PROTOTYPE_FIELDS
from api.services.vlm_extractor import (
    extract_facts_from_vlm_payload,
    merge_vlm_page_facts,
    parse_vlm_json_response,
)


DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "tender_regression_expected.json"
DEFAULT_TEXT_CACHE_DIR = ROOT / "text_cache"
DEFAULT_OUTPUT_DIR = ROOT / "eval" / "vlm_benchmark_results"

LIST_FIELDS = {"administrative_documents", "technical_documents", "financial_documents"}

FIELD_NOTES = {
    "administrative_documents": (
        "Extract the actual required administrative documents, not the section heading. "
        "Examples: CNSS certificate, RNE certificate, signed terms, power of attorney, KYS form."
    ),
    "technical_documents": (
        "Extract the actual required technical documents, not the section heading. "
        "Examples: response form, manufacturer datasheets, brochures, warranty certificate, new/original equipment commitment."
    ),
    "financial_documents": (
        "Extract the actual required financial documents, not the section heading. "
        "Examples: soumission form, price schedule, financial offer, annex price table."
    ),
    "manufacturer_authorization": (
        "Only extract an explicit manufacturer/vendor authorization, reseller authorization, "
        "partner certificate, or named authorization allowing the bidder to sell or support. "
        "Do NOT extract warranty certificates, manufacturer brochures, or generic notices as authorization."
    ),
    "guarantee": (
        "Extract warranty duration only. Do NOT confuse warranty duration with manufacturer authorization."
    ),
    "caution": "Extract provisional/temporary bid bond only, not definitive guarantee.",
    "definitive_caution": "Extract definitive guarantee / performance bond only, not provisional bid bond.",
    "references": (
        "Extract required bidder references or similar-project references only. "
        "Do NOT extract legal article references or annex references."
    ),
    "variants": "Extract whether variants are allowed or forbidden only when explicitly stated.",
}


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    chars = []
    for char in text:
        if unicodedata.combining(char):
            continue
        category = unicodedata.category(char)
        if category.startswith(("L", "N")):
            chars.append(char)
        else:
            chars.append(" ")
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def token_set(value: object) -> set[str]:
    normalized = normalize_text(value)
    return {token for token in normalized.split() if len(token) > 1}


def expected_contains_match(actual: object, expected_contains: list[str]) -> bool:
    normalized_actual = normalize_text(actual)
    return all(normalize_text(part) in normalized_actual for part in expected_contains)


def resolve_text_cache_path(filename: str, text_cache_dir: Path) -> Path | None:
    candidates = [
        text_cache_dir / f"{filename}.txt",
        text_cache_dir / f"{filename.replace('_', ' ')}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    wanted = normalize_text(Path(filename).stem.removesuffix(".pdf"))
    for candidate in text_cache_dir.glob("*.txt"):
        if normalize_text(candidate.stem.removesuffix(".pdf")) == wanted:
            return candidate
    return None


def load_fixture_document(fixture_path: Path, filename: str) -> dict[str, Any] | None:
    if not fixture_path.exists():
        return None
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    wanted = normalize_text(Path(filename).stem)
    for document in fixture.get("documents", []):
        if normalize_text(Path(str(document.get("filename", ""))).stem) == wanted:
            return document
    return None


def pdf_page_count(pdf_path: Path) -> int:
    import fitz

    with fitz.open(pdf_path) as document:
        return int(document.page_count)


def render_pdf_page_png(pdf_path: Path, page_number: int, *, dpi: int) -> bytes:
    import fitz

    with fitz.open(pdf_path) as document:
        page = document.load_page(page_number - 1)
        scale = dpi / 72
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return pixmap.tobytes("png")


def extract_direct_page_texts(pdf_path: Path, max_pages: int) -> dict[str, str]:
    import fitz

    page_texts: dict[str, str] = {}
    with fitz.open(pdf_path) as document:
        for index in range(min(document.page_count, max_pages)):
            page_texts[str(index + 1)] = document.load_page(index).get_text("text") or ""
    return page_texts


def parse_page_selection(value: str, *, page_count: int, max_pages: int) -> list[int]:
    value = str(value or "all").strip().lower()
    upper = min(page_count, max_pages)
    if value in {"all", "auto"}:
        return list(range(1, upper + 1))

    pages: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return sorted({page for page in pages if 1 <= page <= upper})


def build_gemma_prompt(fields: tuple[str, ...] = PROTOTYPE_FIELDS) -> str:
    field_lines = []
    for field in fields:
        note = FIELD_NOTES.get(field, "")
        suffix = f" Note: {note}" if note else ""
        field_lines.append(f"- {field}: {FIELD_LABELS.get(field, field)}.{suffix}")

    return (
        "You are reading one page of a Tunisian procurement tender. "
        "Extract only procurement facts that are explicitly visible on this page.\n\n"
        "Return JSON only. Do not use markdown. Use exactly this shape:\n"
        "{\n"
        '  "facts": [\n'
        '    {"field": "payment", "text": "...", "source_quote": "...", "confidence": "high"}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Omit fields that are not explicitly visible on this page.\n"
        "- field must be one of the allowed field keys below.\n"
        "- text is a concise normalized answer for the field.\n"
        "- source_quote must be the exact visible phrase or sentence supporting the answer.\n"
        "- confidence must be high, medium, or low.\n"
        "- Preserve Arabic, French, numbers, dates, percentages, amounts, and deadlines.\n"
        "- Read Arabic right-to-left and keep Arabic source quotes in Arabic.\n"
        "- If the page has a required-document list, include an items array with each item.\n"
        "- For administrative_documents, technical_documents, and financial_documents, never return only a heading such as "
        "\"offre technique\" or \"documents administratifs\"; return the concrete documents listed under that heading.\n"
        "- source_quote must not contain ellipses like '...' unless the visible document literally contains ellipses.\n"
        "- Do not infer facts from general context, cover pages, headers, or table of contents.\n\n"
        "Allowed fields:\n"
        + "\n".join(field_lines)
    )


def call_ollama_generate(
    *,
    image_png: bytes,
    prompt: str,
    base_url: str,
    model: str,
    timeout: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(image_png).decode("ascii")],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    endpoint = base_url.rstrip("/") + "/api/generate"
    req = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("response") or "")


def build_openai_chat_payload(*, image_png: bytes, prompt: str, model: str, max_tokens: int) -> dict[str, Any]:
    image_b64 = base64.b64encode(image_png).decode("ascii")
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def call_openai_chat_completions(
    *,
    image_png: bytes,
    prompt: str,
    base_url: str,
    model: str,
    timeout: float,
    max_tokens: int,
) -> str:
    payload = build_openai_chat_payload(image_png=image_png, prompt=prompt, model=model, max_tokens=max_tokens)
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    req = request.Request(
        f"{url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        return "\n".join(str(part.get("text") or "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def extract_current_pipeline_facts(
    *,
    pdf_path: Path,
    filename: str,
    text_cache_dir: Path,
    no_pdf_fallback: bool,
) -> tuple[dict[str, Any], str]:
    from backfill_document_facts import _chunks_from_text_cache
    from ingest import extract_and_chunk, extract_document_facts

    cache_path = resolve_text_cache_path(filename, text_cache_dir)
    if cache_path:
        text = cache_path.read_text(encoding="utf-8", errors="replace")
        chunks, metas = _chunks_from_text_cache(filename, text)
        return extract_document_facts(chunks, metas), f"text_cache:{cache_path}"

    if no_pdf_fallback:
        return {}, "missing_text_cache"

    chunks, metas, _ids = extract_and_chunk(str(pdf_path), filename)
    return extract_document_facts(chunks, metas), f"pdf:{pdf_path}"


def fact_text(fact: Any) -> str:
    if not isinstance(fact, dict):
        return ""
    text = str(fact.get("text") or "").strip()
    if text:
        return text
    items = fact.get("items")
    if isinstance(items, list):
        return "\n".join(
            str(item.get("text") if isinstance(item, dict) else item).strip()
            for item in items
            if str(item.get("text") if isinstance(item, dict) else item).strip()
        )
    return ""


def compact_fact(fact: Any) -> dict[str, Any]:
    if not isinstance(fact, dict):
        return {"present": False}

    compact: dict[str, Any] = {
        "present": bool(fact_text(fact)),
        "text": fact_text(fact) or None,
        "page": fact.get("page"),
        "confidence": fact.get("confidence"),
    }
    if fact.get("source"):
        compact["source"] = fact.get("source")
    if fact.get("source_quote"):
        compact["source_quote"] = fact.get("source_quote")
    if isinstance(fact.get("items"), list):
        compact["items"] = [
            str(item.get("text") if isinstance(item, dict) else item).strip()
            for item in fact["items"]
            if str(item.get("text") if isinstance(item, dict) else item).strip()
        ]
    return compact


def verify_quote(source_quote: str, page_text: str) -> dict[str, Any]:
    quote = str(source_quote or "").strip()
    text = str(page_text or "").strip()
    if not quote:
        return {"status": "missing_quote", "token_overlap": 0.0}
    if not text:
        return {"status": "unchecked_no_page_text", "token_overlap": 0.0}

    normalized_quote = normalize_text(quote)
    normalized_text = normalize_text(text)
    if normalized_quote and normalized_quote in normalized_text:
        return {"status": "exact", "token_overlap": 1.0}

    quote_tokens = token_set(quote)
    text_tokens = token_set(text)
    if not quote_tokens:
        return {"status": "missing_quote", "token_overlap": 0.0}
    overlap = len(quote_tokens & text_tokens) / len(quote_tokens)
    status = "approximate" if overlap >= 0.55 else "not_found"
    return {"status": status, "token_overlap": round(overlap, 3)}


def compare_field(
    *,
    field: str,
    current_fact: Any,
    vlm_fact: Any,
    expected: dict[str, Any] | None,
    page_texts: dict[str, str],
) -> dict[str, Any]:
    current = compact_fact(current_fact)
    vlm = compact_fact(vlm_fact)

    expected_status = (expected or {}).get("expected_status")
    expected_contains = list((expected or {}).get("expected_contains") or [])
    current_match = None
    vlm_match = None
    false_positive = None
    if expected_status == "present":
        current_match = expected_contains_match(current.get("text") or "", expected_contains)
        vlm_match = expected_contains_match(vlm.get("text") or "", expected_contains)
    elif expected_status == "not_found":
        false_positive = {
            "current": bool(current.get("present")),
            "vlm": bool(vlm.get("present")),
        }

    quote_check = None
    if vlm.get("source_quote"):
        page_text = page_texts.get(str(vlm.get("page") or ""), "")
        quote_check = verify_quote(str(vlm.get("source_quote") or ""), page_text)

    winner = "unknown"
    reason = "no trusted ground truth for this field"
    if expected_status == "present":
        if vlm_match and not current_match:
            winner = "vlm"
            reason = "VLM matches expected_contains and current does not"
        elif current_match and not vlm_match:
            winner = "current"
            reason = "Current pipeline matches expected_contains and VLM does not"
        elif current_match and vlm_match:
            winner = "both"
            reason = "Both match expected_contains"
        else:
            winner = "none"
            reason = "Neither output matches expected_contains"
    elif expected_status == "not_found":
        if not current.get("present") and vlm.get("present"):
            winner = "current"
            reason = "Expected absent; VLM produced a possible false positive"
        elif current.get("present") and not vlm.get("present"):
            winner = "vlm"
            reason = "Expected absent; current produced a possible false positive"
        elif not current.get("present") and not vlm.get("present"):
            winner = "both"
            reason = "Both correctly omitted an absent field"
        else:
            winner = "none"
            reason = "Both produced possible false positives"
    elif current.get("present") or vlm.get("present"):
        winner = "needs_review"
        reason = "No trusted expected_status; manual review needed"

    return {
        "current": current,
        "vlm": vlm,
        "expected_status": expected_status,
        "expected_contains": expected_contains,
        "current_expected_match": current_match,
        "vlm_expected_match": vlm_match,
        "false_positive": false_positive,
        "vlm_source_quote_check": quote_check,
        "winner": winner,
        "reason": reason,
    }


def run_vlm_pages(
    *,
    pdf_path: Path,
    pages: list[int],
    prompt: str,
    base_url: str,
    model: str,
    provider: str,
    dpi: int,
    timeout: float,
    max_tokens: int,
    include_raw: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], float]:
    merged_facts: dict[str, dict[str, Any]] = {}
    page_reports: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.perf_counter()

    for page_number in pages:
        page_started = time.perf_counter()
        content = ""
        try:
            image_png = render_pdf_page_png(pdf_path, page_number, dpi=dpi)
            if provider == "openai":
                content = call_openai_chat_completions(
                    image_png=image_png,
                    prompt=prompt,
                    base_url=base_url,
                    model=model,
                    timeout=timeout,
                    max_tokens=max_tokens,
                )
            else:
                content = call_ollama_generate(
                    image_png=image_png,
                    prompt=prompt,
                    base_url=base_url,
                    model=model,
                    timeout=timeout,
                    max_tokens=max_tokens,
                )
            payload = parse_vlm_json_response(content)
            page_facts = extract_facts_from_vlm_payload(payload, page_number, PROTOTYPE_FIELDS)
            merged_facts = merge_vlm_page_facts(merged_facts, page_facts)
            page_report: dict[str, Any] = {
                "page": str(page_number),
                "seconds": round(time.perf_counter() - page_started, 3),
                "fields": sorted(page_facts),
            }
            if include_raw:
                page_report["raw_response"] = content
            page_reports.append(page_report)
            print(
                f"page {page_number}: {len(page_facts)} field(s) "
                f"in {page_report['seconds']}s -> {','.join(sorted(page_facts)) or '-'}"
            )
        except Exception as exc:  # pragma: no cover - exercised by real DGX runs.
            error = {
                "page": str(page_number),
                "seconds": round(time.perf_counter() - page_started, 3),
                "error": str(exc),
            }
            if include_raw and content:
                error["raw_response"] = content
            errors.append(error)
            print(f"page {page_number}: ERROR {exc}")

    return merged_facts, page_reports, errors, round(time.perf_counter() - started, 3)


def build_report(
    *,
    pdf_path: Path,
    filename: str,
    selected_pages: list[int],
    current_facts: dict[str, Any],
    current_source: str,
    vlm_facts: dict[str, dict[str, Any]],
    page_reports: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    vlm_seconds: float,
    fixture_document: dict[str, Any] | None,
    page_texts: dict[str, str],
    model: str,
) -> dict[str, Any]:
    fields_report = {}
    fixture_fields = (fixture_document or {}).get("fields") or {}
    for field in PROTOTYPE_FIELDS:
        fields_report[field] = compare_field(
            field=field,
            current_fact=current_facts.get(field),
            vlm_fact=vlm_facts.get(field),
            expected=fixture_fields.get(field),
            page_texts=page_texts,
        )

    trusted_fields = [
        field
        for field, expected in fixture_fields.items()
        if expected.get("expected_status") in {"present", "not_found"}
    ]
    vlm_wins = [field for field, report in fields_report.items() if report["winner"] == "vlm"]
    current_wins = [field for field, report in fields_report.items() if report["winner"] == "current"]
    possible_vlm_false_positives = [
        field
        for field, report in fields_report.items()
        if (report.get("false_positive") or {}).get("vlm")
    ]

    return {
        "schema": "gemma_vlm_benchmark.v1",
        "document": filename,
        "pdf": str(pdf_path),
        "model": model,
        "current_source": current_source,
        "pages_processed": selected_pages,
        "processing_time_seconds": vlm_seconds,
        "fixture": {
            "matched": bool(fixture_document),
            "review_status": (fixture_document or {}).get("review_status"),
            "trusted_field_count": len(trusted_fields),
        },
        "summary": {
            "current_present_fields": [
                field for field in PROTOTYPE_FIELDS if fact_text(current_facts.get(field))
            ],
            "vlm_present_fields": sorted(vlm_facts),
            "vlm_wins": vlm_wins,
            "current_wins": current_wins,
            "possible_vlm_false_positives": possible_vlm_false_positives,
            "errors": len(errors),
        },
        "fields": fields_report,
        "pages": page_reports,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Gemma/Ollama VLM structured extraction against the current tender extractor."
    )
    parser.add_argument("--pdf", type=Path, required=True, help="PDF to benchmark.")
    parser.add_argument("--filename", default="", help="Logical filename for fixture/text-cache matching.")
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--openai-url", default="http://127.0.0.1:8003")
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--candidate-pages", default="all", help="'all', 'auto', '1,6,12', or '1-5'.")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--text-cache-dir", type=Path, default=DEFAULT_TEXT_CACHE_DIR)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--no-pdf-fallback", action="store_true")
    args = parser.parse_args()

    pdf_path = args.pdf.resolve()
    filename = args.filename or pdf_path.name
    page_count = pdf_page_count(pdf_path)
    selected_pages = parse_page_selection(args.candidate_pages, page_count=page_count, max_pages=args.max_pages)
    if not selected_pages:
        raise SystemExit("No pages selected for VLM benchmark.")

    print(f"Benchmarking {filename} with {args.model}")
    print(f"Selected pages: {','.join(str(page) for page in selected_pages)}")

    current_facts, current_source = extract_current_pipeline_facts(
        pdf_path=pdf_path,
        filename=filename,
        text_cache_dir=args.text_cache_dir,
        no_pdf_fallback=args.no_pdf_fallback,
    )
    print(f"Current pipeline source: {current_source}")

    prompt = build_gemma_prompt(PROTOTYPE_FIELDS)
    vlm_facts, page_reports, errors, vlm_seconds = run_vlm_pages(
        pdf_path=pdf_path,
        pages=selected_pages,
        prompt=prompt,
        base_url=args.openai_url if args.provider == "openai" else args.ollama_url,
        model=args.model,
        provider=args.provider,
        dpi=args.dpi,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        include_raw=args.include_raw,
    )

    fixture_document = load_fixture_document(args.fixture, filename)
    page_texts = extract_direct_page_texts(pdf_path, min(page_count, args.max_pages))
    report = build_report(
        pdf_path=pdf_path,
        filename=filename,
        selected_pages=selected_pages,
        current_facts=current_facts,
        current_source=current_source,
        vlm_facts=vlm_facts,
        page_reports=page_reports,
        errors=errors,
        vlm_seconds=vlm_seconds,
        fixture_document=fixture_document,
        page_texts=page_texts,
        model=args.model,
    )

    output_json = args.output_json
    if output_json is None:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename).stem).strip("_") or "document"
        output_json = args.output_dir / f"{safe_name}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(f"\nSaved report to {output_json}")
    print(f"VLM present fields: {len(summary['vlm_present_fields'])}")
    print(f"Current present fields: {len(summary['current_present_fields'])}")
    print(f"VLM wins: {summary['vlm_wins'] or '-'}")
    print(f"Current wins: {summary['current_wins'] or '-'}")
    print(f"Possible VLM false positives: {summary['possible_vlm_false_positives'] or '-'}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
