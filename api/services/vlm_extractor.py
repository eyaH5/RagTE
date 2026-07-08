from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from api.services.llm_fact_extractor import FIELD_LABELS, PROTOTYPE_FIELDS


CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class VLMExtractionResult:
    facts: dict[str, dict[str, Any]]
    pages: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def parse_promoted_fields(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    allowed = set(PROTOTYPE_FIELDS)
    fields = []
    for raw in str(value).split(","):
        field = raw.strip()
        if field and field in allowed and field not in fields:
            fields.append(field)
    return tuple(fields)


def build_vlm_extraction_prompt(fields: tuple[str, ...]) -> str:
    field_lines = "\n".join(f"- {field}: {FIELD_LABELS.get(field, field)}" for field in fields)
    return (
        "You are reading one page of a Tunisian procurement tender. "
        "Extract only facts that are explicitly visible on this page.\n\n"
        "Return strict JSON only, with this shape:\n"
        "{\n"
        '  "facts": [\n'
        '    {"field": "payment", "text": "...", "source_quote": "...", "confidence": "high"}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Omit fields that are not visible on this page.\n"
        "- field must be one of the allowed field keys below.\n"
        "- text is the normalized answer for the field.\n"
        "- source_quote is the exact visible phrase or sentence supporting the answer.\n"
        "- confidence is high, medium, or low.\n"
        "- Preserve Arabic and French text. Read Arabic right-to-left.\n"
        "- Preserve numbers, dates, percentages, and deadlines exactly when visible.\n"
        "- If a table contains required documents, summarize each required item in text.\n\n"
        "Allowed fields:\n"
        f"{field_lines}"
    )


def parse_vlm_json_response(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def _list_text_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, dict):
            item_text = str(item.get("text") or item.get("value") or item.get("answer") or "").strip()
        else:
            item_text = str(item or "").strip()
        if item_text:
            items.append(item_text)
    return items


def standardize_vlm_fact(field: str, raw_fact: dict[str, Any], page: int) -> dict[str, Any] | None:
    raw_text = raw_fact.get("text")
    if raw_text is None:
        raw_text = raw_fact.get("answer")
    if raw_text is None:
        raw_text = raw_fact.get("value")
    text_items = _list_text_items(raw_text)
    text = "\n".join(text_items) if text_items else str(raw_text or "").strip()
    quote = str(raw_fact.get("source_quote") or raw_fact.get("quote") or "").strip()
    confidence = str(raw_fact.get("confidence") or "medium").strip().lower()
    if confidence not in CONFIDENCE_ORDER:
        confidence = "medium"
    if not text and quote:
        text = quote
    if not text:
        return None
    fact = {
        "text": text,
        "page": str(raw_fact.get("page") or page),
        "section": "vlm_extracted",
        "source": "vlm",
        "confidence": confidence,
    }
    if quote:
        fact["source_quote"] = quote
    if field in {"administrative_documents", "technical_documents", "financial_documents"}:
        items = raw_fact.get("items")
        if not isinstance(items, list) and text_items:
            items = text_items
        if isinstance(items, list):
            fact["items"] = [
                {
                    "text": str(item.get("text") if isinstance(item, dict) else item).strip(),
                    "page": str(raw_fact.get("page") or page),
                    "section": "vlm_extracted",
                }
                for item in items
                if str(item.get("text") if isinstance(item, dict) else item).strip()
            ]
    return fact


def extract_facts_from_vlm_payload(payload: dict[str, Any], page: int, fields: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    allowed = set(fields)
    facts: dict[str, dict[str, Any]] = {}

    raw_facts = payload.get("facts")
    if isinstance(raw_facts, list):
        iterable = raw_facts
    else:
        iterable = []
        for field, raw_fact in payload.items():
            if field in allowed:
                if isinstance(raw_fact, dict):
                    iterable.append({"field": field, **raw_fact})
                else:
                    iterable.append({"field": field, "text": raw_fact})

    for raw_fact in iterable:
        if not isinstance(raw_fact, dict):
            continue
        field = str(raw_fact.get("field") or "").strip()
        if field not in allowed:
            continue
        fact = standardize_vlm_fact(field, raw_fact, page)
        if fact:
            facts[field] = fact
    return facts


def _fact_rank(fact: dict[str, Any]) -> tuple[int, int, int]:
    confidence = CONFIDENCE_ORDER.get(str(fact.get("confidence") or "medium").lower(), 1)
    quote_len = len(str(fact.get("source_quote") or ""))
    text_len = len(str(fact.get("text") or ""))
    return (confidence, quote_len, text_len)


def merge_vlm_page_facts(current: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged = dict(current)
    for field, fact in candidate.items():
        existing = merged.get(field)
        if not existing or _fact_rank(fact) > _fact_rank(existing):
            merged[field] = fact
    return merged


def render_pdf_page_data_url(pdf_path: str, page_number: int, *, dpi: int) -> str:
    import fitz

    with fitz.open(pdf_path) as document:
        page = document.load_page(page_number - 1)
        scale = dpi / 72
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def pdf_page_count(pdf_path: str) -> int:
    import fitz

    with fitz.open(pdf_path) as document:
        return int(document.page_count)


async def extract_vlm_facts_from_pdf(
    *,
    pdf_path: str,
    client: AsyncOpenAI,
    model: str,
    fields: tuple[str, ...] = PROTOTYPE_FIELDS,
    max_pages: int = 20,
    dpi: int = 160,
    timeout: float = 180,
    max_output_tokens: int = 1200,
) -> VLMExtractionResult:
    page_count = min(pdf_page_count(pdf_path), max_pages)
    prompt = build_vlm_extraction_prompt(fields)
    merged_facts: dict[str, dict[str, Any]] = {}
    pages: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for page_number in range(1, page_count + 1):
        try:
            image_url = render_pdf_page_data_url(pdf_path, page_number, dpi=dpi)
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                timeout=timeout,
                max_tokens=max_output_tokens,
            )
            content = response.choices[0].message.content or ""
            payload = parse_vlm_json_response(content)
            page_facts = extract_facts_from_vlm_payload(payload, page_number, fields)
            merged_facts = merge_vlm_page_facts(merged_facts, page_facts)
            pages.append(
                {
                    "page": str(page_number),
                    "fields": sorted(page_facts),
                    "raw_response": content,
                }
            )
        except Exception as exc:
            errors.append({"page": str(page_number), "error": str(exc)})

    return VLMExtractionResult(facts=merged_facts, pages=pages, errors=errors)
