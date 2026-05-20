#!/usr/bin/env python3
"""
Diagnostic trace for Arabic extraction failures.

Usage (on DGX):
    python scripts/diagnose_arabic_extraction.py /data/text_cache/CDC_01-2026.pdf.txt

Outputs a per-field report showing exactly where each field fails:
  - regex extraction result
  - evidence pages selected for the LLM
  - raw LLM JSON response (before validation)
  - validator accept/reject with reason
  - final merged result

This tells us whether the bottleneck is OCR quality, evidence selection,
validator strictness, or prompt weakness — before committing to any fix.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("APP_ENV_FILE", os.environ.get("APP_ENV_FILE", ".env.local"))
os.environ.setdefault("JWT_SECRET_KEY", "diagnostic-only")

from api.config import get_settings
from api.services.llm_fact_extractor import (
    PROTOTYPE_FIELDS,
    build_extraction_prompt,
    derive_facts_from_list_evidence,
    derive_list_facts_from_page_evidence,
    evidence_fields_for_group,
    group_chunks_by_page,
    group_fields_for_llm,
    is_arabic_dominant_pages,
    is_fact_strong,
    llm_fact_rejection_reason,
    max_pages_for_group,
    parse_llm_json_response,
    select_evidence_pages,
    standardize_llm_fact,
    text_quality_mode_for_pages,
    validate_llm_fact,
    weak_fields_for_llm,
)
from ingest import (
    _clean_chunk_text,
    _fold_fact_text,
    _is_noise_chunk,
    detect_section,
    extract_document_facts,
)


# ── Parse cached text file into chunks/metas ────────────────────────────

def _parse_text_cache(path: str) -> tuple[list[str], list[dict]]:
    """Parse a [Page N] formatted text cache file into chunks + metas."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    entries: list[dict] = []

    # Split on [Page N] markers
    parts = re.split(r"\[Page\s+(\d+)\]", text)
    # parts[0] is preamble (usually empty), then alternating page_num, text
    for i in range(1, len(parts), 2):
        page = parts[i]
        content = _clean_chunk_text(parts[i + 1]) if i + 1 < len(parts) else ""
        if content and not _is_noise_chunk(content):
            entries.append({"page": page, "text": content})

    if not entries:
        # Fallback: treat entire file as one chunk
        entries = [{"page": "1", "text": _clean_chunk_text(text)}]

    chunks = []
    metas = []
    for entry in entries:
        chunks.append(entry["text"])
        metas.append({
            "source": Path(path).name,
            "page": entry["page"],
            "section": detect_section(entry["text"]),
            "chunk_index": len(chunks) - 1,
        })

    return chunks, metas


# ── Diagnostic report ───────────────────────────────────────────────────

def _normalize_for_display(text: str) -> str:
    """Truncate text for readable output."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) > 200:
        return text[:200] + "..."
    return text


def _llm_mentioned(raw_fact: object) -> bool:
    if not isinstance(raw_fact, dict):
        return False
    mentioned = raw_fact.get("mentioned")
    if isinstance(mentioned, bool):
        return mentioned
    if isinstance(mentioned, str):
        return mentioned.strip().lower() in {"true", "yes", "oui", "1"}
    return bool(mentioned)


async def _run_llm_extraction(
    chunks: list[str],
    metas: list[dict],
    draft_facts: dict,
    *,
    reasoning_effort: str = "low",
) -> dict:
    """Run LLM extraction and capture raw responses per group."""
    from openai import AsyncOpenAI

    settings = get_settings()
    pages = group_chunks_by_page(chunks, metas)
    arabic_dominant = is_arabic_dominant_pages(pages)
    quality_mode = text_quality_mode_for_pages(pages)
    arabic_context = arabic_dominant or quality_mode == "arabic_noisy"

    # Pre-derive list facts
    derived = derive_list_facts_from_page_evidence(pages, draft_facts, PROTOTYPE_FIELDS)
    working_facts = {**draft_facts, **derived}
    scalar_derived = derive_facts_from_list_evidence(working_facts, PROTOTYPE_FIELDS)
    derived.update(scalar_derived)
    working_facts.update(scalar_derived)

    weak = weak_fields_for_llm(working_facts, PROTOTYPE_FIELDS)

    results = {
        "arabic_dominant": arabic_dominant,
        "text_quality_mode": quality_mode,
        "arabic_context": arabic_context,
        "total_pages": len(pages),
        "total_chars": sum(len(p.get("text", "")) for p in pages),
        "derived_fields": sorted(derived.keys()),
        "weak_fields": weak,
        "strong_fields": [f for f in PROTOTYPE_FIELDS if f not in weak],
        "groups": {},
    }

    if not weak:
        results["note"] = "No weak fields — all fields already strong from regex+derivation"
        return results

    client = AsyncOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key="none",
    )

    try:
        for group_name, group_fields in group_fields_for_llm(weak):
            evidence_fields = evidence_fields_for_group(group_name, group_fields)
            max_p = max_pages_for_group(
                group_name, 5,
                arabic_dominant=arabic_dominant,
                text_quality_mode=quality_mode,
            )
            evidence_pages = select_evidence_pages(pages, evidence_fields, max_pages=max_p)
            prompt = build_extraction_prompt(
                evidence_pages, group_fields, arabic_context=arabic_context,
            )

            group_result = {
                "fields_requested": group_fields,
                "evidence_page_ids": [p["page"] for p in evidence_pages],
                "max_pages": max_p,
                "prompt_length": len(prompt),
                "prompt_first_500": prompt[:500],
                "raw_response": None,
                "parsed_response": None,
                "per_field": {},
            }

            try:
                response = await client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "Extract structured facts from tender text. Return JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=1800,
                    extra_body={"reasoning_effort": reasoning_effort},
                    timeout=120,
                )
                raw_content = response.choices[0].message.content or ""
                group_result["raw_response"] = raw_content[:3000]
                parsed = parse_llm_json_response(raw_content)
                group_result["parsed_response"] = parsed
            except Exception as exc:
                group_result["llm_error"] = str(exc)
                parsed = {}

            for field in group_fields:
                field_result: dict = {"field": field}
                raw_fact = parsed.get(field)
                field_result["llm_raw"] = raw_fact

                std_fact = standardize_llm_fact(field, raw_fact, evidence_pages)
                field_result["llm_standardized"] = std_fact

                if std_fact:
                    valid = validate_llm_fact(field, std_fact, evidence_pages)
                    field_result["llm_validated"] = valid
                    if not valid:
                        field_result["rejection_reason"] = llm_fact_rejection_reason(
                            field, std_fact, evidence_pages,
                        )
                    field_result["is_strong"] = is_fact_strong(field, std_fact)
                else:
                    field_result["llm_validated"] = False
                    field_result["rejection_reason"] = "standardization_returned_none"

                # What regex produced
                field_result["regex_fact_text"] = _normalize_for_display(
                    draft_facts.get(field, {}).get("text") if isinstance(draft_facts.get(field), dict) else None
                )
                field_result["regex_is_strong"] = is_fact_strong(field, draft_facts.get(field))

                # Derived?
                if field in derived:
                    field_result["derived_fact_text"] = _normalize_for_display(
                        derived[field].get("text") if isinstance(derived[field], dict) else None
                    )

                group_result["per_field"][field] = field_result

            results["groups"][group_name] = group_result
    finally:
        await client.close()

    return results


def _print_report(results: dict, regex_facts: dict) -> None:
    """Print a human-readable diagnostic report."""
    print("\n" + "=" * 80)
    print("ARABIC EXTRACTION DIAGNOSTIC REPORT")
    print("=" * 80)

    print(f"\nDocument: arabic_dominant={results['arabic_dominant']}, "
          f"quality_mode={results['text_quality_mode']}, "
          f"pages={results['total_pages']}, chars={results['total_chars']}")
    print(f"Strong from regex: {results['strong_fields']}")
    print(f"Weak (sent to LLM): {results['weak_fields']}")
    print(f"Derived from lists: {results['derived_fields']}")

    if results.get("note"):
        print(f"\nNote: {results['note']}")
        return

    # Summary table
    print(f"\n{'Field':<30} {'Regex':>6} {'LLM Raw':>10} {'Validated':>10} {'Reason'}")
    print("-" * 90)

    all_fields = []
    for group_name, group_data in results.get("groups", {}).items():
        for field, fd in group_data.get("per_field", {}).items():
            regex_ok = "✓" if fd.get("regex_is_strong") else "✗"
            llm_raw = "—"
            if fd.get("llm_raw"):
                mentioned = _llm_mentioned(fd["llm_raw"])
                llm_raw = "✓ mentioned" if mentioned else "✗ not_mentioned"
            validated = "✓" if fd.get("llm_validated") else "✗"
            reason = fd.get("rejection_reason", "")
            print(f"{field:<30} {regex_ok:>6} {llm_raw:>10} {validated:>10} {reason}")
            all_fields.append(fd)

    # Diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    llm_mentioned_count = sum(
        1 for fd in all_fields
        if _llm_mentioned(fd.get("llm_raw"))
    )
    llm_validated_count = sum(1 for fd in all_fields if fd.get("llm_validated"))
    llm_rejected_count = llm_mentioned_count - llm_validated_count

    print(f"\nLLM mentioned {llm_mentioned_count} fields, validated {llm_validated_count}, rejected {llm_rejected_count}")

    if llm_rejected_count >= 3:
        print("\n⚠ VALIDATORS ARE THE BOTTLENECK")
        print("  The LLM found fields but validators rejected them.")
        print("  Fix: relax Arabic validator keywords/thresholds.")
        rejected_fields = [
            fd["field"] for fd in all_fields
            if _llm_mentioned(fd.get("llm_raw")) and not fd.get("llm_validated")
        ]
        print(f"  Rejected fields: {rejected_fields}")

    not_mentioned = [
        fd["field"] for fd in all_fields
        if isinstance(fd.get("llm_raw"), dict) and not _llm_mentioned(fd.get("llm_raw"))
    ]
    if len(not_mentioned) >= 3:
        print(f"\n⚠ LLM DID NOT FIND {len(not_mentioned)} FIELDS: {not_mentioned}")
        print("  Possible causes:")
        print("    - Evidence pages don't contain the relevant clauses")
        print("    - OCR text is too corrupted for LLM to parse")
        print("    - Prompt doesn't guide Arabic OCR interpretation")
        print("  Next: check evidence page content for these fields")

    no_response = [fd["field"] for fd in all_fields if fd.get("llm_raw") is None]
    if no_response:
        print(f"\n⚠ LLM RETURNED NO DATA for: {no_response}")
        print("  The field was not in the LLM's JSON response at all.")

    # Detail per group
    for group_name, group_data in results.get("groups", {}).items():
        print(f"\n{'─' * 80}")
        print(f"GROUP: {group_name}")
        print(f"  Evidence pages: {group_data['evidence_page_ids']} (max={group_data['max_pages']})")
        print(f"  Prompt length: {group_data['prompt_length']} chars")

        if group_data.get("llm_error"):
            print(f"  ❌ LLM ERROR: {group_data['llm_error']}")
            continue

        for field, fd in group_data.get("per_field", {}).items():
            print(f"\n  ── {field} ──")
            if fd.get("regex_fact_text"):
                print(f"    Regex: {fd['regex_fact_text']}")
            if fd.get("derived_fact_text"):
                print(f"    Derived: {fd['derived_fact_text']}")
            if isinstance(fd.get("llm_raw"), dict):
                print(f"    LLM mentioned: {fd['llm_raw'].get('mentioned')}")
                answer = fd['llm_raw'].get('answer')
                if answer:
                    print(f"    LLM answer: {_normalize_for_display(str(answer))}")
                print(f"    LLM page: {fd['llm_raw'].get('page')}")
            if fd.get("rejection_reason"):
                print(f"    ❌ Rejected: {fd['rejection_reason']}")
            if fd.get("llm_validated"):
                print(f"    ✓ Validated and accepted")


async def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <text_cache_file> [reasoning_effort]")
        print(f"  reasoning_effort: low (default), medium, high")
        sys.exit(1)

    cache_path = sys.argv[1]
    effort = sys.argv[2] if len(sys.argv) > 2 else "low"

    if not Path(cache_path).exists():
        print(f"Error: {cache_path} not found")
        sys.exit(1)

    print(f"Loading cached text from {cache_path}...")
    chunks, metas = _parse_text_cache(cache_path)
    print(f"Parsed {len(chunks)} chunks from {len(set(m['page'] for m in metas))} pages")

    print(f"\nRunning regex extraction...")
    regex_facts = extract_document_facts(chunks, metas)
    regex_field_count = len([
        k for k in regex_facts
        if k not in {"summary", "tender_profile", "extraction_warning", "_hybrid_extraction"}
    ])
    print(f"Regex extracted {regex_field_count} fields")

    print(f"\nRunning LLM extraction (reasoning_effort={effort})...")
    results = await _run_llm_extraction(
        chunks, metas, regex_facts, reasoning_effort=effort,
    )

    _print_report(results, regex_facts)

    # Save full results as JSON
    out_path = Path(cache_path).with_suffix(".diagnostic.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
