from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pytest

from ingest import detect_section, extract_document_facts


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tender_regression_expected.json"
TEXT_CACHE_DIR = Path(__file__).parents[1] / "text_cache"
PAGE_MARKER_RE = re.compile(r"^\s*\[Page\s+([^\]]+)\]\s*$", re.IGNORECASE | re.MULTILINE)
PAGE_BANNER_RE = re.compile(r"\s*(?:-|\u2014)\s*Page\s+\d+\s*(?:-|\u2014)\s*", re.IGNORECASE)


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).lower())
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _resolve_text_cache_path(filename: str) -> Path | None:
    candidates = [
        TEXT_CACHE_DIR / f"{filename}.txt",
        TEXT_CACHE_DIR / f"{filename.replace('_', ' ')}.txt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    wanted = _normalize(Path(filename).stem)
    for candidate in TEXT_CACHE_DIR.glob("*.txt"):
        if _normalize(candidate.stem.removesuffix(".pdf")) == wanted:
            return candidate

    return None


def _fact_text(facts: dict, field: str) -> str:
    value = facts.get(field)
    if isinstance(value, dict):
        return str(value.get("text") or "")
    return ""


def _chunks_from_text_cache(filename: str, text: str) -> tuple[list[str], list[dict]]:
    matches = list(PAGE_MARKER_RE.finditer(text))
    entries = []

    if matches:
        for index, match in enumerate(matches):
            page = match.group(1).strip()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            page_text = PAGE_BANNER_RE.sub(" ", text[start:end])
            page_text = re.sub(r"[ \t]+", " ", page_text)
            page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()
            if page_text:
                entries.append((page, page_text))
    else:
        cleaned = PAGE_BANNER_RE.sub(" ", text).strip()
        if cleaned:
            entries.append(("1", cleaned))

    chunks = []
    metas = []
    for index, (page, chunk) in enumerate(entries):
        chunks.append(chunk)
        metas.append(
            {
                "source": filename,
                "page": page,
                "section": detect_section(chunk),
                "chunk_index": index,
            }
        )

    return chunks, metas


def _matches_expected_contains(actual_text: str, expected_contains: list[str]) -> bool:
    normalized_actual = _normalize(actual_text)
    return all(_normalize(part) in normalized_actual for part in expected_contains)


def test_trusted_tender_regression_current_extractor_baseline():
    fixture = _load_fixture()
    scored_documents = []
    failures = []

    for document in fixture["documents"]:
        if document["review_status"] != "trusted":
            continue

        cache_path = _resolve_text_cache_path(document["filename"])
        if cache_path is None:
            continue

        chunks, metas = _chunks_from_text_cache(
            document["filename"],
            cache_path.read_text(encoding="utf-8", errors="replace"),
        )
        facts = extract_document_facts(chunks, metas)

        matched = []
        missing = []
        mismatched = []
        present_fields = [
            (field, expected)
            for field, expected in document["fields"].items()
            if expected["expected_status"] == "present"
        ]

        for field, expected in present_fields:
            actual_text = _fact_text(facts, field)
            if not actual_text:
                missing.append(field)
                continue

            if _matches_expected_contains(actual_text, expected["expected_contains"]):
                matched.append(field)
            else:
                mismatched.append(field)

        scored_documents.append(document["filename"])
        minimum = document["extractor_baseline"]["min_present_matches"]
        if len(matched) < minimum:
            failures.append(
                "\n".join(
                    [
                        f"{document['filename']}: {len(matched)}/{len(present_fields)} "
                        f"matched present fields, below baseline {minimum}.",
                        f"matched={matched}",
                        f"missing={missing}",
                        f"mismatched={mismatched}",
                    ]
                )
            )

    if not scored_documents:
        pytest.skip("No trusted tender text caches are available for extractor regression scoring.")

    assert failures == []


def test_trusted_tender_regression_negative_guards():
    fixture = _load_fixture()
    failures = []
    checked = []

    for document in fixture["documents"]:
        if document["review_status"] != "trusted":
            continue

        guarded_fields = [
            field
            for field, expected in document["fields"].items()
            if expected.get("negative_guard") and expected["expected_status"] == "not_found"
        ]
        if not guarded_fields:
            continue

        cache_path = _resolve_text_cache_path(document["filename"])
        if cache_path is None:
            continue

        chunks, metas = _chunks_from_text_cache(
            document["filename"],
            cache_path.read_text(encoding="utf-8", errors="replace"),
        )
        facts = extract_document_facts(chunks, metas)

        for field in guarded_fields:
            checked.append(f"{document['filename']}::{field}")
            actual_text = _fact_text(facts, field)
            if actual_text:
                failures.append(f"{document['filename']}::{field} unexpectedly extracted: {actual_text}")

    if not checked:
        pytest.skip("No negative guard fields with local text caches are available.")

    assert failures == []


def test_topnet_noisy_scalar_answers_are_polished():
    cache_path = _resolve_text_cache_path("TOPNET.pdf")
    if cache_path is None:
        pytest.skip("TOPNET text cache is not available.")

    chunks, metas = _chunks_from_text_cache(
        "TOPNET.pdf",
        cache_path.read_text(encoding="utf-8", errors="replace"),
    )
    facts = extract_document_facts(chunks, metas)

    assert facts["subject"]["text"] == (
        "La société TOPNET lance une consultation auprès des sociétés opérant dans la "
        "sécurité des infrastructures informatiques pour le renouvellement de la Solution "
        "Antivirale, avec installation et activation des licences et mise en production de l'EDR."
    )
    assert "Lesobjectifs" not in facts["subject"]["text"]

    assert facts["caution"]["text"] == (
        "Le cautionnement provisoire est de Mille Dinars Tunisiens (1 000 DT), sous forme "
        "de caution bancaire inconditionnelle émise par une banque tunisienne, payable à "
        "première demande de TOPNET et valable 120 jours à partir de la date de réception "
        "des offres."
    )
    assert "s'élève à Le cautionnement" not in facts["caution"]["text"]

    assert facts["payment"]["text"] == (
        "100 % sont réglés après signature du PV de réception provisoire, à 60 jours "
        "suivant la présentation de la facture originale et des attachements signés, "
        "par virement bancaire."
    )


def test_ubci_noisy_scalar_and_list_answers_are_polished():
    cache_path = _resolve_text_cache_path("UBCI.pdf")
    if cache_path is None:
        pytest.skip("UBCI text cache is not available.")

    chunks, metas = _chunks_from_text_cache(
        "UBCI.pdf",
        cache_path.read_text(encoding="utf-8", errors="replace"),
    )
    facts = extract_document_facts(chunks, metas)

    assert _matches_expected_contains(
        _fact_text(facts, "submission_method"),
        ["bureau d'ordre central", "UBCI", "139 Avenue de la Liberté", "1001 Tunis"],
    )
    assert _matches_expected_contains(
        _fact_text(facts, "validity"),
        ["validité de la soumission"],
    )
    assert _matches_expected_contains(
        _fact_text(facts, "administrative_documents"),
        ["paraphé", "signé", "daté", "cachet"],
    )
    assert _matches_expected_contains(
        _fact_text(facts, "technical_documents"),
        [
            "documentation technique",
            "formulaire technique",
            "document officiel du constructeur",
            "date de fin de support",
            "validité de la soumission",
        ],
    )
    assert _matches_expected_contains(
        _fact_text(facts, "manufacturer_authorization"),
        ["document officiel du constructeur", "Attestation Constructeur"],
    )
    assert _matches_expected_contains(
        _fact_text(facts, "financial_documents"),
        [
            "offre financière par unité",
            "Prix catalogue",
            "extension de garantie",
            "contrat cadre de service",
        ],
    )
    assert _matches_expected_contains(
        _fact_text(facts, "payment"),
        ["virement", "R.I.B", "trente (30) jours", "factures conformes"],
    )
