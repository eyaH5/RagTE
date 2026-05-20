from __future__ import annotations

from api.services.vlm_extractor import (
    extract_facts_from_vlm_payload,
    merge_vlm_page_facts,
    parse_promoted_fields,
    parse_vlm_json_response,
)


def test_parse_vlm_json_response_strips_markdown_fence():
    payload = parse_vlm_json_response(
        """```json
        {"facts": [{"field": "payment", "text": "Paiement a 30 jours"}]}
        ```"""
    )

    assert payload["facts"][0]["field"] == "payment"


def test_extract_facts_from_vlm_payload_keeps_page_quote_and_confidence():
    facts = extract_facts_from_vlm_payload(
        {
            "facts": [
                {
                    "field": "payment",
                    "text": "Paiement a 30 jours",
                    "source_quote": "paiement dans un delai de 30 jours",
                    "confidence": "high",
                },
                {"field": "unknown", "text": "ignored"},
            ]
        },
        page=11,
        fields=("payment", "deadline"),
    )

    assert facts["payment"]["page"] == "11"
    assert facts["payment"]["source"] == "vlm"
    assert facts["payment"]["source_quote"] == "paiement dans un delai de 30 jours"
    assert facts["payment"]["confidence"] == "high"
    assert "unknown" not in facts


def test_merge_vlm_page_facts_prefers_higher_confidence():
    merged = merge_vlm_page_facts(
        {
            "payment": {
                "text": "weak",
                "page": "4",
                "source": "vlm",
                "confidence": "low",
            }
        },
        {
            "payment": {
                "text": "strong",
                "page": "8",
                "source": "vlm",
                "source_quote": "strong quote",
                "confidence": "high",
            }
        },
    )

    assert merged["payment"]["text"] == "strong"
    assert merged["payment"]["page"] == "8"


def test_parse_promoted_fields_filters_unknowns_and_duplicates():
    assert parse_promoted_fields("payment,unknown,payment,technical_documents") == (
        "payment",
        "technical_documents",
    )
