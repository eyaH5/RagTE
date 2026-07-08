from __future__ import annotations

from scripts.benchmark_gemma_vlm import (
    build_openai_chat_payload,
    build_report,
    compare_field,
    expected_contains_match,
    parse_page_selection,
    verify_quote,
)


def test_parse_page_selection_supports_all_ranges_and_bounds():
    assert parse_page_selection("all", page_count=12, max_pages=5) == [1, 2, 3, 4, 5]
    assert parse_page_selection("2,4-6,99", page_count=10, max_pages=8) == [2, 4, 5, 6]


def test_expected_contains_match_handles_accents_and_arabic():
    assert expected_contains_match("Caution provisoire fixee a 500,000 DT", ["caution provisoire", "500"])
    assert expected_contains_match("الضمان النهائي بنسبة 3%", ["الضمان النهائي", "3"])


def test_verify_quote_marks_exact_approximate_and_missing():
    assert verify_quote("paiement dans un delai de 30 jours", "Le paiement dans un delai de 30 jours.")[
        "status"
    ] == "exact"

    approximate = verify_quote(
        "bordereau des prix et soumission",
        "Le dossier financier contient la soumission et le bordereau des prix.",
    )
    assert approximate["status"] == "approximate"

    assert verify_quote("garantie definitive", "Le document parle uniquement du paiement.")["status"] == "not_found"


def test_build_openai_chat_payload_uses_image_data_url():
    payload = build_openai_chat_payload(
        image_png=b"fake-png",
        prompt="Extract fields as JSON.",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        max_tokens=512,
    )

    content = payload["messages"][0]["content"]
    assert payload["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "Extract fields as JSON."}


def test_compare_field_uses_ground_truth_to_choose_winner():
    report = compare_field(
        field="penalties",
        current_fact={"text": "", "page": None},
        vlm_fact={
            "text": "Penalite de retard de 1 pour mille par jour.",
            "page": "12",
            "source_quote": "penalite de retard de 1 pour mille",
        },
        expected={"expected_status": "present", "expected_contains": ["penalite", "1 pour mille"]},
        page_texts={"12": "La penalite de retard de 1 pour mille sera appliquee."},
    )

    assert report["winner"] == "vlm"
    assert report["vlm_expected_match"] is True
    assert report["vlm_source_quote_check"]["status"] == "exact"


def test_compare_field_flags_vlm_false_positive_for_absent_field():
    report = compare_field(
        field="manufacturer_authorization",
        current_fact=None,
        vlm_fact={"text": "Attestation de garantie constructeur", "page": "6"},
        expected={"expected_status": "not_found", "expected_contains": []},
        page_texts={},
    )

    assert report["winner"] == "current"
    assert report["false_positive"] == {"current": False, "vlm": True}


def test_build_report_summarizes_field_winners(tmp_path):
    fixture_document = {
        "review_status": "trusted",
        "fields": {
            "payment": {"expected_status": "present", "expected_contains": ["30 jours"]},
            "variants": {"expected_status": "not_found", "expected_contains": []},
        },
    }
    report = build_report(
        pdf_path=tmp_path / "sample.pdf",
        filename="sample.pdf",
        selected_pages=[1],
        current_facts={"payment": {"text": "", "page": None}},
        current_source="text_cache",
        vlm_facts={"payment": {"text": "Paiement a 30 jours", "page": "1", "source_quote": "30 jours"}},
        page_reports=[{"page": "1", "fields": ["payment"]}],
        errors=[],
        vlm_seconds=1.2,
        fixture_document=fixture_document,
        page_texts={"1": "Paiement a 30 jours."},
        model="gemma4:e4b",
    )

    assert report["schema"] == "gemma_vlm_benchmark.v1"
    assert report["summary"]["vlm_wins"] == ["payment"]
    assert "payment" in report["summary"]["vlm_present_fields"]
