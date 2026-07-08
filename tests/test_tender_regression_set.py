from __future__ import annotations

import json
from pathlib import Path

from api.services.rag import TENDER_CHECKLIST_ITEMS


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tender_regression_expected.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_tender_regression_fixture_has_canonical_questions():
    fixture = _load_fixture()
    canonical = {
        field: {
            "question": question,
            "presence_question": presence_question,
        }
        for question, field, presence_question in TENDER_CHECKLIST_ITEMS
    }

    assert fixture["questions_source"] == "api.services.rag.TENDER_CHECKLIST_ITEMS"
    assert len(canonical) == 21
    assert fixture["candidate_documents"]
    assert any(doc["review_status"] == "trusted" for doc in fixture["documents"])

    for document in fixture["documents"]:
        fields = document["fields"]
        present_fields = [
            field
            for field, expected in fields.items()
            if expected["expected_status"] == "present"
        ]

        if document["review_status"] == "trusted":
            baseline = document.get("extractor_baseline")
            assert baseline
            assert 0 <= baseline["min_present_matches"] <= len(present_fields)
            assert baseline["min_present_matches"] <= baseline["target_present_matches"]
            assert baseline["target_present_matches"] == len(present_fields)

        assert set(fields) == set(canonical)

        for field, expected in fields.items():
            assert expected["question"] == canonical[field]["question"]
            assert expected["expected_status"] in {"present", "not_found", "needs_review"}
            assert expected["confidence"] in {"high", "medium", "low"}

            if expected["expected_status"] == "present":
                assert expected["expected"]
                assert expected["expected_contains"]
                assert all(str(part).strip() for part in expected["expected_contains"])
            elif expected["expected_status"] == "not_found":
                assert expected["expected"] is None
                assert expected["expected_contains"] == []
