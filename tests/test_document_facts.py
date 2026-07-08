from __future__ import annotations

from pathlib import Path

import pytest

import ingest
from api.services.rag import (
    HALLUCINATION_SIGNALS,
    _answer_from_mined_facts,
    _detect_answer_language,
    _fact_list_field_for_question,
    _language_instruction,
    answer_from_document_facts,
    build_tender_checklist_answer,
)
from ingest import _entries_need_arabic_ocr, _should_use_direct_pdf_text, extract_document_facts


def _facts_from_text(filename: str, text: str, section: str = "general") -> dict:
    return extract_document_facts(
        [text],
        [{"source": filename, "page": "1", "section": section, "chunk_index": 0}],
    )


def test_text_quality_metadata_detects_page_gaps():
    quality = ingest._build_text_quality_metadata(
        [
            {"page": "1", "text": "Objet du marche et soumission."},
            {"page": "2", "text": "Offre technique."},
            {"page": "5", "text": "Modalites de paiement."},
        ],
        page_count=5,
        text_source="docling_ocr",
    )

    assert quality["mode"] == "partial_pages"
    assert quality["page_gap_count"] == 2
    assert quality["missing_page_ranges"] == ["3-4"]
    assert quality["text_source"] == "docling_ocr"


def test_text_quality_metadata_prioritizes_arabic_noisy_over_page_gaps():
    quality = ingest._build_text_quality_metadata(
        [
            {"page": "1", "text": "طلب عروض لاقتناء مواد إعلامية. " * 20},
            {"page": "4", "text": "العرض الفني والضمان الوقتي. " * 20},
        ],
        page_count=4,
        text_source="docling_ocr",
    )

    assert quality["mode"] == "arabic_noisy"
    assert quality["page_gap_count"] == 2
    assert quality["missing_page_ranges"] == ["2-3"]


def test_text_quality_metadata_keeps_french_page_gaps_as_partial_pages():
    quality = ingest._build_text_quality_metadata(
        [
            {"page": "1", "text": "Objet du marche et soumission."},
            {"page": "4", "text": "Modalites de paiement."},
        ],
        page_count=4,
        text_source="docling_ocr",
    )

    assert quality["mode"] == "partial_pages"
    assert quality["page_gap_count"] == 2


def test_text_quality_metadata_detects_noisy_non_arabic_ocr_before_page_gaps():
    quality = ingest._build_text_quality_metadata(
        [
            {"page": "1", "text": "@@@ ### !!! ???"},
            {"page": "4", "text": "$$$ *** !!! ???"},
        ],
        page_count=4,
        text_source="docling_ocr",
    )

    assert quality["mode"] == "noisy_ocr"
    assert quality["page_gap_count"] == 2


def test_low_signal_image_scan_ocr_forces_tesseract_recovery():
    entries = [
        {"page": str(page), "text": "Jall Clel aall lll e a o -- (1) (2) %0 %20"}
        for page in range(1, 26)
    ]

    assert _entries_need_arabic_ocr(entries)


def test_clean_arabic_ocr_does_not_force_extra_tesseract_pass():
    entries = [
        {
            "page": str(page),
            "text": (
                "كراس الشروط الخاص بطلب عروض لاقتناء معدات إعلامية. "
                "يحتوي العرض على الوثائق الإدارية والفنية والمالية والضمان الوقتي. "
                "تفتح العروض حسب الشروط المحددة في هذا الكراس."
            ),
        }
        for page in range(1, 10)
    ]

    assert not _entries_need_arabic_ocr(entries)


def test_extract_and_chunk_uses_vlm_ocr_for_low_signal_pdf(monkeypatch):
    low_signal_entries = [
        {"page": str(page), "text": "Jall Clel aall lll e a o -- (1) (2) %0 %20"}
        for page in range(1, 11)
    ]
    vlm_entries = [
        {
            "page": "1",
            "text": (
                "# Cahier des charges\n\n"
                "Article 1: Objet\n"
                "La presente consultation a pour objet l'acquisition de materiel informatique.\n\n"
                "La date limite de reception des offres est fixee au 17 avril 2017 a 12h."
            ),
            "source_type": "vlm_ocr",
        }
    ]
    calls: list[list[int] | None] = []

    monkeypatch.setattr(ingest, "VLM_OCR_ENABLED", True)
    monkeypatch.setattr(ingest, "VLM_OCR_MAX_PAGES", 10)
    monkeypatch.setattr(ingest, "_write_text_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingest, "_extract_text_entries_pypdf", lambda path: ([], 10))
    monkeypatch.setattr(ingest, "_extract_text_entries_pymupdf", lambda path: ([], 10))
    monkeypatch.setattr(ingest, "_extract_text_entries_docling", lambda path, filename: low_signal_entries)

    def fail_tesseract(*args, **kwargs):
        raise AssertionError("Tesseract should not run after successful early VLM OCR")

    monkeypatch.setattr(ingest, "_extract_text_entries_tesseract", fail_tesseract)

    def fake_vlm(file_path: str, filename: str, pages: list[int] | None = None) -> list[dict]:
        calls.append(pages)
        return vlm_entries

    monkeypatch.setattr(ingest, "_extract_text_entries_vlm", fake_vlm)

    chunks, metas, _ids = ingest.extract_and_chunk("scan.pdf", "scan.pdf")

    assert calls == [list(range(1, 11))]
    assert "l'acquisition de materiel informatique" in "\n".join(chunks)
    assert metas[0]["text_quality"]["text_source"] == "hybrid_vlm_ocr"
    assert metas[0]["source_type"] == "vlm_ocr"


def test_extract_and_chunk_keeps_clean_pdf_on_fast_path_when_vlm_ocr_enabled(monkeypatch):
    clean_entries = [
        {
            "page": "1",
            "text": """
            APPEL D'OFFRES N 01/2025 CAHIER DES CHARGES.
            Les offres doivent parvenir par voie postale.
            La date limite de reception des offres est fixee au 24 fevrier 2025.
            L'offre technique et l'offre financiere doivent etre placees dans deux enveloppes separees.
            La caution provisoire, les conditions de paiement et les penalites sont precisees.
            """,
        }
    ]

    monkeypatch.setattr(ingest, "VLM_OCR_ENABLED", True)
    monkeypatch.setattr(ingest, "_write_text_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingest, "_extract_text_entries_pypdf", lambda path: (clean_entries, 1))
    monkeypatch.setattr(ingest, "_extract_text_entries_pymupdf", lambda path: (clean_entries, 1))
    monkeypatch.setattr(ingest, "_extract_text_entries_docling", lambda *args, **kwargs: [])

    def fail_vlm(*args, **kwargs):
        raise AssertionError("VLM OCR should not run for clean direct text")

    monkeypatch.setattr(ingest, "_extract_text_entries_vlm", fail_vlm)

    chunks, metas, _ids = ingest.extract_and_chunk("clean.pdf", "clean.pdf")

    assert chunks
    assert metas[0]["text_quality"]["text_source"] == "pdf_text_layer"
    assert "24 fevrier 2025" in "\n".join(chunks)


def test_arabic_weak_fact_coverage_triggers_vlm_ocr(monkeypatch):
    entries = [
        {
            "page": "1",
            "text": (
                "كراس الشروط الخاص بطلب عروض لاقتناء معدات إعلامية. "
                "يتكون العرض من الوثائق الإدارية والفنية والمالية والضمان الوقتي."
            ),
        }
    ]
    facts = {
        "subject": {"text": "اقتناء معدات إعلامية", "page": "1"},
        "deadline": {"text": "17 أفريل 2017", "page": "1"},
    }

    monkeypatch.setattr(ingest, "VLM_OCR_ENABLED", True)
    monkeypatch.setattr(ingest, "VLM_OCR_ARABIC_FALLBACK_MIN_FACTS", 10)

    assert ingest._entries_need_vlm_ocr(
        entries,
        facts,
        page_count=8,
        text_source="hybrid_ocr",
    )


def test_arabic_weak_fact_coverage_does_not_trigger_for_pdf_text_layer(monkeypatch):
    entries = [
        {
            "page": "1",
            "text": "كراس الشروط الخاص بطلب عروض لاقتناء معدات إعلامية.",
        }
    ]

    monkeypatch.setattr(ingest, "VLM_OCR_ENABLED", True)

    assert not ingest._entries_need_vlm_ocr(
        entries,
        {"subject": {"text": "اقتناء معدات إعلامية", "page": "1"}},
        page_count=8,
        text_source="pdf_text_layer",
    )


def test_arabic_strong_fact_coverage_skips_vlm_ocr(monkeypatch):
    entries = [
        {
            "page": "1",
            "text": "كراس الشروط الخاص بطلب عروض لاقتناء معدات إعلامية.",
        }
    ]
    facts = {
        field: {"text": f"value {field}", "page": "1"}
        for field in (
            "subject",
            "submission_method",
            "deadline",
            "validity",
            "opening",
            "caution",
            "administrative_documents",
            "technical_documents",
            "financial_documents",
            "guarantee",
        )
    }

    monkeypatch.setattr(ingest, "VLM_OCR_ENABLED", True)
    monkeypatch.setattr(ingest, "VLM_OCR_ARABIC_FALLBACK_MIN_FACTS", 10)

    assert not ingest._entries_need_vlm_ocr(
        entries,
        facts,
        page_count=8,
        text_source="hybrid_ocr",
    )


def test_vlm_ocr_text_wins_over_noisy_same_page_candidate():
    merged = ingest._merge_vlm_entries_by_page(
        [
            {"page": "1", "text": "Jall Clel aall lll e a o -- (1) (2) %0 %20"},
            {"page": "2", "text": "Existing useful page text"},
        ],
        [
            {
                "page": "1",
                "text": "الفصل الأول: موضوع طلب العروض. اقتناء معدات إعلامية.",
                "source_type": "vlm_ocr",
            }
        ],
    )

    by_page = {entry["page"]: entry for entry in merged}
    assert "اقتناء معدات إعلامية" in by_page["1"]["text"]
    assert "Jall Clel" not in by_page["1"]["text"]
    assert by_page["1"]["source_type"] == "vlm_ocr"
    assert by_page["2"]["text"] == "Existing useful page text"


def test_vlm_ocr_uses_versioned_page_cache(monkeypatch, tmp_path):
    calls: list[dict] = []

    monkeypatch.setattr(ingest, "VLM_OCR_ENABLED", True)
    monkeypatch.setattr(ingest, "VLM_OCR_CACHE_ENABLED", True)
    monkeypatch.setattr(ingest, "VLM_OCR_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ingest, "VLM_OCR_MODEL", "test-vlm")
    monkeypatch.setattr(ingest, "VLM_OCR_DPI", 180)
    monkeypatch.setattr(ingest, "VLM_OCR_PROMPT_VERSION", "v1")
    monkeypatch.setattr(ingest, "_render_pdf_page_png", lambda *args, **kwargs: b"page-image")

    def fake_post(payload: dict, *, timeout: float) -> dict:
        calls.append(payload)
        return {"choices": [{"message": {"content": "## Page 1\n\nClean VLM transcription"}}]}

    monkeypatch.setattr(ingest, "_post_vlm_ocr_payload", fake_post)

    first = ingest._extract_text_entries_vlm("demo.pdf", "demo.pdf", pages=[1])
    assert calls
    assert first[0]["source_type"] == "vlm_ocr"

    def fail_post(*args, **kwargs):
        raise AssertionError("cached VLM OCR should not call the endpoint")

    monkeypatch.setattr(ingest, "_post_vlm_ocr_payload", fail_post)
    second = ingest._extract_text_entries_vlm("demo.pdf", "demo.pdf", pages=[1])

    assert second[0]["source_type"] == "vlm_ocr_cache"
    assert second[0]["text"] == first[0]["text"]


def test_vlm_ocr_cache_invalidates_on_prompt_version(monkeypatch, tmp_path):
    calls: list[str] = []

    monkeypatch.setattr(ingest, "VLM_OCR_ENABLED", True)
    monkeypatch.setattr(ingest, "VLM_OCR_CACHE_ENABLED", True)
    monkeypatch.setattr(ingest, "VLM_OCR_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ingest, "VLM_OCR_MODEL", "test-vlm")
    monkeypatch.setattr(ingest, "VLM_OCR_DPI", 180)
    monkeypatch.setattr(ingest, "_render_pdf_page_png", lambda *args, **kwargs: b"same-page-image")

    def fake_post(payload: dict, *, timeout: float) -> dict:
        calls.append(ingest.VLM_OCR_PROMPT_VERSION)
        return {"choices": [{"message": {"content": f"transcription {ingest.VLM_OCR_PROMPT_VERSION}"}}]}

    monkeypatch.setattr(ingest, "_post_vlm_ocr_payload", fake_post)

    monkeypatch.setattr(ingest, "VLM_OCR_PROMPT_VERSION", "v1")
    ingest._extract_text_entries_vlm("demo.pdf", "demo.pdf", pages=[1])

    monkeypatch.setattr(ingest, "VLM_OCR_PROMPT_VERSION", "v2")
    result = ingest._extract_text_entries_vlm("demo.pdf", "demo.pdf", pages=[1])

    assert calls == ["v1", "v2"]
    assert result[0]["text"] == "transcription v2"
    assert len(list(tmp_path.rglob("*.md"))) == 2


def test_extract_and_chunk_prefers_tsb_pdf_text_layer_when_cache_has_page_gaps(monkeypatch):
    pdf_path = Path(__file__).parents[1] / "pdfs" / "TUNISIAN SAUDI BANK.pdf"
    if not pdf_path.exists():
        pytest.skip("TSB fixture PDF is not available")

    monkeypatch.setattr(ingest, "TEXT_CACHE_READ_ENABLED", False)
    monkeypatch.setattr(ingest, "_write_text_cache", lambda *args, **kwargs: None)

    chunks, metas, _ids = ingest.extract_and_chunk(str(pdf_path), pdf_path.name)

    assert chunks
    assert metas[0]["text_quality"]["text_source"] == "pdf_text_layer"
    assert metas[0]["text_quality"]["preferred_source"] == "pdf_text_layer"

    joined = "\n".join(chunks).lower()
    assert "voie postale" in joined
    assert "24 février 2025" in joined
    assert "cnss" in joined
    assert "rne" in joined


def test_extract_and_chunk_supports_plain_text_files(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "_write_text_cache", lambda *args, **kwargs: None)

    path = tmp_path / "consultation.txt"
    path.write_text(
        "La presente consultation a pour objet l'acquisition de consommables informatiques.\n"
        "La date limite de reception des offres est fixee au 10/06/2025.",
        encoding="utf-8",
    )

    chunks, metas, ids = ingest.extract_and_chunk(str(path), path.name)

    assert chunks
    assert metas
    assert ids
    assert metas[0]["source"] == path.name
    facts = extract_document_facts(chunks, metas)
    assert facts["subject"]["text"].startswith("l'acquisition de consommables")
    assert facts["deadline"]["text"] == "10/06/2025"


def test_extract_and_chunk_supports_csv_files(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "_write_text_cache", lambda *args, **kwargs: None)

    path = tmp_path / "items.csv"
    path.write_text("Designation,Quantite\nToner HP,20\nScanner,2\n", encoding="utf-8")

    chunks, metas, _ = ingest.extract_and_chunk(str(path), path.name)

    joined = "\n".join(chunks)
    assert "Designation | Quantite" in joined
    assert "Toner HP | 20" in joined
    assert metas[0]["source"] == path.name


def test_extract_and_chunk_supports_docx_files(monkeypatch, tmp_path):
    docx = pytest.importorskip("docx")
    monkeypatch.setattr(ingest, "_write_text_cache", lambda *args, **kwargs: None)

    path = tmp_path / "consultation.docx"
    document = docx.Document()
    document.add_paragraph("La presente consultation a pour objet la fourniture de licences.")
    document.add_heading("Pieces administratives", level=1)
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Document"
    table.cell(0, 1).text = "Exigence"
    table.cell(1, 0).text = "CNSS"
    table.cell(1, 1).text = "Attestation d'affiliation"
    document.save(path)

    chunks, metas, _ = ingest.extract_and_chunk(str(path), path.name)

    joined = "\n".join(chunks)
    assert "fourniture de licences" in joined
    assert "CNSS | Attestation d'affiliation" in joined
    assert metas[0]["source"] == path.name
    assert any(meta.get("doc_type") == "docx" for meta in metas)
    assert any(meta.get("location") == "Section: Pieces administratives" for meta in metas)
    assert any(meta.get("section_heading") == "Pieces administratives" for meta in metas)


def test_direct_pdf_text_checker_rejects_tender_without_deadline_value():
    entries = [
        {
            "page": "1",
            "text": """
            APPEL D'OFFRES N 01/2025 CAHIER DES CHARGES.
            Les offres demeureront valables pendant 90 jours apres la date limite de reception des offres.
            L'offre technique et l'offre financiere doivent etre placees dans deux enveloppes separees.
            Ces enveloppes, en plus de la caution provisoire et du dossier administratif, seront deposees.
            Conditions de paiement et penalite de retard sont precisees dans le cahier des charges.
            """,
        }
    ]

    assert _should_use_direct_pdf_text(entries, page_count=1) is False


def test_direct_pdf_text_checker_accepts_tender_with_deadline_value():
    entries = [
        {
            "page": "1",
            "text": """
            APPEL D'OFFRES N 01/2025 CAHIER DES CHARGES.
            Les offres doivent parvenir par voie postale.
            La date limite de reception des offres est fixee au 24 fevrier 2025.
            L'offre technique et l'offre financiere doivent etre placees dans deux enveloppes separees.
            Conditions de paiement et penalite de retard sont precisees dans le cahier des charges.
            """,
        }
    ]

    assert _should_use_direct_pdf_text(entries, page_count=1) is True


def test_direct_pdf_text_checker_rejects_sparse_large_scanned_document():
    entries = [
        {
            "page": "1",
            "text": """
            REPUBLIQUE TUNISIENNE
            APPEL D'OFFRES N 01/2025 CAHIER DES CHARGES
            Quelques fragments extraits seulement.
            """,
        },
        {"page": "27", "text": "Date limite 20/05/2025."},
        {"page": "50", "text": "Prix Qte."},
    ]

    assert _should_use_direct_pdf_text(entries, page_count=50) is False


def test_extract_document_facts_marks_unusable_ocr_scan():
    chunks = [
        "ZO/S\nEST\nSEE\nIAE\nIIS\nFES\nS/E\nEEE\nSSS\nOO",
        "TEE\nESE\nSOO\nAI\nII\n12\n05\n3032\nNO\nNN",
        "AE\nRE\nSE\nTT\nLL\nPO\nQO\nEE\nSS\n00",
        "II\nII\nIII\nEE\nAA\nRR\nTT\nOO\nSS\nUU",
        "MM\nNN\nPP\nQQ\nRR\nSS\nTT\nVV\nWW\nXX",
    ]
    metas = [
        {"source": "UNKNOWN.pdf", "page": str(index + 1), "section": "general", "chunk_index": index}
        for index in range(len(chunks))
    ]

    facts = extract_document_facts(chunks, metas)

    assert "extraction_warning" in facts
    assert facts["extraction_warning"]["level"] == "warning"
    assert "Qualité d'extraction insuffisante" in facts["extraction_warning"]["text"]


def test_arabic_ocr_checker_retries_sparse_large_document_after_docling():
    entries = [
        {"page": "1", "text": "REPUBLIQUE TUNISIENNE APPEL D'OFFRES quelques fragments."},
        {"page": "25", "text": "TUNEPS tableau incomplet."},
        {"page": "50", "text": "Prix Qte."},
    ]

    assert _entries_need_arabic_ocr(entries) is True


def test_arabic_weak_profile_ocr_reinforcement_keeps_clause_pages():
    entries = [
        {"page": str(page), "text": "نص عربي عام من كراس الشروط."}
        for page in range(1, 23)
    ]
    entries[6]["text"] = (
        "إرسال العرض الفني والعرض المالي على منظومق الشراء العموميه علو الخط. "
        "وثيقة الضمان الوقتي وبطاقة الإرشادات والسجل الوطني."
    )
    entries[8]["text"] = "مدة الضمان من تاريخ القبول الوقتي."
    entries[10]["text"] = "غرامق الت خير وخطايا التأخير."
    entries[12]["text"] = "خلاص صاحب الصفقة وأمر بصرف المبالغ وفاتورة."

    pages = ingest._target_pages_for_ocr_reinforcement(
        entries,
        {"tender_profile": {"coverage": {"core_ratio": 0.25}}},
    )

    assert pages[:14] == list(range(1, 15))
    assert 8 in pages
    assert 11 in pages
    assert 13 in pages
    assert len(pages) <= ingest.OCR_REINFORCE_ARABIC_MAX_PAGES


def test_arabic_weak_profile_ocr_reinforcement_detects_clean_arabic_pages():
    entries = [
        {"page": str(page), "text": "نص عربي عام من كراس الشروط."}
        for page in range(1, 23)
    ]
    entries[7]["text"] = (
        "فتح العروض في نفس اليوم جلسة واحدة. "
        "وثيقة الضمان الوقتي وبطاقة الإرشادات والسجل الوطني للمؤسسات. "
        "التعهد المالي وجدول الأثمان."
    )
    entries[10]["text"] = "غرامة التأخير بنسبة 1000/01 في اليوم وبسقف 5%."
    entries[12]["text"] = "يصدر المشتري العمومي أمر بصرف المبالغ في أجل 30 يوما ثم الخلاص في أجل 15 يوما."

    pages = ingest._target_pages_for_ocr_reinforcement(
        entries,
        {"tender_profile": {"coverage": {"core_ratio": 0.25}}},
    )

    assert pages[:14] == list(range(1, 15))
    assert 8 in pages
    assert 11 in pages
    assert 13 in pages


def test_arabic_weak_profile_ocr_reinforcement_uses_preview_facts():
    entries = [
        {"page": str(page), "text": "latin OCR garbage without useful markers"}
        for page in range(1, 23)
    ]

    pages = ingest._target_pages_for_ocr_reinforcement(
        entries,
        {
            "subject": {"text": "طلب عروض لاقتناء مواد إعلامية"},
            "submission_method": {"text": "منظومة الشراء العمومي على الخط تونابس"},
            "tender_profile": {"coverage": {"core_ratio": 0.25}},
        },
    )

    assert pages[:14] == list(range(1, 15))
    assert 8 in pages
    assert 11 in pages
    assert 13 in pages


def test_weak_long_profile_ocr_reinforcement_keeps_front_clause_pages():
    entries = [
        {"page": str(page), "text": "fragment OCR de cahier des charges"}
        for page in range(1, 23)
    ]

    pages = ingest._target_pages_for_ocr_reinforcement(
        entries,
        {"tender_profile": {"coverage": {"core_ratio": 0.25}}},
    )

    assert pages[:14] == list(range(1, 15))
    assert 8 in pages
    assert 11 in pages
    assert 13 in pages


def test_extract_and_chunk_reinforces_weak_core_facts_with_targeted_ocr(monkeypatch):
    direct_entries = [
        {
            "page": "1",
            "text": "CAHIER DES CHARGES. Les offres sont deposees via TUNEPS. Validite indiquee au DPC.",
        },
        {
            "page": "23",
            "text": "Pieces objet de changement dans le cadre du marche: 1070119891Developpingunit, = 1",
        },
    ]
    seen = {}

    monkeypatch.setattr(ingest, "_extract_text_entries_pypdf", lambda path: (direct_entries, 23))
    monkeypatch.setattr(ingest, "_should_use_direct_pdf_text", lambda entries, page_count: True)
    monkeypatch.setattr(ingest, "_entries_need_arabic_ocr", lambda entries: False)
    monkeypatch.setattr(ingest, "_extracted_facts_need_ocr_reinforcement", lambda facts, entries: True)
    monkeypatch.setattr(ingest, "_target_pages_for_ocr_reinforcement", lambda entries, facts: [1, 2, 3])
    monkeypatch.setattr(ingest, "_write_text_cache", lambda filename, entries: None)

    def fake_tesseract(path, filename, pages=None):
        seen["pages"] = pages
        return [
            {
                "page": "1",
                "text": "Objet : Acquisition de pieces de rechange pour imprimantes. Date limite : 20/05/2026.",
            }
        ]

    monkeypatch.setattr(ingest, "_extract_text_entries_tesseract", fake_tesseract)

    chunks, metas, ids_out = ingest.extract_and_chunk("demo.pdf", "demo.pdf")

    assert seen["pages"] == [1, 2, 3]
    assert any("Acquisition de pieces de rechange" in chunk for chunk in chunks)
    assert len(chunks) == len(metas) == len(ids_out)


def test_requested_items_are_extracted_from_short_license_request():
    facts = _facts_from_text(
        "BH_ASSURANCE.pdf",
        """
        BH ASSURANCE CONSULTATION-DSI-05-2025 Objet : Consultation pour le renouvellement
        des Licences Veeam. Rubriques Quantite Id de Licence Support Id
        Veeam Backup for Microsoft 365 125 utilisateurs 99D1B297 #02714153
        Veeam Availability Suite 100 Instances E98D6984 #02714151
        """,
    )

    requested = facts["requested_items"]
    normalized = requested["text"].lower()
    assert "veeam backup for microsoft 365 : 125 utilisateurs" in normalized
    assert "id licence 99d1b297" in normalized
    assert "support id #02714153" in normalized
    assert "veeam availability suite : 100 instances" in normalized
    assert "support id #02714151" in normalized
    assert _fact_list_field_for_question("Quel est le support id ?") == "requested_items"
    assert _fact_list_field_for_question("Quelles licences sont demandées ?") == "requested_items"


def test_requested_items_are_extracted_from_multiline_license_table():
    facts = _facts_from_text(
        "BH_ASSURANCE.pdf",
        """
        BH ASSURANCE
        CONSULTATION-DSI-05-2025
        Objet : Consultation pour le renouvellement des Licences Veeam.
        Rubriques Quantite Id de Licence Support Id
        Veeam Backup for
        Microsoft 365
        125
        utilisateurs
        99D1B297-263D-B66F-3705-
        C50657F317E0
        #02714153
        Veeam Availability
        Suite
        100 Instances E98D6984-C8E3-B99A-
        2D9D-9C501F16DBF3
        #02714151
        """,
    )

    normalized = facts["requested_items"]["text"].lower()
    assert "veeam backup for microsoft 365 : 125 utilisateurs" in normalized
    assert "support id #02714153" in normalized
    assert "veeam availability suite : 100 instances" in normalized
    assert "support id #02714151" in normalized


def test_requested_items_and_deadline_are_extracted_from_quantity_first_table():
    facts = _facts_from_text(
        "BANQUE_CENTRALE_DE_TUNISIE.pdf",
        """
        Quantite Designation
        20 TONER LEXMARK CX 431ADW (20NOX20) : BLEU
        20 TONER LEXMARK CX 431 ADW (20NOX10) : NOIR
        20 BOUTEILLE RECUP TONER (WASTE TONER BOTTLE) LEXMARK CX 431ADW (20NOW00)
        Votre offre doit etre expediee par voie postale ou deposee directement au Bureau d'Ordre
        Central au plus tard le 27/01/2025, sous plis ferme et anonyme.
        """,
    )

    normalized = facts["requested_items"]["text"].lower()
    assert "toner lexmark cx 431adw" in normalized
    assert ": 20" in normalized
    assert "bouteille" in normalized
    assert facts["deadline"]["text"] == "27/01/2025"


def test_requested_items_are_extracted_from_generic_product_table():
    facts = _facts_from_text(
        "GENERIC_TABLE.pdf",
        """
        Rubriques Quantite Reference Support ID
        Microsoft 365 Business Premium 50 licences MS365-BP-50 #112233
        Fortinet FortiGate 100F 2 unites FG100F-BDL #445566
        """,
    )

    normalized = facts["requested_items"]["text"].lower()
    assert "microsoft 365 business premium : 50 licences" in normalized
    assert "support id #112233" in normalized
    assert "fortinet fortigate 100f : 2 unites" in normalized
    assert "support id #445566" in normalized


def test_requested_items_are_extracted_from_item_designation_quantity_table():
    facts = _facts_from_text(
        "GENERIC_BORDEREAU.pdf",
        """
        Bordereau des prix
        Item Designation Unite Qte
        1 Firewall FortiGate 100F Unite 2
        2 Switch 24 ports PoE Unite 4
        3 Licence antivirus poste 36 mois Licences 150
        """,
    )

    normalized = facts["requested_items"]["text"].lower()
    assert "firewall fortigate 100f : 2 unite" in normalized
    assert "switch 24 ports poe : 4 unite" in normalized
    assert "licence antivirus poste 36 mois : 150 licences" in normalized

    answer, _ = _answer_from_mined_facts(
        "GENERIC_BORDEREAU.pdf",
        "Quels articles sont demandés ?",
        facts,
    )
    assert "Firewall FortiGate 100F" in answer
    assert "Switch 24 ports PoE" in answer


def test_requested_items_ignore_metric_and_annex_table_noise():
    facts = _facts_from_text(
        "GENERIC_BORDEREAU.pdf",
        """
        Bordereau des prix
        Item Designation Unite Qte
        1 et versions ulterieures Windows Server 2016 Unite 11
        2 Admin externes / Unite 35
        3 de ressources a proteger Unite 25
        4 DC 02 ANNEXE 3 Modele de bordereau des prix LOT 2
        5 Mise en Place d'une Solution PAM Unite 1
        """,
    )

    normalized = facts["requested_items"]["text"].lower()
    assert "mise en place d'une solution pam : 1 unite" in normalized
    assert "windows server" not in normalized
    assert "admin externes" not in normalized
    assert "ressources a proteger" not in normalized
    assert "modele de bordereau" not in normalized


def test_rose_blanche_financial_documents_and_subject_are_extracted():
    facts = _facts_from_text(
        "Rose_Blanche.pdf",
        """
        ARTICLE 1 - OBJET DE L'APPEL D'OFFRES
        Le présent appel d'offres a pour objet la fourniture, l'installation, la configuration,
        l'intégration et la mise en production de solutions de cybersécurité répondant aux
        besoins de la rose Blanche selon la répartition suivante :
        1. LOT 1 Solution de type Antivirus + EDR (EndPoint Detection and Response)
        2. Lot 2 : Mise en place d'une Solution PAM

        3. Fichier Zip INTERIEURE « F » OFFRE FINANCIERE :
        N° DOCUMENTS OPERATION A REALISER AUTHENTIFICATION
        F1 La soumission Remplir le modèle fourni en ANNEXE 2 Original du document remis
        par Rose Blanche dûment complété par le soumissionnaire Datée et portant signature
        et cachet du soumissionnaire à la fin du document.
        F2 Le bordereau des prix (Remplir le modèle fourni en annexe) Original du document
        remis par Rose Blanche dûment complété par le soumissionnaire Paraphe, signature
        & cachet du soumissionnaire.
        """,
    )

    assert "Antivirus + EDR" in facts["subject"]["text"]
    assert "PAM" in facts["subject"]["text"]
    normalized_financial = facts["financial_documents"]["text"].lower()
    assert "la soumission" in normalized_financial
    assert "bordereau des prix" in normalized_financial


def _rose_blanche_table_facts() -> dict:
    chunks = [
        """
        ARTICLE 1 - OBJET DE L'APPEL D'OFFRES
        Le present appel d'offres a pour objet la fourniture, l'installation, la configuration,
        l'integration et la mise en production de solutions de cybersecurite repondant aux
        besoins de la rose Blanche selon la repartition suivante :
        1. LOT 1 Solution de type Antivirus + EDR (EndPoint Detection and Response)
        2. Lot 2 : Mise en place d'une Solution PAM
        """,
        """
        LOT 1 : Solution de type Antivirus + EDR
        Nombre de Endpoint a proteger :
        PC : 1300
        Serveur : 120
        Mobile : 100
        """,
        """
        LOT 2 : Mise en place d'une Solution PAM
        Nombre des utilisateurs de la plateforme 35 Admin externes / 25 Admin internes
        Nombre de ressources a proteger 150
        """,
        """
        ANNEXE 3 Modele de bordereau des prix (Lot 1)
        LOT 1: Une solution Antivirus + EDR
        Item Designation Unite Qte Prix U. HT Total HT
        1 Mise en Place dune Solution Antivirus + EDR Unite 1
        4 Installation et mise en place de la solution Unite 1
        5 Formation et Transfert de competences Unite 1
        6 Support sur 36 mois a payer annuellement sur Trois ans Unite 1
        """,
        """
        ANNEXE 3 Modele de bordereau des prix (Lot 2)
        LOT 2: Une solution PAM
        Item Designation Unite Qte Prix U. HT Total HT
        1 Mise en Place dune Solution PAM selon le sizing propose Unite 1
        4 Installation et mise en place de la solution Unite 1
        5 Formation et Transfert de competences Unite 1
        6 Support sur 36 mois a payer annuellement sur Trois ans Unite 1
        """,
    ]
    metas = [
        {"source": "Rose_Blanche.pdf", "page": str(page), "section": "general", "chunk_index": page - 1}
        for page in (2, 12, 31, 39, 40)
    ]
    return extract_document_facts(chunks, metas)


def test_rose_blanche_mined_facts_cover_lots_metrics_and_bordereaux():
    facts = _rose_blanche_table_facts()

    mined = facts["mined_facts"]
    normalized = mined["text"].lower()
    assert "lot 1" in normalized
    assert "antivirus" in normalized
    assert "lot 2" in normalized
    assert "pam" in normalized
    assert "nombre de endpoint pc : 1300" in normalized
    assert "nombre de endpoint serveur : 120" in normalized
    assert "nombre de endpoint mobile : 100" in normalized
    assert "35 admin externes" in normalized
    assert "ressources" in normalized
    assert "150" in normalized
    assert "support sur 36 mois" in normalized


def test_mined_facts_answer_table_questions_without_generic_rag():
    facts = _rose_blanche_table_facts()

    answer, _ = _answer_from_mined_facts(
        "Rose_Blanche.pdf",
        "Quels sont les lots demandes ?",
        facts,
    )
    assert "Lot 1" in answer
    assert "Antivirus + EDR" in answer
    assert "Lot 2" in answer
    assert "PAM" in answer

    answer, metas = _answer_from_mined_facts(
        "Rose_Blanche.pdf",
        "Combien de endpoints sont demandes pour le lot Antivirus + EDR ?",
        facts,
    )
    assert "1300" in answer
    assert "120" in answer
    assert "100" in answer
    assert metas[0]["source"] == "Rose_Blanche.pdf"

    answer, _ = _answer_from_mined_facts(
        "Rose_Blanche.pdf",
        "Combien d'administrateurs sont prevus pour la solution PAM ?",
        facts,
    )
    assert "35 Admin externes" in answer
    assert "25 Admin internes" in answer

    answer, _ = _answer_from_mined_facts(
        "Rose_Blanche.pdf",
        "Quels sont les elements du bordereau des prix pour le lot 2 ?",
        facts,
    )
    assert "Mise en Place dune Solution PAM" in answer
    assert "Support sur 36 mois" in answer


def _republique_tunisienne_table_facts() -> dict:
    chunks = [
        """
        OFFICE DE LA TOPOGRAPHIE ET DU CADASTRE
        إستشارة عدد 2025/17 لإقتناء 04 Coupeuses de plans A0
        يتم تقديم العروض عبر منظومة الشراء العمومي على الخط TUNEPS على الموقع www.tuneps.tn
        لقبول العروض يوم 2025/04/10
        """,
        """
        Caractéristiques Techniques Coupeuse de Plan A0 (36 ")
        Caractéristique technique Minimum demandé
        Fonctionnalité professionnelle grand format papier A0
        Type de coupe Manuelle
        Coupe bidirectionnel oui
        Longueur de coupe : Papier A0 | > 1190 mm
        Orientation papier A0 à découper | Portrait et paysage
        Graduations mm
        Dispositif de pression Automatique
        Table de coupe Métallique
        Bac de récupération des chutes papiers Avec bac de récupération des chutes papiers
        Equipement de sécurité Tête de coupe carénée
        Protection intégrale de la lame Oui
        Chariot porte lame Oui
        Lame interchangeable Oui
        """,
        """
        Prix unitaire TVA Désignation Qté Prix Total HTVA
        Coupeuse de plans grand format papier A0 (36 ") 04
        PRIX TOTALE HORS TVA
        """,
    ]
    metas = [
        {"source": "REPUBLIQUE_TUNISIENNE.pdf", "page": "1", "section": "deadline", "chunk_index": 0},
        {"source": "REPUBLIQUE_TUNISIENNE.pdf", "page": "3", "section": "technical", "chunk_index": 1},
        {"source": "REPUBLIQUE_TUNISIENNE.pdf", "page": "4", "section": "financial", "chunk_index": 2},
    ]
    return extract_document_facts(chunks, metas)


def test_republique_tunisienne_arabic_ocr_style_facts_are_extracted():
    facts = _republique_tunisienne_table_facts()

    assert "04 Coupeuses de plans A0" in facts["subject"]["text"]
    assert facts["deadline"]["text"] == "2025/04/10"
    assert "TUNEPS" in facts["submission_method"]["text"]

    mined = facts["mined_facts"]["text"].lower()
    assert "coupeuse de plans grand format papier a0" in mined
    assert "orientation papier a0" in mined
    assert "portrait et paysage" in mined
    assert "longueur de coupe papier a0" in mined
    assert "1190 mm" in mined


def test_extract_document_facts_handles_arabic_tender_checklist_fields():
    chunks = [
        """
        الفصل 1  :موضوع الاستشارة
        تعتزم شركة اللحوم بصفتها "المشتري العمومي" إجراء استشارة تتعلق باقتناء ووضع في طور إستخدام
        رخص مضاد للفيروسات لـ03 مستعمل و1 خادم windows server 2016.
        الفصل4 :تقديم العروض
        يتم إيداع العرض المتكون من الوثائق الإدارية والعرض الفني و المالي عبر منظومة الشراء العمومي على الخط TUNEPS.
        وقد حدد آخر أجل لقبول العروض بواسطة منظومة الشراء العمومي على الخط يوم 03 ماي 2025 على الساعة التاسعة والنصف صباحا.
        الفصل5 :صلوحية العروض
        يلتزم العارض بعرضه لمدة تسعون يوما بداية من اليوم الموالي للتاريخ الأقصى المحدد لقبول العروض.
        الفصل 6 :الوثائق المكونة لملف طلب العروض
        I. الوثائق الإداريةوتتكون من :
        1. كراس الشروط الإدارية والفنية في نسخته الأصلية مؤشر عليه الكترونيا.
        2. بطاقة إرشادات عامة حول العارض طبقا للملحق عدد 01.
        3. وثيقة الضمان المالي الوقتي بمبلغ قدره 125 دينار.
        4. نظير أصلي من السجل الوطني للمؤسسات.
        """,
        """
        II. الوثائق الخاصة بالعرض الفني:
        1. جدول الخصائص الفنية معمرة بدقة ويكون ممضى ومختوما.
        2. البطاقات الفنية Prospectus technique.
        3. إحدى شهادات المطابقة للمواصفات العالمية.
        III. وثائق الخاصة بالعرض المالي:
        1. التعهد المالي معمر بدقة ويكون ممضى ومختوما.
        2. جدول الأثمان بالدينار التونسي باحتساب جميع الأداءات.
        3. مشروع عقد الصيانة معمر بكل دقة.
        الفصل 8 :فتح العروض:
        يتم فتح العروض يوم 03 ماي 2025 على الساعة العاشرة صباحا بمقر شركة اللحوم.
        الفصل 12 :الضمان المالي النهائي:
        يجب على المزود المقبول أن يقدم ضمانا ماليا نهائيا بنسبة عشرة بالمائة (10%) من القيمة الجملية للطلبية.
        الفصل 14 :خلاص الطلبية:
        يتم تسديد مستحقات المزود في أجل أقصاه خمسة عشر يوما بعد تقديم فاتورة ومحضر التسليم الوقتي.
        الفصل15 :عقوبة التأخير
        غرامة تأخير قدرها ثلاثة بالألف (3‰) عن كل يوم تأخير.
        """,
    ]
    metas = [
        {"source": "YESNETWORKS_TECHNOLOGIES.pdf", "page": "2", "section": "general", "chunk_index": 0},
        {"source": "YESNETWORKS_TECHNOLOGIES.pdf", "page": "3", "section": "general", "chunk_index": 1},
    ]

    facts = extract_document_facts(chunks, metas)

    assert "رخص مضاد للفيروسات" in facts["subject"]["text"]
    assert "TUNEPS" in facts["submission_method"]["text"]
    assert "03 ماي 2025" in facts["deadline"]["text"]
    assert "تسعون يوما" in facts["validity"]["text"]
    assert "فتح العروض" in facts["opening"]["text"]
    assert "125 دينار" in facts["caution"]["text"]
    assert "بطاقة إرشادات" in facts["administrative_documents"]["text"]
    assert "السجل الوطني للمؤسسات" in facts["administrative_documents"]["text"]
    assert "جدول الخصائص الفنية" in facts["technical_documents"]["text"]
    assert "جدول الأثمان" in facts["financial_documents"]["text"]
    assert "10%" in facts["definitive_caution"]["text"]
    assert "فاتورة" in facts["payment"]["text"]
    assert "غرامة تأخير" in facts["penalties"]["text"]


def test_extract_document_facts_handles_cdc_01_arabic_ocr_noise():
    chunks = [
        """
        طلب عروض عدد 2026/01 خاص باقتناء مواد اإعلامية لفائدة وزارة العدل.
        على المترشح ارسال العرض الفني والعرض المالي وكراس الشروط الإدارية والفنية والتصاريح على الشرف
        على منظومق الشراء العموميه علو الخط "توزيبس" وبفوات التاريخ والساعة المحدداز يغلق باب الإيداع.
        يتضمن الوثائق التاليقة: كراس الشروط الإدارية والفنية، العرض الفني حسب كل قسط، شهادق المطابقة
        للمواصفات الفنيق 7509001 نسخة 2015، شهادات المطابقة لمواصفات 14001 ISO، تقرير اختبار لعدد
        الصفحات، تعمير جداول الخاصيات الفنية وتقديم جذاذات فنية للمواد المطلوبة.
        العرض المالي يتضمن التعهد المالي حسب كل قسط وجدول الأشمان حسب كل قسط.
        الوثائق التي ترسل مباشرة: وشيقة الضمان الوقتي، نظير من لسجل الوطني للمؤسسات، الوشائق المثبتة
        للمؤسسات الصغرى، بطاقق الإرشادات، تصريح علو الشرف باستقلالية المؤسسة الصغرى.
        ترسل في ظرف مغلق عبر البريد مضمون الوصول أو البريد السريع أو تسلم مباشرة إلى مكتب الضبط.
        تنعقد جلسق فتح العروض وجوبا في نفس اليوم المحدد كتاريخ أقصى لقبول العروض وتجتمع لجنة فتح
        العروض في جلسق واحدق وتكون هذه الجلسة علنية.
        """,
        """
        يشترط أن يكون هذا الضمان صالح لمدق 120 يوما ابتداء من التاريخ الأقصى لقبول العروض.
        يطالب المترشح الذي تم الاحتفاظ بعرضه بتقديم ضمان نهائي يساوي 3 96 من القيمق الأصلية للصفقة
        خلال العشرين (20) يوما الموالية لإعلامه بالموافقة على الصفقة.
        مدق الضمان يجب أن لا تكون أقل مز سنق مز تاريخ القبول الوقتي ويتعهد بتعويض المواد الإعلامية
        التي بها عيوب في الصنع في أجل 7 أيام.
        يتم اعداد محضر الاستلام وامضاؤه كنتيجة لذلك. ب- الاستلام لنهائي: شريطة أن لا تكون هناك تحفظات.
        لفصل 15: غرامق الت خير تطبق عقوبة مالية على المزود على أساس واحد مز الألف (1000/01)
        عز كل يوم تأخير ولا يمكن أن تتجاوز جملة خطايا التأخير نسبة 5 96.
        الفصل 24: خلاص Jess يلتزم صاحب الصفقة بتقديم فاتورق إلى الإدارة في 4 نظائر.
        وعلى المشتري العمومي اصدار أمر بصرف المبالغ الراجعة لصاحب الصفقة في أجل ثلاثون (30) يوما
        ويتعين على المحاسب العمومي خلاص صاحب الصفقة في أجل خمس عشر (15) يوما.
        """,
    ]
    metas = [
        {"source": "CDC_01-2026.pdf", "page": "7", "section": "general", "chunk_index": 0},
        {"source": "CDC_01-2026.pdf", "page": "12", "section": "general", "chunk_index": 1},
    ]

    facts = extract_document_facts(chunks, metas)

    assert "اقتناء مواد" in facts["subject"]["text"]
    assert "منظومة الشراء العمومية" in facts["submission_method"]["text"]
    assert "مكتب الضبط" in facts["submission_method"]["text"]
    assert "فتح العروض" in facts["opening"]["text"]
    assert "جلسة واحدة" in facts["opening"]["text"]
    assert "الضمان الوقتي" in facts["caution"]["text"]
    assert "120 يوما" in facts["caution"]["text"]
    assert "بطاقة الإرشادات" in facts["information_sheet"]["text"]
    assert "السجل الوطني للمؤسسات" in facts["rne"]["text"]
    assert "تصريح على الشرف" in facts["administrative_documents"]["text"]
    assert "ISO 9001" in facts["technical_documents"]["text"]
    assert "ISO 14001" in facts["technical_documents"]["text"]
    assert "التعهد المالي" in facts["financial_documents"]["text"]
    assert "جدول الأثمان" in facts["financial_documents"]["text"]
    assert "3 96" in facts["definitive_caution"]["text"]
    assert "20" in facts["definitive_caution"]["text"]
    assert "سنة" in facts["guarantee"]["text"]
    assert "الاستلام النهائي" in facts["reception"]["text"]
    assert "غرامة التأخير" in facts["penalties"]["text"]
    assert "1000/01" in facts["penalties"]["text"]
    assert "أمر بصرف" in facts["payment"]["text"]
    assert "30" in facts["payment"]["text"]


def test_extract_document_facts_handles_arabic_vlm_markdown_sections():
    chunks = [
        """
        الفصل الأول: موضوع طلب العروض.
        تعتزم وزارة التنمية والاستثمار والتعاون الدولي إجراء طلب عروض قصد اقتناء معدات إعلامية في إطار مشروع الدعم المؤسسي.
        ويتكون طلب العروض من ثلاث حصص:
        - الحصة عدد1: جدار ناري ونقاط وصول لاسلكية "Wireless Access Point(WAP)"
        - الحصة عدد2: معدات لحفظ المعطيات
        - الحصة عدد3: معدات وبرامج فيديوية للمؤتمرات "Système de visioconférence"
        الفصل 04: كيفية تقديم العروض.
        يتكون العرض من:
        - العرض الفني.
        - العرض المالي.
        يجب تضمين العرض الفني والعرض المالي في ظرفين منفصلين.
        يتضمن الطرف الخارجي إلى جانب العرضين الفني والمالي الوثائق التالية:
        - كراس الشروط الإدارية الخاص مؤشر عليه وممضى في آخر صفحة.
        - شهادة الخراط في نظام للضمان الإجتماعي.
        - شهادة في الوضعية الجبائية.
        - نظير من السجل التجاري.
        - الضمان الوقتي (الملحق عدد 1).
        - بطاقة إرشادات عامة حول المشارك (الملحق عدد 4).
        ويتكون العرض الفني من الوثائق التالية:
        - قائمة في الحصص المشارك فيها (الملحق عدد 5).
        - جداول الخصائص الفنية.
        - المطويات الفنية «Prospectus Techniques».
        - الوثائق المتعلقة بالسلامة والجودة.
        يتكون العرض المالي من الوثائق التالية:
        - التعهد المالي ممضى ومختوم (الملحق عدد 6).
        - العرض المالي ممضى ومختوم (تعمير الملحق عدد 7).
        ترسل الظروف المحتوية على العروض الفنية والمالية عن طريق البريد المضمون الوصول أو عن طريق البريد السريع أو تسليم مباشرة إلى مكتب الضبط المركزي.
        وذلك في أجل أقصاه يوم الإثنين 17 أفريل 2017 على الساعة منتصف النهار.
        الفصل 05: الوثائق التعاقدية.
        """,
        """
        الفصل 06: صلاحيّة العروض.
        يتعهد العارض بالإبقاء على صلاحيّته عرضه لمدة تسعة (90) يوما بداية من اليوم الموالي للتاريخ الأقصى المحدّد لقبول العروض.
        ويمكن طلب التمديد في آجال صلاحيّة العروض إلى أجل أقصاه مائة وعشرون (120) يوما.
        الفصل 07: الضمان الوقتي.
        على كل عارض أن يقدم ضمانا وقتيا بمبلغ ثابت للحصة الأولى والثانية والثالثة كما هو مبين بالجدول التالي.
        مبلغ الضمان الوقتي (بالدينار).
        | الحصة | عدد |
        | --- | --- |
        | 500 | 1 |
        | 350 | 2 |
        | 400 | 3 |
        الفصل 08: فتح الظروف.
        تجمع لجنة الشراءات لفتح الظروف المحتوية على العروض الفنية والمالية في جلسة واحدة علنية وذلك يوم الإثنين 17 أبريل 2017 على الساعة الثالثة والنصف.
        الفصل 09: منهجية تقييم العروض.
        """,
        """
        - الاستلام الودي.
        يتم الاستلام الودي بحضور ممثلين عن وزارة التنمية والاستثمار والتعاون الدولي وبحضور المزود.
        - الاستلام النهائي.
        يتم تحرير محضر الاستلام النهائي ويتم إمضاؤه من طرف الإدارة والمزود بعد انقضاء مدة الضمان.
        الفصل 19: ضمان المعدات.
        يجب ألا تقل مدة الضمان عن سنة ابتداء من تاريخ القبول الودي للمعدات.
        الفصل 20: غرامات التأخير
        يتم توظيف غرامة التأخير حسب الصيغة: غ = (ق × ت × 3) / 1000.
        ولا يمكن أن يتجاوز المبلغ الجملي لغرامة التأخير 5% من المبلغ الأصلي للصفقة.
        الفصل 21: آجال خلاص صاحب الصفقة
        يتم خلاص الصفقة في أجل أقصاه ثلاثون (30) يوما بعد تقديم فاتورة في أربعة نظائر ومصحوبة بأذون التسليم.
        الفصل 22: فسخ الصفقة
        """,
    ]
    metas = [
        {"source": "MDCI.pdf", "page": "3", "section": "general", "chunk_index": 0},
        {"source": "MDCI.pdf", "page": "4", "section": "general", "chunk_index": 1},
        {"source": "MDCI.pdf", "page": "8", "section": "general", "chunk_index": 2},
    ]

    facts = extract_document_facts(chunks, metas)

    assert "Wireless Access Point" in facts["subject"]["text"]
    assert "البريد المضمون الوصول" in facts["submission_method"]["text"]
    assert "مكتب الضبط المركزي" in facts["submission_method"]["text"]
    assert "17 أفريل 2017" in facts["deadline"]["text"]
    assert "90" in facts["validity"]["text"]
    assert "120" in facts["validity"]["text"]
    assert "فتح الظروف" in facts["opening"]["text"]
    assert "جلسة واحدة علنية" in facts["opening"]["text"]
    assert "500" in facts["caution"]["text"]
    assert "350" in facts["caution"]["text"]
    assert "400" in facts["caution"]["text"]
    assert "الضمان الإجتماعي" in facts["cnss"]["text"]
    assert "الوضعية الجبائية" in facts["fiscal_certificate"]["text"]
    assert "السجل التجاري" in facts["rne"]["text"]
    assert "كراس الشروط" in facts["administrative_documents"]["text"]
    assert "بطاقة إرشادات" in facts["administrative_documents"]["text"]
    assert "جداول الخصائص الفنية" in facts["technical_documents"]["text"]
    assert "Prospectus Techniques" in facts["technical_documents"]["text"]
    assert "التعهد المالي" in facts["financial_documents"]["text"]
    assert "الملحق عدد 7" in facts["financial_documents"]["text"]
    assert "مدة الضمان" in facts["guarantee"]["text"]
    assert "سنة" in facts["guarantee"]["text"]
    assert "الاستلام النهائي" in facts["reception"]["text"]
    assert "1000" in facts["penalties"]["text"]
    assert "5%" in facts["penalties"]["text"]
    assert "خلاص الصفقة" in facts["payment"]["text"]
    assert "فاتورة" in facts["payment"]["text"]


def test_extract_document_facts_handles_arabic_tuneps_offer_document_tables():
    chunks = [
        """
        طلب عروض عدد 2016/droi/01
        للتزود بحبر ومتممات وقطع غيار للتجهيزات الإعلامية
        آخر أجل لقبول العروض : 04 أفريل 2016 الساعة 10:00
        جلسة فتح الظروف : 04 أفريل 2016 الساعة 11:00
        """,
        """
        موضوع طلب العروض
        يتمثل موضوع طلب العروض هذا في عقد صفقة عادية للتزود بحبر ومتممات وقطع غيار للتجهيزات الإعلامية لفائدة الإدارات المركزية التابعة للوزارة، متكونة من الحصص التالية:
        حصة عدد 1: حبر لآلات طباعة من نوع HPLaser et traceurs HP.
        حصة عدد 6: قطع غيار ومستلزمات إعلامية.
        شروط المشاركة
        صلاحية العروض يصبح المشاركون ملزمون بعروضهم بمجرد تقديمها لمدة تسعين (90) يوما ابتداء من اليوم الموالي لآخر أجل محدد لقبول العروض.
        الضمان الوقتي حدد مبلغ الضمان الوقتي بــ: حصة عدد 1: ثلاثمائة (300) دينار.
        طريقة تقديم العروض
        يتمّ تقديم كل من العرض الفني والعرض المالي عبر منظومة الشراء العمومي على الخط TUNEPS على الموقع www.tuneps.tn.
        يجب أن توجه الظروف المحتوية على العروض الفنية والمالية عن طريق البريد مضمون الوصول أو عن طريق البريد السريع أو تسلمّ مباشرة إلى مكتب الضبط المركزي وزارة التجهيز والإسكان والتهيئة الترابية.
        الوثائق المكوّنة للعرض
        1.10- الظرف المتضمن للوثائق الإدارية والضّمان الوقتي والعرض الفنّي
        عدد | اسم الوثيقة | طريقة التقديم | منظومة تونيبس
        1 | شروط طلب العروض (CAO) | ختم المشارك | إضافته بالمرفقات
        2 | كرّاس الشّروط الإدارية الخاصة (CCAP) | ختم المشارك | إضافته بالمرفقات
        3 | كراس الشروط لتسويق المعدات الإعلامية ممضاة لدى وزارة التجارة (للحصة عدد 6) | نسخة | إرسال نسخة خارج الخط
        4 | الضّمان الوقتي أو التزام الكفيل بالتضامن | نسخة تحتوي على إمضاء و ختم المؤسسة البنكية | إرسال نسخة خارج الخط
        5 | شهادة في الانخراط في إحدى الصناديق الإجتماعية | نسخة | غير مطلوبة (توفرها تونيبس)
        6 | شهادة في الوضعية الجبائية صالحة إلى آخر أجل لقبول العروض | نسخة | غير مطلوبة (توفرها تونيبس)
        7 | تصريح على الشرف في عدم الانتماء (الملحق عدد 7) | تعمير وتأريخ وإمضاء وختم | إضافته بالمرفقات
        8 | تصريح على الشرف يقدمه المشارك بأنه ليس في حالة إفلاس أو تسوية قضائية (ملحق عدد 3) | تعمير وتأريخ وإمضاء وختم | إضافته بالمرفقات
        9 | تصريح على الشرف بعدم التأثير في إجراءات إبرام الصفقة. (ملحق عدد 4) | تعمير وتأريخ وإمضاء وختم | إضافته بالمرفقات
        10 | جداول الخاصيات الفنية | تعمير وتأريخ وإمضاء وختم | إضافتها بالمرفقات
        11 | بالنسبة للحصص من 1 إلى 5 : تعهد كتابي من المشارك بأن جميع مكونات الحصة هي مواد أصلية | تعهد كتابي ممضى ومؤرخ ومختوم | يضاف بالمرفقات
        2.10- الظرف المتضمن للعرض المالي:
        عدد | اسم الوثيقة | طريقة التقديم | منظومة تونيبس
        1 | وثيقة التعهد (ملحق عدد 1) | تعمير وتأريخ وإمضاء وختم | تعميرها على منظومة تونيبس
        2 | جداول الأثمان الفردية | تعمير وتأريخ وإمضاء وختم | إضافتها بالمرفقات
        3 | جداول الأثمان التفصيلية | تعمير وتأريخ وإمضاء وختم | إضافتها بالمرفقات
        فتح الظروف
        تفتح العروض في جلسة علنية وعلى الخط.
        بالنسبة للحصة عدد 06 يتم الفرز الفني للعروض على أساس مطابقتها للخصائص الفنية المدرجة بالجداول الفنية كما يجب على الأقل تقديم الوثائق الفنية للمواد الستة الأولى في هذه الحصة مع ضرورة ذكر العلامة التجارية لكل مكونات هذه الحصة.
        """,
        """
        جدول الخاصيات المطلوبة حصة عدد 1: حبر لآلات طباعة من نوع HPLaser et traceurs HP.
        N° Référence Article Réponse 1 CB380A 2 CB381A 3 CB382A
        جدول الأثمان التفصيلية حصة عدد 6: قطع غيار ومستلزمات إعلامية.
        N° Référence Article Qtité TVA Prix Unitaires Prix Totaux
        الاستلام والتسليم
        لا يتم استلام المواد موضوع إذن التزود إلا بعد التثبت من مطابقتها للمواصفات المطلوبة. ويقوم المسؤول عن استلام المواد بإمضاء وصل الاستلام (Bon de livraison) مع ذكر التاريخ.
        غرامات التأخير
        يتم، دون أي سابق إعلام، إحتساب غرامة مالية عن كل يوم تأخير غير مبرر في تسليم المواد موضوع الطلبية وذلك في حدود(5‰) من قيمة مبلغ المواد غير المسلمة في آجالها. ويكون الاحتساب ابتداء من تاريخ إذن التزود دون أن يتعدى 5% من قيمة المبلغ الجملي للطلبية.
        مدة الضمان- الضمان النهائي – الحجز بعنوان الضمان
        يتم خلاص الفواتير بتحويل بنكي أو بريدي بحساب المزود المذكور بوثيقة التعهد وذلك في أجل أقصاه 45 يوما (30 يوما لإعداد الأمر بالصرف و15 يوما للدفع).
        """,
    ]
    metas = [
        {"source": "Cahier_des_Charges.docx", "page": "1", "section": "general", "chunk_index": 0},
        {"source": "Cahier_des_Charges.docx", "page": "2", "section": "general", "chunk_index": 1},
        {"source": "Cahier_des_Charges.docx", "page": "3", "section": "general", "chunk_index": 2},
    ]

    facts = extract_document_facts(chunks, metas)

    assert "للتزود بحبر" in facts["subject"]["text"]
    assert "حصة عدد 6" in facts["subject"]["text"]
    assert "TUNEPS" in facts["submission_method"]["text"]
    assert "مكتب الضبط المركزي" in facts["submission_method"]["text"]
    assert "10:00" in facts["deadline"]["text"]
    assert "الضمان الوقتي" not in facts["validity"]["text"]
    assert "شهادة في الانخراط" in facts["cnss"]["text"]

    admin_text = facts["administrative_documents"]["text"]
    assert facts["administrative_documents"]["source"] == "arabic_offer_documents_table"
    assert "شهادة في الانخراط" in admin_text
    assert "الوضعية الجبائية" in admin_text
    assert "تصريح على الشرف" in admin_text
    assert "جدول الخاصيات المطلوبة" not in admin_text

    technical_text = facts["technical_documents"]["text"]
    assert facts["technical_documents"]["source"] == "arabic_offer_documents_table"
    assert "جداول الخاصيات الفنية" in technical_text
    assert "مواد أصلية" in technical_text
    assert "الوثائق الفنية" in technical_text

    financial_text = facts["financial_documents"]["text"]
    assert facts["financial_documents"]["source"] == "arabic_offer_documents_table"
    assert "وثيقة التعهد" in financial_text
    assert "جداول الأثمان الفردية" in financial_text
    assert "جداول الأثمان التفصيلية" in financial_text
    assert "Qtité" not in financial_text

    assert "وصل الاستلام" in facts["reception"]["text"]
    assert "5‰" in facts["penalties"]["text"]
    assert "مدة الضمان" not in facts["penalties"]["text"]
    assert "45 يوما" in facts["payment"]["text"]


def test_extract_document_facts_handles_vlm_validity_salahiyya_ocr_variant():
    chunks = [
        """
        الفصل 06: صلويته العروض.
        يتعهد العارض بالإبقاء على صلويته عرضه لمدة تسعين (90) يوما بداية من اليوم الموالي
        للتاريخ الأقصى المحدد لقبول العروض.
        ويمكن طلب التمديد في آجال صلويته العروض إلى أجل أقصاه مائة وعشرون (120) يوما.
        الفصل 07: الضمان الوقتي.
        """,
    ]
    metas = [{"source": "MDCI.pdf", "page": "4", "section": "general", "chunk_index": 0}]

    facts = extract_document_facts(chunks, metas)

    assert "90" in facts["validity"]["text"]
    assert "120" in facts["validity"]["text"]


def test_definitive_caution_percent_ocr_does_not_match_year_ending_96():
    assert ingest._is_reliable_scalar_fact(
        "definitive_caution",
        {"text": "Le titulaire doit fournir une garantie definitive de 3 96 du montant."},
    )
    assert not ingest._is_reliable_scalar_fact(
        "definitive_caution",
        {"text": "La garantie definitive est mentionnee dans un decret de 1996 sans montant."},
    )


def test_extract_document_facts_prefers_arabic_opening_session_clause():
    chunks = [
        """
        لجنة فتح العروض تتولى التثبت من وثيقة الضمان الوقتي والوثائق الإدارية.
        لا تتضمن هذه الفقرة تاريخ أو كيفية جلسة فتح العروض.
        """,
        """
        تنعقد جلسة فتح العروض وجوبا في نفس اليوم المحدد كتاريخ أقصى لقبول العروض.
        وتجتمع لجنة فتح العروض في جلسة واحدة لفتح العروض وتكون هذه الجلسة علنية.
        """,
    ]
    metas = [
        {"source": "CDC_01-2026.pdf", "page": "6", "section": "general", "chunk_index": 0},
        {"source": "CDC_01-2026.pdf", "page": "8", "section": "general", "chunk_index": 1},
    ]

    facts = extract_document_facts(chunks, metas)

    assert "نفس اليوم" in facts["opening"]["text"]
    assert "جلسة واحدة" in facts["opening"]["text"]
    assert facts["opening"]["page"] == "8"


def test_extract_document_facts_prefers_arabic_reception_minutes_clause():
    chunks = [
        """
        مدة الضمان يجب أن لا تكون أقل من سنة من تاريخ الاستلام الوقتي.
        يتعهد المزود بتعويض المواد الإعلامية التي بها عيوب في الصنع.
        """,
        """
        الفصل 19: تسليم المواد:
        أ- الاستلام الوقتي: يتم إعداد محضر الاستلام وامضاؤه من طرف أعضاء اللجنة.
        ب- الاستلام النهائي: بعد سنة من تاريخ الاستلام الوقتي، يتم تحرير محضر الاستلام النهائي
        شريطة أن لا تكون هناك تحفظات.
        """,
    ]
    metas = [
        {"source": "CDC_01-2026.pdf", "page": "9", "section": "general", "chunk_index": 0},
        {"source": "CDC_01-2026.pdf", "page": "12", "section": "general", "chunk_index": 1},
    ]

    facts = extract_document_facts(chunks, metas)

    assert "محضر الاستلام" in facts["reception"]["text"]
    assert "الاستلام النهائي" in facts["reception"]["text"]
    assert facts["reception"]["page"] == "12"


def test_extract_document_facts_handles_cdc_01_clean_arabic_ocr_pages():
    chunks = [
        """
        الفصل 6: طريقة تقديم العروض:
        مكونات العرض الفني والمالي:
        1-العرض الفني:
        يتضمن الوثائق التالية:
        - العرض الفني والوثائق الفنية حسب كل قسط.
        - شهادة المطابقة للمواصفات الفنية 1509001 نسخة 2015.
        - شهادات المطابقة لمواصفات 14001 1590 نسخة 2015.
        - تقرير إختبار لعدد الصفحات المنتجة حسب معيار ISO/IEC 19752.
        - تعمير جداول الخاصيات الفنية الواردة بكراس الشروط.
        - يجب على العارض تقديم جذاذات فنية للمواد المطلوبة.
        """,
        """
        يطالب المترشح الذي تم الاحتفاظ بعرضه بتقديم ضمان نهائي يساوي 903 من القيمة الأصلية للصفقة
        خلال العشرين (20) يوما الموالية لإعلامه بالموافقة على الصفقة.
        الفصل 15: غرامة التأخير:
        تطبق عقوبة مالية على المزود على أساس واحد من الألف (1000/01) عن كل يوم تأخير
        ولا يمكن أن تتجاوز جملة خطايا التأخير نسبة 965 من مبلغ الحساب النهائي.
        الفصل 24: خلاص الصفقة:
        وعلى المشتري العمومي إصدار أمر بصرف المبالغ الراجعة لصاحب الصفقة في أجل ثلاثون (30) يوما.
        ويتعين على المحاسب العمومي خلاص صاحب الصفقة في أجل خمسة عشر(15) يوما من تاريخ تلقيه الأمر بالصرف.
        """,
    ]
    metas = [
        {"source": "CDC_01-2026.pdf", "page": "7", "section": "general", "chunk_index": 0},
        {"source": "CDC_01-2026.pdf", "page": "13", "section": "general", "chunk_index": 1},
    ]

    facts = extract_document_facts(chunks, metas)

    assert "ISO 9001" in facts["technical_documents"]["text"]
    assert "ISO 14001" in facts["technical_documents"]["text"]
    assert "تقرير اختبار" in facts["technical_documents"]["text"]
    assert "جذاذات فنية" in facts["technical_documents"]["text"]
    assert "3 96" in facts["definitive_caution"]["text"]
    assert "20" in facts["definitive_caution"]["text"]
    assert "غرامة التأخير" in facts["penalties"]["text"]
    assert "1000/01" in facts["penalties"]["text"]
    assert "5 96" in facts["penalties"]["text"]
    assert "أمر بصرف" in facts["payment"]["text"]
    assert "30" in facts["payment"]["text"]
    assert "15" in facts["payment"]["text"]


def test_marker_facts_do_not_replace_reliable_arabic_definitive_caution():
    chunks = [
        """
        ARTICLE 1 : OBJET DU MARCHE
        Le present marche a pour objet la fourniture de (IM (1)).
        ARTICLE 19 : CAUTION DEFINITIVE
        Le texte du modele renvoie a (IM (7)).
        ARTICLE 21 : PENALITES
        Une penalite de (IM (11)) est appliquee.
        ANNEXE III INSTRUCTIONS DU MARCHE
        ARTICLE 1 : OBJET DU MARCHE IM (1) Acquisition de materiel informatique
        ARTICLE 19 : CAUTION DEFINITIVE IM (7) annexe sans montant
        ARTICLE 21 : PENALITES DE RETARD IM (11) 0,2%
        """,
        """
        يطالب المترشح الذي تم الاحتفاظ بعرضه بتقديم ضمان نهائي يساوي 903 من القيمة الأصلية للصفقة
        خلال العشرين (20) يوما الموالية لإعلامه بالموافقة على الصفقة.
        """,
    ]
    metas = [
        {"source": "CDC_01-2026.pdf", "page": "1", "section": "general", "chunk_index": 0},
        {"source": "CDC_01-2026.pdf", "page": "10", "section": "general", "chunk_index": 1},
    ]

    facts = extract_document_facts(chunks, metas)

    assert "3 96" in facts["definitive_caution"]["text"]
    assert "20" in facts["definitive_caution"]["text"]


def test_select_best_entries_merges_arabic_payment_signal_from_weaker_ocr_candidate():
    selected = ingest._select_best_entries_by_page(
        [
            {
                "page": "13",
                "text": "الفصل 24: خلاص الصفقة. يتعين على المحاسب العمومي خلاص صاحب الصفقة في أجل خمسة عشر(15) يوما من تاريخ تلقيه الأمر بالصرف.",
            }
        ],
        [
            {
                "page": "13",
                "text": "وعلى المشتري العمومي إصدار أمر بصرف المبالغ الراجعة لصاحب الصفقة في أجل ثلاثون (30) يوما. ويتعين على المحاسب العمومي خلاص صاحب الصفقة في أجل خمسة عشر(15) يوما.",
            }
        ],
    )

    assert len(selected) == 1
    assert "ثلاثون (30)" in selected[0]["text"]


def test_arabic_fact_signal_count_detects_payment_sentence_gain():
    weak = "يتعين على المحاسب العمومي خلاص صاحب الصفقة في أجل خمسة عشر(15) يوما من تاريخ تلقيه الأمر بالصرف."
    stronger = (
        "وعلى المشتري العمومي إصدار أمر بصرف المبالغ الراجعة لصاحب الصفقة في أجل ثلاثون (30) يوما. "
        "ويتعلق الدفع بأجل خمسة عشر(15) يوما من تاريخ تلقيه الأمر بالصرف."
    )

    assert ingest._arabic_fact_signal_count(stronger) > ingest._arabic_fact_signal_count(weak)


def test_extract_document_facts_handles_stb2_arabic_offer_components():
    chunks = [
        """
        الفصل 5: طريقة تقديم العروض:
        مكونات العرض الفني والمالي:
        1-العرض الفني: يتضمن الوثائق التالية:
        - كراس الشروط الإدارية والفنية الخاصة ممضى ومؤشر على جميع صفحاته؛
        - العرض الفني والوثائق الفنية؛
        - ترخيص في تمثيل المصنع لمنظومة الحماية من الفيروسات؛
        - تقديم قائمة اسمية في المهندسين والتقنيين المكلفين بالتكوين ونقل الخبرة؛
        - وثائق لمشاريع مماثلة في الميدان بالنسبة للشركة؛
        - تعمير جداول الخاصيات الفنية الواردة بكراس الشروط بكل دقة.
        2-العرض المالي: يتضمن الوثائق التالية:
        - التعهد المالي مؤشر عليه ويحمل الإمضاء الأصلي وختم الشركة؛
        - جدول الأثمان مؤشر عليه ويحمل الإمضاء الأصلي وختم الشركة؛
        - عقد الصيانة مختوم وممضى.
        الوثائق التي ترسل مباشرة للإدارة:
        - وثيقة الضمان الوقتي.
        - نظير من السجل الوطني للمؤسّسات لم يمضي على استخراجه ثلاثة أشهر.
        الفصل 6: فتح العروض.
        """,
        """
        الفصل 9: صلوحية العروض:
        يبقى العارض ملتزما بمحتوى عرضه لمدة مائة وعشرين يوما (120)
        بداية من اليوم الموالي للتاريخ الأقصى المحدد لقبول العروض.
        """,
    ]
    metas = [
        {"source": "stb2.pdf", "page": "3", "section": "admin", "chunk_index": 0},
        {"source": "stb2.pdf", "page": "5", "section": "general", "chunk_index": 1},
    ]

    facts = extract_document_facts(chunks, metas)

    assert "مائة وعشرين يوما" in facts["validity"]["text"]
    assert "السجل الوطني للمؤسسات" in facts["rne"]["text"]
    assert "وثيقة الضمان الوقتي" in facts["administrative_documents"]["text"]
    assert "السجل الوطني للمؤسسات" in facts["administrative_documents"]["text"]
    assert "ترخيص في تمثيل المصنع" in facts["technical_documents"]["text"]
    assert "مشاريع مماثلة" in facts["technical_documents"]["text"]
    assert "جداول الخاصيات الفنية" in facts["technical_documents"]["text"]
    assert "عقد الصيانة" in facts["financial_documents"]["text"]
    assert "تمثيل المصنع" in facts["manufacturer_authorization"]["text"]
    assert "مشاريع مماثلة" in facts["references"]["text"]


def test_extract_document_facts_handles_arabic_tuneps_accessories_scan_style():
    chunks = [
        """
        إستشارة 2026/31 لإقتناء لوازم أصلية لآلات سحب الأمثلة
        يعلن ديوان قيس الأراضي و المسح العقاري إجراء إستشارة لإقتناء لوازم أصلية لآلات سحب الأمثلة التالية:
        Plotwave gamme TDS CANON 3500 et (OCé 365), ROWE I4.
        لا تجوز المشاركة في هذه الاستشارة إلا عبر منظومة الشراء العمومي على الخط.
        يبقى المتعهدون ملتزمون بما قدموه من عروض لمدة (60) يوما من تاريخ آخر أجل لقبول العروض.
        يتم تقديم العروض عبر منظومة الشراء العمومي على الخط TUNEPS www.tuneps.tn.
        آخر اجل لقبول العروض 2026/05/20.
        يتم إرسال الضمان المالي الوقتي في ظرف مغلق إلى مكتب الضبط المركزي في أجل أقصاه يوم 2026/05/20.
        الفصل الثاني: الوثائق المكونة للعرض
        الشروط العامة للاستشارة تكون ممضاة ومختومة.
        شهادة تثبت أنّ المزود هو الممثل الرسمي بتونس المرخص له لبيع اللوازم.
        شهادة إثبات أصلية للوازم الأصلية Certificat d’authenticité.
        الفصل الرابع: الضمانات المالية
        الضمان المالي الوقتي بقيمة 0 دينارا صالحا لمدة 60 يوما. عدم تقديم الضمان المالي الوقتي يؤدي إلى إقصاء العرض.
        الضمان المالي النهائي مقداره ثلاثة بالمائة (903) من المبلغ الجملي للعقد.
        """,
        """
        الفصل السادس: أجل التسليم
        يسلم صاحب العقد الطلبيات في أجل أقصاه ستون (60) يوما.
        غرامات التأخير يتم احتسابها عن كل يوم تأخير وفي جميع الحالات لا يمكن أن تتجاوز جملة غرامات التأخير سقف 5% من المبلغ النهائي.
        الفصل العاشر: الإستلام الوقتي
        يتم الإستلام الوقتي بعد التثبت من مطابقة اللوازم للمواصفات وتقديم وصولات التسليم.
        الفصل الحادي عشر: الإستلام النهائي
        يتم الإستلام النهائي بعد رفع التحفظات.
        الفصل الثالث عشر: مدة الضمان
        مدة الضمان تكون لمدة (6) أشهر بداية من تاريخ الإستلام الوقتي.
        الفصل الرابع عشر: كيفية الخلاص
        يتم خلاص صاحب العقد بعد التصريح بالاستلام الوقتي وإثر تقديم فاتورة في أربعة نظائر.
        يتم إصدار الأمر بصرف المبالغ في أجل أقصاه خمسة و أربعون (45) يوما بتحويل بريدي أو بنكي.
        """,
        """
        الخصائص الفنية المطلوبة
        N° article Code Réference Proposés Tireuses de plans Plotwave et ROWE I4.
        جدول الأثمان
        السعر الفردي المبلغ الجملي دون إحتساب الآداءات مبلغ الآداءات على القيمة المضافة.
        """,
    ]
    metas = [
        {"source": "cons-31-2026-accessoires-tireuses_0001.pdf", "page": "1", "section": "general", "chunk_index": 0},
        {"source": "cons-31-2026-accessoires-tireuses_0001.pdf", "page": "3", "section": "general", "chunk_index": 1},
        {"source": "cons-31-2026-accessoires-tireuses_0001.pdf", "page": "5", "section": "technical", "chunk_index": 2},
    ]

    facts = extract_document_facts(chunks, metas)

    assert "لوازم أصلية" in facts["subject"]["text"]
    assert "سحب الأمثلة" in facts["subject"]["text"]
    assert "TUNEPS" in facts["submission_method"]["text"]
    assert "2026/05/20" in facts["deadline"]["text"]
    assert "60" in facts["validity"]["text"]
    assert "تقديم العروض" not in facts["validity"]["text"]
    assert "الضمان المالي الوقتي" in facts["caution"]["text"]
    assert "0 دينارا" not in facts["caution"]["text"]
    assert "ثلاثة بالمائة" in facts["definitive_caution"]["text"]
    assert "3%" in facts["definitive_caution"]["text"]
    assert "الشروط العامة للاستشارة" in facts["administrative_documents"]["text"]
    assert "الممثل الرسمي" in facts["administrative_documents"]["text"]
    assert "الخصائص الفنية" in facts["technical_documents"]["text"]
    assert "السعر الفردي" in facts["financial_documents"]["text"]
    assert "غرامات التأخير" not in facts["financial_documents"]["text"]
    assert "6" in facts["guarantee"]["text"]
    assert "الإستلام الوقتي" in facts["reception"]["text"]
    assert "كيفية الخلاص" not in facts["reception"]["text"]
    assert "غرامات التأخير" in facts["penalties"]["text"]
    assert "905" not in facts["penalties"]["text"]
    assert "خمسة و أربعون" in facts["payment"]["text"]


def test_mined_facts_answer_republique_tunisienne_table_questions():
    facts = _republique_tunisienne_table_facts()

    answer, _ = _answer_from_mined_facts(
        "REPUBLIQUE_TUNISIENNE.pdf",
        "Quelle est la quantité demandée ?",
        facts,
    )
    assert "4" in answer
    assert "Coupeuse de plans grand format papier A0" in answer

    answer, _ = _answer_from_mined_facts(
        "REPUBLIQUE_TUNISIENNE.pdf",
        "Quelle est la longueur de coupe minimale ?",
        facts,
    )
    assert "1190 mm" in answer

    answer, _ = _answer_from_mined_facts(
        "REPUBLIQUE_TUNISIENNE.pdf",
        "Quelle orientation papier A0 est demandée ?",
        facts,
    )
    assert "Portrait et paysage" in answer


@pytest.mark.parametrize(
    ("filename", "text", "checks"),
    [
        (
            "UBCI.pdf",
            """
            ARTICLE 1 : OBJET DE LA CONSULTATION : L'UBCI se propose d'acquerir 160 UC, 20 Laptop et 320 ecrans.
            La premiere enveloppe interieure Enveloppe A portant la mention Offre Technique doit contenir sous peine
            de nullite les pieces suivantes : La documentation technique de chaque type de materiel propose.
            Le formulaire technique annexe au present cahier des charges. Date de commercialisation du modele propose.
            Date previsionnelle d'arret de commercialisation du modele propose. Date de fin de support du modele propose.
            La validite de la soumission. La deuxieme enveloppe interieure Enveloppe B portant la mention Offre Financiere
            doit contenir : L'offre financiere par unite et selon les quantites proposees. Prix catalogue des pieces de rechange.
            L'offre d'extension de garantie. Proposition d'un contrat cadre de service.
            ARTICLE 3. DATE LIMITE DE RECEPTION DES OFFRES : Les soumissions doivent parvenir au bureau d'ordre central.
            La date limite de reception des offres est arretee au 18 Juillet 2025, le cachet du bureau d'ordre fait foi.
            ARTICLE 13. REGLEMENT DES FACTURES : Les paiements s'effectueront par virement a la banque nommee par le
            soumissionnaire retenu. A defaut, des penalites seront fixees dans le contrat d'acquisition (5% du montant
            d'acquisition de l'article est impute par chaque jour de retard).
            """,
            {
                "subject": "160 UC",
                "deadline": "18 Juillet 2025",
                "technical_documents": "documentation technique",
                "financial_documents": "Prix catalogue",
                "payment": "virement",
                "penalties": "5%",
            },
        ),
        (
            "STEG.pdf",
            """
            Nous vous prions d'accompagner votre offre par les pieces administratives suivantes : un extrait du registre
            de commerce (original) ; un certificat d'affiliation a la CNSS (copie certifiee conforme) ; une attestation
            de situation fiscale valable a la date limite de reception des offres ; une declaration de non influence.
            Article 14 DELAI DE VALIDITE DES OFFRES Les offres techniques et financieres resteront valables durant la
            periode indiquee au DPC a compter du lendemain de la date limite de reception des offres.
            """,
            {
                "administrative_documents": "certificat d'affiliation a la CNSS",
                "cnss": "CNSS",
                "rne": "registre de commerce",
                "validity": "resteront valables",
            },
        ),
        (
            "Orange Tunisie.pdf",
            """
            Date limite de remise des offres : 30/04/2025 avant 23h45.
            Objet de la consultation : Orange Tunisie souhaite acquerir un serveur performant pour les besoins de mise
            en place d'un projet d'intelligence artificielle.
            Partie 2 : Offre financiere Votre offre financiere doit comporter : - Offre de prix detaille.
            Specificite de paiement Virement ou Traite a xxx jours apres date signature du PV de reception et de depot
            de la facture.
            """,
            {
                "subject": "serveur performant",
                "deadline": "30/04/2025",
                "financial_documents": "Offre de prix detaille",
                "payment": "Virement ou Traite",
            },
        ),
        (
            "BH ASSURANCE.pdf",
            """
            BH ASSURANCE CONSULTATION-DSI-05-2025 Objet : Consultation pour le renouvellement des Licences Veeam.
            Messieurs, La compagnie BH ASSURANCE se propose de Renouveler les Licences Veeam Backup.
            """,
            {
                "subject": "renouvellement des Licences Veeam",
            },
        ),
        (
            "BANQUE_ZITOUNA.pdf",
            """
            Cahier des Clauses administratives Particulieres
            ARTICLE 1 - OBJET DU MARCHE
            Cette consultation a pour objet l'acquisition de 150 ordinateurs de bureau avec ecran et
            20 Workstation avec ecran au titre de l'annee 2025.
            Je joins a la presente soumission le CCAP, le CCTP, une attestation fiscale et une attestation CNSS.
            """,
            {
                "subject": "150 ordinateurs de bureau",
            },
        ),
    ],
)
def test_extract_document_facts_from_representative_pdf_snippets(filename, text, checks):
    facts = _facts_from_text(filename, text)

    for field, expected_text in checks.items():
        assert field in facts
        assert expected_text.lower() in facts[field]["text"].lower()


def test_extract_document_facts_prefers_article_object_over_cover_title():
    facts = extract_document_facts(
        [
            """
            UBCI
            Cahier des Charges
            Marche d'acquisition des UC, PC Portables et des ecrans - 2025
            """,
            """
            ARTICLE 1 : OBJET DE LA CONSULTATION : L'UBCI se propose d'acquerir 160 UC,
            20 Laptop et 320 ecrans.
            """,
        ],
        [
            {"source": "UBCI.pdf", "page": "1", "section": "general", "chunk_index": 0},
            {"source": "UBCI.pdf", "page": "2", "section": "admin", "chunk_index": 1},
        ],
    )

    assert "160 UC" in facts["subject"]["text"]
    assert "20 Laptop" in facts["subject"]["text"]
    assert facts["subject"]["page"] == "2"


def test_extract_document_facts_prefers_explicit_ubci_deadline_date():
    facts = _facts_from_text(
        "UBCI.pdf",
        """
        ARTICLE 3. DATE LIMITE DE RECEPTION DES OFFRES : Les soumissions doivent parvenir au bureau d'ordre central de
        L'UBCI a l'adresse suivante : UBCI 139 Avenue de la Liberte 1001 Tunis.
        La date limite de réception des offres est arrêtée au 18 Juillet 2025, le cachet du bureau d'ordre de L'UBCI fait foi.
        """,
        section="deadline",
    )

    assert facts["deadline"]["text"].startswith("18 Juillet 2025")


def test_extract_document_facts_handles_envelope_tables_and_article_clauses():
    chunks = [
        """
        ARTICLE 1 : OBJET D'APPEL D'OFFRES
        L'Agence de Protection et d'Amenagement du littoral se propose de lancer un appel d'offres
        pour l'acquisition des equipements informatiques ayant pour objet la fourniture de :
        Lot A : Ordinateurs de bureau, PCs portables et disques durs externes.
        A1 : Equipements assujettis a la TVA.
        1) Sous-lot A1-a : 14 ordinateurs de bureau.
        Lot B : Imprimantes et scanners.
        ARTICLE 2 : CONDITIONS DE PARTICIPATION
        """,
        """
        4.8. "Offre technique : enveloppe A"
        L'enveloppe A portant la mention "Offre technique et justificatif" en double exemplaire et contenant :
        | N | Document Appellation | Operation a realiser | Authentification |
        |---|----------------------|----------------------|------------------|
        | 1 | Les tableaux des caracteristiques techniques minimales exigees dans le present cahier remplis en se conformant a leur structure. | A fournir par le soumissionnaire | Paraphe et cachet. |
        | 2 | Toutes pieces techniques justificatives necessaires (prospectus). | A fournir par le soumissionnaire | Date, signature et cachet. |

        4.9. "Offre financiere : enveloppe B"
        L'enveloppe B portant la mention "Offre Financiere" contiendra les documents suivants en double exemplaire :
        | N | Document Appellation | Operation a realiser | Authentification |
        |---|----------------------|----------------------|------------------|
        | 1 | L'acte d'engagement pour chaque lot complete par les montants de l'offre en toutes lettres et en chiffres | A fournir par le soumissionnaire | Date, signature, et cachet |
        | 2 | Les bordereaux des prix pour chaque lot/detail estimatif | A fournir par le soumissionnaire | signes et portant le cachet |
        """,
        """
        4.10. "Enveloppe exterieure : enveloppe C"
        L'enveloppe exterieure : enveloppe C, contiendra en double exemplaire les documents suivants :
        | N | Document Appellation | Operation a realiser | Authentication |
        |---|----------------------|----------------------|----------------|
        | 1 | Le montant de la Caution Provisoire est fixe a : 400 DT pour le lot A et 400 DT pour le lot B | Original de la caution bancaire | Date, signature |
        | 2 | Fiche de renseignements generaux sur le soumissionnaire | Copie du modele | Date, signature |
        | 3 | Le present cahier des charges avec paraphe et cachet sur chaque page. | Original du document | Date, signature |
        | 4 | Extrait du registre de commerce | A fournir par les soumissionnaires residents | Original. |
        | 5 | Attestation fiscale valable certifiant que le soumissionnaire est en regle avec la Direction des Impots. | Valable a la date limite | Date, signature |
        | 6 | Certificat d'affiliation a la caisse nationale de securite sociale (CNSS). | Valable a la date limite | Date, signature |
        | 7 | Un certificat de non faillite ou de redressement judiciaire | A fournir par le soumissionnaire | Date signature |
        """,
        """
        ARTICLE 7 : CAUTIONNEMENT DEFINITIF :
        Le montant de la caution est egal a trois pour cent (3%) du montant initial du marche,
        doit etre etabli suivant le modele de l'annexe, enregistre et remis dans un delai de vingt (20) jours.
        """,
        """
        ARTICLE 13 : PENALITES DE RETARD
        Le fournisseur subira une penalite sur la base des dispositions suivantes :
        Le montant des penalites par jour calendrier de retard est egal au 1/1000ème du montant final du marche.
        Le montant total de ces penalites sera plafonne a cinq pour cent (5 %) du montant final du marche.
        """,
        """
        ARTICLE15 : RECEPTIONS
        15.1 - Reception provisoire. Il sera procede a la reception provisoire et un proces-verbal sera etabli.
        15.2 - Reception definitive. Le cautionnement devient caduc apres reception definitive des prestations.
        ARTICLE 17 : GARANTIE
        Le delai de garantie est fixe a 12 mois a partir de la reception provisoire sans reserve.
        """,
        """
        Lesmodalitesdepaiementsontlessuivantes:
        100%dumontantdumarchecontrepresentationduproces-verbal dereceptionprovisoiredechaque prestation sansreserves.
        Lemandatementdessommesduesautitulairedumarche doit intervenir dansun delaimaximumdetrentejours.
        Le comptable public doit payer dansundelaimaximumdequinze joursapartirdelareceptiondelordredepaiement.
        """,
        """
        ANNEXE - Retenue de garantie
        Paiement a la premiere demande ecrite de l'administration sans contestation par l'etablissement bancaire.
        """,
    ]
    metas = [
        {"source": "DOSSIER_A.O.pdf", "page": str(page), "section": "general", "chunk_index": index}
        for index, page in enumerate((2, 3, 4, 6, 7, 8, 11, 30))
    ]

    facts = extract_document_facts(chunks, metas)

    assert "Lot A" in facts["subject"]["text"]
    assert "Lot B" in facts["subject"]["text"]
    assert "Extrait du registre de commerce" in facts["rne"]["text"]
    admin_text = facts["administrative_documents"]["text"]
    assert "Fiche de renseignements" in admin_text
    assert "registre de commerce" in admin_text
    assert "CNSS" in admin_text
    tech_text = facts["technical_documents"]["text"]
    assert "caracteristiques techniques" in tech_text
    assert "prospectus" in tech_text
    assert "acte d'engagement" not in tech_text.lower()
    financial_text = facts["financial_documents"]["text"]
    assert "acte d'engagement" in financial_text
    assert "bordereaux des prix" in financial_text
    assert "3%" in facts["definitive_caution"]["text"]
    assert "1/1000" in facts["penalties"]["text"]
    assert "5 %" in facts["penalties"]["text"]
    assert "Reception provisoire" in facts["reception"]["text"]
    assert "Reception definitive" in facts["reception"]["text"]
    assert "proces-verbal" in facts["reception"]["text"]
    assert "100%" in facts["payment"]["text"]
    assert "trente jours" in facts["payment"]["text"]
    assert "premiere demande" not in facts["payment"]["text"].lower()


def test_extract_document_facts_resolves_steg_style_im_placeholders():
    facts = _facts_from_text(
        "STEG.pdf",
        """
        ARTICLE 1 : OBJET DU MARCHE
        Le present marche a pour objet la fourniture de (IM (1)), specifies dans le CCTP.

        ARTICLE 24 GARANTIE ET RECEPTION DEFINITIVE
        Le delai de garantie pour le materiel objet du marche est fixe a (IM (13)).

        ARTICLE 21 PENALITES DE RETARD
        Une penalite de retard de (IM (11)) par jour est appliquee. Le montant total
        ne doit pas depasser 5% du montant du marche.

        L'offre financiere doit preciser notamment : - La designation, les quantites ;
        - Les prix unitaires, les prix totaux ; - Le mode de paiement choisi ;
        - Le mode et le delai de livraison.

        ANNEXE III INSTRUCTIONS DU MARCHE
        ARTICLE 1 : OBJET DU MARCHE IM (1) Acquisition de materiel informatique
        ARTICLE 21: PENALITES DE RETARD IM (11) 0,2%
        ARTICLE 24: GARANTIE IM (13) 12 mois
        """,
    )

    assert "Acquisition de materiel informatique" in facts["subject"]["text"]
    assert "IM" not in facts["subject"]["text"]
    assert "12 mois" in facts["guarantee"]["text"]
    assert "0,2%" in facts["penalties"]["text"]
    assert "prix unitaires" in facts["financial_documents"]["text"]


def test_extract_document_facts_resolves_steg_style_dpc_placeholders():
    facts = _facts_from_text(
        "Consultation_N_2025_C020_02.pdf",
        """
        Article 1. Le present cahier a pour objet de definir les conditions de participation
        et de soumission a la consultation applicable aux travaux et/ou fourniture de biens
        et services tel que specifie dans les donnees particulieres de la consultation (DPC).

        Article 8 RECEPTION TECHNIQUE
        La reception provisoire sera prononcee en une seule fois apres la livraison des
        equipements et de la documentation technique associee. La reception definitive sera
        prononcee a l'expiration du delai de garantie.

        Article 9 GARANTIE
        A defaut d'un meilleur delai propose par le fournisseur, le delai de garantie est
        fixe a 6 mois pieces et main d'oeuvre. Ce delai commence a courir a compter de la
        date de la reception provisoire.

        Article 11 PENALITES DE RETARD
        Il sera applique une penalite de 0,2% du montant de la commande par jour calendaire
        de retard avec un maximum de 5% du montant total definitif de la commande hors TVA.

        Article 14 CAUTION BANCAIRE DE BONNE FIN
        Le titulaire doit fournir une caution bancaire a premiere demande de bonne fin.
        Le montant de cette caution doit etre egal a 5% du montant TTC de la commande.

        Article 16 CONDITIONS DE PAIEMENT
        Les factures regulierement emises sont payables a 45 jours. Mode de paiement :
        Virement Bancaire.

        Donnees particulieres de la consultation(DPC)
        Article 1 2025/C020/02 « ACQUISITION DES PIECES DE RECHANGE POUR LES IMPRIMANTES
        DE PRODUCTION CANON VP140 TUNIS ET SFAX » Objet
        (7) Elements ayant trait a l'evaluation technique et financiere devant etre
        telecharges sur TUNEPS : Le bordereau des prix dument rempli.
        (9) Extrait du registre national des entreprises.
        (10) Questionnaire technique (voir CST page 24).
        (12) Date et heure limite de reception des offres : .U../ FE ZJ/]
        (14) 90 jours.
        """,
    )

    assert "ACQUISITION DES PIECES DE RECHANGE" in facts["subject"]["text"]
    assert "CANON VP140" in facts["subject"]["text"]
    assert "TUNIS ET SFAX" in facts["subject"]["text"]
    assert "conditions de participation" not in facts["subject"]["text"].lower()
    assert "90 jours" in facts["validity"]["text"]
    assert "6 mois" in facts["guarantee"]["text"]
    assert "0,2%" in facts["penalties"]["text"]
    assert "5%" in facts["definitive_caution"]["text"]
    assert "45 jours" in facts["payment"]["text"]


def test_extract_document_facts_handles_stb_printing_solution_ocr():
    facts = _facts_from_text(
        "Societe_Tunisienne_de_Banque.pdf",
        """
        ARTICLE 1 : OBJET ET COMPOSITION DE L'APPEL D'OFFRES 1.1 Objet de l'appel d'offres
        Le present appel d'offres a pour objet l'acquisition, la fourniture, l'installation de
        equipements, materiels et logiciels, necessaires pour beneficier d'une solution d'impression
        a la banque ainsi que les prestations de Maintenance sur une Periode de trois {03} ans
        apres l'expiration de la periode de garantie. Composante 1: La fourniture, la livraison,
        l'installation, les tests de conformite et la mise en service de 50 equipements Multifonction
        impression, copie et scan. La fourniture de Consommable supplementaires necessaire pour
        imprimer >= 100 000 pages. Composante 2 : l'Application d'administration et de gestion
        d'impression pour une capacite minimale de 100 equipements.

        L'offre doit etre constituee de: des documents administratifs ci-apres : Une fiche kys
        (Know Your Supplier) etablie conformement au modele en annexe 2. Un certificat
        d'affiliation a la Caisse Nationale de Securite Sociale. L'original du certificat
        d'inscription au registre national des entreprises.

        L'offre technique ne comporte pas le formulaire de reponses dument rempli conformement
        au modele de l'annexe 3. Les justificatifs des references du soumissionnaire. Une
        documentation technique complete des equipements a fournir. Un engagement attestant la
        disponibilite des pieces de rechange. Une autorisation du constructeur en originale.
        Certification des equipements aux normes suivantes : ISO9001 et IEEE 2600.

        Le delai global des prestations d'entretien continu des equipements multifonction
        Impression-Copie-Scan est fixe a trois ans a partir de l'expiration d'une annee de garantie.
        La STB procedera au reglement du prix du marche par virement bancaire apres signature
        des PV de reception. ARTICLE 14 : PENALITES DE RETARD Les penalites de retard sont
        calculees a raison de 3%o (trois pour mille).
        """,
    )

    assert "50 équipements multifonction" in facts["subject"]["text"]
    assert "fiche kys" in facts["administrative_documents"]["text"].lower()
    assert "registre national des entreprises" in facts["administrative_documents"]["text"].lower()
    assert "documentation technique" in facts["technical_documents"]["text"].lower()
    assert "autorisation du constructeur" in facts["technical_documents"]["text"].lower()
    assert "année" in facts["guarantee"]["text"]
    assert "50 équipements multifonction" in facts["requested_items"]["text"]
    assert "100 000 pages" in facts["requested_items"]["text"]
    assert "virement bancaire" in facts["payment"]["text"]
    assert "3‰" in facts["penalties"]["text"]


def test_extract_document_facts_rejects_stb_toc_subject_fragment():
    facts = _facts_from_text(
        "Societe_Tunisienne_de_Banque.pdf",
        """
        ETCOMPOSITIONDEL'APPELD'OFFRES....., 1 = 14.. , 1 = 14

        ARTICLE 1 : OBJET ET COMPOSITION DE L'APPEL D'OFFRES
        Le present appel d'offres a pour objet l'acquisition, la fourniture,
        l'installation et la mise en service de 50 equipements multifonction
        impression-copie-scan, avec une solution d'administration et des prestations
        de maintenance.
        """,
    )

    assert "ETCOMPOSITION" not in facts["subject"]["text"]
    assert "50" in facts["subject"]["text"]
    assert "multifonction" in facts["subject"]["text"].lower()


def test_extract_document_facts_polishes_stb_subject_title_fragment():
    facts = _facts_from_text(
        "Societe_Tunisienne_de_Banque.pdf",
        """
        Etatderealisatlon/sort 'Acquisition et mise en place d'une solution
        d'impression a la STB' Page : 34/ 50
        """,
    )

    assert facts["subject"]["text"] == (
        "L'appel d'offres a pour objet l'acquisition et la mise en place "
        "d'une solution d'impression a la STB."
    )


def test_extract_document_facts_drops_caution_procedure_as_admin_documents():
    facts = _facts_from_text(
        "Societe_Tunisienne_de_Banque.pdf",
        """
        Documents administratifs :
        La caution provisoire doit etre etablie conformement au modele etabli en Annexe 1.
        La caution provisoire sera restituee aux soumissionnaires dont les offres sont eliminees.
        La caution provisoire sera mise en paiement de plein droit au profit de la STB.
        La caution provisoire sera restituee au titulaire du marche apres constitution de la caution definitive.
        Si le soumissionnaire refuse de signer le marche, la caution definitive est appelee.
        """,
    )

    assert "administrative_documents" not in facts


def test_extract_document_facts_builds_tender_profile():
    facts = _facts_from_text(
        "UBCI.pdf",
        """
        ARTICLE 1 : OBJET DE LA CONSULTATION : L'UBCI se propose d'acquerir 160 UC,
        20 Laptop et 320 ecrans.
        La date limite de reception des offres est arretee au 18 Juillet 2025.
        Les soumissions doivent parvenir au bureau d'ordre central de l'UBCI.
        L'offre financiere par unite et selon les quantites proposees.
        """,
    )

    profile = facts["tender_profile"]

    assert profile["schema"] == "tender_profile.v1"
    assert "object" in profile["fields"]
    assert "deadline" in profile["fields"]
    assert profile["fields"]["object"]["page"] == facts["subject"]["page"]
    assert "160 UC" in profile["fields"]["object"]["text"]
    assert profile["coverage"]["core_present"] >= 3
    assert "payment" in profile["coverage"]["missing_core_fields"]


def test_extract_document_facts_supports_tender_checklist_fields():
    facts = _facts_from_text(
        "CDC.pdf",
        """
        Les soumissions doivent parvenir par voie postale ou par depot direct au bureau d'ordre central.
        Les variantes ne sont pas autorisees.
        Le dossier administratif comprend une fiche de renseignements, une attestation de situation fiscale,
        une attestation d'affiliation a la CNSS et un extrait du registre de commerce.
        L'offre technique doit contenir une autorisation du constructeur et une liste des references similaires.
        La reception provisoire sera prononcee apres installation et la reception definitive apres garantie.
        Une caution definitive de 10% du montant du marche est exigee.
        """,
    )

    assert "voie postale" in facts["submission_method"]["text"]
    assert "ne sont pas autorisees" in facts["variants"]["text"]
    assert "fiche de renseignements" in facts["information_sheet"]["text"]
    assert "situation fiscale" in facts["fiscal_certificate"]["text"]
    assert "CNSS" in facts["cnss"]["text"]
    assert "registre de commerce" in facts["rne"]["text"]
    assert "autorisation du constructeur" in facts["manufacturer_authorization"]["text"]
    assert "references similaires" in facts["references"]["text"]
    assert "reception provisoire" in facts["reception"]["text"]
    assert "caution definitive" in facts["definitive_caution"]["text"]


def test_extract_document_facts_handles_tunisian_saudi_bank_article_layout():
    facts = _facts_from_text(
        "TUNISIAN_SAUDI_BANK.pdf",
        """
        ARTICLE 0. OBJET DU DOSSIER Ils fixent les procédures de l'appel d'offres et stipulent
        les conditions du marché.

        ARTICLE 1. OBJET DU MARCHE La TSB envisage de mettre a niveau l'Infrastructure Systeme
        (Site Principal et de Backup a Kairouan). A cet effet la TSB lance le present appel d'offres
        en lot unique pour l'acquisition, la mise en place et la migration des serveurs et de la solution
        de virtualisation VMWARE.

        ARTICLE 7. VALIDITE DES OFFRES Les offres demeureront valables pour une periode de 90 jours
        apres la date limite de reception des offres fixee par TSB.

        ARTICLE 14. CAUTIONNEMENT 1. Caution provisoire Chaque offre doit etre accompagnee d'une
        caution bancaire provisoire. Le montant de la caution provisoire s'elevera a 12.000 DT.
        2. Caution definitive Le titulaire du marche devra fournir une caution definitive d'une valeur
        egale a 3% du montant total du marche toutes taxes comprises.

        ARTICLE 16. PRESENTATION & RECEPTION DE LA SOUMISSION
        1. DOSSIER ADMINISTRATIF : La caution bancaire provisoire. RNE recent valable a la date
        d'ouverture des offres. Une attestation d'affiliation a la CNSS.
        2. OFFRE TECHNIQUE Le dossier de l'offre technique doit contenir les pieces suivantes :
        La liste des equipements. Documentation techniques. Engagement concernant l'origine des fournitures.
        La certification sur HPE Synergy. La certification sur Vmware.
        3. OFFRE FINANCIERE Le dossier de l'offre financiere doit obligatoirement comporter :
        La lettre de soumission. Le bordereau des prix. Le recapitulatif des prix.

        ARTICLE 17. DATE LIMITE DE RECEPTION DES OFFRES Les offres doivent parvenir par voie postale
        ou remise directement au bureau d'ordre de T.S.B. La date limite de la réception des offres est
        fixée au 24 février 2025.

        ARTICLE 26. GARANTIE Le délai de garantie est de 3 (trois) ans a compter de la date de la reception
        provisoire sans reserve.
        ARTICLE 28. RECEPTION La reception provisoire sera prononcee apres essais satisfaisants.
        La reception definitive sera prononcee un an apres la reception provisoire sans reserves.
        ARTICLE 29. CONDITIONS DE PAIEMENT Les conditions de paiement sont fixees comme suit :
        20% a la livraison du materiel. 50% a la finalisation du site principal. 20% a la finalisation
        du site de backup. 10% retenue de garantie.
        ARTICLE 30. MODALITES DE PAIEMENT La facture sera payable par chèque ou virement bancaire
        dans un delai de 30 jours.
        ARTICLE 34. PENALITE DE RETARD Le fournisseur devra payer une penalite calculee a raison
        d'un pour mille pour chaque jour de retard. Le montant total de la penalite ne doit pas exceder
        cinq pour cent de la valeur totale du marche.
        """,
    )

    assert "Infrastructure Systeme" in facts["subject"]["text"]
    assert facts["deadline"]["text"] == "24 février 2025"
    assert "90 jours" in facts["validity"]["text"]
    assert "12.000 DT" in facts["caution"]["text"]
    assert "voie postale" in facts["submission_method"]["text"]
    assert "RNE" in facts["administrative_documents"]["text"]
    assert "CNSS" in facts["administrative_documents"]["text"]
    assert "Documentation techniques" in facts["technical_documents"]["text"]
    assert "Vmware" in facts["technical_documents"]["text"]
    assert "lettre de soumission" in facts["financial_documents"]["text"].lower()
    assert "3 (trois) ans" in facts["guarantee"]["text"]
    assert "reception definitive" in facts["reception"]["text"].lower()
    assert "50%" in facts["payment"]["text"]
    assert "pour mille" in facts["penalties"]["text"]


def test_extract_document_facts_rejects_correspondence_as_submission_method():
    facts = _facts_from_text(
        "CORRESPONDENCE.pdf",
        """
        Les demandes d'eclaircissement doivent parvenir par courrier electronique au secretariat.
        Toute correspondance doit etre envoyee par voie postale.

        Les offres doivent parvenir sous pli ferme au bureau d'ordre central avant la date limite.
        """,
    )

    assert "pli ferme" in facts["submission_method"]["text"]
    assert "eclaircissement" not in facts["submission_method"]["text"].lower()


def test_extract_document_facts_rejects_execution_validity_for_offer_validity():
    facts = _facts_from_text(
        "VALIDITY.pdf",
        """
        ARTICLE 4 VALIDITE DU CONTRAT
        La validite du contrat couvre toute la periode d'execution du marche.

        ARTICLE 5 VALIDITE DES OFFRES
        Les offres resteront valables pendant 120 jours a compter de la date limite de reception.
        """,
    )

    assert "120 jours" in facts["validity"]["text"]
    assert "execution" not in facts["validity"]["text"].lower()


def test_extract_document_facts_penalizes_subject_toc_and_forms():
    facts = _facts_from_text(
        "SUBJECT.pdf",
        """
        ARTICLE 1 OBJET ........................................ 3
        ARTICLE 2 CONDITIONS ................................... 4
        ARTICLE 3 VALIDITE ..................................... 5

        ANNEXE N 1 MODELE DE SOUMISSION Objet : formulaire de reponse.

        Article 1 Objet du marche
        Le present appel d'offres a pour objet l'acquisition et la mise en place
        d'une solution de sauvegarde centralisee.
        """,
    )

    assert "solution de sauvegarde centralisee" in facts["subject"]["text"]
    assert "ARTICLE 1 OBJET" not in facts["subject"]["text"]


def test_extract_document_facts_handles_bct_article_tender_fields():
    facts = _facts_from_text(
        "soumissionner_ARTICLE_2026-03-05.pdf",
        """
        SOMMAIRE
        ARTICLE1, 1 = Objet du marche
        ARTICLE2, 1 = : Composition du marche
        ARTICLE3, 1 = Pieces constitutives du dossier

        ARTICLE 1ER : OBJET DU MARCHE
        Le present marche a pour objet de definir les conditions generales et speciales
        pour la fourniture et la livraison d'imprimes simples, de fournitures de bureaux,
        de fournitures informatiques, de fournitures d'imprimerie et de fournitures de caisse.

        ARTICLE 4 : PRESENTATION DES OFFRES
        Une sous-enveloppe fermee pour l'offre financiere contenant :
        La soumission. Le bordereau des prix. Le sous-detail des prix par lot.
        Le cautionnement provisoire de 1,5% du montant de la soumission doit etre joint.
        Les offres doivent etre adressees par voie postale ou deposees au Bureau d'Ordre Central
        au plus tard le 07/07/2025 a 12h00.

        ARTICLE 5 : DELAI DE VALIDITE DES OFFRES
        Les soumissionnaires sont engages par leurs offres pendant 120 jours a compter
        de la date limite fixee pour la reception des plis.

        Toute enveloppe comportant une reference relative au nom du soumissionnaire est automatiquement rejetee.
        """,
    )

    assert facts["subject"]["text"].startswith("Le present marche a pour objet") or facts["subject"][
        "text"
    ].startswith("Le présent marché a pour objet")
    assert "fourniture et la livraison d'imprimes simples" in facts["subject"]["text"]
    assert "Composition du marche" not in facts["subject"]["text"]
    assert "120 jours" in facts["validity"]["text"]
    assert "reception des plis" in facts["validity"]["text"].lower()
    assert "1,5%" in facts["caution"]["text"]
    assert "Bureau d'Ordre Central" in facts["submission_method"]["text"]
    assert "07/07" not in facts["submission_method"]["text"]
    assert "La soumission" in facts["financial_documents"]["text"]
    assert "bordereau des prix" in facts["financial_documents"]["text"].lower()
    assert "sous-detail des prix" in facts["financial_documents"]["text"].lower()
    assert "references" not in facts


def test_extract_document_facts_keeps_designation_column_from_ocr_tables():
    facts = _facts_from_text(
        "TUNISIAN_SAUDI_BANK.pdf",
        """
        1. DOSSIER ADMINISTRATIF : N° de la pièce, 1 = Désignations. N° de la pièce, 2 = Authentifications.
        1, 1 = La caution bancaire provisoire d'un montant égal à Douze mille (12 000) Dinars.
        1, 2 = Cachet signature du soumissionnaire.
        2, 1 = joint en annexe(2). Les tableaux portant sur les références, clairement remplis.
        2, 2 = Dûment signé paraphé et daté par soumissionnaire.
        3, 1 = Déclaration sur l'honneur concernant l'exactitude des informations fournies selon le modèle joint en annexe(7).
        3, 2 = Dûment signé paraphé et daté par soumissionnaire.

        2. OFFRE TECHNIQUE Le dossier de l'offre technique doit contenir les pieces suivantes :
        Documentation technique. La certification sur Vmware.
        """,
    )

    admin_text = facts["administrative_documents"]["text"]
    assert "La caution bancaire provisoire" in admin_text
    assert "Les tableaux portant sur les références" in admin_text
    assert "Déclaration sur l'honneur" in admin_text
    assert "Authentifications" not in admin_text
    assert "Cachet signature" not in admin_text
    assert "Dûment signé" not in admin_text


def test_extract_document_facts_keeps_financial_designations_from_ocr_tables():
    facts = _facts_from_text(
        "TUNISIAN_SAUDI_BANK.pdf",
        """
        3. OFFRE FINANCIERE Le dossier de l'offre financière doit obligatoirement comporter :
        1, Désignations = La lettre de soumission conformément au modèle joint en annexe(1).
        1, Authentifications = Date, signature et cachet du soumissionnaire.
        2, Désignations = Le bordereau des prix conformément au modèle joint en annexe(10).
        2, Authentifications = Date, signature et cachet du soumissionnaire.
        3, Désignations = Le récapitulatif des prix conformément au modèle joint en annexe(11).
        3, Authentifications = Date, signature et cachet du soumissionnaire.
        """,
    )

    financial_text = facts["financial_documents"]["text"]
    assert "La lettre de soumission" in financial_text
    assert "Le bordereau des prix" in financial_text
    assert "Le récapitulatif des prix" in financial_text
    assert "Authentifications" not in financial_text
    assert "signature et cachet" not in financial_text


def test_extract_document_facts_strips_ocr_prefixes_from_designation_items():
    facts = _facts_from_text(
        "TUNISIAN_SAUDI_BANK.pdf",
        """
        3. OFFRE FINANCIERE Le dossier de l'offre financière doit obligatoirement comporter :
        1, Désignations = a , oa La lettre de soumission conformément au modèlejointenannexe(1).
        1, Authentifications = Date, signature et cachet du soumissionnaire.
        2, Désignations = , un Le bordereau des prix Conformément au modèlejointenannexe(10).
        2, Authentifications = Date, signature et cachet du soumissionnaire.
        3, Désignations = Ve . . , oo. Le récapitulatif des prix. Conformément au modèle joint enannexe(11).
        3, Authentifications = Date, signature et cachet du soumissionnaire.

        1. DOSSIER ADMINISTRATIF :
        5, 1 = ' ' 4; PrésentationduSoumissionnaire.
        5, 2 = Dûment signé paraphé et daté par soumissionnaire.
        """,
    )

    financial_text = facts["financial_documents"]["text"]
    assert "- La lettre de soumission" in financial_text
    assert "- Le bordereau des prix" in financial_text
    assert "- Le récapitulatif des prix" in financial_text
    assert "modèle joint en annexe(1)" in financial_text
    assert "annexe(11)" in financial_text
    assert "a , oa" not in financial_text
    assert ", un Le" not in financial_text
    assert "Ve . ." not in financial_text

    admin_text = facts["administrative_documents"]["text"]
    assert "Présentation du Soumissionnaire" in admin_text
    assert "' ' 4" not in admin_text


def test_extract_document_facts_handles_messy_tsb_ocr_fragments():
    facts = _facts_from_text(
        "TUNISIAN_SAUDI_BANK.pdf",
        """
        Les offres doivent obligatoirement parvenir par voie postale recommandee ou par rapide-poste
        ou remise directement au bureau d'ordre de T.S.B (cachet du bureau d'ordre faisant foi) a
        l'adresse suivante. La date limite de la réception des offres est fixée au 24 février 2025.

        La caution bancaire provisoire d'un montant égal 4 Douze mille (12 000) Dinars en original
        et établi conformément aux dispositions du cahier des charges.

        L' offre technique et l'offre financière doivent être placées dans deux enveloppes séparées.
        Le dossier de l' offre technique doit contenir sous peine de nullité les pièces suivantes :
        La liste de l'équipe intervenante accompagnée de leur CV et copie des diplômes et des certifications.
        Documentation technique. Engagement concernant l'origine des fournitures.
        La certification sur HPE Synergy. La certification sur Vmware.
        Le dossier de l'offre financière doit obligatoirement comporter : La lettre de soumission.
        """,
    )

    assert "voie postale" in facts["submission_method"]["text"]
    assert "bureau d'ordre" in facts["submission_method"]["text"]
    assert "Douze mille" in facts["caution"]["text"]
    assert "12 000" in facts["caution"]["text"]
    assert "Documentation technique" in facts["technical_documents"]["text"]
    assert "Vmware" in facts["technical_documents"]["text"]


def test_extract_document_facts_handles_cetime_dense_consultation_articles():
    facts = _facts_from_text(
        "CETIME.pdf",
        """
        Article 1. Objet de la consultation : Le present cahier des charges a pour objet
        l'accompagnement a la mise en place d'un systeme de gestion des documents et des archives,
        Article 2. CONDITION DE SOUMISSION : La participation a la consultation est ouverte
        a toutes les personnes physiques ou morales etablies en Tunisie.

        Pieces administratives a fournir :
        + Une fiche de renseignement.
        + Le present cahier des charges paraphe, signe et cachete avec la mention lu et approuve.
        + un original du registre national de l'entreprise.
        + Attestation de la situation fiscale valide.
        + CV des intervenants.
        + Une declaration de non faillite.

        Les candidats sont lies par leurs offres pour une periode de soixante jours (60) jours
        a compter du jour suivant la date limite fixee pour la reception des offres.
        Toute offre ne contenant pas la liste d'au moins 3 travaux similaires durant les cinq
        dernieres annees sera rejetee.

        La reception est prononcee suite a: La validation par le CETIME des prestations requises;
        La fin de la formation; La remise de la documentation technique. Le PV de reception doit
        etre signe par les deux parties sans reserves.
        Le reglement est effectue par virement suite au depot de la facture au bureau d'ordre
        central du CETIME et a la fourniture du PV signe par les deux parties sans reserves.
        """,
    )

    assert "systeme de gestion des documents et des archives" in facts["subject"]["text"]
    assert "participation" not in facts["subject"]["text"].lower()
    assert "soixante jours" in facts["validity"]["text"]
    assert "fiche de renseignement" in facts["administrative_documents"]["text"].lower()
    assert "registre national" in facts["administrative_documents"]["text"].lower()
    assert "situation fiscale" in facts["administrative_documents"]["text"].lower()
    assert "non faillite" in facts["administrative_documents"]["text"].lower()
    assert "3 travaux similaires" in facts["references"]["text"]
    assert "PV de reception" in facts["reception"]["text"]
    assert "virement" in facts["payment"]["text"].lower()


def test_extract_document_facts_handles_tunisie_telecom_consumables_consultation():
    facts = _facts_from_text(
        "Tunisie_Telecom_DCSI.pdf",
        """
        ARTICLE 1-2 : OBJET
        La presente consultation a pour objet la conclusion d'un marche cadre pour l'acquisition des
        consommables et accessoires informatiques, au profit Tunisie Telecom, dont les specifications
        techniques sont definies ci-apres dans le cahier des charges des clauses techniques.
        Cette acquisition est repartie en trois (03) lots separes comme suit:
        N Designation Ref Qte MIN Qte MAX Lot N01 Consommables imprimante HP Toner d'origine HP.

        ARTICLE 3.2 : DOCUMENTS CONSTITUTIFS DE L'OFFRE
        Les offres contiennent les pieces administratives suivantes :
        Presentation du Soumissionnaire.
        Une attestation d'affiliation a la CNSS.
        Extrait du registre national des entreprises actualisees datant de moins de 30 jours.
        Le cahier des charges signe paraphe par le soumissionnaire.
        Attestation constructeur attestant l'originalite des produits.

        A- DOSSIER DE L'OFFRE FINANCIERE
        Le(s) lettre(s) de soumission pour chaque lot(s).
        Devis estimatif detaille : le soumissionnaire est tenu d'indiquer les quantites
        et les prix unitaires (PU) de chaque article et le prix total (PT).

        B- DOSSIER DE L'OFFRE TECHNIQUE
        La deuxieme enveloppe doit porter la mention Offre technique Consultation N 02/DCSI/2025
        et sera composee des pieces techniques suivantes dans l'ordre indique :
        Presentation de l'offre technique.
        Presentation des specifications techniques conformement aux tableaux des clauses techniques particuliers.
        Les delais de livraison des articles.

        ARTICLE 4.1 : DATE LIMITE DE RECEPTION DES OFFRES
        Les soumissionnaires doivent disposer leurs offres au bureau d'ordre a l'adresse suivante :
        TUNISIE TELECOM Direction Centrale des Systemes d'Information Les Jardins du Lac.
        Au plus tard, le 10/06/2025. La date et le numero d'enregistrement sur le registre
        du bureau d'ordre de TUNISIE TELECOM faisant foi.

        ARTICLE 4.2 : DELAI DE VALIDITE DE L'OFFRE
        Les offres seront valables pendant 90 jours a compter de la date limite de reception des offres.

        ARTICLE 5.1 : OUVERTURE DES PLIS
        L'ouverture des plis aura lieu dans les locaux de TUNISIE TELECOM.

        ARTICLE 9 : RECEPTION PROVISOIRE-RECEPTION DEFINITIVE
        9.1 Reception quantitative. 9.2 Reception provisoire. 9.3 Reception definitive.

        ARTICLE 13 : GARANTIE
        Tunisie Telecom informera le fournisseur pour le remplacement dans un delai de 48 heures
        des articles defectueux pendant une periode de 6 mois a partir de la date de la reception
        provisoire pour les Lots 1 et 2. Pour le Lot 3 : la garantie est de 2 ans pour les
        casques et souris sans fil et 3 annees pour les douchettes.

        ARTICLE 14 : CONDITIONS DE PAIEMENT
        Pour chaque Appel de commande, Le paiement se fera 100% sera regle a 60 jours sur
        presentation de l'originale de la facture et de(s) bon(s) de livraison et le PV de
        reception provisoire.

        ARTICLE 15 : PENALITES POUR RETARD
        Il sera applique une penalite pour retard de cinq pour mille (5‰) par jour sur le
        montant des articles non livres avec un maximum de 10% du montant definitif du marche.
        """,
    )

    assert "marche cadre" in facts["subject"]["text"].lower()
    assert "consommables et accessoires informatiques" in facts["subject"]["text"].lower()
    assert "toner" not in facts["subject"]["text"].lower()
    assert "bureau d'ordre" in facts["submission_method"]["text"].lower()
    assert "tunisie telecom" in facts["submission_method"]["text"].lower()
    assert "10/06/2025" in facts["deadline"]["text"]
    assert "90 jours" in facts["validity"]["text"]
    assert "locaux de TUNISIE TELECOM" in facts["opening"]["text"]
    assert "soumissionnaire" in facts["administrative_documents"]["text"].lower()
    assert "cnss" in facts["administrative_documents"]["text"].lower()
    assert "registre national" in facts["administrative_documents"]["text"].lower()
    assert "originalite des produits" in facts["manufacturer_authorization"]["text"].lower()
    assert "lettre" in facts["financial_documents"]["text"].lower()
    assert "devis estimatif" in facts["financial_documents"]["text"].lower()
    assert "presentation de l'offre technique" in facts["technical_documents"]["text"].lower()
    assert "specifications techniques" in facts["technical_documents"]["text"].lower()
    assert "6 mois" in facts["guarantee"]["text"]
    assert "2 ans" in facts["guarantee"]["text"]
    assert "3 annees" in facts["guarantee"]["text"]
    assert "Reception quantitative" in facts["reception"]["text"]
    assert "60 jours" in facts["payment"]["text"]
    assert "facture" in facts["payment"]["text"].lower()
    assert "cinq pour mille" in facts["penalties"]["text"].lower()
    assert "10%" in facts["penalties"]["text"]


def test_extract_document_facts_accepts_intt_guarantee_and_decimal_penalties():
    facts = _facts_from_text(
        "INTT.pdf",
        """
        Garantie et maintenance
        Outre les garanties a preciser pour chacun des equipements proposes selon les tableaux
        y afferents fournis en annexe, le soumissionnaire doit garantir la solution globale
        du Lot A ainsi fournie et mise en place et ce, pour une duree d'un (01) an a partir
        de la date de la reception provisoire.

        ARTICLE 24 : Penalites de retard
        L'INT se reserve le droit, au cas ou le delai contractuel pour la realisation de la
        mission ne serait pas respecte du fait du titulaire du marche, d'appliquer une penalite
        de retard de 0,1% par jour calendaire de retard, du montant contractuel total.
        Ces penalites ne peuvent toutefois depasser le plafond de 5% du montant total.
        """,
    )

    assert "solution globale" in facts["guarantee"]["text"]
    assert "un (01) an" in facts["guarantee"]["text"]
    assert "0,1%" in facts["penalties"]["text"]
    assert "5%" in facts["penalties"]["text"]


def test_extract_document_facts_accepts_cimf_caution_and_service_guarantee_articles():
    facts = _facts_from_text(
        "cahier des charges CIMF.pdf",
        """
        Article 9. CAUTIONNEMENT PROVISOIRE
        La caution provisoire est fixee a un montant de Trois Mille dinars (3 000,000 DT).
        Le cautionnement provisoire sera presente sous la forme d'une caution bancaire
        inconditionnelle emise par une banque tunisienne et payable a premiere demande du CIMF.
        Cette caution devra etre valable pendant cent vingt (120) jours.

        Article 31. GARANTIE & SERVICE APRES-VENTE
        Le titulaire garantit que tous les equipements proposes seront fournis a l'etat neuf.
        La periode de garantie commence a courir a compter de la date de reception provisoire.
        Sa duree minimale est fixee a douze (12) mois. Le titulaire est responsable du bon
        fonctionnement, des vices caches et des corrections necessaires.
        """,
    )

    assert "Trois Mille" in facts["caution"]["text"]
    assert "3 000,000 DT" in facts["caution"]["text"]
    assert "GARANTIE & SERVICE" in facts["guarantee"]["text"]
    assert "douze (12) mois" in facts["guarantee"]["text"]


def test_extract_document_facts_accepts_cni_cloud_caution_and_guarantee():
    facts = _facts_from_text(
        "CC_AO_CLOUD_PRIVE_CNI_VF.pdf",
        """
        ARTICLE 8 : CAUTION PROVISOIRE
        Un cautionnement provisoire, selon modele figurant en ANNEXE 1, est exige de tout
        soumissionnaire d'un montant fixe de 25000 dinars (25 Milles dinars). Le dit
        cautionnement doit etre constitue aupres d'une banque agree par le ministere des
        finances, il devra etre valable pendant cent vingt (120) jours, a compter du lendemain
        de la date limite fixee pour la reception des offres.

        ARTICLE 10 : GARANTIE ET SERVICES APRES VENTE
        9-1 : Garantie Le titulaire du marche garantit que tous les equipements proposes seront
        fournis d'origine constructeur et a l'etat neuf. La periode de garantie est de (12) mois
        au minimum pour toutes les composantes du marche et commence a courir a partir de la
        date de prononciation de la reception provisoire sans reserve. Durant cette periode le
        titulaire du marche assurera le bon etat de fonctionnement et executera toutes les
        reparations et tous les remplacements necessaires.
        """,
    )

    assert "25000 dinars" in facts["caution"]["text"]
    assert "cent vingt" in facts["caution"]["text"]
    assert "GARANTIE ET SERVICES" in facts["guarantee"]["text"]
    assert "(12) mois" in facts["guarantee"]["text"]


def test_extract_document_facts_prefers_offer_deadline_over_clarification_deadline():
    facts = _facts_from_text(
        "CloudNational-AO-CdC.pdf",
        """
        Date limite de reception des demandes d'eclaircissement : 23/04/2017
        Dernier delai de reception des offres : 12/05/2017 - 09h30
        Date de la reunion d'ouverture des offres (publique) : 12/05/2017 - 10h00

        Article 11 : Duree de validite des offres
        Tout soumissionnaire sera lie par son offre pendant cent-vingt (120) jours a compter
        du jour suivant la date limite fixee pour la reception des offres.
        """,
    )

    assert facts["deadline"]["text"] == "12/05/2017 - 09h30"
    assert "cent-vingt (120) jours" in facts["validity"]["text"]


def test_extract_document_facts_handles_tunisie_telecom_international_offer_documents():
    facts = _facts_from_text(
        "CC A.O INTERNATIONAL 5.2017.docx",
        """
        ARTICLE 8 - DOCUMENTS CONSTITUTIFS DE L'OFFRE
        L'offre preparee par le soumissionnaire doit etre presentee au plus tard le 09 fevrier
        2017 et obligatoirement en deux parties distinctes, offre technique (Enveloppe A) et
        offre financiere (Enveloppe B), mises sous enveloppes separees et cachetees portant
        l'intitule de l'Appel d'Offres International N 05/2017, lesquelles devront etre inserees
        dans une seule enveloppe exterieure anonyme portant l'adresse de Tunisie Telecom.
        En dehors de ces deux enveloppes, le soumissionnaire est tenu de presenter les documents
        suivants : Le cautionnement provisoire d'un montant egal a Six mille Dinars Tunisiens
        (6000 DT), la non presentation du cautionnement provisoire constitue un motif pour rejet
        d'office de l'offre. Les cahiers des charges administratifs et techniques, signes avec
        la mention Lu et approuve. Presentation detaillee du soumissionnaire. Delegations de
        pouvoir et de signature. Un engagement sur l'honneur selon le modele de l'annexe 3.
        Attestation fiscale valable a la date limite de reception des offres. Certificat
        d'affiliation a la CNSS, valable a la date de reception des offres.
        A - ENVELOPPE A : OFFRE TECHNIQUE Des tableaux de l'annexe 7 clairement remplis,
        concernant les references du soumissionnaire (au moins 5 par lot) dans l'elaboration
        de missions similaires datant d'au moins des trois dernieres annees, a partir de l'annee 2010.

        ARTICLE 9 - CAUTIONNEMENT PROVISOIRE
        Le soumissionnaire doit, sous peine de rejet de l'offre, fournir avec son offre un
        cautionnement provisoire de six mille Dinars Tunisiens (6000 DT). Le cautionnement
        provisoire se presentera sous la forme d'une caution bancaire inconditionnelle emise par
        une banque tunisienne et payable a premiere demande de Tunisie Telecom. Cette caution
        devra etre valable pendant 120 jours a compter de la date limite de reception des offres.
        """,
    )

    assert "6000 DT" in facts["caution"]["text"]
    assert "TOPNET" not in facts["caution"]["text"]
    assert "cahiers des charges" in facts["administrative_documents"]["text"].lower()
    assert "CNSS" in facts["administrative_documents"]["text"]
    assert "annexe 7" in facts["technical_documents"]["text"].lower()
    assert "references du soumissionnaire" in facts["technical_documents"]["text"].lower()




def test_extract_document_facts_handles_ao03_cert_multilot_demo_fields():
    facts = _facts_from_text(
        "AO03-2019Equipements-et-accessoires-informatiques.PDF",
        """
        le 27/06/2019 a 10h le 27/06/2019 a 10 h. Leur envoi se fera a travers
        la procedure materiel et ce au plus tard delai rigueur. La participation a
        travers la procedure en ligne TUNEPS sera fermee automatiquement.
        Le fournisseur est tenu d'etre engage par son offre pendant 60 jours a partir
        du jour suivant la date limite fixee pour la reception des offres.

        11-1 Pieces Administratives
        L'offre du soumissionnaire doit renfermer l'ensemble des pieces administratives suivantes :
        1 Un cautionnement provisoire valable pour une periode de 60 jours a compter du jour
        suivant la date limite fixee pour la reception des offres. Le Montant de la dite caution
        est fixe a : Lot N1 :500 DT Lot N2 :520 DT Lot N3 :130 DT Lot N4 :850 DT
        Lot N5 :250 DT Lot N6 :20 DT Lot N7 :30 DT.
        2 situation fiscale reglee A verifier via le systeme TUNEPS.
        3 affiliation a la CNSS en activite A verifier via le systeme TUNEPS.
        4 Original de l'extrait du registre de commerce valide a la date limite de remise
        des offres a envoyer via le systeme TUNEPS.
        5 Declaration sur l'honneur de non influence Cacher la case j'accepte sur TUNEPS.
        6 Declaration sur l'honneur attestant que le soumissionnaire n'etait pas un agent
        au sein du CERT ou l'ait quittee depuis au moins cinq ans. Cocher la case j'accepte sur TUNEPS.
        7 Le present document d'Appel d'Offres (CCAP et CCTP) Lu, approuve et cocher la
        case y afferente a travers le systeme TUNEPS.

        11-2 PIECES TECHNIQUES
        1 Tableaux de conformite conformement au modele joint-en Annexe 1.
        Accompagne par les justificatifs, les catalogues et prospectus techniques des equipements a fournir.
        2 Certification ISO 9001 valable delivre par le constructeur pour les LOT 1, 2,3,4,5 et 7.
        3 Norme de securite EN 60950 pour les LOT 1, 2, 3,4,5 et 7.
        4 Certificat de conformite delivre par un organisme accredite pour les Normes EN 55022, EN 55024.
        5 Modele d'engagement concernant le service Apres vente et la disponibilite des pieces de rechange Annexe 5.
        6 Le contrat de maintenance a ajouter en pieces jointes sur le systeme TUNEPS.

        Le fournisseur garantit gratuitement tous les travaux executes au sein de cet Appel d'offre
        pendant un delai minimum de douze (12) mois a compter de la date de la prononciation
        de la reception provisoire des travaux sans reserves.
        Le paiements relatif a l'execution du marche sera effectue en une seule fois et ce apres
        services fait et receptionne provisoirement sans reserve par virement bancaire ou postale.
        Sur production des pieces suivantes : une facture et le proces-verbal de reception provisoire.
        Pour chaque jour de retard non justifie par ecrit et a l'avance, apporte dans l'execution
        de la commande, le fournisseur devra payer une penalite a raison de un pour mille (1 %o)
        du montant du lot non livre pour chaque jour de retard. Toutefois, le montant total de ces
        penalites ne doit pas exceder cinq pour cent (5%) du montant total du marche.
        """,
    )

    assert facts["deadline"]["text"] == "27/06/2019 a 10h"
    assert "60 jours" in facts["validity"]["text"]
    assert "Lot N4 :850 DT" in facts["caution"]["text"]
    assert "Lot N7 :30 DT" in facts["caution"]["text"]
    assert "situation fiscale" in facts["administrative_documents"]["text"].lower()
    assert "CNSS" in facts["administrative_documents"]["text"]
    assert "CCAP et CCTP" in facts["administrative_documents"]["text"]
    assert "Tableaux de conformite" in facts["technical_documents"]["text"]
    assert "Certification ISO 9001" in facts["technical_documents"]["text"]
    assert "douze (12) mois" in facts["guarantee"]["text"]
    assert "virement bancaire" in facts["payment"]["text"].lower()
    assert "un pour mille" in facts["penalties"]["text"].lower()
    assert "5%" in facts["penalties"]["text"]


def test_extract_document_facts_handles_arru_ministere_tables_and_penalties():
    facts = _facts_from_text(
        "Ministere de l'Equipement.pdf",
        """
        Article 8 - Documents de l'appel d'offres et pieces a fournir.
        8-1 l'enveloppe exterieure devra contenir en plus des 2 enveloppes interieures A et B
        les documents suivants: 1 Cautionnement provisoire Une caution de 1 400 dinars sera
        fournie selon modele ci-joint en Annexe1. Cautionnement valable 90 jours a partir du
        lendemain de la date limite de reception des offres.
        2 Attestation de situation fiscale Copie.
        3 Attestation d'affiliation a la Caisse Nationale de la Securite Sociale Copie.
        4 Une declaration sur l'honneur presentee par le soumissionnaire attestant qu'il n'etait
        pas un employe au sein de l'administration selon modele indique en annexe 2.
        5 declaration sur l'honneur de non influence comportant la confirmation de ne pas faire
        des promesses ou des dons selon modele en annexe 3.
        6 copie du registre de commerce en cours de validite.
        7 Le present cahier des charges Paraphe et signe, avec cachet du soumissionnaire.

        8-2 l'Enveloppe interieure A: L'offre technique Celle-ci devra contenir les pieces suivantes:
        A.1 Les fiches techniques des equipements informatiques. Remplies conformement au modele
        en annexe 6 paraphees et signees avec cachet du soumissionnaire.
        A.2 Les prospectus techniques des equipements informatiques proposes micro-ordinateurs,
        imprimantes, scanners et onduleurs. paraphees et signes et avec cachet du soumissionnaire.

        ARTICLE 6 : Penalites de retard
        En cas de depassement des delais prevus au niveau de l'article 5 ci-dessus, le titulaire
        du marche sera passible d'une penalite de retard de 50 dinars par jour. Cette penalite
        est plafonnee a 5 % du montant du marche.
        """,
    )

    assert "1 400 dinars" in facts["administrative_documents"]["text"]
    assert "Attestation de situation fiscale" in facts["administrative_documents"]["text"]
    assert "registre de commerce" in facts["administrative_documents"]["text"].lower()
    assert "fiches techniques" in facts["technical_documents"]["text"].lower()
    assert "prospectus techniques" in facts["technical_documents"]["text"].lower()
    assert "50 dinars par jour" in facts["penalties"]["text"]
    assert "5 %" in facts["penalties"]["text"]

def test_extract_document_facts_opening_handles_huis_clos_and_commission_sections():
    bfpm_e = _facts_from_text(
        "bfpm-e.pdf",
        """
        Atticle 8. OUVERTURE DES PLIS
        a) La commission des consultations se reunit en seance unique (a huis clos)
        pour ouvrir les enveloppes contenant les offres techniques et financieres.
        b) La date de l'ouverture des plis techniques et financiers doit avoir lieu
        dans un delai maximum d'un jour ouvrable suivant la date limite de reception des offres.
        """,
    )

    assert "huis clos" in bfpm_e["opening"]["text"].lower()
    assert "offres techniques" in bfpm_e["opening"]["text"].lower()

    cc_cnss = _facts_from_text(
        "cc_cnss.docx",
        """
        Ouverture des plis

        Pendant cette seance la commission d'ouverture des offres procedera a l'ouverture
        simultanement des enveloppes parvenues dans les delais au bureau d'ordre central
        et le decryptage des offres parvenues en ligne.

        Montant des offres
        Les montants doivent etre presentes hors taxes et TTC.
        """,
    )

    assert "commission d'ouverture" in cc_cnss["opening"]["text"].lower()
    assert "decryptage des offres" in cc_cnss["opening"]["text"].lower()
    assert "montants doivent etre presentes" not in cc_cnss["opening"]["text"].lower()


def test_extract_document_facts_references_handles_hpe_frame_reference_table():
    facts = _facts_from_text(
        "cdc_hpe.pdf",
        """
        ARTICLE 4 - Soumissionnaire et Equipe intervenante :
        Soumissionnaire : N° Designation Exigence minimale (*)
        1 Nombre de reference 3 References dans l'installation des frames HPE
        durant les 3 dernieres annees (avec justificatifs)
        2 Nombre d'effectif dedie aux projets d'installation, de configuration
        et de maintenance des Frames HPE 2
        (*) Seules les references justifiees par une commande, un contrat ou facture
        seront prises en consideration.
        """,
    )

    assert "frames HPE" in facts["references"]["text"]
    assert "justificatifs" in facts["references"]["text"].lower()


def test_extract_document_facts_references_handles_cnss_reference_list_row():
    facts = _facts_from_text(
        "CC_CNSS.docx",
        """
        Composition de l'offre
        Liste des references du soumissionnaire (au minimum, deux references pour la vente,
        la configuration des memes marques des produits proposes dans le cadre de cette
        Consultation durant la periode du 1er janvier 2020 jusqu'a la date limite de
        reception des offres) le soumissionnaire doit presenter les justificatifs necessaires :
        Contrats ou PV de reception provisoires ou PV de reception definitifs.
        Les formulaires de reponses techniques avec les prospectus techniques des logiciels proposes.
        """,
    )

    assert "deux references" in facts["references"]["text"].lower()
    assert "memes marques" in facts["references"]["text"].lower()
    assert "justificatifs" in facts["references"]["text"].lower()


def test_extract_document_facts_submission_method_handles_address_delivery_clause():
    facts = _facts_from_text(
        "cdc_hpe.pdf",
        """
        ARTICLE 8 - DATE LIMITE DE RECEPTION DES OFFRES
        Les soumissionnaires doivent envoyer leurs offres a l'adresse suivante :
        BANQUE ZITOUNA 02 Boulevard Qualite de la vie - LE KRAM - TUNIS.
        La date limite de reception des offres est arretee au 06/04/2026.
        Le registre du bureau d'ordre de la BANQUE ZITOUNA faisant foi.
        """,
    )

    assert "adresse suivante" in facts["submission_method"]["text"].lower()
    assert "BANQUE ZITOUNA" in facts["submission_method"]["text"]
    assert "LE KRAM" in facts["submission_method"]["text"]


def test_extract_document_facts_reception_and_payment_handle_cnss_sections():
    facts = _facts_from_text(
        "cc_cnss.docx",
        """
        Reception provisoire
        Apres la mise en exploitation des composantes de l'offre, il sera procede
        au cours d'une periode de deux mois a l'utilisation en reel des solutions installees.
        A l'issue de cette periode et si aucune anomalie n'est constatee, un proces-verbal
        de reception provisoire sera etabli par les 2 parties.

        Reception definitive
        La reception definitive sera prononcee apres quatre mois de la reception provisoire
        et donnera lieu a un proces-verbal de reception definitive.

        Modalites de paiement
        Le reglement se fera integralement apres la signature du proces-verbal de la reception
        definitive sans reserve. La facture doit etre deposee au Bureau d'Ordre Central de
        la CNSS. La facture emises par le fournisseur est payable a 45 jours de la date de
        reception de la facture par la CNSS.
        """,
    )

    assert "reception provisoire" in facts["reception"]["text"].lower()
    assert "reception definitive" in facts["reception"]["text"].lower()
    assert "45 jours" in facts["payment"]["text"]
    assert "Bureau d'Ordre Central" in facts["payment"]["text"]


def test_extract_document_facts_classifies_contenu_offre_table():
    facts = _facts_from_text(
        "ATI.pdf",
        """
        ARTICLE 1 : OBJET DE LA PRESENTE CONSULTATION
        L'Agence Tunisienne d'Internet se propose de lancer une consultation pour
        l'acquisition de materiels informatiques repartis en 05 lots independants.

        ARTICLE 2 : PRESENTATION DES OFFRES
        L'offre est placee dans une enveloppe contenant outre les documents administratifs
        et techniques, la soumission ainsi que le bordereau detaille des prix. Les offres
        devront parvenir par voie postale ou par remise directe au bureau d'ordre.

        ARTICLE 3 : CONTENU DE L'OFFRE
        L'offre doit comporter les pieces suivantes :
        N DESIGNATION AUTHENTIFICATION PIECE
        1 Fiche de renseignements generaux sur le soumissionnaire dument complete Selon le modele figurant en annexe n1
        2 Cahier des charges paraphe sur chaque page date signature et cachet du soumissionnaire
        3 Tableaux des specificites techniques dument remplis et signes et les documents techniques y afferant
        4 La soumission dument remplie et signee par le soumissionnaire Selon le modele figurant en annexe n2
        5 Les Bordereaux des prix dument remplis et signes par le soumissionnaire Selon le modele figurant en annexe n3
        6 Engagement de garantie complete
        7 Un extrait du registre de commerce/certificat RNE

        ARTICLE 4 : SPECIFICATION TECHNIQUE
        Garantie 1an.
        """,
    )

    assert "fiche de renseignements" in facts["administrative_documents"]["text"].lower()
    assert "cahier des charges" in facts["administrative_documents"]["text"].lower()
    assert "registre de commerce" in facts["administrative_documents"]["text"].lower()
    assert "tableaux des specificites techniques" in facts["technical_documents"]["text"].lower()
    assert "documents techniques y afferant" in facts["technical_documents"]["text"].lower()
    assert "soumission" in facts["financial_documents"]["text"].lower()
    assert "bordereaux des prix" in facts["financial_documents"]["text"].lower()


def test_extract_document_facts_handles_stb_direct_consultation_clauses():
    facts = _facts_from_text(
        "STB.pdf",
        """
        ARTICLE 2 : PARTICIPATION
        Les soumissionnaires sont tenus de presenter une seule et unique offre technique et sans variantes.
        Les soumissionnaires sont tenus de presenter une seule et unique offre financiere et sans variantes.

        ARTICLE 8 : CAUTION PROVISOIRE
        Le montant de la caution provisoire est fixe a 500,000 DT (Cinq cent) Dinars Tunisiens.

        ARTICLE 10 : PRESENTATION DES OFFRES
        2-Des documents administratifs ci-apres :
        o La fiche « Know your supplier », dument remplie et signee.
        o Un certificat d'affiliation a la Caisse Nationale de Securite Sociale.
        o L'original, de date ne depassant pas 3 mois, du certificat d'inscription au Registre National
        d'Entreprise.
        o La decision, la procuration ou le pouvoir du signataire.

        3-D'une offre technique constituee de ce que suit :
        - Le formulaire de reponses techniques.
        - les fiches, prospectus et les notices d'utilisation du fabricant libelles en langue francaise ou anglaise.
        - L'attestation de garantie de constructeur de trois annees des ordinateurs portables.
        - engagement du soumissionnaire specifiant que les equipements proposes sont neufs et d'origine.

        4-D'une offre financiere constituee des documents ci-apres :
        - La soumission.
        - Le bordereau des prix.

        ARTICLE 2 : CAUTION DEFINITIVE
        Le soumissionnaire dont l'offre sera retenue doit fournir obligatoirement une caution bancaire
        definitive de 3% du montant initial
        du marche dans les 10 jours a compter de la notification.

        ARTICLE 12 : PENALITES DE RETARD
        En cas de retard, il sera applique une penalite de retard de un pour mille par jour de retard.
        """,
    )

    assert "ne sont pas autorisees" in facts["variants"]["text"].lower()
    assert "know your supplier" in facts["administrative_documents"]["text"].lower()
    assert "Caisse Nationale" in facts["administrative_documents"]["text"]
    assert "Registre National" in facts["administrative_documents"]["text"]
    technical_text = facts["technical_documents"]["text"].lower()
    assert "formulaire de reponses" in technical_text
    assert "fiches, prospectus" in technical_text
    assert "attestation de garantie" in technical_text
    assert "la soumission" not in technical_text
    assert "bordereau des prix" not in technical_text
    assert "manufacturer_authorization" not in facts
    assert "3%" in facts["definitive_caution"]["text"]
    assert "pour mille" in facts["penalties"]["text"].lower()
    assert "soumission" in facts["financial_documents"]["text"].lower()
    assert "bordereau des prix" in facts["financial_documents"]["text"].lower()


def test_extract_document_facts_handles_switch_san_structured_consultation_docx():
    facts = _facts_from_text(
        "Consultation_31-2026_SWITCH_SAN.docx",
        """
        La presente consultation a pour objet la revision de la plateforme SAN Switch.
        A cet effet, le present dossier de consultation comprend : 1) L'acte de soumission
        2) Un cahier des charges 3) Un bordereau des prix 4) Une fiche de renseignements.
        En cas de desaccord entre deux documents differents, la priorite sera donnee au document prioritaire.

        ARTICLE 2 : PROCEDURE DE PARTICIPATION
        2.1 Presentation et envoi des offres
        Les offres contiendront les documents ci-apres divises en deux groupes, places dans deux enveloppes
        separees, et portant respectivement les mentions enveloppe (A) et enveloppe (B). Ces deux enveloppes
        seront placees dans une troisieme enveloppe fermee qui ne doit faire aucune mention de l'identite du
        soumissionnaire et porter l'indication : ARAB TUNISIAN BANK Direction des Services Generaux
        9, Rue Hedi NOUIRA - 1001 TUNIS. Cette derniere enveloppe devra parvenir au plus tard le 02/06/2026 inclus.
        Enveloppe A: L'enveloppe A - OFFRE FINANCIERE - contiendra les documents suivants dans l'ordre :
        Document N 1: Soumission.
        Document N 2 : Un cautionnement provisoire de 2000 DT T.T.C. sous forme de caution bancaire.
        Document N 3 : Bordereau des prix.
        +++ offre technico-financiere sur support numerique.
        Enveloppe B: L'enveloppe B - OFFRE TECHNIQUE - fermee et scellee portant le nom de l'entreprise
        soumissionnaire et contiendra les documents suivants dans l'ordre :
        Document N 1 : Le cahier des charges.
        Document N 2 : Un formulaire de reponse.
        Document N 3 : La fiche de renseignements sur le soumissionnaire.
        Document N 4 : Une copie recente de l'RNE.
        Document N 5 : Une declaration sur l'honneur justifiant que le soumissionnaire n'est pas en etat de cessation de paiement.
        Document N 6 : Une attestation de solde de la CNSS valable a la date limite de reception des offres.
        +++ offre technique sur support numerique.
        2.2 Recevabilite des offres
        Le soumissionnaire doit envoyer a l'ATB sa soumission sous pli ferme au plus tard le 02/06/2026 inclus.
        2.3 Validite des soumissions
        Les soumissions doivent rester valables pendant trente (30) jours calendaires a compter du jour suivant
        la date limite de reception des offres.

        ARTICLE 4 : DISPOSITIONS CONTRACTUELLES
        Le soumissionnaire dont l'offre aura ete retenue doit, fournir obligatoirement a l'ATB un cautionnement
        dont le montant total sera egal a 3% du montant total TTC arrondi au dinar inferieur.
        Le cautionnement definitif reste affecte a la garantie de la bonne execution du marche.
        ARTICLE 6 : MODALITE DE PAIEMENT
        Le reglement sera effectue au plus tard 60(soixante) jours a compter de la reception de la facture par l'ATB,
        le cachet du Bureau d'Ordre Central faisant foi.

        ARTICLE 5 : SERVICES ET EXIGENCES
        Le prestataire doit fournir a l'ATB une documentation complete de la solution (Low level design / high level design)
        ainsi que les procedures d'installation, configuration.
        Le prestataire doit fournir a l'ATB un plan d'action des differentes phases d'installation.
        Le soumissionnaire devra mener les tests appropries afin d'assurer la haute disponibilite.
        Le soumissionnaire s'engage a faire un transfert de competences a l'equipe ATB.
        L'entreprise soumissionnaire devra justifier d'une experience averee et de references pertinentes dans les domaines
        objet de la presente consultation.
        ARTICLE 6 : FORMATION
        Le soumissionnaire est appele a nous fournir une proposition de formation relative a l'administration des SAN switch.

        ARTICLE 7 : DELAI D'EXECUTION ET PENALITES DE RETARD
        Si les delais de livraison et d'execution des travaux objet du present marche ne sont respectes, le soumissionnaire
        retenu encourra une penalite d'un pour mille (1/1000) du montant du marche par jour de retard sans toutefois
        depasser 10% du montant du marche.
        9.2. Reception
        La reception technique provisoire sera prononcee dans un delai maximum de 30 jours et sera sanctionnee par un
        compte rendu signe par les deux parties. La reception definitive sera prononcee au plus tard 3 mois apres la
        reception technique provisoire et sera sanctionnee par un compte rendu signe par les deux parties.
        9.3. Reglement
        Le reglement du prix du marche sera effectue comme suit : 50% a la livraison de la solution et la reception
        technique provisoire sans reserves; 40% a la reception definitive sans reserves; l0% a l'expiration du delai
        de garantie technique. Il sera procede aux retenues a la source pour chaque paiement.
        """,
    )

    assert "troisieme enveloppe" in facts["submission_method"]["text"].lower()
    assert "02/06/2026" in facts["deadline"]["text"]
    assert "trente (30) jours" in facts["validity"]["text"].lower()
    assert "2000 DT" in facts["caution"]["text"]
    assert "fiche de renseignements" in facts["information_sheet"]["text"].lower()
    assert "sur le soumissionnaire" in facts["information_sheet"]["text"].lower()
    assert "attestation de solde de la CNSS" in facts["cnss"]["text"]
    assert "RNE" in facts["rne"]["text"]
    admin_text = facts["administrative_documents"]["text"].lower()
    assert "cahier des charges" in admin_text
    assert "formulaire de reponse" in admin_text
    assert "fiche de renseignements" in admin_text
    assert "declaration sur l'honneur" in admin_text
    assert "attestation de solde de la cnss" in admin_text
    tech_text = facts["technical_documents"]["text"].lower()
    assert "documentation complete" in tech_text
    assert "procedures d'installation" in tech_text
    assert "ainsi que les procedures" not in tech_text
    assert "plan d'action" in tech_text
    assert "transfert de competences" in tech_text
    assert "proposition de formation" in tech_text
    assert "references pertinentes" in facts["references"]["text"].lower()
    financial_text = facts["financial_documents"]["text"].lower()
    assert "soumission" in financial_text
    assert "cautionnement provisoire" in financial_text
    assert "t.t.c" in financial_text
    assert "bordereau des prix" in financial_text
    assert "support numerique" in financial_text
    assert "3%" in facts["definitive_caution"]["text"]
    assert "1/1000" in facts["penalties"]["text"]
    assert "10%" in facts["penalties"]["text"]
    assert "reception technique provisoire" in facts["reception"]["text"].lower()
    assert "reception definitive" in facts["reception"]["text"].lower()
    assert "60" in facts["payment"]["text"]
    assert "50%" in facts["payment"]["text"]
    assert "40%" in facts["payment"]["text"]
    assert "l0%" in facts["payment"]["text"]


def test_tender_checklist_answer_uses_extracted_facts():
    answer = build_tender_checklist_answer(
        "UBCI.pdf",
        {
            "subject": {"text": "L'UBCI se propose d'acquerir 160 UC", "page": "2", "section": "admin"},
            "deadline": {"text": "18 Juillet 2025", "page": "3", "section": "deadline"},
            "variants": {"text": "Les variantes ne sont pas autorisees", "page": "4", "section": "admin"},
        },
    )

    assert "Analyse de consultation" in answer
    assert "L'UBCI se propose d'acquerir 160 UC" in answer
    assert "18 Juillet 2025" in answer
    assert "Non - Les variantes ne sont pas autorisees" in answer
    assert "Non mentionne dans ce document." in answer


def test_language_detection_defaults_to_french_for_tender_questions():
    assert _detect_answer_language("Une attestation d'affiliation a la CNSS est-elle exigee ?") == "fr"
    assert _detect_answer_language("Un extrait du registre de commerce est-il exige ?") == "fr"
    assert _detect_answer_language("Existe-t-il des penalites de retard ?") == "fr"
    assert _language_instruction("RNE ?") == "Reponds uniquement en francais."


def test_meta_language_answers_are_hallucination_signals():
    answer = "The user requested the response to be in English."

    assert any(signal.lower() in answer.lower() for signal in HALLUCINATION_SIGNALS)


@pytest.mark.asyncio
async def test_facts_first_answers_new_scalar_fields(
    initialized_db,
    seed_department,
    seed_user,
    seed_document,
):
    await seed_department()
    await seed_user()
    await seed_document(
        filename="CDC 01-2026.pdf",
        status="indexed",
        extracted_facts={
            "caution": {
                "text": "5 000 DT",
                "page": "6",
                "section": "guarantee",
            },
            "validity": {
                "text": "120 jours a compter de la date limite de reception des offres",
                "page": "8",
                "section": "deadline",
            },
        },
    )

    async with initialized_db.session_factory() as session:
        answer, metas = await answer_from_document_facts(
            db=session,
            question="Quelle est la caution provisoire ?",
            source_filter=["CDC 01-2026.pdf"],
            department_filter=["commerciale"],
            universe_id=None,
            user_id="user-1",
            is_admin=True,
        )

    assert "Caution : 5 000 DT" in answer
    assert "Source: CDC 01-2026.pdf, page 6." in answer
    assert metas == [
        {
            "source": "CDC 01-2026.pdf",
            "page": "6",
            "section": "guarantee",
            "score": 1.0,
        }
    ]


@pytest.mark.asyncio
async def test_facts_first_answers_lobjet_question_with_apostrophe(
    initialized_db,
    seed_department,
    seed_user,
    seed_document,
):
    await seed_department()
    await seed_user()
    await seed_document(
        filename="UBCI.pdf",
        status="indexed",
        extracted_facts={
            "subject": {
                "text": "L'UBCI se propose d'acquerir 160 UC, 20 Laptop et 320 ecrans",
                "page": "2",
                "section": "admin",
            }
        },
    )

    async with initialized_db.session_factory() as session:
        answer, metas = await answer_from_document_facts(
            db=session,
            question="Quel est l'objet de la consultation ?",
            source_filter=["UBCI.pdf"],
            department_filter=["commerciale"],
            universe_id=None,
            user_id="user-1",
            is_admin=True,
        )

    assert "160 UC" in answer
    assert "Source: UBCI.pdf, page 2." in answer
    assert metas[0]["source"] == "UBCI.pdf"


@pytest.mark.asyncio
async def test_facts_first_answers_structured_document_lists(
    initialized_db,
    seed_department,
    seed_user,
    seed_document,
):
    await seed_department()
    await seed_user()
    await seed_document(
        filename="STEG.pdf",
        status="indexed",
        extracted_facts={
            "administrative_documents": {
                "text": "- un extrait du registre de commerce\n- un certificat d'affiliation a la CNSS",
                "items": [
                    {
                        "text": "un extrait du registre de commerce",
                        "page": "1",
                        "section": "admin",
                    },
                    {
                        "text": "un certificat d'affiliation a la CNSS",
                        "page": "1",
                        "section": "admin",
                    },
                ],
                "page": "1",
                "section": "admin",
            }
        },
    )

    async with initialized_db.session_factory() as session:
        answer, metas = await answer_from_document_facts(
            db=session,
            question="Quels documents administratifs faut-il fournir ?",
            source_filter=["STEG.pdf"],
            department_filter=["commerciale"],
            universe_id=None,
            user_id="user-1",
            is_admin=True,
        )

    assert "Documents administratifs :" in answer
    assert "- un extrait du registre de commerce" in answer
    assert "- un certificat d'affiliation a la CNSS" in answer
    assert "Source: STEG.pdf, page 1." in answer
    assert metas == [
        {
            "source": "STEG.pdf",
            "page": "1",
            "section": "admin",
            "score": 1.0,
        }
    ]
