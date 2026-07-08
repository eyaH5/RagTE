from __future__ import annotations

import ingest


def test_pdf_text_only_mode_uses_relaxed_direct_text_without_docling(monkeypatch):
    entries = [{"page": "1", "text": "Short but useful cahier des charges content for direct text extraction."}]

    monkeypatch.setattr(ingest, "PDF_TEXT_ONLY_MODE", True)
    monkeypatch.setattr(ingest, "_read_fresh_text_cache_entries", lambda file_path, filename: [])
    monkeypatch.setattr(ingest, "_extract_text_entries_pypdf", lambda file_path: (entries, 12))
    monkeypatch.setattr(ingest, "_extract_text_entries_pymupdf", lambda file_path: ([], 12))
    monkeypatch.setattr(ingest, "_should_use_direct_pdf_text", lambda candidate_entries, page_count: False)
    monkeypatch.setattr(ingest, "_direct_pdf_candidate_score", lambda candidate_entries, page_count: 10)
    monkeypatch.setattr(ingest, "_write_text_cache", lambda filename, entries: None)

    def fail_docling(file_path, filename):
        raise AssertionError("Docling/OCR fallback should not run in PDF_TEXT_ONLY_MODE")

    monkeypatch.setattr(ingest, "_extract_text_entries_docling", fail_docling)

    chunks, metas, ids = ingest.extract_and_chunk("demo.pdf", "demo.pdf")

    assert chunks == [entries[0]["text"]]
    assert ids == ["demo.pdf_c0"]
    assert metas[0]["text_quality_text_source"] == "pdf_text_layer"


def test_pdf_text_only_mode_returns_empty_when_no_direct_text(monkeypatch):
    monkeypatch.setattr(ingest, "PDF_TEXT_ONLY_MODE", True)
    monkeypatch.setattr(ingest, "_read_fresh_text_cache_entries", lambda file_path, filename: [])
    monkeypatch.setattr(ingest, "_extract_text_entries_pypdf", lambda file_path: ([], 3))
    monkeypatch.setattr(ingest, "_extract_text_entries_pymupdf", lambda file_path: ([], 3))

    def fail_docling(file_path, filename):
        raise AssertionError("Docling/OCR fallback should not run in PDF_TEXT_ONLY_MODE")

    monkeypatch.setattr(ingest, "_extract_text_entries_docling", fail_docling)

    assert ingest.extract_and_chunk("scanned.pdf", "scanned.pdf") == ([], [], [])
