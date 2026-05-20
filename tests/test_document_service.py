from __future__ import annotations

from types import SimpleNamespace

import pytest

from api.services import document_service as document_service_module


def test_attach_document_metadata_adds_expected_fields():
    metas = [{"page": "1"}, {"page": "2", "universe_id": "stale"}]

    document_service_module.DocumentService._attach_document_metadata(
        metas,
        doc_id="doc-123",
        department="commerciale",
        uploaded_by="user-1",
        visibility="department",
        universe_id="universe-1",
    )

    assert metas[0]["doc_id"] == "doc-123"
    assert metas[0]["department"] == "commerciale"
    assert metas[0]["uploaded_by"] == "user-1"
    assert metas[0]["visibility"] == "department"
    assert metas[0]["chunk_index"] == 0
    assert metas[0]["universe_id"] == "universe-1"
    assert metas[1]["chunk_index"] == 1


def test_attach_document_metadata_removes_universe_when_none():
    metas = [{"page": "1", "universe_id": "old-universe"}]

    document_service_module.DocumentService._attach_document_metadata(
        metas,
        doc_id="doc-123",
        department="commerciale",
        uploaded_by="user-1",
        visibility="department",
        universe_id=None,
    )

    assert "universe_id" not in metas[0]


@pytest.mark.asyncio
async def test_run_async_from_sync_works_inside_running_event_loop():
    async def _work():
        return "ok"

    assert document_service_module.DocumentService._run_async_from_sync(lambda: _work()) == "ok"


def test_maybe_enrich_facts_rebuilds_tender_profile_after_llm(monkeypatch, test_settings, settings_override):
    settings_override(document_service_module)
    test_settings.LLM_FACT_EXTRACTION_ENABLED = True
    test_settings.LLM_FACT_EXTRACTION_FIELDS = "subject,technical_documents,financial_documents,payment"
    test_settings.LLM_FACT_EXTRACTION_MAX_PAGES = 5
    test_settings.LLM_FACT_EXTRACTION_MAX_OUTPUT_TOKENS = 800
    test_settings.LLM_REASONING_EFFORT = "low"

    async def fake_extract_llm_facts_for_weak_fields(**kwargs):
        final_facts = {
            **kwargs["draft_facts"],
            "technical_documents": {
                "text": "- Offre technique\n- Fiches techniques",
                "items": [{"text": "Offre technique"}, {"text": "Fiches techniques"}],
                "page": "7",
                "section": "llm_extracted",
            },
            "financial_documents": {
                "text": "- Soumission\n- Bordereau des prix",
                "items": [{"text": "Soumission"}, {"text": "Bordereau des prix"}],
                "page": "8",
                "section": "llm_extracted",
            },
            "payment": {"text": "Paiement par virement apres facture.", "page": "9", "section": "llm_extracted"},
        }
        return SimpleNamespace(
            regex_facts=kwargs["draft_facts"],
            llm_facts={
                "technical_documents": final_facts["technical_documents"],
                "financial_documents": final_facts["financial_documents"],
                "payment": final_facts["payment"],
            },
            derived_facts={},
            rejected_llm_facts={},
            final_facts=final_facts,
            weak_fields=["technical_documents", "financial_documents", "payment"],
        )

    monkeypatch.setattr(
        document_service_module,
        "extract_llm_facts_for_weak_fields",
        fake_extract_llm_facts_for_weak_fields,
    )

    enriched = document_service_module.DocumentService._maybe_enrich_facts_with_llm(
        chunks=["page text"],
        metas=[{"source": "doc.pdf", "page": "1", "section": "general"}],
        facts={
            "subject": {"text": "Objet: acquisition de materiels informatiques.", "page": "1", "section": "general"},
            "tender_profile": {"fields": {"object": {"text": "stale"}}},
        },
    )

    profile_fields = enriched["tender_profile"]["fields"]
    assert "technical_documents" in profile_fields
    assert "financial_documents" in profile_fields
    assert "payment" in profile_fields
    assert profile_fields["object"]["text"] == "Objet: acquisition de materiels informatiques"


def test_maybe_enrich_facts_expands_llm_fields_for_arabic_documents(
    monkeypatch,
    test_settings,
    settings_override,
):
    settings_override(document_service_module)
    test_settings.LLM_FACT_EXTRACTION_ENABLED = True
    test_settings.LLM_FACT_EXTRACTION_FIELDS = "subject,payment"
    test_settings.LLM_FACT_EXTRACTION_AUTO_ARABIC = True
    test_settings.LLM_FACT_EXTRACTION_ARABIC_REASONING_EFFORT = "medium"

    captured = {}

    async def fake_extract_llm_facts_for_weak_fields(**kwargs):
        captured["fields"] = kwargs["fields"]
        captured["arabic_reasoning_effort"] = kwargs["arabic_reasoning_effort"]
        return SimpleNamespace(
            regex_facts=kwargs["draft_facts"],
            llm_facts={},
            derived_facts={},
            rejected_llm_facts={},
            final_facts=kwargs["draft_facts"],
            weak_fields=[],
        )

    monkeypatch.setattr(
        document_service_module,
        "extract_llm_facts_for_weak_fields",
        fake_extract_llm_facts_for_weak_fields,
    )

    document_service_module.DocumentService._maybe_enrich_facts_with_llm(
        chunks=["طلب عروض خاص باقتناء مواد إعلامية. العرض الفني والعرض المالي والضمان الوقتي."],
        metas=[{"source": "doc.pdf", "page": "1", "section": "general", "text_quality_mode": "arabic_noisy"}],
        facts={},
    )

    assert "subject" in captured["fields"]
    assert "payment" in captured["fields"]
    assert "caution" in captured["fields"]
    assert "definitive_caution" in captured["fields"]
    assert "administrative_documents" in captured["fields"]
    assert captured["arabic_reasoning_effort"] == "medium"


def test_maybe_enrich_facts_respects_auto_arabic_flag(
    monkeypatch,
    test_settings,
    settings_override,
):
    settings_override(document_service_module)
    test_settings.LLM_FACT_EXTRACTION_ENABLED = True
    test_settings.LLM_FACT_EXTRACTION_FIELDS = "subject,payment"
    test_settings.LLM_FACT_EXTRACTION_AUTO_ARABIC = False

    captured = {}

    async def fake_extract_llm_facts_for_weak_fields(**kwargs):
        captured["fields"] = kwargs["fields"]
        return SimpleNamespace(
            regex_facts=kwargs["draft_facts"],
            llm_facts={},
            derived_facts={},
            rejected_llm_facts={},
            final_facts=kwargs["draft_facts"],
            weak_fields=[],
        )

    monkeypatch.setattr(
        document_service_module,
        "extract_llm_facts_for_weak_fields",
        fake_extract_llm_facts_for_weak_fields,
    )

    document_service_module.DocumentService._maybe_enrich_facts_with_llm(
        chunks=["طلب عروض خاص باقتناء مواد إعلامية."],
        metas=[{"source": "doc.pdf", "page": "1", "section": "general"}],
        facts={},
    )

    assert captured["fields"] == ("subject", "payment")


def test_maybe_enrich_facts_expands_fields_for_arabic_noisy_quality_mode(
    monkeypatch,
    test_settings,
    settings_override,
):
    settings_override(document_service_module)
    test_settings.LLM_FACT_EXTRACTION_ENABLED = True
    test_settings.LLM_FACT_EXTRACTION_FIELDS = "subject,payment"
    test_settings.LLM_FACT_EXTRACTION_AUTO_ARABIC = True

    captured = {}

    async def fake_extract_llm_facts_for_weak_fields(**kwargs):
        captured["fields"] = kwargs["fields"]
        return SimpleNamespace(
            regex_facts=kwargs["draft_facts"],
            llm_facts={},
            derived_facts={},
            rejected_llm_facts={},
            final_facts=kwargs["draft_facts"],
            weak_fields=[],
        )

    monkeypatch.setattr(
        document_service_module,
        "extract_llm_facts_for_weak_fields",
        fake_extract_llm_facts_for_weak_fields,
    )

    document_service_module.DocumentService._maybe_enrich_facts_with_llm(
        chunks=["mostly garbled text"],
        metas=[{"source": "doc.pdf", "page": "1", "section": "general", "text_quality_mode": "arabic_noisy"}],
        facts={},
    )

    assert "caution" in captured["fields"]
    assert "definitive_caution" in captured["fields"]


def test_maybe_enrich_facts_expands_fields_for_arabic_partial_pages_metadata(
    monkeypatch,
    test_settings,
    settings_override,
):
    settings_override(document_service_module)
    test_settings.LLM_FACT_EXTRACTION_ENABLED = True
    test_settings.LLM_FACT_EXTRACTION_FIELDS = "subject,payment"
    test_settings.LLM_FACT_EXTRACTION_AUTO_ARABIC = True

    captured = {}

    async def fake_extract_llm_facts_for_weak_fields(**kwargs):
        captured["fields"] = kwargs["fields"]
        return SimpleNamespace(
            regex_facts=kwargs["draft_facts"],
            llm_facts={},
            derived_facts={},
            rejected_llm_facts={},
            final_facts=kwargs["draft_facts"],
            weak_fields=[],
        )

    monkeypatch.setattr(
        document_service_module,
        "extract_llm_facts_for_weak_fields",
        fake_extract_llm_facts_for_weak_fields,
    )

    document_service_module.DocumentService._maybe_enrich_facts_with_llm(
        chunks=["mostly readable mixed OCR text"],
        metas=[
            {
                "source": "doc.pdf",
                "page": "1",
                "section": "general",
                "text_quality_mode": "partial_pages",
                "text_quality_arabic_ratio": 0.17,
                "text_quality_readable_ratio": 0.89,
                "text_quality_page_gap_count": 2,
            }
        ],
        facts={},
    )

    assert "caution" in captured["fields"]
    assert "definitive_caution" in captured["fields"]


@pytest.mark.asyncio
async def test_save_upload_to_disk_allocates_unique_name_after_repeated_collision(
    settings_override,
    test_settings,
):
    settings_override(document_service_module)
    upload_dir = test_settings.UPLOAD_DIR
    first_name = "same.pdf"
    hash_prefix = "abcdef12"

    from pathlib import Path

    Path(upload_dir, first_name).write_bytes(b"existing")
    Path(upload_dir, f"{hash_prefix}_{first_name}").write_bytes(b"existing")

    _, safe_name = await document_service_module.DocumentService.save_upload_to_disk(
        b"new",
        first_name,
        f"{hash_prefix}999999",
    )

    assert safe_name == f"{hash_prefix}_2_{first_name}"
    assert Path(upload_dir, safe_name).read_bytes() == b"new"


def test_index_document_file_returns_zero_when_no_chunks_extracted(monkeypatch, test_settings, settings_override):
    settings_override(document_service_module)
    monkeypatch.setattr(document_service_module, "extract_and_chunk", lambda file_path, filename: ([], [], []))

    result = document_service_module.DocumentService.index_document_file(
        doc_id="doc-123",
        file_path="C:/tmp/test.pdf",
        department="commerciale",
        uploaded_by="user-1",
    )

    assert result == {"chunk_count": 0, "facts": {}}


def test_index_document_file_replaces_existing_and_adds_vectors(
    monkeypatch,
    test_settings,
    settings_override,
    stub_embedder,
    stub_extract_and_chunk,
    fake_vector_store_factory,
):
    settings_override(document_service_module)
    monkeypatch.setattr(document_service_module, "extract_and_chunk", stub_extract_and_chunk())
    monkeypatch.setattr(document_service_module, "get_embedder", lambda: stub_embedder)

    state, fake_vector_store, _ = fake_vector_store_factory()
    monkeypatch.setattr(document_service_module, "VectorStore", fake_vector_store)

    result = document_service_module.DocumentService.index_document_file(
        doc_id="doc-123",
        file_path="C:/tmp/test.pdf",
        department="commerciale",
        uploaded_by="user-1",
        visibility="shared",
        universe_id="universe-1",
        replace_existing=True,
    )

    assert result["chunk_count"] == 2
    assert result["facts"] == {}
    assert state["delete_by_doc_id"] == ["doc-123"]
    assert state["delete_by_source"] == ["test.pdf"]
    assert len(state["adds"]) == 1
    added = state["adds"][0]
    assert added["metadatas"][0]["doc_id"] == "doc-123"
    assert added["metadatas"][0]["department"] == "commerciale"
    assert added["metadatas"][0]["uploaded_by"] == "user-1"
    assert added["metadatas"][0]["visibility"] == "shared"
    assert added["metadatas"][0]["universe_id"] == "universe-1"


def test_index_document_file_continues_when_one_cleanup_path_fails(
    monkeypatch,
    test_settings,
    settings_override,
    stub_embedder,
    stub_extract_and_chunk,
    fake_vector_store_factory,
):
    settings_override(document_service_module)
    monkeypatch.setattr(document_service_module, "extract_and_chunk", stub_extract_and_chunk(chunks=["chunk only"], metas=[{"source": "test.pdf", "page": "1", "section": "general"}], ids=["test.pdf_c0"]))
    monkeypatch.setattr(document_service_module, "get_embedder", lambda: stub_embedder)

    state, fake_vector_store, _ = fake_vector_store_factory(delete_doc_exc=RuntimeError("doc cleanup failed"))
    monkeypatch.setattr(document_service_module, "VectorStore", fake_vector_store)

    result = document_service_module.DocumentService.index_document_file(
        doc_id="doc-123",
        file_path="C:/tmp/test.pdf",
        department="commerciale",
        uploaded_by="user-1",
        replace_existing=True,
    )

    assert result["chunk_count"] == 1
    assert result["facts"] == {}
    assert state["delete_by_doc_id"] == ["doc-123"]
    assert state["delete_by_source"] == ["test.pdf"]
    assert len(state["adds"]) == 1


def test_index_document_file_aborts_when_both_cleanup_paths_fail(
    monkeypatch,
    test_settings,
    settings_override,
    stub_embedder,
    stub_extract_and_chunk,
    fake_vector_store_factory,
):
    settings_override(document_service_module)
    monkeypatch.setattr(document_service_module, "extract_and_chunk", stub_extract_and_chunk(chunks=["chunk only"], metas=[{"source": "test.pdf", "page": "1", "section": "general"}], ids=["test.pdf_c0"]))
    monkeypatch.setattr(document_service_module, "get_embedder", lambda: stub_embedder)

    state, fake_vector_store, _ = fake_vector_store_factory(
        delete_doc_exc=RuntimeError("doc cleanup failed"),
        delete_source_exc=RuntimeError("source cleanup failed"),
    )
    monkeypatch.setattr(document_service_module, "VectorStore", fake_vector_store)

    with pytest.raises(RuntimeError, match="Both Qdrant cleanup paths failed"):
        document_service_module.DocumentService.index_document_file(
            doc_id="doc-123",
            file_path="C:/tmp/test.pdf",
            department="commerciale",
            uploaded_by="user-1",
            replace_existing=True,
        )

    assert state["delete_by_doc_id"] == ["doc-123"]
    assert state["delete_by_source"] == ["test.pdf"]
    assert state["adds"] == []
