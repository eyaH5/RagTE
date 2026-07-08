from __future__ import annotations

import importlib
import sys

import pytest

from api.database import Document


def _load_module():
    module = importlib.import_module("reindex_documents")
    return importlib.reload(module)


@pytest.mark.asyncio
async def test_run_prints_no_matches_when_filters_empty(
    monkeypatch,
    capsys,
    initialized_db,
    test_settings,
    settings_override,
):
    module = _load_module()
    settings_override(module)
    monkeypatch.setattr(module, "async_session", initialized_db.session_factory)
    monkeypatch.setattr(module, "init_db", lambda: initialized_db.engine.dispose())
    monkeypatch.setattr(sys, "argv", ["reindex_documents.py", "--status", "processing"])

    await module.run()

    captured = capsys.readouterr()
    assert "No documents matched the requested filters." in captured.out


@pytest.mark.asyncio
async def test_dry_run_preserves_status(
    monkeypatch,
    capsys,
    initialized_db,
    test_settings,
    settings_override,
    seed_department,
    seed_user,
    seed_document,
    temp_upload_dir,
):
    await seed_department()
    await seed_user()
    await seed_document(doc_id="doc-1", filename="dry-run.pdf", status="processing")
    (temp_upload_dir / "dry-run.pdf").write_text("pdf", encoding="utf-8")

    module = _load_module()
    settings_override(module)
    monkeypatch.setattr(module, "async_session", initialized_db.session_factory)

    async def fake_init_db():
        return None

    monkeypatch.setattr(module, "init_db", fake_init_db)
    monkeypatch.setattr(module.DocumentService, "index_document_file", lambda **kwargs: 99)
    monkeypatch.setattr(sys, "argv", ["reindex_documents.py", "--status", "processing", "--dry-run"])

    await module.run()

    captured = capsys.readouterr()
    assert "DRY RUN doc_id=doc-1 file=dry-run.pdf status=processing" in captured.out

    async with initialized_db.session_factory() as session:
        stored = await session.get(Document, "doc-1")
        assert stored is not None
        assert stored.status == "processing"
        assert stored.chunk_count == 0


@pytest.mark.asyncio
async def test_successful_reindex_sets_indexed_and_chunk_count(
    monkeypatch,
    capsys,
    initialized_db,
    test_settings,
    settings_override,
    seed_department,
    seed_user,
    seed_document,
    temp_upload_dir,
):
    await seed_department()
    await seed_user()
    await seed_document(doc_id="doc-2", filename="success.pdf", status="processing")
    (temp_upload_dir / "success.pdf").write_text("pdf", encoding="utf-8")

    module = _load_module()
    settings_override(module)
    monkeypatch.setattr(module, "async_session", initialized_db.session_factory)

    async def fake_init_db():
        return None

    monkeypatch.setattr(module, "init_db", fake_init_db)
    monkeypatch.setattr(module.DocumentService, "index_document_file", lambda **kwargs: 7)
    monkeypatch.setattr(sys, "argv", ["reindex_documents.py", "--doc-id", "doc-2"])

    await module.run()

    captured = capsys.readouterr()
    assert "Indexed 7 chunk(s)." in captured.out

    async with initialized_db.session_factory() as session:
        stored = await session.get(Document, "doc-2")
        assert stored is not None
        assert stored.status == "indexed"
        assert stored.chunk_count == 7


@pytest.mark.asyncio
async def test_missing_file_marks_failed(
    monkeypatch,
    capsys,
    initialized_db,
    test_settings,
    settings_override,
    seed_department,
    seed_user,
    seed_document,
):
    await seed_department()
    await seed_user()
    await seed_document(doc_id="doc-3", filename="missing.pdf", status="processing")

    module = _load_module()
    settings_override(module)
    monkeypatch.setattr(module, "async_session", initialized_db.session_factory)

    async def fake_init_db():
        return None

    monkeypatch.setattr(module, "init_db", fake_init_db)
    monkeypatch.setattr(sys, "argv", ["reindex_documents.py", "--doc-id", "doc-3"])

    await module.run()

    captured = capsys.readouterr()
    assert "Missing file on disk" in captured.out

    async with initialized_db.session_factory() as session:
        stored = await session.get(Document, "doc-3")
        assert stored is not None
        assert stored.status == "failed"
        assert stored.chunk_count == 0


@pytest.mark.asyncio
async def test_limit_applies_before_processing(
    monkeypatch,
    capsys,
    initialized_db,
    test_settings,
    settings_override,
    seed_department,
    seed_user,
    seed_document,
    temp_upload_dir,
):
    await seed_department()
    await seed_user()
    await seed_document(doc_id="doc-4", filename="first.pdf", file_hash="hash-4", status="processing")
    await seed_document(doc_id="doc-5", filename="second.pdf", file_hash="hash-5", status="processing")
    (temp_upload_dir / "first.pdf").write_text("pdf", encoding="utf-8")
    (temp_upload_dir / "second.pdf").write_text("pdf", encoding="utf-8")

    calls = []

    module = _load_module()
    settings_override(module)
    monkeypatch.setattr(module, "async_session", initialized_db.session_factory)

    async def fake_init_db():
        return None

    monkeypatch.setattr(module, "init_db", fake_init_db)
    monkeypatch.setattr(module.DocumentService, "index_document_file", lambda **kwargs: calls.append(kwargs["doc_id"]) or 3)
    monkeypatch.setattr(sys, "argv", ["reindex_documents.py", "--status", "processing", "--limit", "1"])

    await module.run()

    assert calls == ["doc-4"]
