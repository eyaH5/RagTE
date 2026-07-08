from __future__ import annotations

import importlib
import sys

import pytest

from api.database import Document


def _load_module():
    module = importlib.import_module("backfill_document_facts")
    return importlib.reload(module)


def _write_cache(cache_dir, filename: str, body: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{filename}.txt").write_text(body, encoding="utf-8")


def test_force_docling_is_passed_to_pdf_fallback(monkeypatch, tmp_path):
    module = _load_module()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "forced.pdf").write_bytes(b"%PDF-1.4")

    seen = {}

    def fake_extract_and_chunk(path, filename, **kwargs):
        seen["path"] = path
        seen["filename"] = filename
        seen["force_docling"] = kwargs.get("force_docling")
        return (
            ["Objet : Consultation pour les serveurs."],
            [{"source": filename, "page": "1", "section": "general", "chunk_index": 0}],
            ["forced.pdf_c0"],
        )

    monkeypatch.setattr(module, "extract_and_chunk", fake_extract_and_chunk)

    facts, source = module.extract_facts_for_document(
        "forced.pdf",
        text_cache_dir=tmp_path / "text_cache",
        pdf_dir=pdf_dir,
        use_text_cache=False,
        pdf_fallback=True,
        force_docling=True,
    )

    assert seen["force_docling"] is True
    assert seen["filename"] == "forced.pdf"
    assert source.startswith("docling-pdf:")
    assert "subject" in facts


def test_pdf_fallback_reextracts_facts_from_generated_text_cache(monkeypatch, tmp_path):
    module = _load_module()
    pdf_dir = tmp_path / "pdfs"
    cache_dir = tmp_path / "text_cache"
    pdf_dir.mkdir()
    (pdf_dir / "generated.pdf").write_bytes(b"%PDF-1.4")

    def fake_extract_and_chunk(path, filename, **kwargs):
        _write_cache(
            cache_dir,
            filename,
            """
            [Page 1]
            Objet : weak chunk should not be used.
            [Page 2]
            ARTICLE 17. DATE LIMITE DE RECEPTION DES OFFRES
            La date limite de reception des offres est fixee au 24 fevrier 2025.
            """,
        )
        return (
            ["Objet : weak chunk should not be used."],
            [{"source": filename, "page": "1", "section": "general", "chunk_index": 0}],
            ["generated.pdf_c0"],
        )

    monkeypatch.setattr(module, "extract_and_chunk", fake_extract_and_chunk)

    facts, source = module.extract_facts_for_document(
        "generated.pdf",
        text_cache_dir=cache_dir,
        pdf_dir=pdf_dir,
        use_text_cache=False,
        pdf_fallback=True,
    )

    assert source.startswith("pdf+text-cache:")
    assert facts["deadline"]["text"] == "24 fevrier 2025"


@pytest.mark.asyncio
async def test_dry_run_extracts_from_text_cache_without_updating_db(
    monkeypatch,
    capsys,
    initialized_db,
    settings_override,
    seed_department,
    seed_user,
    seed_document,
    tmp_path,
):
    await seed_department()
    await seed_user()
    await seed_document(doc_id="doc-1", filename="cached.pdf", status="indexed")

    cache_dir = tmp_path / "text_cache"
    _write_cache(
        cache_dir,
        "cached.pdf",
        """
        [Page 1]
        Objet : Consultation pour le renouvellement des licences.
        Date limite de remise des offres : 30/04/2025 avant 23h45.
        """,
    )

    module = _load_module()
    settings_override(module)
    monkeypatch.setattr(module, "async_session", initialized_db.session_factory)

    async def fake_init_db():
        return None

    monkeypatch.setattr(module, "init_db", fake_init_db)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill_document_facts.py",
            "--doc-id",
            "doc-1",
            "--text-cache-dir",
            str(cache_dir),
            "--no-pdf-fallback",
            "--dry-run",
        ],
    )

    await module.run()

    captured = capsys.readouterr()
    assert "DRY RUN doc_id=doc-1 file=cached.pdf" in captured.out
    assert "source=text-cache:" in captured.out

    async with initialized_db.session_factory() as session:
        stored = await session.get(Document, "doc-1")
        assert stored is not None
        assert stored.extracted_facts is None
        assert stored.status == "indexed"


@pytest.mark.asyncio
async def test_backfill_updates_extracted_facts_from_text_cache_only(
    monkeypatch,
    capsys,
    initialized_db,
    settings_override,
    seed_department,
    seed_user,
    seed_document,
    tmp_path,
):
    await seed_department()
    await seed_user()
    await seed_document(doc_id="doc-2", filename="ubci.pdf", status="indexed", chunk_count=12)

    cache_dir = tmp_path / "text_cache"
    _write_cache(
        cache_dir,
        "ubci.pdf",
        """
        [Page 1]
        ARTICLE 1 : OBJET DE LA CONSULTATION : L'UBCI se propose d'acquerir 160 UC, 20 Laptop et 320 ecrans.
        [Page 2]
        DATE LIMITE DE RECEPTION DES OFFRES : La date limite de reception des offres est arretee au 18 Juillet 2025.
        """,
    )

    module = _load_module()
    settings_override(module)
    monkeypatch.setattr(module, "async_session", initialized_db.session_factory)
    monkeypatch.setattr(module, "extract_and_chunk", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pdf fallback used")))

    async def fake_init_db():
        return None

    monkeypatch.setattr(module, "init_db", fake_init_db)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill_document_facts.py",
            "--doc-id",
            "doc-2",
            "--text-cache-dir",
            str(cache_dir),
            "--no-pdf-fallback",
        ],
    )

    await module.run()

    captured = capsys.readouterr()
    assert "Updated ubci.pdf" in captured.out
    assert "via text-cache:" in captured.out

    async with initialized_db.session_factory() as session:
        stored = await session.get(Document, "doc-2")
        assert stored is not None
        assert stored.status == "indexed"
        assert stored.chunk_count == 12
        assert stored.extracted_facts["subject"]["text"].startswith("L'UBCI se propose")
        assert "18 Juillet 2025" in stored.extracted_facts["deadline"]["text"]


@pytest.mark.asyncio
async def test_missing_only_skips_documents_that_already_have_facts(
    monkeypatch,
    capsys,
    initialized_db,
    settings_override,
    seed_department,
    seed_user,
    seed_document,
    tmp_path,
):
    await seed_department()
    await seed_user()
    await seed_document(
        doc_id="doc-existing",
        filename="existing.pdf",
        status="indexed",
        extracted_facts={"subject": {"text": "already done", "page": "1", "section": "general"}},
    )
    await seed_document(doc_id="doc-missing", filename="missing.pdf", file_hash="hash-2", status="indexed")

    cache_dir = tmp_path / "text_cache"
    _write_cache(cache_dir, "existing.pdf", "[Page 1]\nObjet : Should not be touched.")
    _write_cache(cache_dir, "missing.pdf", "[Page 1]\nObjet : Consultation pour les equipements reseau.")

    module = _load_module()
    settings_override(module)
    monkeypatch.setattr(module, "async_session", initialized_db.session_factory)

    async def fake_init_db():
        return None

    monkeypatch.setattr(module, "init_db", fake_init_db)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill_document_facts.py",
            "--status",
            "indexed",
            "--missing-only",
            "--text-cache-dir",
            str(cache_dir),
            "--no-pdf-fallback",
        ],
    )

    await module.run()

    captured = capsys.readouterr()
    assert "Matched 1 document(s)" in captured.out
    assert "Updated missing.pdf" in captured.out
    assert "existing.pdf" not in captured.out

    async with initialized_db.session_factory() as session:
        existing = await session.get(Document, "doc-existing")
        missing = await session.get(Document, "doc-missing")
        assert existing.extracted_facts["subject"]["text"] == "already done"
        assert "equipements reseau" in missing.extracted_facts["subject"]["text"]
