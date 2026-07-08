from __future__ import annotations

import hashlib
import importlib
import sys

import pytest

from api.database import Document


def _load_module():
    module = importlib.import_module("import_legacy_pdfs")
    return importlib.reload(module)


@pytest.mark.asyncio
async def test_run_requires_explicit_uploader_without_universe(
    monkeypatch,
    initialized_db,
    test_settings,
    settings_override,
):
    module = _load_module()
    settings_override(module)
    monkeypatch.setattr(module, "async_session", initialized_db.session_factory)

    async def fake_init_db():
        return None

    monkeypatch.setattr(module, "init_db", fake_init_db)
    monkeypatch.setattr(sys, "argv", ["import_legacy_pdfs.py", "--dry-run"])

    with pytest.raises(ValueError, match="Could not infer the uploader"):
        await module.run()


@pytest.mark.asyncio
async def test_run_does_not_auto_select_single_active_non_admin(
    monkeypatch,
    initialized_db,
    test_settings,
    settings_override,
    seed_department,
    seed_user,
):
    await seed_department()
    await seed_user(user_id="solo-user", email="solo@example.com", role="analyst")

    module = _load_module()
    settings_override(module)
    monkeypatch.setattr(module, "async_session", initialized_db.session_factory)

    async def fake_init_db():
        return None

    monkeypatch.setattr(module, "init_db", fake_init_db)
    monkeypatch.setattr(sys, "argv", ["import_legacy_pdfs.py", "--dry-run"])

    with pytest.raises(ValueError, match="Could not infer the uploader"):
        await module.run()


@pytest.mark.asyncio
async def test_run_does_not_auto_select_single_admin(
    monkeypatch,
    initialized_db,
    test_settings,
    settings_override,
    seed_department,
    seed_user,
):
    await seed_department()
    await seed_user(user_id="solo-admin", email="admin@example.com", role="admin")

    module = _load_module()
    settings_override(module)
    monkeypatch.setattr(module, "async_session", initialized_db.session_factory)

    async def fake_init_db():
        return None

    monkeypatch.setattr(module, "init_db", fake_init_db)
    monkeypatch.setattr(sys, "argv", ["import_legacy_pdfs.py", "--dry-run"])

    with pytest.raises(ValueError, match="Could not infer the uploader"):
        await module.run()


@pytest.mark.asyncio
async def test_resolve_context_uses_universe_creator_when_universe_provided(
    monkeypatch,
    initialized_db,
    test_settings,
    settings_override,
    seed_department,
    seed_user,
    seed_universe,
):
    await seed_department()
    creator = await seed_user(user_id="creator-1", email="creator@example.com")
    await seed_universe(universe_id="u-1", created_by=creator.id)

    module = _load_module()
    settings_override(module)
    monkeypatch.setattr(module, "async_session", initialized_db.session_factory)

    args = module.build_parser().parse_args(["--universe-id", "u-1"])
    context = await module.resolve_context(args)

    assert context.uploaded_by.id == "creator-1"
    assert context.department_id == "commerciale"
    assert context.universe is not None
    assert context.universe.id == "u-1"


@pytest.mark.asyncio
async def test_resolve_context_rejects_department_conflict(
    monkeypatch,
    initialized_db,
    test_settings,
    settings_override,
    seed_department,
    seed_user,
    seed_universe,
):
    await seed_department(department_id="commerciale")
    await seed_department(department_id="technique", name="Direction Technique")
    creator = await seed_user(user_id="creator-2", email="creator2@example.com", department_id="commerciale")
    await seed_universe(universe_id="u-2", department_id="commerciale", created_by=creator.id)

    module = _load_module()
    settings_override(module)
    monkeypatch.setattr(module, "async_session", initialized_db.session_factory)

    args = module.build_parser().parse_args(["--universe-id", "u-2", "--department-id", "technique"])

    with pytest.raises(ValueError, match="conflicts with universe department"):
        await module.resolve_context(args)


@pytest.mark.asyncio
async def test_non_admin_uploader_must_match_department(
    monkeypatch,
    initialized_db,
    test_settings,
    settings_override,
    seed_department,
    seed_user,
):
    await seed_department(department_id="commerciale")
    await seed_department(department_id="technique", name="Direction Technique")
    await seed_user(user_id="user-2", email="analyst2@example.com", department_id="commerciale")

    module = _load_module()
    settings_override(module)
    monkeypatch.setattr(module, "async_session", initialized_db.session_factory)

    args = module.build_parser().parse_args([
        "--department-id",
        "technique",
        "--uploaded-by-id",
        "user-2",
    ])

    with pytest.raises(ValueError, match="belongs to department 'commerciale'"):
        await module.resolve_context(args)


@pytest.mark.asyncio
async def test_dry_run_reports_duplicate_hash_skip(
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
    user = await seed_user(user_id="user-3", email="analyst3@example.com")
    content = b"same-pdf"
    file_hash = hashlib.sha256(content).hexdigest()
    await seed_document(doc_id="doc-dup", filename="tracked.pdf", file_hash=file_hash, uploaded_by=user.id, status="indexed")
    (temp_upload_dir / "candidate.pdf").write_bytes(content)

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
            "import_legacy_pdfs.py",
            "--department-id",
            "commerciale",
            "--uploaded-by-id",
            user.id,
            "--dry-run",
        ],
    )

    await module.run()

    captured = capsys.readouterr()
    assert "Skipping candidate.pdf because the same file hash already exists as tracked.pdf" in captured.out


@pytest.mark.asyncio
async def test_reindex_path_sets_document_indexed_and_chunk_count(
    monkeypatch,
    capsys,
    initialized_db,
    test_settings,
    settings_override,
    seed_department,
    seed_user,
    temp_upload_dir,
):
    await seed_department()
    user = await seed_user(user_id="user-4", email="analyst4@example.com")
    (temp_upload_dir / "new.pdf").write_bytes(b"new-pdf")

    module = _load_module()
    settings_override(module)
    monkeypatch.setattr(module, "async_session", initialized_db.session_factory)

    async def fake_init_db():
        return None

    monkeypatch.setattr(module, "init_db", fake_init_db)
    monkeypatch.setattr(module.DocumentService, "index_document_file", lambda **kwargs: 4)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import_legacy_pdfs.py",
            "--department-id",
            "commerciale",
            "--uploaded-by-id",
            user.id,
            "--reindex",
        ],
    )

    await module.run()

    captured = capsys.readouterr()
    assert "Imported and indexed new.pdf (4 chunks)." in captured.out

    async with initialized_db.session_factory() as session:
        docs = list((await session.execute(Document.__table__.select())).all())
        assert len(docs) == 1
        stored = await session.get(Document, docs[0].id)
        assert stored is not None
        assert stored.filename == "new.pdf"
        assert stored.status == "indexed"
        assert stored.chunk_count == 4
