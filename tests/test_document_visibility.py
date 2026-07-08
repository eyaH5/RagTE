from __future__ import annotations

import pytest

from api.policy import AccessPolicy
from api.repositories import document_repo
from api.routers.documents import _visible_duplicate_for_policy


@pytest.mark.asyncio
async def test_list_for_user_returns_department_and_own_private_docs(
    initialized_db,
    seed_department,
    seed_user,
    seed_document,
):
    await seed_department()
    user = await seed_user(user_id="user-1", email="user1@example.com")
    await seed_user(user_id="user-2", email="user2@example.com")

    await seed_document(
        doc_id="dept-doc",
        filename="dept.pdf",
        file_hash="hash-dept",
        uploaded_by="user-2",
        visibility="department",
        status="indexed",
    )
    await seed_document(
        doc_id="my-private",
        filename="mine.pdf",
        file_hash="hash-mine",
        uploaded_by="user-1",
        visibility="private",
        status="indexed",
    )
    await seed_document(
        doc_id="other-private",
        filename="other.pdf",
        file_hash="hash-other",
        uploaded_by="user-2",
        visibility="private",
        status="indexed",
    )

    async with initialized_db.session_factory() as session:
        docs = await document_repo.list_for_user(session, AccessPolicy(user))

    assert {doc.id for doc in docs} == {"dept-doc", "my-private"}


@pytest.mark.asyncio
async def test_get_by_id_hides_another_users_private_doc(
    initialized_db,
    seed_department,
    seed_user,
    seed_document,
):
    await seed_department()
    user = await seed_user(user_id="user-1", email="user1@example.com")
    await seed_user(user_id="user-2", email="user2@example.com")
    await seed_document(
        doc_id="other-private",
        filename="other.pdf",
        file_hash="hash-other",
        uploaded_by="user-2",
        visibility="private",
        status="indexed",
    )

    async with initialized_db.session_factory() as session:
        doc = await document_repo.get_by_id(session, "other-private", policy=AccessPolicy(user))

    assert doc is None


@pytest.mark.asyncio
async def test_duplicate_hash_hidden_private_doc_does_not_block_upload(
    initialized_db,
    seed_department,
    seed_user,
    seed_document,
):
    await seed_department()
    user = await seed_user(user_id="user-1", email="user1@example.com")
    await seed_user(user_id="user-2", email="user2@example.com")
    await seed_document(
        doc_id="other-private",
        filename="hidden.pdf",
        file_hash="same-hash",
        uploaded_by="user-2",
        visibility="private",
        status="indexed",
    )

    async with initialized_db.session_factory() as session:
        existing_docs = await document_repo.list_by_hash(session, "same-hash")
        visible_duplicate = _visible_duplicate_for_policy(existing_docs, AccessPolicy(user))

    assert len(existing_docs) == 1
    assert visible_duplicate is None


@pytest.mark.asyncio
async def test_duplicate_hash_visible_department_doc_blocks_upload(
    initialized_db,
    seed_department,
    seed_user,
    seed_document,
):
    await seed_department()
    user = await seed_user(user_id="user-1", email="user1@example.com")
    await seed_user(user_id="user-2", email="user2@example.com")
    await seed_document(
        doc_id="dept-doc",
        filename="visible.pdf",
        file_hash="same-hash",
        uploaded_by="user-2",
        visibility="department",
        status="indexed",
    )

    async with initialized_db.session_factory() as session:
        existing_docs = await document_repo.list_by_hash(session, "same-hash")
        visible_duplicate = _visible_duplicate_for_policy(existing_docs, AccessPolicy(user))

    assert visible_duplicate is not None
    assert visible_duplicate.id == "dept-doc"


@pytest.mark.asyncio
async def test_count_for_user_excludes_hidden_private_docs_in_universe(
    initialized_db,
    seed_department,
    seed_user,
    seed_universe,
    seed_document,
):
    await seed_department()
    user = await seed_user(user_id="user-1", email="user1@example.com")
    await seed_user(user_id="user-2", email="user2@example.com")
    await seed_universe(universe_id="universe-1")

    await seed_document(
        doc_id="dept-doc",
        filename="dept.pdf",
        file_hash="hash-dept",
        uploaded_by="user-2",
        visibility="department",
        universe_id="universe-1",
        status="indexed",
    )
    await seed_document(
        doc_id="hidden-private",
        filename="private.pdf",
        file_hash="hash-private",
        uploaded_by="user-2",
        visibility="private",
        universe_id="universe-1",
        status="indexed",
    )

    async with initialized_db.session_factory() as session:
        count = await document_repo.count_for_user(session, AccessPolicy(user), universe_id="universe-1")

    assert count == 1


@pytest.mark.asyncio
async def test_admin_can_list_private_documents_from_any_department(
    initialized_db,
    seed_department,
    seed_user,
    seed_document,
):
    await seed_department()
    await seed_department(department_id="admin", name="Administration")
    admin = await seed_user(
        user_id="admin-1",
        email="admin@example.com",
        department_id="admin",
        role="admin",
    )
    await seed_user(user_id="user-2", email="user2@example.com")
    await seed_document(
        doc_id="other-private",
        filename="other.pdf",
        file_hash="hash-other",
        uploaded_by="user-2",
        visibility="private",
        status="indexed",
    )

    async with initialized_db.session_factory() as session:
        docs = await document_repo.list_for_user(session, AccessPolicy(admin))

    assert {doc.id for doc in docs} == {"other-private"}
