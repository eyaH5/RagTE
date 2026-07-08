from __future__ import annotations

import pytest

from api.models import QueryRequest
from api.policy import AccessPolicy
from api.services import query_service as query_service_module
from api.services.query_service import QueryService


@pytest.mark.asyncio
async def test_execute_query_passes_visibility_context_to_rag(
    monkeypatch,
    initialized_db,
    seed_department,
    seed_user,
):
    await seed_department()
    user = await seed_user(user_id="user-1", email="user1@example.com")
    captured = {}

    async def fake_rag_query(**kwargs):
        captured.update(kwargs)
        return "ok", [{"source": "test.pdf", "page": "1", "score": 0.8}]

    async def fake_log_action(**kwargs):
        return None

    monkeypatch.setattr(query_service_module, "rag_query", fake_rag_query)
    monkeypatch.setattr(query_service_module.AuditService, "log_action", fake_log_action)

    async with initialized_db.session_factory() as session:
        response = await QueryService.execute_query(
            QueryRequest(question="Question de test", k=6),
            AccessPolicy(user),
            session,
        )

    assert captured["user_id"] == "user-1"
    assert captured["is_admin"] is False
    assert captured["department_filter"] == ["commerciale"]
    assert response.answer == "ok"
    assert response.sources[0].source == "test.pdf"
