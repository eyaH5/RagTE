from __future__ import annotations

from vector_store import _build_search_filter


def test_build_search_filter_for_non_admin_requires_department_or_own_private_access():
    filt = _build_search_filter(
        source_filter=["a.pdf"],
        section_filter="general",
        department_filter=["commerciale"],
        universe_id="universe-1",
        user_id="user-1",
        is_admin=False,
    )

    dumped = filt.model_dump()

    assert dumped["min_should"]["min_count"] == 1
    assert len(dumped["should"]) == 2
    assert any(cond["key"] == "universe_id" for cond in dumped["must"])
    assert any(cond["key"] == "source" for cond in dumped["must"])
    assert any(cond["key"] == "section" for cond in dumped["must"])

    department_branch = dumped["should"][0]["must"]
    private_branch = dumped["should"][1]["must"]

    assert department_branch[0]["key"] == "visibility"
    assert department_branch[0]["match"]["value"] == "department"
    assert department_branch[1]["key"] == "department"
    assert department_branch[1]["match"]["any"] == ["commerciale"]

    assert private_branch[0]["key"] == "visibility"
    assert private_branch[0]["match"]["value"] == "private"
    assert private_branch[1]["key"] == "uploaded_by"
    assert private_branch[1]["match"]["value"] == "user-1"


def test_build_search_filter_for_admin_has_no_visibility_branches():
    filt = _build_search_filter(
        department_filter=["commerciale"],
        user_id="user-1",
        is_admin=True,
    )

    assert filt is None
