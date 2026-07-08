from __future__ import annotations

import pytest

from api.services import rag as rag_module


def test_normalize_query_text_removes_accents():
    assert rag_module._normalize_query_text("validité de l'offre") == "validite de l'offre"


def test_enhance_query_matches_unaccented_validite_query():
    enhanced = rag_module.enhance_query("Quelle est la validite de l'offre ?")
    assert "offre reste valable" in enhanced
    assert "الالتزام بالعروض" in enhanced


def test_matching_focus_rules_detects_caution_query():
    rules = rag_module._matching_focus_rules("Quelle est la caution provisoire ?")
    assert len(rules) == 1
    assert "caution" in rules[0]["keywords"]


def test_focus_bonus_rewards_validity_commitment_clause():
    rules = rag_module._matching_focus_rules("Quelle est la validite de l'offre ?")
    chunk = "الالتزام ببنود وشروط هذا التعهد لمدة 120 يوما"
    assert rag_module._focus_bonus(rules, chunk) > 0.4


def test_focus_bonus_penalizes_toner_table_validity_clause():
    rules = rag_module._matching_focus_rules("Quelle est la validite de l'offre ?")
    chunk = "Le remplissage de la colonne date minimale de validité est obligatoire. Les emballages des toners doivent..."
    assert rag_module._focus_bonus(rules, chunk) < 0


def test_focus_bonus_rewards_provisional_guarantee_clause():
    rules = rag_module._matching_focus_rules("Quelle est la caution provisoire ?")
    chunk = "يرجع الضمان الوقتي للمشاركين الذين لم يتم اختيار عروضهم وكفلائهم بالتضامن"
    assert rag_module._focus_bonus(rules, chunk) > 0.4


def test_focus_bonus_rewards_opening_session_clause():
    rules = rag_module._matching_focus_rules("Quelle est la date ouverture des offres ?")
    chunk = "تُعقد جلسة فتح العروض في نفس اليوم المحدد كتاريخ أقصى لقبول العروض وتكون هذه الجلسة علنية"
    assert rag_module._focus_bonus(rules, chunk) > 0.5


def test_focus_bonus_penalizes_final_guarantee_clause_for_caution_query():
    rules = rag_module._matching_focus_rules("Quelle est la caution provisoire ?")
    chunk = "هذا النص يتعلق بالضمان النهائي وضمان حسن تنفيذ الصفقة فقط"
    assert rag_module._focus_bonus(rules, chunk) < 0


def test_answer_hint_guides_opening_query_away_from_article_numbers():
    hint = rag_module._answer_hint("Quelle est la date ouverture des offres ?")
    assert "ignore les numéros d'article" in hint.lower()
    assert "même jour que la date limite" in hint.lower()


def test_focused_chunk_excerpt_prefers_explicit_caution_amount_phrase():
    rules = rag_module._matching_focus_rules("Quelle est la caution provisoire ?")
    chunk = (
        "المادة 113 من الأمر ... مبلغ الضمان القار وقدره خمسة آلاف دينار 5000 دينار قصد المشاركة "
        "ثم تفاصيل أخرى غير مهمة عن الوثائق."
    )

    excerpt = rag_module._focused_chunk_excerpt(rules, chunk, window=40)

    assert "خمسة آلاف دينار" in excerpt
    assert "113" not in excerpt


def test_focused_chunk_excerpt_prefers_relative_opening_timing_clause():
    rules = rag_module._matching_focus_rules("Quelle est la date ouverture des offres ?")
    chunk = (
        "ويتم خلال هذه الجلسة فتح العروض الفنية والمالية. تُعقد جلسة فتح العروض وجوبا في نفس اليوم المحدد "
        "كتاريخ أقصى لقبول العروض مع تفاصيل إجرائية إضافية."
    )

    excerpt = rag_module._focused_chunk_excerpt(rules, chunk, window=35)

    assert "في نفس اليوم المحدد كتاريخ أقصى لقبول العروض" in excerpt


class _FlatReranker:
    def __init__(self, score: float = 0.0):
        self.score = score

    def predict(self, pairs):
        return [self.score for _ in pairs]


class _SequenceReranker:
    def __init__(self, scores):
        self.scores = list(scores)

    def predict(self, pairs):
        return list(self.scores)


class _StaticVectorStore:
    def __init__(self, rows):
        self.rows = rows

    async def search(self, **kwargs):
        return list(self.rows)


@pytest.mark.asyncio
async def test_retrieve_fallback_ranks_validity_clause_above_table_chunk(monkeypatch):
    rows = [
        {
            "source": "CDC 01-2026.pdf",
            "page": "9",
            "section": "deadline",
            "score": 0.61,
            "text": "Le remplissage de la colonne date minimale de validité est obligatoire. Les emballages des toners doivent...",
        },
        {
            "source": "CDC 01-2026.pdf",
            "page": "18",
            "section": "payment",
            "score": 0.55,
            "text": "الالتزام ببنود وشروط هذا التعهد لمدة 120 يوما من 90",
        },
    ]

    async def fake_get_embedding(text):
        return [0.1] * 1024

    monkeypatch.setattr(rag_module, "get_embedding", fake_get_embedding)
    monkeypatch.setattr(rag_module, "_get_vector_store", lambda: _StaticVectorStore(rows))
    monkeypatch.setattr(rag_module, "_get_reranker", lambda: _FlatReranker(0.0))

    chunks, metas = await rag_module.retrieve(
        query="Quelle est la validite de l'offre ?",
        k=2,
        source_filter=["CDC 01-2026.pdf"],
        department_filter=["commerciale"],
    )

    assert metas[0]["page"] == "18"
    assert metas[0]["retrieval_score"] > metas[1]["retrieval_score"]
    assert "التعهد" in chunks[0]


@pytest.mark.asyncio
async def test_retrieve_fallback_ranks_provisional_guarantee_chunk_first(monkeypatch):
    rows = [
        {
            "source": "CDC 01-2026.pdf",
            "page": "21",
            "section": "general",
            "score": 0.58,
            "text": "مثال التزام الأشخاص الكافلين بالتضامن المعوض للضمان الوقتي",
        },
        {
            "source": "CDC 01-2026.pdf",
            "page": "6",
            "section": "deadline",
            "score": 0.54,
            "text": "يرجع الضمان الوقتي للمشاركين الذين لم يتم اختيار عروضهم بعد اختيار أو يضع حدا لالتزام كفلائهم بالتضامن",
        },
        {
            "source": "CDC 01-2026.pdf",
            "page": "22",
            "section": "guarantee",
            "score": 0.57,
            "text": "هذا النص يتعلق بالضمان النهائي وضمان حسن تنفيذ الصفقة",
        },
    ]

    async def fake_get_embedding(text):
        return [0.1] * 1024

    monkeypatch.setattr(rag_module, "get_embedding", fake_get_embedding)
    monkeypatch.setattr(rag_module, "_get_vector_store", lambda: _StaticVectorStore(rows))
    monkeypatch.setattr(rag_module, "_get_reranker", lambda: _FlatReranker(0.0))

    chunks, metas = await rag_module.retrieve(
        query="Quelle est la caution provisoire ?",
        k=3,
        source_filter=["CDC 01-2026.pdf"],
        department_filter=["commerciale"],
    )

    assert metas[0]["page"] == "6"
    assert metas[0]["retrieval_score"] > metas[1]["retrieval_score"]
    assert "الضمان الوقتي" in chunks[0]


@pytest.mark.asyncio
async def test_retrieve_with_reranker_still_prefers_explicit_caution_amount_chunk(monkeypatch):
    rows = [
        {
            "source": "CDC 01-2026.pdf",
            "page": "6",
            "section": "deadline",
            "score": 0.58,
            "text": "يرجع الضمان الوقتي للمشاركين الذين لم يتم اختيار عروضهم Cartouche d'encre Noir Disque dur Clavier Câble USB Adaptateur HDMI Switch",
        },
        {
            "source": "CDC 01-2026.pdf",
            "page": "21",
            "section": "guarantee",
            "score": 0.56,
            "text": "حدد مبلغ الضمان الوقتي بخمسة آلاف دينار 5000 دينار قصد المشاركة والكفيل بالتضامن",
        },
    ]

    async def fake_get_embedding(text):
        return [0.1] * 1024

    monkeypatch.setattr(rag_module, "get_embedding", fake_get_embedding)
    monkeypatch.setattr(rag_module, "_get_vector_store", lambda: _StaticVectorStore(rows))
    monkeypatch.setattr(rag_module, "_get_reranker", lambda: _SequenceReranker([0.82, 0.74]))

    chunks, metas = await rag_module.retrieve(
        query="Quelle est la caution provisoire ?",
        k=2,
        source_filter=["CDC 01-2026.pdf"],
        department_filter=["commerciale"],
    )

    assert metas[0]["page"] == "21"
    assert "cinq" not in chunks[0].lower()
    assert "خمسة آلاف دينار" in chunks[0]
