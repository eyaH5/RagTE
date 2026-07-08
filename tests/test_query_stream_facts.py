from __future__ import annotations

import pytest

from api.routers.query import _single_retrieved_source
from api.services import rag as rag_module
from api.services.rag import answer_chatbot_identity, answer_from_document_facts, answer_from_text_cache, answer_organization_from_text_cache, build_tender_checklist_answer


def test_single_retrieved_source_returns_only_source():
    assert _single_retrieved_source(
        [
            {"source": "UBCI.pdf", "page": "3"},
            {"source": "UBCI.pdf", "page": "4"},
        ]
    ) == "UBCI.pdf"


def test_single_retrieved_source_rejects_mixed_sources():
    assert _single_retrieved_source(
        [
            {"source": "UBCI.pdf", "page": "3"},
            {"source": "TOPNET.pdf", "page": "4"},
        ]
    ) is None


def test_single_retrieved_source_ignores_missing_source():
    assert _single_retrieved_source([{"page": "3"}]) is None


def test_build_tender_checklist_answer_prefers_tender_profile():
    answer = build_tender_checklist_answer(
        "Demo.pdf",
        {
            "subject": {
                "text": "Ancien objet bruite",
                "page": "9",
                "section": "subject",
            },
            "tender_profile": {
                "schema": "tender_profile.v1",
                "fields": {
                    "object": {
                        "text": "Objet propre depuis le profil structure",
                        "page": "2",
                        "section": "subject",
                    },
                    "technical_documents": {
                        "text": "Documents techniques",
                        "page": "5",
                        "section": "technical_documents",
                        "items": [
                            {"text": "Documentation technique", "page": "5"},
                            {"text": "Fiche produit", "page": "5"},
                        ],
                    },
                },
            },
        },
    )

    assert "Objet propre depuis le profil structure" in answer
    assert "Source: Demo.pdf, page 2." in answer
    assert "Ancien objet bruite" not in answer
    assert "- Documentation technique" in answer
    assert "- Fiche produit" in answer


def test_build_tender_checklist_answer_falls_back_without_profile_field():
    answer = build_tender_checklist_answer(
        "Demo.pdf",
        {
            "deadline": {
                "text": "18 Juillet 2025",
                "page": "3",
                "section": "deadline",
            },
            "tender_profile": {
                "schema": "tender_profile.v1",
                "fields": {},
            },
        },
    )

    assert "18 Juillet 2025" in answer
    assert "Source: Demo.pdf, page 3." in answer


def test_build_tender_checklist_answer_shows_extraction_warning():
    answer = build_tender_checklist_answer(
        "UNKNOWN.pdf",
        {
            "extraction_warning": {
                "text": "Qualité d'extraction insuffisante : le document semble être un scan tourné.",
                "level": "warning",
                "section": "extraction",
            }
        },
    )

    assert "Avertissement extraction :" in answer
    assert "scan tourné" in answer
    assert "1. Quel est l'objet" in answer
    assert "Reponse: Non mentionne dans ce document." in answer


def test_build_tender_checklist_answer_cleans_noisy_table_rows():
    answer = build_tender_checklist_answer(
        "TSB.pdf",
        {
            "administrative_documents": {
                "text": (
                    "- La caution bancaire provisoire d'un montant egal 4 Douze mille (12 "
                    "- N° de |la pièce, 1 = Désignations. N° de |la pièce, 2 = Authentifications. "
                    "- RNE recent valable a la date d'ouverture des offres.. 6, 2 = par. |7 8 |, 1 = "
                    "- Une attestation d'affiliation a la CNSS du soumissionnaire ainsi que "
                    "- Cachet signature du soumissionnaire REG P "
                ),
                "page": "7",
                "section": "administrative_documents",
            },
            "technical_documents": {
                "items": [
                    {"text": "Lacertification de leurs equipes sur HPE Synergy et pour Vmware", "page": "3"},
                    {"text": "La certification de leurs equipes sur HPE Synergy et pour Vmware", "page": "3"},
                    {"text": "Documentation technique", "page": "3"},
                    {"text": "2, Authentifications = Date signature et cachet du soumissionnaire", "page": "3"},
                ],
                "page": "3",
                "section": "technical_documents",
            },
        },
    )

    assert "N° de" not in answer
    assert "Authentifications" not in answer
    assert "Cachet signature" not in answer
    assert answer.count("certification de leurs equipes sur HPE Synergy et pour Vmware") == 1
    assert "- Documentation technique" in answer
    assert "Source: TSB.pdf, page 7." in answer


def test_build_tender_checklist_answer_trims_incomplete_scalar_tail():
    answer = build_tender_checklist_answer(
        "TSB.pdf",
        {
            "validity": {
                "text": "Offre reste valable pendant 90 jours à partir de",
                "page": "20",
                "section": "validity",
            },
            "submission_method": {
                "text": "Les offres doivent parvenir par voie postale à l'adresse suivante :",
                "page": "8",
                "section": "submission_method",
            },
        },
    )

    assert "Offre reste valable pendant 90 jours" in answer
    assert "Offre reste valable pendant 90 jours à partir" not in answer
    assert "à partir de\n" not in answer
    assert "à l'adresse suivante :" not in answer


def test_build_tender_checklist_answer_normalizes_common_ocr_glue():
    answer = build_tender_checklist_answer(
        "TSB.pdf",
        {
            "technical_documents": {
                "items": [
                    {"text": "Lacertification de leurs équipes sur HPE Synergy", "page": "3"},
                    {"text": "La liste de l'équipe intervenante selonlemodèlejointenannexe(6)", "page": "3"},
                ],
                "page": "3",
                "section": "technical_documents",
            },
            "financial_documents": {
                "items": [
                    {"text": "Le récapitulatif des prix conformément au modèle joint enannexe(11", "page": "8"},
                ],
                "page": "8",
                "section": "financial_documents",
            },
        },
    )

    assert "La certification de leurs équipes" in answer
    assert "selon le modèle joint en annexe(6)" in answer
    assert "modèle joint en annexe(11)" in answer


def test_build_tender_checklist_answer_summarizes_long_reception_clause():
    answer = build_tender_checklist_answer(
        "TunisieTelecom.pdf",
        {
            "reception": {
                "text": (
                    "ARTICLE 9 : RECEPTION PROVISOIRE-RECEPTION DEFINITIVE "
                    "Les receptions des articles seront effectuees de la maniere suivante : "
                    "9.1 Reception quantitative Une reception quantitative sera prononcee pour les articles "
                    "commandes apres la livraison effective des articles dans les locaux. "
                    "9.2 Reception provisoire Une reception provisoire sera prononcee apres la verification "
                    "de la conformite des articles aux specifications techniques et les tests. "
                    "9.3 Reception definitive Une reception definitive sera prononcee a l'expiration du "
                    "delai de garantie."
                ),
                "page": "11",
                "section": "reception",
            },
        },
    )

    assert "Modalites de reception: reception quantitative, reception provisoire puis reception definitive" in answer
    assert "apres livraison" in answer
    assert "apres verification de conformite" in answer
    assert "ARTICLE 9" not in answer
    assert "9.1" not in answer


def test_build_tender_checklist_answer_summarizes_multi_lot_guarantee_clause():
    answer = build_tender_checklist_answer(
        "TunisieTelecom.pdf",
        {
            "guarantee": {
                "text": (
                    "Tunisie Telecom informera le fournisseur pour le remplacement dans un delai de 48 heures "
                    "des articles defectueux pendant une periode de 6 mois a partir de la date de la reception "
                    "provisoire pour les Lots 1 et 2. Pour le Lot 3 : La duree de la garantie, a partir de la "
                    "date d'emission du certificat de reception provisoire, est fixee a : "
                    "a. Deux (02) ans pour les casques et les souris sans fil. "
                    "b. Trois (03) annees pour les douchettes."
                ),
                "page": "12",
                "section": "guarantee",
            },
        },
    )

    assert (
        "Garantie: 6 mois pour les lots 1 et 2; "
        "2 ans pour les casques et souris sans fil; "
        "3 ans pour les douchettes."
    ) in answer
    assert "Tunisie Telecom informera" not in answer


class _FactDoc:
    def __init__(self, facts):
        self.extracted_facts = facts


class _FactResult:
    def __init__(self, doc):
        self.doc = doc

    def scalar_one_or_none(self):
        return self.doc


class _FactDb:
    def __init__(self, doc):
        self.doc = doc

    async def execute(self, stmt):
        return _FactResult(self.doc)


@pytest.mark.asyncio
async def test_answer_from_document_facts_accepts_natural_payment_question_from_profile():
    answer, metas = await answer_from_document_facts(
        db=_FactDb(
            _FactDoc(
                {
                    "tender_profile": {
                        "fields": {
                            "payment": {
                                "text": "Paiement par virement apres reception de la facture.",
                                "page": "12",
                                "section": "payment",
                            }
                        }
                    }
                }
            )
        ),
        question="How do they pay?",
        source_filter=["Demo.pdf"],
        department_filter=["admin"],
        universe_id=None,
        user_id="user-1",
        is_admin=True,
        strict_missing=True,
    )

    assert "Paiement : Paiement par virement" in answer
    assert "Source: Demo.pdf, page 12." in answer
    assert metas == [{"source": "Demo.pdf", "page": "12", "section": "payment", "score": 1.0}]


@pytest.mark.asyncio
async def test_answer_from_document_facts_strict_single_document_returns_fast_missing():
    answer, metas = await answer_from_document_facts(
        db=_FactDb(_FactDoc({"subject": {"text": "Acquisition de materiel", "page": "1"}})),
        question="Who is the project manager?",
        source_filter=["Demo.pdf"],
        department_filter=["admin"],
        universe_id=None,
        user_id="user-1",
        is_admin=True,
        strict_missing=True,
    )

    assert answer == "Not mentioned in this document."
    assert metas == []



def test_answer_from_text_cache_handles_new_document_question(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_module, "TEXT_CACHE_DIR", tmp_path)
    (tmp_path / "Demo.pdf.txt").write_text(
        "[Page 3]\n"
        "ARTICLE 2. CONDITIONS DE PARTICIPATION : Les soumissionnaires doivent avoir "
        "la certification de leurs equipes et une reference au minimum sur des projets similaires.\n"
        "ARTICLE 5. RESERVES T.S.B se reserve le droit de rejeter toute offre non conforme.",
        encoding="utf-8",
    )

    answer, metas = answer_from_text_cache("Demo.pdf", "What are the participation conditions?")

    assert "CONDITIONS DE PARTICIPATION" in answer
    assert "certification" in answer
    assert "Source: Demo.pdf, page 3." in answer
    assert metas == [{"source": "Demo.pdf", "page": "3", "section": "text_cache", "score": 0.7}]


def test_answer_from_text_cache_ignores_unrelated_question(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_module, "TEXT_CACHE_DIR", tmp_path)
    (tmp_path / "Demo.pdf.txt").write_text(
        "[Page 1]\nARTICLE 1. OBJET DU MARCHE Acquisition de serveurs.",
        encoding="utf-8",
    )

    assert answer_from_text_cache("Demo.pdf", "What color is the sky?") is None



def test_answer_organization_from_text_cache_handles_company_identity_question(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_module, "TEXT_CACHE_DIR", tmp_path)
    (tmp_path / "TUNISIAN SAUDI BANK.pdf.txt").write_text(
        "[Page 1]\n"
        "TUNISIAN SAUDI BANK\n"
        "APPEL D'OFFRES N 01/2025\n"
        "CAHIER DES CHARGES\n",
        encoding="utf-8",
    )

    answer, metas = answer_organization_from_text_cache(
        "TUNISIAN SAUDI BANK.pdf",
        "Quelle est ce soci\u00e9t\u00e9 ?",
    )

    assert "TUNISIAN SAUDI BANK" in answer
    assert "Source: TUNISIAN SAUDI BANK.pdf, page 1." in answer
    assert metas == [{"source": "TUNISIAN SAUDI BANK.pdf", "page": "1", "section": "organization", "score": 1.0}]


def test_answer_organization_from_text_cache_tolerates_replacement_chars(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_module, "TEXT_CACHE_DIR", tmp_path)
    (tmp_path / "TUNISIAN SAUDI BANK.pdf.txt").write_text(
        "[Page 1]\nTUNISIAN SAUDI BANK\nAPPEL D'OFFRES\n",
        encoding="utf-8",
    )

    answer, _metas = answer_organization_from_text_cache(
        "TUNISIAN SAUDI BANK.pdf",
        "Quelle est ce soci?t? ?",
    )

    assert "TUNISIAN SAUDI BANK" in answer



def test_answer_chatbot_identity_in_french():
    answer, metas = answer_chatbot_identity("Qui es tu ?")

    assert "TE RAG Assistant" in answer
    assert "Tunisie Electronique" in answer
    assert metas == []


def test_answer_chatbot_identity_in_english():
    answer, metas = answer_chatbot_identity("Who are you?")

    assert answer.startswith("I am TE RAG Assistant")
    assert "selected documents" in answer
    assert metas == []



def test_answer_chatbot_identity_handles_abbreviation():
    answer, metas = answer_chatbot_identity("WHO ARE U")

    assert answer.startswith("I am TE RAG Assistant")
    assert metas == []
