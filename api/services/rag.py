"""
Core RAG service — extracted from app.py for use by FastAPI backend.
Handles query enhancement, retrieval, reranking, and LLM generation.
"""
import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator

import torch
from sentence_transformers import CrossEncoder
import httpx
from openai import AsyncOpenAI
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

import asyncio
from loguru import logger

from vector_store import AsyncVectorStore
from api.config import get_settings
from api.database import Document
from api.embeddings import get_embedder, to_builtin_list

settings = get_settings()
TEXT_CACHE_DIR = Path(settings.CACHE_DIR)


@lru_cache(maxsize=1)
def get_llm_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key="none",
    )

# ── Query Expansion (from app.py) ─────────────────────────────────────────

QUERY_EXPANSIONS = {
    "caution":  "caution cautionnement provisoire définitive garantie bancaire engagement solidaire ضمان وقتي الضمان الوقتي يرجع الضمان الوقتي ضمان نهائي الكفيل بالتضامن كفلائهم بالتضامن كفالة بنكية",
    "délai":  "délai livraison installation période durée أجل تسليم مدة تنفيذ",
    "paiement":  "paiement modalité conditions échéance versement خلاص دفع أقساط فاتورة",
    "offre":  "offre soumission dépôt remise envoi plateforme تونبس تقديم العروض إيداع إرسال",
    "réception":  "réception provisoire définitive livraison validation استلام تسليم قبول",
    "garantie":  "garantie durée maintenance SAV après-vente ضمان مدة الضمان صيانة",
    "document":  "document administratif technique registre attestation وثائق إدارية فنية",
    "cnss":  "CNSS affiliation attestation solde الضمان الاجتماعي",
    "registre":  "registre commerce RNE identification سجل تجاري المؤسسات",
    "variante":  "variante alternative substitut option بديل عروض بديلة",
    "ouverture":  "ouverture plis séance publique ouverture des offres فتح العروض جلسة علنية",
    "soumission":  "soumission dépôt plateforme envoi remise تقديم العروض تونبس إيداع إرسال",
    "validité":  "valable validité offre reste valable délais de validité des offres الالتزام بالعروض الالتزام ببنود وشروط هذا التعهد هذا التعهد لمدة 90 120 يوما",
    "références":  "références projets installation justificatifs مراجع",
    "financier":  "financier prix bordereau soumission lettre عرض مالي أثمان أسعار",
    "constructeur":  "constructeur partenariat officielle lettre مصنع",
    "pénalité":  "pénalité retard sanction amende تغريم غرامة عقوبة تأخير فسخ",
    "objet":  "objet cahier charges acquisition fourniture projet موضوع اقتناء تزويد مشروع",
}

QUERY_FOCUS_RULES = [
    {
        "keywords": ("validité", "validite", "valable", "صلاحية", "سارية المفعول"),
        "primary_terms": (
            "offre reste valable",
            "délais de validité des offres",
            "الالتزام بالعروض",
            "الالتزام ببنود وشروط هذا التعهد",
            "هذا التعهد لمدة",
        ),
        "support_terms": ("90", "120"),
        "penalty_terms": (
            "date minimale de validité",
            "remplissage de la colonne",
            "emballages des toners",
            "nombre de pages",
        ),
    },
    {
        "keywords": ("caution", "cautionnement", "provisoire", "الضمان الوقتي", "ضمان وقتي", "كفيل"),
        "primary_terms": (
            "caution provisoire",
            "cautionnement provisoire",
            "مبلغ الضمان الوقتي",
            "حدد مبلغ الضمان الوقتي",
            "خمسة آلاف دينار",
            "5000 خمسة آلاف دينار",
            "الضمان الوقتي",
            "يرجع الضمان الوقتي",
            "المعوّض للضمان الوقتي",
        ),
        "support_terms": (
            "الكفيل بالتضامن",
            "كفلائهم بالتضامن",
            "قصد المشاركة",
            "صالح لمدة 120 يوما",
        ),
        "penalty_terms": (
            "caution définitive",
            "garantie définitive",
            "الضمان النهائي",
            "ضمان حسن تنفيذ",
            "المعوّض للضمان النهائي",
            "cartouche",
            "disque dur",
            "clavier",
            "câble usb",
            "adaptateur hdmi",
            "switch",
        ),
    },
    {
        "keywords": (
            "ouverture",
            "ouverture des offres",
            "plis",
            "séance publique",
            "seance publique",
            "فتح العروض",
            "جلسة فتح العروض",
        ),
        "primary_terms": (
            "في نفس اليوم المحدد كتاريخ أقصى لقبول العروض",
            "ouverture des offres",
            "séance publique",
            "جلسة فتح العروض",
            "تُعقد جلسة فتح العروض",
            "فتح العروض الفنية والمالية",
            "جلسة علنية",
        ),
        "support_terms": (
            "قبول العروض",
            "جلسة واحدة لفتح العروض",
            "التثبت من تاربخ الوصول",
        ),
        "penalty_terms": (
            "تصريح على الشرف",
            "السجل الوطني للمؤسسات",
        ),
    },
]

HALLUCINATION_SIGNALS = [
    "généralement", "habituellement", "en général", "il est courant",
    "typiquement", "on peut supposer", "il est probable", "dépend du contexte",
    "il faudrait consulter", "je recommande de consulter", "il est important de",
    "peut être requis", "peuvent être demandées", "certaines circonstances",
    "LinkedIn", "Indeed", "Glassdoor", "en France", "selon la nature",
    "il n'y a pas de mention claire", "il est donc recommandé",
    "the user requested", "the user asked", "respond in english", "answer in english",
]

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)

SECTION_HINT_RULES = [
    (("pénalité", "penalite", "retard", "amende", "غرامة", "تأخير"), ("penalty", "technical")),
    (("garantie", "warranty", "ضمان"), ("technical", "guarantee")),
    (("caution", "cautionnement", "كفالة", "كفيل"), ("guarantee",)),
    (("paiement", "payer", "facture", "virement", "خلاص", "دفع"), ("payment", "financial")),
    (("validité", "validite", "valable", "صلاحية", "سارية المفعول"), ("deadline", "guarantee")),
    (("date limite", "heure limite", "remise des offres", "ouverture", "plis", "soumission", "tuneps", "تونيبس", "envoyer", "depot", "dépôt"), ("admin", "deadline")),
    (("réception", "reception", "installation", "livraison", "validation", "استلام", "تسليم"), ("reception", "technical")),
    (("document administratif", "cnss", "registre", "attestation", "fiscal", "administratif"), ("admin",)),
    (("technique", "spécification", "specification", "documentation technique", "fiche technique"), ("technical",)),
    (("financier", "bordereau", "prix", "lettre de soumission"), ("financial",)),
]


# ── Singletons ────────────────────────────────────────────────────────────

client = get_llm_client()

from fastapi import HTTPException

async def get_embedding(text: str | list[str]) -> list[float] | list[list[float]]:
    loop = asyncio.get_running_loop()
    embedder = get_embedder()
    if isinstance(text, str):
        result = await loop.run_in_executor(None, lambda: to_builtin_list(embedder.encode(text)))
        return result
    else:
        result = await loop.run_in_executor(None, lambda: to_builtin_list(embedder.encode(text)))
        return result

@lru_cache()
def _get_reranker():
    if not settings.RERANKER_ENABLED:
        logger.info("Reranker disabled by configuration.")
        return None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return CrossEncoder(settings.RERANKER_MODEL, device=device)
    except Exception as exc:
        logger.warning("Reranker unavailable, falling back to raw Qdrant ranking: {}", exc)
        return None

@lru_cache()
def _get_vector_store():
    return AsyncVectorStore(url=settings.QDRANT_URL, collection=settings.QDRANT_COLLECTION)


# ── Core Functions ────────────────────────────────────────────────────────

def enhance_query(query: str) -> str:
    normalized_query = _normalize_query_text(query)
    expansions = []
    for key, expansion in QUERY_EXPANSIONS.items():
        if _normalize_query_text(key) in normalized_query:
            expansions.append(expansion)
    if expansions:
        return query + " " + " ".join(expansions)
    return query


def _normalize_query_text(text: str) -> str:
    lowered = text.lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    stripped = re.sub(r"[`’]", "'", stripped)
    stripped = re.sub(r"[^\w\s'\u0600-\u06FF]", " ", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def infer_section_hints(query: str) -> list[str]:
    normalized_query = _normalize_query_text(query)
    hints = []
    seen = set()

    for keywords, sections in SECTION_HINT_RULES:
        if any(_normalize_query_text(keyword) in normalized_query for keyword in keywords):
            for section in sections:
                if section not in seen:
                    seen.add(section)
                    hints.append(section)

    return hints


def _matching_focus_rules(query: str) -> list[dict]:
    normalized_query = _normalize_query_text(query)
    matched = []

    for rule in QUERY_FOCUS_RULES:
        if any(_normalize_query_text(keyword) in normalized_query for keyword in rule["keywords"]):
            matched.append(rule)

    return matched


def _focus_bonus(rules: list[dict], chunk: str) -> float:
    if not rules:
        return 0.0

    normalized_chunk = _normalize_query_text(chunk)
    bonus = 0.0

    for rule in rules:
        primary_hits = sum(
            1 for term in rule["primary_terms"]
            if _normalize_query_text(term) in normalized_chunk
        )
        support_hits = sum(
            1 for term in rule["support_terms"]
            if _normalize_query_text(term) in normalized_chunk
        )
        penalty_hits = sum(
            1 for term in rule["penalty_terms"]
            if _normalize_query_text(term) in normalized_chunk
        )

        bonus += (primary_hits * 0.22) + (support_hits * 0.08) - (penalty_hits * 0.25)

    return bonus


def _answer_hint(question: str) -> str:
    normalized_question = _normalize_query_text(question)
    hints = []

    if any(keyword in normalized_question for keyword in ("caution", "cautionnement", "provisoire", "الضمان الوقتي")):
        hints.append(
            "Pour une question sur la caution provisoire, privilégie le passage qui donne explicitement le montant "
            "ou la règle de restitution. Ignore les tableaux d'articles, les listes de produits et les montants "
            "sans lien clair avec le cautionnement."
        )

    if any(keyword in normalized_question for keyword in ("ouverture", "plis", "séance publique", "seance publique", "فتح العروض")):
        hints.append(
            "Pour une question sur l'ouverture des offres, ignore les numéros d'article, de page, de lot ou "
            "d'annexe. Si le contexte dit seulement que la séance a lieu le même jour que la date limite de remise "
            "des offres, réponds avec cette règle au lieu d'inventer une date calendaire."
        )

    if _is_subject_question(question):
        hints.append(
            "Pour une question sur le sujet, l'objet ou l'intitule du document, privilegie le titre, l'objet de la consultation, "
            "la premiere page et les formulations comme 'objet', 'consultation', 'appel d'offres' ou 'demande d'offre de prix'."
        )

    return "\n".join(hints)


def _is_subject_question(question: str) -> bool:
    normalized_question = f" {_normalize_query_text(question)} "
    markers = (
        " sujet ",
        " objet ",
        " l'objet ",
        " l objet ",
        " objet de la consultation ",
        " intitule ",
        " titre ",
        " de quoi s agit ",
        " de quoi s'agit ",
        " de quoi parle ",
        " c'est quoi ",
        " c est quoi ",
        " concerne quoi ",
        " what is this about ",
        " what is the tender about ",
        " what is the document about ",
        " what does this concern ",
        " purpose ",
        " subject ",
        " title ",
    )
    return any(marker in normalized_question for marker in markers)


def _page_position_bonus(question: str, page: str | int | None, chunk: str) -> float:
    if not _is_subject_question(question):
        return 0.0

    try:
        page_num = int(str(page))
    except (TypeError, ValueError):
        page_num = 999

    normalized_chunk = _normalize_query_text(chunk)
    title_markers = (
        "objet",
        "intitule",
        "consultation",
        "appel d offres",
        "appel d'offres",
        "demande d offre de prix",
        "demande d'offre de prix",
    )
    lexical_bonus = 0.25 if any(marker in normalized_chunk for marker in title_markers) else 0.0

    if page_num <= 1:
        return 0.45 + lexical_bonus
    if page_num <= 2:
        return 0.3 + lexical_bonus
    if page_num <= 4:
        return 0.15 + lexical_bonus
    return lexical_bonus - 0.05


def _detect_answer_language(question: str) -> str:
    normalized_question = _normalize_query_text(question)

    if re.search(r"[\u0600-\u06FF]", question):
        return "ar"

    padded = f" {normalized_question} "
    english_markers = (
        " what ", " which ", " when ", " where ", " who ", " whom ", " whose ",
        " how ", " does ", " do ", " is ", " are ", " deadline ", " warranty ",
        " payment ", " penalties ", " documents required ", " bid ", " tender ",
        " offer validity ", " opening of bids ",
    )
    if any(marker in padded for marker in english_markers):
        return "en"

    french_markers = (
        " quel ", " quelle ", " quels ", " quelles ", " objet ", " sujet ",
        " cahier de charge", " cahier des charges", " delai", " delais",
        " garantie", " paiement", " document", " resume", " resumer",
        " mentionne", " mentions", " liste", " fournir", " fournissez",
        " date limite", " echeance", " echeances", " offre", " soumission",
        " est ce que ", " existe", " exig", " attestation", " affiliation",
        " cnss", " fiscal", " fiscale", " registre", " rne", " constructeur",
        " fabricant", " references", " reception", " caution", " penalite",
        " penalites", " modalite", " modalites", " depot", " envoi", " plis",
        " variantes",
    )
    if any(marker in padded for marker in french_markers):
        return "fr"

    return "fr"


def _missing_answer_text(question: str) -> str:
    language = _detect_answer_language(question)
    if language == "ar":
        return "غير مذكور في هذه الوثيقة."
    if language == "en":
        return "Not mentioned in this document."
    return "Non mentionne dans ce document."


def _language_instruction(question: str) -> str:
    language = _detect_answer_language(question)
    if language == "ar":
        return "Reponds uniquement en arabe."
    if language == "en":
        return "Answer only in English."
    return "Reponds uniquement en francais."


def _is_deadline_question(question: str) -> bool:
    normalized_question = f" {_normalize_query_text(question)} "
    markers = (
        " date limite ",
        " date de remise ",
        " remise des offres ",
        " delai de remise ",
        " dernier delai ",
        " dernier jour ",
        " au plus tard ",
        " quand deposer ",
        " quand soumettre ",
        " quand envoyer ",
        " submit the offer ",
        " submit offers ",
        " submission date ",
        " submission deadline ",
        " last date ",
        " last day ",
        " deadline ",
        " due date ",
        " closing date ",
    )
    return any(marker in normalized_question for marker in markers)


def _is_summary_question(question: str) -> bool:
    normalized_question = f" {_normalize_query_text(question)} "
    markers = (
        " resume ",
        " resumer ",
        " resumee ",
        " summary ",
        " summarize ",
        " summarise ",
        " points cles ",
        " points cle ",
        " key points ",
        " main points ",
        " overview ",
        " tell me about ",
        " explique ",
        " explain ",
        " de quoi parle ",
    )
    return any(marker in normalized_question for marker in markers)



def _is_organization_identity_question(question: str) -> bool:
    normalized_question = f" {_normalize_query_text(question)} "
    markers = (
        " quelle est ce societe ",
        " quelle est cette societe ",
        " quel est ce societe ",
        " c est quelle societe ",
        " c'est quelle societe ",
        " quelle societe ",
        " nom de la societe ",
        " quelle entreprise ",
        " nom de l entreprise ",
        " quelle banque ",
        " quel organisme ",
        " quelle organisme ",
        " quelle institution ",
        " what company ",
        " which company ",
        " what is this company ",
        " company name ",
        " which bank ",
        " what organization ",
        " which organization ",
    )
    if any(marker in normalized_question for marker in markers):
        return True

    has_company_word = any(marker in normalized_question for marker in (" societe ", " soci ", " entreprise ", " organisme ", " company "))
    has_identity_shape = any(
        marker in normalized_question
        for marker in (" quelle est ce ", " quelle est cette ", " c est quelle ", " c'est quelle ", " what is this ", " which ")
    )
    return has_company_word and has_identity_shape



def _is_chatbot_identity_question(question: str) -> bool:
    normalized_question = f" {_normalize_query_text(question)} "
    markers = (
        " qui es tu ",
        " qui etes vous ",
        " t es qui ",
        " tu es qui ",
        " c est qui ",
        " c'est qui ",
        " presente toi ",
        " presentez vous ",
        " quel est ton nom ",
        " comment tu t appelles ",
        " comment vous appelez vous ",
        " who are you ",
        " who are u ",
        " who r u ",
        " who r you ",
        " what are you ",
        " what are u ",
        " what r u ",
        " what r you ",
        " your name ",
        " introduce yourself ",
        " chkoun enti ",
        " chkun enti ",
        " chnowa enti ",
    )
    if any(marker in normalized_question for marker in markers):
        return True

    tokens = set(normalized_question.split())
    if ("who" in tokens or "what" in tokens) and ("u" in tokens or "you" in tokens) and ("are" in tokens or "r" in tokens):
        return True
    return False


def answer_chatbot_identity(question: str) -> tuple[str, list[dict]] | None:
    if not _is_chatbot_identity_question(question):
        return None
    language = _detect_answer_language(question)
    if language == "en":
        answer = (
            "I am TE RAG Assistant, the Tunisie Electronique document assistant. "
            "I help analyze cahiers des charges, extract key requirements, and answer only from the selected documents."
        )
    else:
        answer = (
            "Je suis TE RAG Assistant, l'assistant documentaire de Tunisie Electronique. "
            "J'aide a analyser les cahiers des charges, extraire les exigences importantes et repondre a partir des documents selectionnes."
        )
    return answer, []


def _extract_direct_answer(
    question: str,
    chunks: list[str],
    metas: list[dict] | None = None,
) -> str | None:
    if not chunks:
        return None

    source_metas = metas or [{} for _ in chunks]
    sentence_stop = r"(?=(?:\.\s+[A-Z][^a-z]|\.\s+Messieurs\b|\.\s+Madame\b|\.\s+\d+\b|\s+\d+\.\s+[A-Z]|\n|$))"
    subject_patterns = (
        re.compile(
            rf"\barticle\s*\d+\s*[:\-]?\s*objet(?:\s+(?:du|de\s+la|de\s+l['’])\s+(?:march[eé]|consultation|cahier\s+des\s+charges))?\s*[:\-]?\s*(.+?){sentence_stop}",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            rf"\b(?:la\s+)?(?:pr[eé]sente\s+)?consultation\s+a\s+pour\s+objet\s+(.+?){sentence_stop}",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            rf"\b(?:le\s+)?(?:pr[eé]sent\s+)?(?:march[eé]|appel\s+d['’]offres?|cahier\s+des\s+charges)\s+a\s+pour\s+objet\s+(.+?){sentence_stop}",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            rf"\bporte\s+sur\s+(.+?){sentence_stop}",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            rf"\bobjet(?:\s+de\s+la\s+consultation)?\s*[:\-]\s*(.+?){sentence_stop}",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            rf"\bconsultation\s+pour\s+(.+?){sentence_stop}",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            rf"\bdemande d[' ]offre de prix\s+(.+?){sentence_stop}",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            rf"\bmarch[eé]\s+d[' ]acquisition\s+(.+?){sentence_stop}",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            rf"\brenouvellement\s+des\s+licences\s+(.+?){sentence_stop}",
            re.IGNORECASE | re.DOTALL,
        ),
    )
    deadline_patterns = (
        re.compile(
            rf"\bdate\s+limite(?:\s+de\s+remise\s+des\s+offres)?\s*[:\-]\s*(.+?){sentence_stop}",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            rf"\bremise\s+des\s+offres\s*[:\-]\s*(.+?){sentence_stop}",
            re.IGNORECASE | re.DOTALL,
        ),
    )

    if _is_subject_question(question):
        patterns = subject_patterns
    elif _is_deadline_question(question):
        patterns = deadline_patterns
    else:
        return None

    for chunk, meta in zip(chunks, source_metas):
        compact = re.sub(r"\s+", " ", chunk).strip()
        for pattern in patterns:
            match = pattern.search(compact)
            if not match:
                continue

            extracted = re.sub(r"\s+", " ", match.group(1)).strip(" .;:-")
            if not extracted:
                continue

            source = meta.get("source")
            page = meta.get("page")
            if source and page:
                return f"{extracted}\nSource: {source}, page {page}."
            return extracted

    return None


def _extract_summary_answer(
    question: str,
    chunks: list[str],
    metas: list[dict] | None = None,
) -> str | None:
    if not _is_summary_question(question) or not chunks:
        return None

    source_metas = metas or [{} for _ in chunks]

    for chunk, meta in zip(chunks, source_metas):
        compact = re.sub(r"\s+", " ", chunk).strip()
        if not compact:
            continue

        summary_parts: list[str] = []

        def add_part(text: str) -> None:
            cleaned = re.sub(r"\s+", " ", text).strip(" .;:-")
            if not cleaned:
                return

            normalized_cleaned = _normalize_query_text(cleaned)
            for existing in summary_parts:
                normalized_existing = _normalize_query_text(existing)
                if (
                    normalized_cleaned == normalized_existing
                    or normalized_cleaned in normalized_existing
                    or normalized_existing in normalized_cleaned
                ):
                    return
            summary_parts.append(cleaned)

        object_patterns = (
            re.compile(
                rf"\barticle\s*\d+\s*[:\-]?\s*objet(?:\s+(?:du|de\s+la|de\s+l['’])\s+(?:march[eé]|consultation|cahier\s+des\s+charges))?\s*[:\-]?\s*(.+?){sentence_stop}",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                rf"\b(?:la\s+)?(?:pr[eé]sente\s+)?consultation\s+a\s+pour\s+objet\s+(.+?){sentence_stop}",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                rf"\b(?:le\s+)?(?:pr[eé]sent\s+)?(?:march[eé]|appel\s+d['’]offres?|cahier\s+des\s+charges)\s+a\s+pour\s+objet\s+(.+?){sentence_stop}",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                rf"\bobjet(?:\s+de\s+la\s+consultation)?\s*[:\-]\s*(.+?){sentence_stop}",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                rf"\bconsultation\s+pour\s+(.+?){sentence_stop}",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                rf"\bmarch[eé]\s+d[' ]acquisition\s+(.+?){sentence_stop}",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                rf"\brenouvellement\s+des\s+licences\s+(.+?){sentence_stop}",
                re.IGNORECASE | re.DOTALL,
            ),
        )
        for pattern in object_patterns:
            match = pattern.search(compact)
            if not match:
                continue
            extracted = match.group(1)
            if "consultation pour" in pattern.pattern.lower():
                add_part(f"Consultation pour {extracted}")
            elif "mar" in pattern.pattern.lower():
                add_part(f"Marche d'acquisition {extracted}")
            elif "renouvellement des licences" in pattern.pattern.lower():
                add_part(f"Renouvellement des licences {extracted}")
            else:
                add_part(extracted)

        bullets = re.findall(r"[•\-]\s*([^•]+?)(?=(?:[•\-]\s*)|$)", compact)
        cleaned_bullets = [
            re.sub(r"\s+", " ", bullet).strip(" .;:-")
            for bullet in bullets
            if len(re.sub(r"\s+", " ", bullet).strip()) > 3
        ]
        if cleaned_bullets:
            add_part("Elements cites : " + ", ".join(cleaned_bullets[:3]) + ".")

        if not summary_parts:
            sentences = [
                re.sub(r"\s+", " ", sentence).strip(" .;:-")
                for sentence in re.split(r"(?<=[.!?])\s+", compact)
                if len(sentence.strip()) > 20
            ]
            for sentence in sentences[:2]:
                add_part(sentence)

        summary_parts = [part for part in summary_parts if part]
        if not summary_parts:
            continue

        summary_text = " ".join(summary_parts[:2]).strip()
        source = meta.get("source")
        page = meta.get("page")
        if source and page:
            return f"{summary_text}\nSource: {source}, page {page}."
        return summary_text

    return None


def _fact_to_source_meta(source: str, fact: dict, score: float = 1.0) -> dict:
    meta = {
        "source": source,
        "page": str(fact.get("page", "?")),
        "section": fact.get("section", "general"),
        "score": score,
    }
    if fact.get("location"):
        meta["location"] = str(fact.get("location"))
    return meta


def _fact_source_reference(source: str, fact: dict) -> str:
    location = str(fact.get("location") or "").strip()
    if location:
        return f"Source: {source}, {location}."
    page = fact.get("page")
    if page:
        return f"Source: {source}, page {page}."
    return f"Source: {source}."


def _format_fact_answer(source: str, fact: dict) -> str | None:
    text = re.sub(r"\s+", " ", str(fact.get("text", ""))).strip(" .;:-")
    if not text:
        return None
    return f"{text}\n{_fact_source_reference(source, fact)}"


FACT_FIELD_LABELS = {
    "validity": "Validite des offres",
    "opening": "Ouverture des plis",
    "caution": "Caution",
    "guarantee": "Garantie",
    "payment": "Paiement",
    "penalties": "Penalites",
    "cnss": "CNSS",
    "rne": "RNE / registre",
    "submission_method": "Mode de depot",
    "variants": "Variantes",
    "information_sheet": "Fiche de renseignements",
    "fiscal_certificate": "Situation fiscale",
    "manufacturer_authorization": "Autorisation constructeur",
    "references": "References",
    "reception": "Reception",
    "definitive_caution": "Caution definitive",
}

FACT_FIELD_QUERY_MARKERS = (
    ("submission_method", ("mode d'envoi", "mode d envoi", "mode de depot", "depot de la soumission", "depot des offres", "envoi", "bureau d'ordre", "tuneps", "comment deposer", "comment soumettre", "how to submit", "submission method", "where submit")),
    ("definitive_caution", ("caution definitive", "garantie definitive", "garantie de bonne execution")),
    ("caution", ("caution", "cautionnement", "garantie provisoire", "garantie bancaire provisoire", "bid bond", "provisional bond", "provisional guarantee", "temporary guarantee")),
    ("validity", ("validite", "valable", "delai de validite", "duree de validite", "offer validity", "bid validity", "validity period", "how long valid")),
    ("opening", ("ouverture", "ouverture des plis", "ouverture des offres", "seance publique", "plis", "bid opening", "opening date", "opening session")),
    ("variants", ("variante", "variantes", "autorisees", "admises")),
    ("information_sheet", ("fiche de renseignements", "fiche signaletique", "formulaire de renseignements")),
    ("cnss", ("cnss", "affiliation cnss", "certificat cnss")),
    ("fiscal_certificate", ("solde", "situation fiscale", "attestation fiscale", "certificat fiscal")),
    ("rne", ("rne", "registre de commerce", "registre national", "extrait du registre")),
    ("manufacturer_authorization", ("autorisation constructeur", "autorisation fabricant", "lettre constructeur", "lettre fabricant")),
    ("references", ("references", "liste de references", "liste des references", "similar projects", "past projects")),
    ("reception", ("reception", "reception provisoire", "reception definitive", "pv de reception")),
    ("payment", ("paiement", "paiements", "reglement", "facture", "virement", "traite", "payment", "payment terms", "how paid", "how do they pay")),
    ("penalties", ("penalite", "penalites", "retard", "amende", "sanction", "late penalty", "delay penalty", "penalties", "penalty")),
    (
        "guarantee",
        (
            "garantie constructeur",
            "garantie definitive",
            "garantie contractuelle",
            "periode de garantie",
            "duree de garantie",
            "warranty",
            "warranty period",
            "guarantee period",
            "maintenance",
            "sav",
        ),
    ),
)

FACT_LIST_FIELD_LABELS = {
    "administrative_documents": "Documents administratifs",
    "technical_documents": "Documents techniques",
    "financial_documents": "Documents financiers",
    "requested_items": "Licences / quantites demandees",
}

FACT_LIST_QUERY_MARKERS = (
    ("administrative_documents", ("administratif", "administrative", "pieces administratives", "documents administratifs", "admin documents", "administrative documents")),
    ("technical_documents", ("technique", "techniques", "offre technique", "documents techniques", "technical documents", "technical offer")),
    ("financial_documents", ("financier", "financiere", "financiers", "offre financiere", "documents financiers", "financial documents", "financial offer")),
    (
        "requested_items",
        (
            "licence",
            "licences",
            "license",
            "licenses",
            "quantite",
            "quantites",
            "produit",
            "produits",
            "article",
            "articles",
            "materiel",
            "equipements",
            "support id",
            "support ids",
            "id support",
            "id de support",
            "id licence",
            "id de licence",
        ),
    ),
)

FACT_LIST_INTENT_MARKERS = (
    "combien",
    "donne",
    "donner",
    "quel",
    "quelle",
    "document",
    "documents",
    "id",
    "piece",
    "pieces",
    "dossier",
    "offre",
    "liste",
    "lister",
    "fournir",
    "contient",
    "contenir",
    "comporter",
    "quels",
    "quelles",
    "support",
    "required",
    "needed",
    "need",
    "requirements",
)


def _query_has_any(normalized_question: str, markers: tuple[str, ...]) -> bool:
    return any(_normalize_query_text(marker) in normalized_question for marker in markers)


def _fact_scalar_field_for_question(question: str) -> str | None:
    normalized_question = _normalize_query_text(question)
    for field, markers in FACT_FIELD_QUERY_MARKERS:
        if _query_has_any(normalized_question, markers):
            return field
    return None


def _fact_list_field_for_question(question: str) -> str | None:
    normalized_question = _normalize_query_text(question)
    if not _query_has_any(normalized_question, FACT_LIST_INTENT_MARKERS):
        return None

    for field, markers in FACT_LIST_QUERY_MARKERS:
        if _query_has_any(normalized_question, markers):
            return field
    return None


def _format_labeled_fact_answer(source: str, label: str, fact: dict) -> str | None:
    text = re.sub(r"\s+", " ", str(fact.get("text", ""))).strip(" .;:-")
    if not text:
        return None

    answer = f"{label} : {text}"
    return f"{answer}\n{_fact_source_reference(source, fact)}"


def _fact_source_metas(source: str, fact: dict, limit: int = 4) -> list[dict]:
    raw_items = fact.get("items")
    items = raw_items if isinstance(raw_items, list) and raw_items else [fact]
    metas = []
    seen = set()

    for item in items:
        page = str(item.get("page", fact.get("page", "?")))
        section = item.get("section", fact.get("section", "general"))
        location = item.get("location", fact.get("location"))
        key = (page, section, location)
        if key in seen:
            continue
        seen.add(key)
        meta = {
            "source": source,
            "page": page,
            "section": section,
            "score": 1.0,
        }
        if location:
            meta["location"] = str(location)
        metas.append(meta)
        if len(metas) >= limit:
            break

    return metas


def _format_list_fact_answer(source: str, label: str, fact: dict) -> str | None:
    raw_items = fact.get("items")
    if isinstance(raw_items, list) and raw_items:
        items = [
            re.sub(r"\s+", " ", str(item.get("text", ""))).strip(" .;:-")
            for item in raw_items
        ]
        items = [item for item in items if item]
    else:
        text = str(fact.get("text", "")).strip()
        items = [
            re.sub(r"\s+", " ", line.lstrip("-").strip()).strip(" .;:-")
            for line in text.splitlines()
            if line.strip()
        ]

    if not items:
        return None

    body = "\n".join([f"{label} :"] + [f"- {item}" for item in items])
    return f"{body}\n{_fact_source_reference(source, fact)}"


TEXT_CACHE_PAGE_RE = re.compile(r"(?m)^\s*\[Page\s+([^\]]+)\]\s*$", re.IGNORECASE)
TEXT_CACHE_PAGE_BANNER_RE = re.compile(r"\s*(?:-|\u2014)\s*Page\s+\d+\s*(?:-|\u2014)\s*", re.IGNORECASE)
TEXT_CACHE_STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "au", "aux", "avec", "be", "by", "ce", "ces",
    "comment", "dans", "de", "des", "do", "does", "du", "en", "est", "et", "for", "from", "how",
    "is", "it", "la", "le", "les", "leur", "leurs", "l", "of", "on", "or", "ou", "par", "pour",
    "que", "quel", "quelle", "quels", "quelles", "qui", "sont", "sur", "the", "this", "to", "un",
    "une", "what", "when", "where", "which", "who", "why", "with",
}
TEXT_CACHE_MOJIBAKE_MARKERS = ("?", "?", "??", "???", "?")


def _mojibake_score(value: str) -> int:
    return sum(str(value).count(marker) for marker in TEXT_CACHE_MOJIBAKE_MARKERS)


def _repair_text_cache_mojibake(value: str) -> str:
    best = str(value)
    best_score = _mojibake_score(best)
    for encoding in ("cp1252", "latin1"):
        try:
            candidate = best.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        candidate_score = _mojibake_score(candidate)
        if candidate_score < best_score:
            best = candidate
            best_score = candidate_score
    return best


TEXT_CACHE_SYNONYMS = {
    "address": ("adresse", "rue", "bp", "siege", "contact"),
    "adresse": ("address", "rue", "bp", "siege", "contact"),
    "amount": ("montant", "prix", "dinars", "dinar"),
    "bank": ("banque", "tsb"),
    "buyer": ("acheteur", "maitre", "ouvrage", "tsb"),
    "client": ("maitre", "ouvrage", "acheteur", "beneficiaire", "tsb"),
    "contact": ("email", "mail", "telephone", "tel", "fax", "adresse"),
    "delivery": ("livraison", "installation", "delai", "execution"),
    "manager": ("responsable", "chef", "maitre", "ouvrage"),
    "owner": ("maitre", "ouvrage", "acheteur", "beneficiaire"),
    "price": ("prix", "montant", "bordereau", "financier"),
    "project": ("projet", "marche", "objet", "maitre", "ouvrage"),
    "qualification": ("participation", "certification", "reference", "references"),
    "required": ("exige", "exiges", "obligatoire", "demande", "demandes"),
    "requirements": ("exigences", "exige", "obligatoire", "conditions"),
    "say": ("article", "stipule", "prevoit", "indique"),
    "who": ("qui", "maitre", "ouvrage", "fournisseur"),
}


def _normalize_filename_cache_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).casefold())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"\.(?:pdf|docx?|txt)$", "", normalized)
    return re.sub(r"[^\w]+", " ", normalized).strip()


def _resolve_text_cache_path(filename: str) -> Path | None:
    candidates = [
        TEXT_CACHE_DIR / f"{filename}.txt",
        TEXT_CACHE_DIR / f"{filename.replace('_', ' ')}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    wanted = _normalize_filename_cache_key(filename)
    if not wanted or not TEXT_CACHE_DIR.exists():
        return None
    for candidate in TEXT_CACHE_DIR.glob("*.txt"):
        if _normalize_filename_cache_key(candidate.stem) == wanted:
            return candidate
    return None


def _read_text_cache_pages(filename: str) -> list[dict]:
    cache_path = _resolve_text_cache_path(filename)
    if not cache_path:
        return []
    try:
        cached_text = _repair_text_cache_mojibake(cache_path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        logger.debug("Text-cache fallback skipped for {}: {}", filename, exc)
        return []

    matches = list(TEXT_CACHE_PAGE_RE.finditer(cached_text))
    pages = []
    if matches:
        for index, match in enumerate(matches):
            page = match.group(1).strip()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(cached_text)
            page_text = TEXT_CACHE_PAGE_BANNER_RE.sub(" ", cached_text[start:end])
            page_text = re.sub(r"[ \t]+", " ", page_text)
            page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()
            if page_text:
                pages.append({"page": page, "text": page_text})
    else:
        cleaned = TEXT_CACHE_PAGE_BANNER_RE.sub(" ", cached_text).strip()
        if cleaned:
            pages.append({"page": "1", "text": cleaned})
    return pages


def _query_terms_for_text_cache(question: str) -> set[str]:
    normalized = _normalize_query_text(question)
    terms = {
        token
        for token in normalized.split()
        if token not in TEXT_CACHE_STOPWORDS and (len(token) > 2 or token.isdigit())
    }
    expanded = set(terms)
    for token in list(terms):
        expanded.update(_normalize_query_text(value) for value in TEXT_CACHE_SYNONYMS.get(token, ()))
    return {term for term in expanded if term and term not in TEXT_CACHE_STOPWORDS}


def _text_cache_article_number(question: str) -> str | None:
    match = re.search(r"\barticle\s*(\d{1,3})\b", _normalize_query_text(question))
    return match.group(1) if match else None


def _split_text_cache_passages(text: str) -> list[str]:
    chunks = [piece.strip() for piece in re.split(r"(?=\bARTICLE\s+\d{1,3}\b)", text, flags=re.IGNORECASE) if piece.strip()]
    if not chunks:
        chunks = [piece.strip() for piece in re.split(r"\n{2,}", text) if piece.strip()]
    passages = []
    for chunk in chunks or [text]:
        if len(chunk) <= 1800:
            passages.append(chunk)
            continue
        sentences = re.split(r"(?<=[.!?:;])\s+", chunk)
        buffer = ""
        for sentence in sentences:
            if len(buffer) + len(sentence) > 1600 and buffer:
                passages.append(buffer.strip())
                buffer = sentence
            else:
                buffer = f"{buffer} {sentence}".strip()
        if buffer:
            passages.append(buffer.strip())
    return passages


def _score_text_cache_passage(question: str, passage: str, terms: set[str], article_number: str | None) -> float:
    normalized_question = _normalize_query_text(question)
    normalized_passage = _normalize_query_text(passage)
    if not normalized_passage:
        return 0.0
    score = 0.0
    for term in terms:
        if term.isdigit():
            if re.search(rf"\b{re.escape(term)}\b", normalized_passage):
                score += 0.8
        elif term in normalized_passage:
            score += 1.0
    if article_number and re.search(rf"\barticle\s*{re.escape(article_number)}\b", normalized_passage):
        score += 8.0

    asks_owner = any(
        marker in normalized_question
        for marker in ("project manager", "maitre d'ouvrage", "maitre d ouvrage", "client", "buyer", "owner")
    )
    if asks_owner:
        if re.search(r"\bmaitre\s+d'?\s*ouvrage\s+designe\b", normalized_passage):
            score += 8.0
        elif re.search(r"\bmaitre\s+de\s+l'?\s*ouvrage\s+designe\b", normalized_passage):
            score += 8.0
        elif re.search(r"\bmaitre\s+(?:d'?|de\s+l'?)\s*ouvrage\b", normalized_passage):
            score += 2.0

    if "participation" in normalized_question and "condition" in normalized_question:
        if "conditions de participation" in normalized_passage:
            score += 5.0
    if "contact" in normalized_question and any(marker in normalized_passage for marker in ("contact@", "fax", "telephone", "tel")):
        score += 4.0

    if len(passage) > 2600:
        score -= 1.0
    return score


def _compact_text_cache_passage(question: str, passage: str, max_chars: int = 900) -> str:
    passage = re.sub(r"\s+", " ", str(passage)).strip(" .;:-")
    if len(passage) <= max_chars:
        return passage

    terms = _query_terms_for_text_cache(question)
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?:;])\s+", passage) if sentence.strip()]
    if not sentences:
        return passage[:max_chars].rstrip(" .;:-") + "..."

    scored = [(_score_text_cache_passage(question, sentence, terms, _text_cache_article_number(question)), index, sentence) for index, sentence in enumerate(sentences)]
    scored.sort(key=lambda item: item[0], reverse=True)
    keep_indexes = sorted(index for score, index, _sentence in scored[:3] if score > 0)
    if not keep_indexes:
        return passage[:max_chars].rstrip(" .;:-") + "..."

    excerpt = " ".join(sentences[index] for index in keep_indexes)
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rstrip(" .;:-") + "..."
    return excerpt


def answer_from_text_cache(filename: str, question: str) -> tuple[str, list[dict]] | None:
    pages = _read_text_cache_pages(filename)
    if not pages:
        return None

    terms = _query_terms_for_text_cache(question)
    article_number = _text_cache_article_number(question)
    if not terms and not article_number:
        return None

    scored = []
    for page in pages:
        for passage in _split_text_cache_passages(page["text"]):
            score = _score_text_cache_passage(question, passage, terms, article_number)
            if score > 0:
                scored.append((score, page["page"], passage))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, page, passage = scored[0]
    threshold = 2.0 if not article_number else 4.0
    if best_score < threshold:
        return None

    excerpt = _compact_text_cache_passage(question, passage)
    if not excerpt:
        return None

    language = _detect_answer_language(question)
    if language == "en":
        answer = f"Relevant passage found: {excerpt}\nSource: {filename}, page {page}."
    else:
        answer = f"Passage pertinent trouve : {excerpt}\nSource: {filename}, page {page}."
    return answer, [{"source": filename, "page": str(page), "section": "text_cache", "score": min(best_score / 10.0, 1.0)}]


GENERIC_ORGANIZATION_FILENAME_MARKERS = {
    "ao",
    "appel",
    "appel d offres",
    "cahier",
    "charges",
    "cc",
    "cdc",
    "consultation",
    "dossier",
}


def _clean_organization_candidate(line: str) -> str:
    value = re.sub(r"\s+", " ", str(line)).strip(" .;:-??\"'")
    value = re.sub(r"^(?:l[ea]?|the)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+(?:ma\s+banque\s+et\s+plus|annee\s+\d{4}).*$", "", value, flags=re.IGNORECASE)
    return value.strip(" .;:-??\"'")


def _organization_candidate_score(line: str, index: int, filename: str) -> float:
    candidate = _clean_organization_candidate(line)
    normalized = _normalize_query_text(candidate)
    if len(candidate) < 3 or len(candidate) > 100:
        return -10.0
    if not re.search(r"[A-Za-z?-??-??-?]", candidate):
        return -10.0
    if re.search(r"\d{4}|\+\d|@|www|\.com|fax|tel\b", normalized):
        return -5.0
    if any(marker in normalized for marker in ("appel d offres", "cahier des charges", "clauses administratives", "article ", "page ", "capital", "rue ")):
        return -4.0

    score = max(0.0, 8.0 - index * 0.15)
    alpha_chars = [char for char in candidate if char.isalpha()]
    upper_chars = [char for char in alpha_chars if char.isupper()]
    if alpha_chars and len(upper_chars) / len(alpha_chars) > 0.55:
        score += 2.0
    if any(marker in normalized for marker in ("banque", "bank", "ministere", "ministry", "office", "centre", "institut", "societe", "tunisian", "saudi")):
        score += 3.0

    filename_key = _normalize_filename_cache_key(Path(filename).stem)
    if filename_key and normalized == filename_key:
        score += 8.0
    elif filename_key and normalized in filename_key:
        score += 3.0
    elif filename_key and filename_key in normalized:
        score += 3.0

    if len(normalized.split()) == 1 and normalized not in {"tsb", "stb", "intt", "cimf", "cni"}:
        score -= 3.0
    return score


def _organization_from_filename(filename: str) -> str | None:
    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", stem).strip()
    normalized = _normalize_filename_cache_key(cleaned)
    if not cleaned or not normalized:
        return None
    if any(marker == normalized or normalized.startswith(f"{marker} ") for marker in GENERIC_ORGANIZATION_FILENAME_MARKERS):
        return None
    return cleaned


def answer_organization_from_text_cache(filename: str, question: str) -> tuple[str, list[dict]] | None:
    if not _is_organization_identity_question(question):
        return None

    pages = _read_text_cache_pages(filename)
    scored: list[tuple[float, str, str]] = []
    for page in pages[:2]:
        lines = [line.strip() for line in page["text"].splitlines() if line.strip()]
        for index, line in enumerate(lines[:35]):
            candidate = _clean_organization_candidate(line)
            score = _organization_candidate_score(candidate, index, filename)
            if score > 0:
                scored.append((score, page["page"], candidate))

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        _score, page, name = scored[0]
    else:
        name = _organization_from_filename(filename)
        page = "1"
        if not name:
            return None

    language = _detect_answer_language(question)
    if language == "en":
        answer = f"The identified company/organization is: {name}.\nSource: {filename}, page {page}."
    else:
        answer = f"La societe / l'organisme identifie est : {name}.\nSource: {filename}, page {page}."
    return answer, [{"source": filename, "page": str(page), "section": "organization", "score": 1.0}]


MINED_FACT_STOPWORDS = {
    "a",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "dans",
    "de",
    "des",
    "du",
    "est",
    "et",
    "la",
    "le",
    "les",
    "l",
    "un",
    "une",
    "pour",
    "que",
    "quel",
    "quelle",
    "quels",
    "quelles",
    "sont",
    "sur",
}


def _mined_fact_items(facts: dict) -> list[dict]:
    mined = facts.get("mined_facts")
    if not isinstance(mined, dict):
        return []

    raw_items = mined.get("items")
    if isinstance(raw_items, list):
        return [item for item in raw_items if isinstance(item, dict)]
    return []


def _mined_query_tokens(text: str) -> set[str]:
    normalized = _normalize_query_text(str(text))
    tokens = set()
    for token in normalized.split():
        if token in MINED_FACT_STOPWORDS:
            continue
        if len(token) > 2 or token in {"pc", "id", "edr", "pam"}:
            tokens.add(token)
    return tokens


def _mined_fact_text(fact: dict) -> str:
    parts = [
        str(fact.get("type", "")),
        str(fact.get("label", "")),
        str(fact.get("value", "")),
        str(fact.get("quote", "")),
        str(fact.get("text", "")),
    ]
    return _normalize_query_text(" ".join(parts))


def _mined_fact_type_hint(question: str) -> str | None:
    normalized = _normalize_query_text(question)
    if any(marker in normalized for marker in ("bordereau", "prix", "element", "elements", "item", "poste")):
        return "pricing_item"
    if any(
        marker in normalized
        for marker in (
            "materiel",
            "materiels",
            "equipement",
            "equipements",
            "quantite",
            "quantites",
            "qte",
            "article",
            "articles",
            "produit",
            "produits",
            "licence",
            "licences",
        )
    ):
        return "requested_item"
    if any(
        marker in normalized
        for marker in (
            "caracteristique",
            "caracteristiques",
            "technique",
            "longueur",
            "orientation",
            "type de coupe",
            "bidirection",
            "securite",
            "pression",
            "table de coupe",
            "lame",
            "bac",
            "graduation",
        )
    ):
        return "technical_requirement"
    if any(marker in normalized for marker in ("endpoint", "endpoints", "administrateur", "administrateurs", "admin", "ressource", "ressources", "combien")):
        return "metric"
    padded = f" {normalized} "
    if " lot " in padded or " lots " in padded or normalized.startswith(("lot ", "lots ")):
        return "lot"
    return None


def _mined_question_lot_number(question: str) -> str | None:
    match = re.search(r"\blot\s*(\d+)\b", _normalize_query_text(question))
    return match.group(1) if match else None


def _mined_question_has_any(question: str, markers: tuple[str, ...]) -> bool:
    normalized = _normalize_query_text(question)
    return any(marker in normalized for marker in markers)


def _score_mined_fact(question: str, fact: dict) -> int:
    normalized_question = _normalize_query_text(question)
    fact_text = _mined_fact_text(fact)
    type_hint = _mined_fact_type_hint(question)
    lot_number = _mined_question_lot_number(question)
    score = len(_mined_query_tokens(question) & _mined_query_tokens(fact_text))

    if type_hint and fact.get("type") == type_hint:
        score += 4

    if lot_number:
        if f"lot {lot_number}" in fact_text:
            score += 5
        elif "lot " in fact_text:
            score -= 4

    if any(marker in normalized_question for marker in ("antivirus", "edr", "endpoint", "endpoints")):
        if any(marker in fact_text for marker in ("antivirus", "edr", "endpoint", "pc", "serveur", "mobile")):
            score += 3
    if "pam" in normalized_question and "pam" in fact_text:
        score += 4
    if any(marker in normalized_question for marker in ("administrateur", "administrateurs", "admin")):
        if any(marker in fact_text for marker in ("administrateur", "administrateurs", "admin")):
            score += 4
    if "ressource" in normalized_question or "ressources" in normalized_question:
        if "ressource" in fact_text or "ressources" in fact_text:
            score += 4

    return score


def _format_mined_fact_answer(source: str, title: str, items: list[dict]) -> tuple[str, list[dict]] | None:
    cleaned_items = []
    for item in items:
        text = str(item.get("text") or "").strip()
        if not text:
            label = str(item.get("label") or "").strip()
            value = str(item.get("value") or "").strip()
            text = f"{label} : {value}" if label and value else value
        text = re.sub(r"\s+", " ", text).strip(" .;:-")
        if text and text not in cleaned_items:
            cleaned_items.append(text)

    if not cleaned_items:
        return None

    lines = [f"{title} :"]
    lines.extend(f"- {item}" for item in cleaned_items[:10])

    metas = _fact_source_metas(source, {"items": items}, limit=4)
    pages = []
    for meta in metas:
        page = str(meta.get("page", "")).strip()
        if page and page != "?" and page not in pages:
            pages.append(page)

    if len(pages) == 1:
        lines.append(f"Source: {source}, page {pages[0]}.")
    elif pages:
        lines.append(f"Sources: {source}, pages {', '.join(pages)}.")

    return "\n".join(lines), metas


def _answer_from_mined_facts(source: str, question: str, facts: dict) -> tuple[str, list[dict]] | None:
    items = _mined_fact_items(facts)
    if not items:
        return None

    type_hint = _mined_fact_type_hint(question)
    lot_number = _mined_question_lot_number(question)
    if not type_hint:
        return None

    normalized_question = _normalize_query_text(question)
    selected: list[dict] = []

    if type_hint == "lot":
        lot_items = [
            item for item in items
            if item.get("type") == "lot"
            and (not lot_number or f"lot {lot_number}" in _mined_fact_text(item))
        ]
        seen_lots = set()
        for item in lot_items:
            label = _normalize_query_text(str(item.get("label", "")))
            if label in seen_lots:
                continue
            seen_lots.add(label)
            selected.append(item)
        title = "Lots"
    elif type_hint == "pricing_item":
        selected = [
            item for item in items
            if item.get("type") == "pricing_item"
            and (not lot_number or f"lot {lot_number}" in _mined_fact_text(item))
        ]
        if not selected:
            selected = [item for item in items if item.get("type") == "requested_item"]
        title = "Elements du bordereau"
    elif type_hint == "requested_item":
        selected = [item for item in items if item.get("type") == "requested_item"]
        title = "Materiel / quantites"
    elif type_hint == "technical_requirement":
        selected = [item for item in items if item.get("type") == "technical_requirement"]
        keyword_filters = (
            ("longueur", ("longueur",)),
            ("orientation", ("orientation",)),
            ("type de coupe", ("type de coupe",)),
            ("bidirection", ("bidirection",)),
            ("securite", ("securite", "sécurité", "protection", "lame")),
            ("pression", ("pression",)),
            ("table de coupe", ("table de coupe",)),
            ("bac", ("bac", "chutes")),
            ("graduation", ("graduation",)),
            ("lame", ("lame",)),
        )
        for question_marker, fact_markers in keyword_filters:
            if question_marker in normalized_question:
                filtered = [
                    item for item in selected
                    if any(
                        marker in _normalize_query_text(str(item.get("label", "")))
                        for marker in fact_markers
                    )
                ]
                if filtered:
                    selected = filtered
                    break
        title = "Caracteristiques techniques"
    else:
        selected = [item for item in items if item.get("type") == "metric"]
        if _mined_question_has_any(question, ("endpoint", "endpoints")):
            selected = [item for item in selected if "endpoint" in _mined_fact_text(item)]
        elif _mined_question_has_any(question, ("administrateur", "administrateurs", "admin")):
            selected = [
                item for item in selected
                if any(marker in _mined_fact_text(item) for marker in ("administrateur", "administrateurs", "admin"))
            ]
        elif _mined_question_has_any(question, ("ressource", "ressources")):
            selected = [
                item for item in selected
                if "ressource" in _mined_fact_text(item) or "ressources" in _mined_fact_text(item)
            ]

        if lot_number:
            lot_selected = [item for item in selected if f"lot {lot_number}" in _mined_fact_text(item)]
            if lot_selected:
                selected = lot_selected
        elif "pam" in normalized_question:
            pam_selected = [item for item in selected if "pam" in _mined_fact_text(item) or "lot 2" in _mined_fact_text(item)]
            if pam_selected:
                selected = pam_selected
        elif any(marker in normalized_question for marker in ("antivirus", "edr", "endpoint", "endpoints")):
            endpoint_selected = [
                item for item in selected
                if any(marker in _mined_fact_text(item) for marker in ("antivirus", "edr", "endpoint", "lot 1"))
            ]
            if endpoint_selected:
                selected = endpoint_selected

        title = "Donnees chiffrees"

    if not selected:
        scored = [
            (_score_mined_fact(question, item), item)
            for item in items
        ]
        selected = [item for score, item in sorted(scored, key=lambda pair: pair[0], reverse=True) if score >= 4]

    if not selected:
        return None

    return _format_mined_fact_answer(source, title, selected)


TENDER_CHECKLIST_ITEMS = (
    ("Quel est l'objet du cahier des charges / de la presente consultation ?", "subject", False),
    ("Quel est le mode d'envoi ou de depot de la soumission ?", "submission_method", False),
    ("Quelle est la date limite reelle de soumission ?", "deadline", False),
    ("Quelle est la duree de validite de l'offre ?", "validity", False),
    ("Quelle est la date / modalite d'ouverture des plis ?", "opening", False),
    ("Les variantes sont-elles autorisees ?", "variants", False),
    ("Quel est le montant / l'exigence de la caution provisoire ?", "caution", False),
    ("Une fiche de renseignements est-elle exigee ?", "information_sheet", True),
    ("Une attestation d'affiliation a la CNSS est-elle exigee ?", "cnss", True),
    ("Une attestation de solde / situation fiscale est-elle exigee ?", "fiscal_certificate", True),
    ("Un extrait du registre de commerce / certificat RNE est-il exige ?", "rne", True),
    ("Quels sont les documents administratifs exiges ?", "administrative_documents", False),
    ("Quelle documentation technique est exigee ?", "technical_documents", False),
    ("Une autorisation du constructeur / fabricant est-elle exigee ?", "manufacturer_authorization", True),
    ("Une liste de references est-elle exigee ?", "references", True),
    ("Quels sont les documents financiers exiges ?", "financial_documents", False),
    ("Quelle est la periode de garantie exigee ?", "guarantee", False),
    ("Quelles sont les modalites / types de reception ?", "reception", False),
    ("Une caution definitive est-elle exigee ?", "definitive_caution", True),
    ("Existe-t-il des penalites de retard ?", "penalties", True),
    ("Quelles sont les modalites de paiement ?", "payment", False),
)


TENDER_CHECKLIST_PROFILE_FIELDS = {
    "subject": "object",
    "submission_method": "submission_method",
    "deadline": "deadline",
    "validity": "validity",
    "opening": "opening",
    "variants": "variants",
    "caution": "provisional_caution",
    "information_sheet": "information_sheet",
    "cnss": "cnss_certificate",
    "fiscal_certificate": "fiscal_certificate",
    "rne": "rne_certificate",
    "administrative_documents": "administrative_documents",
    "technical_documents": "technical_documents",
    "manufacturer_authorization": "manufacturer_authorization",
    "references": "references",
    "financial_documents": "financial_documents",
    "guarantee": "guarantee",
    "reception": "reception",
    "definitive_caution": "definitive_caution",
    "penalties": "penalties",
    "payment": "payment",
}


def _clean_checklist_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", str(text))
    text = str(text)
    text = re.sub(r"\s*\|\s*", " ", text)
    text = re.sub(r"\b\d+\s*,\s*\d+\s*=", " ", text)
    text = re.sub(r"\b\d+\s*,\s*(?:D[ée]signations|Authentifications)\s*=", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:N[°o]\s*de\s+la\s*pi[èe]ce|D[ée]signations|Authentifications)\b\s*=?\.?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:Cachet\s+signature\s+du\s+soumissionnaire|signature\s+et\s+cachet\s+du\s+soumissionnaire)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:D[uû]ment|Dament)\s+sign[ée]\s+paraph[ée]\s+et\s+dat[ée]\s+par\s+soumissionnaire\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:REG\s+P|Fu\s+\d+)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .;:-")
    text = re.sub(r"\bLa(?=certification|documentation|liste|lettre|caution|date|livraison)", "La ", text)
    text = re.sub(r"\bselonlemod[eè]lejointenannexe\b", "selon le modèle joint en annexe", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmod[eè]lejointenannexe\b", "modèle joint en annexe", text, flags=re.IGNORECASE)
    text = re.sub(r"\benannexe\b", "en annexe", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:à partir de|a partir de|à partir|a partir|à l'adresse suivante|a l'adresse suivante)\s*:?\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:a|à|de|d|du|des|par|pour|avec|selon|suivant)\s*$", "", text, flags=re.IGNORECASE)
    if text.count("(") > text.count(")"):
        text = text.rstrip(" (")
        if text.count("(") > text.count(")"):
            text += ")"
    return text.strip(" .;:-")


def _checklist_item_key(text: str) -> str:
    normalized = _normalize_query_text(text)
    normalized = re.sub(r"\bla(?=certification|documentation|liste|lettre|caution|date|livraison)", "la ", normalized)
    normalized = re.sub(r"\b(?:la|le|les|un|une|des|de|du|d|et|a|à)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:120]


def _is_noisy_checklist_item(text: str) -> bool:
    normalized = _normalize_query_text(text)
    if len(normalized) < 8:
        return True
    noise_markers = (
        "authentifications",
        "designations",
        "cachet signature",
        "soumissionnaire reg p",
        "fu 3",
    )
    if any(marker in normalized for marker in noise_markers):
        return True
    table_symbol_count = text.count("|") + text.count("=")
    return table_symbol_count >= 3


def _format_checklist_items(raw_items: list, field: str) -> str | None:
    cleaned_items = []
    seen = set()

    for item in raw_items:
        if not isinstance(item, dict):
            continue
        text = _clean_checklist_text(item.get("text", ""))
        if not text or _is_noisy_checklist_item(text):
            continue
        key = _checklist_item_key(text)
        if key in seen:
            continue
        seen.add(key)
        cleaned_items.append(text)

    if not cleaned_items:
        return None

    if field in {"administrative_documents", "technical_documents", "financial_documents"}:
        cleaned_items = cleaned_items[:10]

    return "\n      - " + "\n      - ".join(cleaned_items)


def _join_checklist_phrases(phrases: list[str]) -> str:
    if len(phrases) <= 1:
        return phrases[0] if phrases else ""
    return ", ".join(phrases[:-1]) + " puis " + phrases[-1]


def _unique_checklist_phrases(phrases: list[str]) -> list[str]:
    unique = []
    seen = set()
    for phrase in phrases:
        key = _normalize_query_text(phrase)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(phrase)
    return unique


def _summarize_reception_checklist_text(value: str) -> str:
    folded = _normalize_query_text(value)
    if len(value) < 350 or "reception" not in folded:
        return value

    reception_types = []
    if "reception quantitative" in folded:
        reception_types.append("reception quantitative")
    if "reception provisoire" in folded:
        reception_types.append("reception provisoire")
    if "reception definitive" in folded:
        reception_types.append("reception definitive")

    reception_types = _unique_checklist_phrases(reception_types)
    if len(reception_types) < 2:
        return value

    details = []
    if "livraison" in folded:
        details.append("apres livraison")
    if any(marker in folded for marker in ("verification", "conformite", "specifications techniques", "tests")):
        details.append("apres verification de conformite")
    if "delai de garantie" in folded or "expiration du delai" in folded:
        details.append("a l'expiration du delai de garantie")

    summary = f"Modalites de reception: {_join_checklist_phrases(reception_types)}"
    details = _unique_checklist_phrases(details)
    if details:
        summary += f" ({'; '.join(details)})"
    return summary + "."


def _extract_checklist_durations(value: str) -> list[str]:
    duration_pattern = re.compile(
        r"\b(?:\d{1,3}|un|une|deux|trois|six|douze|vingt\s+quatre|trente\s+six)"
        r"(?:\s*\(\s*\d{1,3}\s*\))?\s*(?:mois|ans?|ann[ée]es?)\b",
        re.IGNORECASE,
    )
    durations = []
    for match in duration_pattern.finditer(value):
        duration = _clean_checklist_text(match.group(0))
        if duration:
            durations.append(duration)
    return _unique_checklist_phrases(durations)


def _summarize_guarantee_checklist_text(value: str) -> str:
    folded = _normalize_query_text(value)
    if len(value) < 260 or "garantie" not in folded:
        return value

    parts = []
    has_six_months = "6 mois" in folded or "six 6 mois" in folded
    has_two_years = "2 ans" in folded or "deux 02 ans" in folded or "deux ans" in folded
    has_three_years = "3 annees" in folded or "trois 03 annees" in folded or "trois annees" in folded or "3 ans" in folded

    if ("lots 1 et 2" in folded or "lot 1 et 2" in folded) and has_six_months:
        parts.append("6 mois pour les lots 1 et 2")
    if "lot 3" in folded:
        if ("casques" in folded or "souris" in folded) and has_two_years:
            parts.append("2 ans pour les casques et souris sans fil")
        if "douchettes" in folded and has_three_years:
            parts.append("3 ans pour les douchettes")

    parts = _unique_checklist_phrases(parts)
    if parts:
        return "Garantie: " + "; ".join(parts) + "."

    durations = _extract_checklist_durations(value)
    if durations:
        return "Garantie: " + "; ".join(durations[:4]) + "."
    return value


def _polish_checklist_fact_value(field: str, value: str) -> str:
    if field == "reception":
        return _summarize_reception_checklist_text(value)
    if field == "guarantee":
        return _summarize_guarantee_checklist_text(value)
    return value


def _profile_fact_for_checklist(facts: dict, field: str) -> dict | None:
    profile = facts.get("tender_profile")
    if not isinstance(profile, dict):
        return None

    profile_fields = profile.get("fields")
    if not isinstance(profile_fields, dict):
        return None

    profile_key = TENDER_CHECKLIST_PROFILE_FIELDS.get(field, field)
    profile_value = profile_fields.get(profile_key)
    if not isinstance(profile_value, dict):
        return None

    text = profile_value.get("text")
    items = profile_value.get("items")
    if not text and not items:
        return None

    fact = {
        "text": text or "",
        "page": profile_value.get("page"),
        "section": profile_value.get("section") or field,
    }
    if isinstance(items, list) and items:
        fact["items"] = items
    return fact


def _fact_for_answer(facts: dict, field: str) -> dict | None:
    fact = _profile_fact_for_checklist(facts, field) or facts.get(field)
    return fact if isinstance(fact, dict) else None


def _format_checklist_fact_value(field: str, fact: dict, presence_question: bool) -> str | None:
    raw_items = fact.get("items")
    if isinstance(raw_items, list) and raw_items:
        value = _format_checklist_items(raw_items, field)
        if not value:
            return None
    else:
        value = _clean_checklist_text(fact.get("text", ""))
        if not value:
            return None
        if field in {"administrative_documents", "technical_documents", "financial_documents"}:
            split_items = [
                {"text": item}
                for item in re.split(r"\s+-\s+|(?:\.\s+)(?=(?:La|Le|Les|Une|Un|L'|l'|RNE|CNSS)\b)", value)
            ]
            formatted_items = _format_checklist_items(split_items, field)
            if formatted_items:
                value = formatted_items

    value = _polish_checklist_fact_value(field, value)
    normalized_value = _normalize_query_text(value)
    if field == "variants":
        if any(marker in normalized_value for marker in ("ne sont pas", "pas autorise", "interdite", "non autorise", "sans variante")):
            return f"Non - {value}"
        return f"Oui / a verifier - {value}"

    if presence_question:
        return f"Oui - {value}"

    return value


def build_tender_checklist_answer(filename: str, facts: dict | None) -> str:
    facts = facts or {}
    lines = [
        "Analyse de consultation",
        f"Document: {filename}",
        "",
    ]
    extraction_warning = facts.get("extraction_warning")
    if isinstance(extraction_warning, dict):
        warning_text = str(extraction_warning.get("text") or "").strip()
        if warning_text:
            lines.extend([
                "Avertissement extraction :",
                f"   {warning_text}",
                "",
            ])

    for index, (question, field, presence_question) in enumerate(TENDER_CHECKLIST_ITEMS, start=1):
        fact = _profile_fact_for_checklist(facts, field) or facts.get(field)
        answer = None
        source_ref = None
        if isinstance(fact, dict):
            answer = _format_checklist_fact_value(field, fact, presence_question)
            source_ref = _fact_source_reference(filename, fact)

        lines.append(f"{index}. {question}")
        if answer:
            lines.append(f"   Reponse: {answer}")
            if source_ref:
                lines.append(f"   {source_ref}")
        else:
            lines.append("   Reponse: Non mentionne dans ce document.")
        lines.append("")

    return "\n".join(lines).strip()


def analyze_tender_checklist_document(doc: Document) -> str:
    return build_tender_checklist_answer(doc.filename, doc.extracted_facts or {})


def _build_summary_from_facts(source: str, facts: dict) -> tuple[str, list[dict]] | None:
    summary_fact = facts.get("summary")
    if summary_fact:
        answer = _format_fact_answer(source, summary_fact)
        if answer:
            return answer, [_fact_to_source_meta(source, summary_fact)]

    subject_fact = facts.get("subject")
    deadline_fact = facts.get("deadline")
    if not subject_fact and not deadline_fact:
        return None

    parts = []
    sources = []
    if subject_fact:
        parts.append(re.sub(r"\s+", " ", str(subject_fact.get("text", ""))).strip(" .;:-"))
        sources.append(_fact_to_source_meta(source, subject_fact))
    if deadline_fact:
        deadline_text = re.sub(r"\s+", " ", str(deadline_fact.get("text", ""))).strip(" .;:-")
        if deadline_text:
            parts.append(f"Date limite : {deadline_text}.")
            sources.append(_fact_to_source_meta(source, deadline_fact))

    if not parts:
        return None

    page = None
    if subject_fact:
        page = subject_fact.get("page")
    elif deadline_fact:
        page = deadline_fact.get("page")

    answer = " ".join(parts).strip()
    if page:
        answer = f"{answer}\nSource: {source}, page {page}."
    return answer, sources[:2]


async def answer_from_document_facts(
    *,
    db: AsyncSession | None,
    question: str,
    source_filter: list[str] | None,
    department_filter: list[str] | None,
    universe_id: str | None,
    user_id: str | None,
    is_admin: bool,
    strict_missing: bool = False,
) -> tuple[str, list[dict]] | None:
    if db is None or not source_filter or len(source_filter) != 1:
        return None

    filename = source_filter[0]
    stmt = select(Document).where(
        Document.filename == filename,
        Document.status == "indexed",
    )

    if universe_id is None:
        stmt = stmt.where(Document.universe_id.is_(None))
    else:
        stmt = stmt.where(Document.universe_id == universe_id)

    if not is_admin:
        department_ids = department_filter or []
        dept_clause = (
            and_(Document.visibility == "department", Document.department_id.in_(department_ids))
            if department_ids
            else None
        )
        private_clause = and_(Document.visibility == "private", Document.uploaded_by == user_id)
        if dept_clause is not None:
            stmt = stmt.where(or_(dept_clause, private_clause))
        else:
            stmt = stmt.where(private_clause)

    stmt = stmt.order_by(Document.created_at.desc()).limit(1)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc or not doc.extracted_facts:
        return None

    facts = doc.extracted_facts or {}

    chatbot_answer = answer_chatbot_identity(question)
    if chatbot_answer:
        return chatbot_answer

    organization_answer = answer_organization_from_text_cache(filename, question)
    if organization_answer:
        return organization_answer

    if _is_subject_question(question):
        subject_fact = _fact_for_answer(facts, "subject")
        if subject_fact:
            answer = _format_fact_answer(filename, subject_fact)
            if answer:
                return answer, [_fact_to_source_meta(filename, subject_fact)]

    if _is_deadline_question(question):
        deadline_fact = _fact_for_answer(facts, "deadline")
        if deadline_fact:
            answer = _format_fact_answer(filename, deadline_fact)
            if answer:
                return answer, [_fact_to_source_meta(filename, deadline_fact)]

    list_field = _fact_list_field_for_question(question)
    if list_field:
        list_fact = _fact_for_answer(facts, list_field)
        if list_fact:
            answer = _format_list_fact_answer(
                filename,
                FACT_LIST_FIELD_LABELS[list_field],
                list_fact,
            )
            if answer:
                return answer, _fact_source_metas(filename, list_fact)

    scalar_field = _fact_scalar_field_for_question(question)
    if scalar_field:
        scalar_fact = _fact_for_answer(facts, scalar_field)
        if scalar_fact:
            answer = _format_labeled_fact_answer(
                filename,
                FACT_FIELD_LABELS[scalar_field],
                scalar_fact,
            )
            if answer:
                return answer, [_fact_to_source_meta(filename, scalar_fact)]

    mined_answer = _answer_from_mined_facts(filename, question, facts)
    if mined_answer:
        return mined_answer

    if _is_summary_question(question):
        summary_answer = _build_summary_from_facts(filename, facts)
        if summary_answer:
            return summary_answer

    cache_answer = answer_from_text_cache(filename, question)
    if cache_answer:
        return cache_answer

    if strict_missing:
        return _missing_answer_text(question), []

    return None


def _build_extraction_messages(
    question: str,
    context: str,
    answer_hint: str,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    if _is_summary_question(question):
        return _build_summary_messages(question, context, system_prompt=system_prompt)

    missing_text = _missing_answer_text(question)
    language_rule = _language_instruction(question)

    extraction_rules = f"""Tu es un extracteur d'information specialise dans les cahiers des charges tunisiens.
TON SEUL TRAVAIL: extraire l'information exacte demandee depuis le contexte fourni.

REGLES ABSOLUES:
1. Utilise uniquement le contexte fourni.
2. {language_rule}
3. Ne produis jamais de balises <think>, de raisonnement, d'analyse intermediaire, ni de preambule.
4. Si l'information est dans le contexte, donne-la de facon concise et exacte.
5. Cite la source en mentionnant le fichier et la page.
6. Si l'information n'est pas presente, reponds uniquement: "{missing_text}"
7. Ne commente jamais la langue demandee par l'utilisateur. Ne dis jamais "The user requested..." ou une phrase equivalente.
8. Format maximum: 3 lignes courtes.

CONSIGNE SPECIFIQUE:
{answer_hint or "Aucune"}"""

    user_prompt = f"""CONTEXTE:
{context}

QUESTION:
{question}

REPONSE FINALE:"""

    if system_prompt:
        return [
            {"role": "system", "content": f"{system_prompt}\n\n{extraction_rules}"},
            {"role": "user", "content": user_prompt},
        ]

    return [
        {"role": "system", "content": extraction_rules},
        {"role": "user", "content": user_prompt},
    ]


def _build_summary_messages(
    question: str,
    context: str,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    language_rule = _language_instruction(question)

    summary_rules = f"""Tu resumes un document a partir du contexte fourni.

REGLES ABSOLUES:
1. Utilise uniquement le contexte fourni.
2. {language_rule}
3. Ne produis jamais de balises <think>, de raisonnement, d'analyse intermediaire, ni de preambule.
4. Donne un resume bref et factuel des points les plus importants.
5. N'invente aucune information qui n'apparait pas dans le contexte.
6. Si le contexte est insuffisant, resume uniquement ce qui est visible au lieu de repondre que ce n'est pas mentionne.
7. Ne commente jamais la langue demandee par l'utilisateur. Ne dis jamais "The user requested..." ou une phrase equivalente.
8. Format maximum: 4 lignes courtes.
9. Termine par une citation de source avec le fichier et la page."""

    user_prompt = f"""CONTEXTE:
{context}

QUESTION:
{question}

RESUME FACTUEL:"""

    if system_prompt:
        return [
            {"role": "system", "content": f"{system_prompt}\n\n{summary_rules}"},
            {"role": "user", "content": user_prompt},
        ]

    return [
        {"role": "system", "content": summary_rules},
        {"role": "user", "content": user_prompt},
    ]


class _StreamReasoningStripper:
    def __init__(self) -> None:
        self._buffer = ""
        self._inside_think = False

    def feed(self, text: str) -> str:
        self._buffer += text
        output: list[str] = []

        while self._buffer:
            lower_buffer = self._buffer.lower()
            if self._inside_think:
                end_idx = lower_buffer.find("</think>")
                if end_idx == -1:
                    self._buffer = self._buffer[-8:]
                    break
                self._buffer = self._buffer[end_idx + len("</think>"):]
                self._inside_think = False
                continue

            start_idx = lower_buffer.find("<think>")
            if start_idx == -1:
                safe_len = max(0, len(self._buffer) - 6)
                if safe_len:
                    output.append(self._buffer[:safe_len])
                    self._buffer = self._buffer[safe_len:]
                break

            if start_idx > 0:
                output.append(self._buffer[:start_idx])
            self._buffer = self._buffer[start_idx + len("<think>"):]
            self._inside_think = True

        return "".join(output)

    def flush(self) -> str:
        if self._inside_think:
            return ""
        remaining = self._buffer
        self._buffer = ""
        return remaining


def _rerank_sort_key(rerank_score: float, meta: dict, *, apply_focus: bool) -> tuple[float, float, float]:
    retrieval_score = meta.get("retrieval_score", meta.get("score", 0.0))
    focus_bonus = meta.get("focus_bonus", 0.0) if apply_focus else 0.0
    return (rerank_score + focus_bonus, focus_bonus, retrieval_score)


def _focused_chunk_excerpt(rules: list[dict], chunk: str, window: int = 220) -> str:
    if not rules:
        return chunk

    chunk_lower = chunk.lower()
    normalized_chunk = _normalize_query_text(chunk)
    terms = []

    for rule in rules:
        terms.extend(rule["primary_terms"])
    for rule in rules:
        terms.extend(rule["support_terms"])

    match_index = -1
    match_length = 0
    for term in terms:
        term_lower = term.lower()
        match_index = chunk_lower.find(term_lower)
        if match_index == -1:
            match_index = normalized_chunk.find(_normalize_query_text(term))
        if match_index != -1:
            match_length = len(term)
            break

    if match_index == -1:
        return chunk

    start = max(0, match_index - window)
    end = min(len(chunk), match_index + max(match_length, 1) + window)
    excerpt = chunk[start:end].strip()

    if start > 0:
        excerpt = "..." + excerpt
    if end < len(chunk):
        excerpt = excerpt + "..."

    return excerpt


def _strip_reasoning_markup(text: str) -> str:
    """Remove hidden reasoning blocks from models that emit <think> tags."""
    cleaned = THINK_BLOCK_RE.sub("", text).strip()
    return cleaned


async def retrieve(
    query: str,
    k: int = 6,
    source_filter: list[str] | None = None,
    department_filter: list[str] | None = None,
    universe_id: str | None = None,
    user_id: str | None = None,
    is_admin: bool = True,
) -> tuple[list[str], list[dict]]:
    """
    Retrieve relevant chunks from Qdrant, then rerank.
    Returns (chunks, source_metas).
    
    department_filter is injected by the auth middleware — never by the user.
    """
    vs = _get_vector_store()
    scoped_to_single_source = len(source_filter or []) == 1
    effective_k = min(k, 2) if scoped_to_single_source else k

    section_hints = infer_section_hints(query)
    focus_rules = _matching_focus_rules(query)
    enhanced_query = enhance_query(query)

    t0 = time.perf_counter()
    query_vec = await get_embedding(enhanced_query)
    embed_ms = int((time.perf_counter() - t0) * 1000)
    logger.debug(f"Query embedding took {embed_ms}ms")

    # ── Qdrant dense search (with department isolation) ───────────────
    t0 = time.perf_counter()
    results = []
    seen = set()
    if scoped_to_single_source and _is_subject_question(query):
        search_plan = [(section, 8) for section in section_hints[:1]] + [(None, 24)]
    elif scoped_to_single_source and _is_deadline_question(query):
        search_plan = [(section, 8) for section in section_hints[:1]] + [(None, 20)]
    elif scoped_to_single_source:
        search_plan = [(section, 4) for section in section_hints[:1]] + [(None, 8)]
    else:
        search_plan = [(section, 10) for section in section_hints[:2]] + [(None, 25)]

    for section_hint, search_k in search_plan:
        partial_results = await vs.search(
            query_embedding=query_vec,
            k=search_k,
            source_filter=source_filter,
            section_filter=section_hint,
            department_filter=department_filter,
            universe_id=universe_id,
            user_id=user_id,
            is_admin=is_admin,
        )
        for row in partial_results:
            dedupe_key = (row["source"], row["page"], row["text"])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            results.append(row)

    qdrant_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(f"Qdrant returned {len(results)} results in {qdrant_ms}ms (dept_filter={department_filter}, section_hints={section_hints})")

    raw_chunks = [r["text"] for r in results]
    raw_metas = [{"source": r["source"], "page": r["page"], "section": r["section"], "score": r["score"]} for r in results]

    for chunk, meta in zip(raw_chunks, raw_metas):
        focus_bonus = _focus_bonus(focus_rules, chunk)
        page_bonus = _page_position_bonus(query, meta.get("page"), chunk)
        meta["focus_bonus"] = focus_bonus + page_bonus
        meta["retrieval_score"] = meta["score"] + focus_bonus + page_bonus

    if not raw_chunks:
        return [], []

    # ── Rerank ────────────────────────────────────────────────────────
    if len(raw_chunks) <= effective_k:
        logger.debug("Skipping reranker because {} candidate chunks <= requested k={}", len(raw_chunks), effective_k)
        ordered = sorted(
            zip(raw_chunks, raw_metas),
            key=lambda item: item[1].get("retrieval_score", item[1].get("score", 0.0)),
            reverse=True,
        )
        top = ordered[:effective_k]
        return [chunk for chunk, _ in top], [meta for _, meta in top]

    if scoped_to_single_source:
        logger.debug("Skipping reranker for single-source query and returning boosted retrieval ranking")
        ordered = sorted(
            zip(raw_chunks, raw_metas),
            key=lambda item: item[1].get("retrieval_score", item[1].get("score", 0.0)),
            reverse=True,
        )
        top = ordered[:effective_k]
        return [chunk for chunk, _ in top], [meta for _, meta in top]

    reranker = _get_reranker()
    if reranker and not scoped_to_single_source:
        t0 = time.perf_counter()
        pairs = [[query, chunk] for chunk in raw_chunks]
        scores = reranker.predict(pairs)
        rerank_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(f"Reranker scored {len(pairs)} pairs in {rerank_ms}ms")

        max_score = max(scores) if len(scores) > 0 else 0
        if max_score < 0.01:
            scored = sorted(zip(scores, raw_chunks, raw_metas),
                            key=lambda x: x[2].get("retrieval_score", x[2].get("score", 0)), reverse=True)
        else:
            scored = sorted(zip(scores, raw_chunks, raw_metas),
                            key=lambda x: _rerank_sort_key(x[0], x[2], apply_focus=bool(focus_rules)), reverse=True)

        top = scored[:effective_k]
        return [c for _, c, _ in top], [m for _, _, m in top]
    else:
        logger.debug("No reranker available - returning raw ranked results")
        return raw_chunks[:effective_k], raw_metas[:effective_k]


async def ask_llm(question: str, chunks: list[str], metas: list[dict] | None = None,
                  system_prompt: str | None = None) -> str:
    """Generate answer using LLM with hallucination detection.
    
    When metas is provided, each chunk is prefixed with its source
    attribution so the LLM can cite specific documents and pages.
    """
    if False and not chunks:
        return "⚠️ Non mentionné dans ce document."

    if len(chunks) == 0:
        return _missing_answer_text(question)

    direct_answer = _extract_direct_answer(question, chunks, metas)
    if direct_answer:
        logger.info("Direct answer extraction matched for question: {}", question[:80])
        return direct_answer

    summary_answer = _extract_summary_answer(question, chunks, metas)
    if summary_answer:
        logger.info("Direct summary extraction matched for question: {}", question[:80])
        return summary_answer

    focus_rules = _matching_focus_rules(question)

    # ── Contextual enrichment (Task 1.1) ──────────────────────────────
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        focused_chunk = _focused_chunk_excerpt(focus_rules, chunk)
        if metas and i < len(metas):
            m = metas[i]
            source = m.get("source", "inconnu")
            page = m.get("page", "?")
            section = m.get("section", "general")
            header = f"[Source: {source}, Page: {page}, Section: {section}]"
            enriched_chunks.append(f"{header}\n{focused_chunk}")
        else:
            enriched_chunks.append(focused_chunk)

    context = "\n\n---\n\n".join(enriched_chunks)
    answer_hint = _answer_hint(question)

    messages = _build_extraction_messages(question, context, answer_hint, system_prompt=system_prompt)

    try:
        t0 = time.perf_counter()
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
                extra_body={"reasoning_effort": settings.LLM_REASONING_EFFORT},
            ),
            timeout=float(settings.LLM_TIMEOUT_SECONDS)
        )
        llm_ms = int((time.perf_counter() - t0) * 1000)
        answer = _strip_reasoning_markup(response.choices[0].message.content or "").strip()
        if not answer:
            answer = _missing_answer_text(question)
        logger.info(f"LLM ({settings.LLM_MODEL}) responded in {llm_ms}ms - {len(answer)} chars")
    except asyncio.TimeoutError:
        logger.error("LLM timeout (> {}s)", settings.LLM_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail="Timeout du LLM")
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"Erreur LLM: {str(e)}"

    answer_lower = answer.lower()
    if any(signal.lower() in answer_lower for signal in HALLUCINATION_SIGNALS):
        logger.warning("Hallucination detected - answer suppressed (matched signal in response)")
        return _missing_answer_text(question)

    return answer

    if system_prompt:
        # Use persona-based prompting when a system prompt is provided
        user_prompt = f"""CONTEXTE:
{context}

QUESTION: {question}

RÉPONSE DIRECTE (extraite du contexte uniquement, avec citation de source):"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        # Legacy: single-message prompt (backward compatible)
        prompt = f"""Tu es un extracteur d'information spécialisé dans les cahiers des charges tunisiens.
TON SEUL TRAVAIL: extraire l'information EXACTE demandée depuis le contexte fourni.

RÈGLES ABSOLUES — aucune exception:
1. Si l'information est dans le contexte → donne-la EXACTEMENT telle qu'elle est écrite.
2. CITE la source: mentionne le fichier et la page (ex: "CDC_STEG.pdf, page 4").
3. Si l'information N'EST PAS dans le contexte → réponds UNIQUEMENT: "Non mentionné dans ce document."
4. FORMAT: 1 à 4 lignes maximum. Pas d'introduction. Pas de conclusion.
5. Ignore les numéros d'article, de page, de lot, d'annexe, de compte bancaire ou les numéros isolés s'ils ne sont pas la réponse exacte demandée.
6. Si le contexte donne seulement une règle relative (par ex. "le même jour que la date limite"), réponds avec cette règle exacte au lieu d'inventer une date.

CONSIGNE SPÉCIFIQUE:
{answer_hint or "Aucune"}

CONTEXTE:
{context}

QUESTION: {question}

RÉPONSE DIRECTE (extraite du contexte uniquement, avec citation de source):"""
        messages = [{"role": "user", "content": prompt}]

    try:
        t0 = time.perf_counter()
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
                extra_body={"reasoning_effort": settings.LLM_REASONING_EFFORT},
            ),
            timeout=float(settings.LLM_TIMEOUT_SECONDS)
        )
        llm_ms = int((time.perf_counter() - t0) * 1000)
        answer = _strip_reasoning_markup(response.choices[0].message.content or "")
        logger.info(f"LLM ({settings.LLM_MODEL}) responded in {llm_ms}ms — {len(answer)} chars")
    except asyncio.TimeoutError:
        logger.error("LLM timeout (> {}s)", settings.LLM_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail="Timeout du LLM")
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"⚠️ Erreur LLM: {str(e)}"

    # Hallucination guard
    answer_lower = answer.lower()
    if any(signal.lower() in answer_lower for signal in HALLUCINATION_SIGNALS):
        logger.warning(f"Hallucination detected — answer suppressed (matched signal in response)")
        return "⚠️ Non mentionné dans ce document."

    return answer


# ── Public API ────────────────────────────────────────────────────────────

async def stream_llm_answer(
    question: str,
    chunks: list[str],
    metas: list[dict] | None = None,
    system_prompt: str | None = None,
) -> AsyncIterator[str]:
    """Stream answer tokens from the LLM for chat UIs."""
    if not chunks:
        yield _missing_answer_text(question)
        return

    direct_answer = _extract_direct_answer(question, chunks, metas)
    if direct_answer:
        logger.info("Direct answer extraction matched for streamed question: {}", question[:80])
        yield direct_answer
        return

    summary_answer = _extract_summary_answer(question, chunks, metas)
    if summary_answer:
        logger.info("Direct summary extraction matched for streamed question: {}", question[:80])
        yield summary_answer
        return

    focus_rules = _matching_focus_rules(question)
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        focused_chunk = _focused_chunk_excerpt(focus_rules, chunk)
        if metas and i < len(metas):
            m = metas[i]
            source = m.get("source", "inconnu")
            page = m.get("page", "?")
            section = m.get("section", "general")
            header = f"[Source: {source}, Page: {page}, Section: {section}]"
            enriched_chunks.append(f"{header}\n{focused_chunk}")
        else:
            enriched_chunks.append(focused_chunk)

    context = "\n\n---\n\n".join(enriched_chunks)
    answer_hint = _answer_hint(question)

    messages = _build_extraction_messages(question, context, answer_hint, system_prompt=system_prompt)
    stripper = _StreamReasoningStripper()
    answer_parts: list[str] = []

    stream = await client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=0.0,
        max_completion_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        extra_body={"reasoning_effort": settings.LLM_REASONING_EFFORT},
        stream=True,
    )

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        cleaned = stripper.feed(delta)
        if cleaned:
            answer_parts.append(cleaned)

    tail = stripper.flush().strip()
    if tail:
        answer_parts.append(tail)

    answer = _strip_reasoning_markup("".join(answer_parts)).strip()
    answer_lower = answer.lower()
    if not answer:
        yield _missing_answer_text(question)
        return
    if any(signal.lower() in answer_lower for signal in HALLUCINATION_SIGNALS):
        logger.warning("Hallucination/meta answer detected in streamed response - answer suppressed")
        yield _missing_answer_text(question)
        return
    yield answer
    return

    if system_prompt:
        user_prompt = f"""CONTEXTE:
{context}

QUESTION: {question}

RÇ%PONSE DIRECTE (extraite du contexte uniquement, avec citation de source):"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        prompt = f"""Tu es un extracteur d'information spÇ¸cialisÇ¸ dans les cahiers des charges tunisiens.
TON SEUL TRAVAIL: extraire l'information EXACTE demandÇ¸e depuis le contexte fourni.

RÇ^GLES ABSOLUES ƒ?" aucune exception:
1. Si l'information est dans le contexte ƒÅ' donne-la EXACTEMENT telle qu'elle est Ç¸crite.
2. CITE la source: mentionne le fichier et la page (ex: "CDC_STEG.pdf, page 4").
3. Si l'information N'EST PAS dans le contexte ƒÅ' rÇ¸ponds UNIQUEMENT: "Non mentionnÇ¸ dans ce document."
4. FORMAT: 1 Çÿ 4 lignes maximum. Pas d'introduction. Pas de conclusion.
5. Ignore les numÇ¸ros d'article, de page, de lot, d'annexe, de compte bancaire ou les numÇ¸ros isolÇ¸s s'ils ne sont pas la rÇ¸ponse exacte demandÇ¸e.
6. Si le contexte donne seulement une rÇùgle relative (par ex. "le mÇ¦me jour que la date limite"), rÇ¸ponds avec cette rÇùgle exacte au lieu d'inventer une date.

CONSIGNE SPÇ%CIFIQUE:
{answer_hint or "Aucune"}

CONTEXTE:
{context}

QUESTION: {question}

RÇ%PONSE DIRECTE (extraite du contexte uniquement, avec citation de source):"""
        messages = [{"role": "user", "content": prompt}]

    stream = await client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=0.0,
        max_completion_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        extra_body={"reasoning_effort": settings.LLM_REASONING_EFFORT},
        stream=True,
    )

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        cleaned = THINK_BLOCK_RE.sub("", delta)
        if cleaned:
            yield cleaned


async def rag_query(
    question: str,
    source_filter: list[str] | None = None,
    department_filter: list[str] | None = None,
    universe_id: str | None = None,
    user_id: str | None = None,
    is_admin: bool = True,
    db: AsyncSession | None = None,
    universe_department_id: str | None = None,
    universe_description: str = "",
    k: int = 6,
) -> tuple[str, list[dict]]:
    """
    Full RAG pipeline: retrieve → rerank → generate.
    Called by the query router.
    """
    pipeline_start = time.perf_counter()

    chatbot_answer = answer_chatbot_identity(question)
    if chatbot_answer:
        answer, metas = chatbot_answer
        total_ms = int((time.perf_counter() - pipeline_start) * 1000)
        logger.info("Chatbot identity answer resolved in {}ms", total_ms)
        return answer, metas

    fact_answer = await answer_from_document_facts(
        db=db,
        question=question,
        source_filter=source_filter,
        department_filter=department_filter,
        universe_id=universe_id,
        user_id=user_id,
        is_admin=is_admin,
        strict_missing=len(source_filter or []) == 1,
    )
    if fact_answer:
        answer, metas = fact_answer
        total_ms = int((time.perf_counter() - pipeline_start) * 1000)
        logger.info("Facts-first answer resolved in {}ms - {}", total_ms, question[:80])
        return answer, metas

    chunks, metas = await retrieve(
        query=question,
        k=k,
        source_filter=source_filter,
        department_filter=department_filter,
        universe_id=universe_id,
        user_id=user_id,
        is_admin=is_admin,
    )

    if not chunks:
        logger.info(f"No chunks found for query: {question[:80]}...")
        return "Aucune information trouvée dans les documents autorisés.", []

    # Build persona system prompt if universe context is available
    system_prompt = None
    if universe_department_id:
        from api.services.prompts import get_system_prompt
        system_prompt = get_system_prompt(universe_department_id, universe_description)

    answer = await ask_llm(question, chunks, metas, system_prompt=system_prompt)

    total_ms = int((time.perf_counter() - pipeline_start) * 1000)
    logger.info(f"RAG pipeline complete in {total_ms}ms — {len(chunks)} chunks → {len(answer)} char answer")

    return answer, metas


async def analyze_document(doc_id: str, analysis_type: str, prompt: str | None = None) -> str:
    """
    Analyze an entire document by pulling all its chunks sorted by chunk_index.
    Bypasses semantic search to look at the document holistically.
    """
    vs = _get_vector_store()
    
    # 1. Pull all chunks from Vector Store for this doc_id
    chunks_meta = await vs.get_document_chunks(doc_id)
    if not chunks_meta:
        return "⚠️ Aucun contenu trouvé pour ce document dans la base vectorielle."
        
    chunks = [c["text"] for c in chunks_meta]
    
    # Provide the context of all chunks
    context = "\n\n---\n\n".join(chunks)
    
    # Map-reduce / batching logic could be added here if context window overflows
    # Assume the DGX-hosted model has a large enough context window for batching.
    
    analysis_prompts = {
        "summary": "Résume ce document de manière concise en extrayant les points clés et l'objet principal.",
        "risks": "Identifie tous les risques (juridiques, techniques ou financiers) mentionnés dans ce document.",
        "deadlines": "Extrait toutes les échéances, dates limites et durées mentionnées.",
        "financials": "Extrait les aspects financiers : montants, pénalités, conditions de paiement, garanties financières.",
        "action_items": "Liste toutes les actions requises, livrables attendus et responsabilités du prestataire."
    }
    
    task_description = prompt if prompt else analysis_prompts.get(analysis_type, analysis_prompts["summary"])
    
    system_prompt = f"""Tu es un analyste expert de documents d'entreprise.
TON SEUL TRAVAIL: Analyser le document fourni et répondre à la demande de l'utilisateur.

RÈGLES ABSOLUES:
1. Base ton analyse UNIQUEMENT sur le contexte fourni. Ne génère pas d'informations externes.
2. Structure ta réponse avec du Markdown (titres, listes à puces, gras) pour la lisibilité.
3. Ne fais pas d'introduction inutile ("Voici l'analyse...").
4. Si l'information n'est pas présente dans le document, dis-le explicitement.

DEMANDE D'ANALYSE:
{task_description}
"""

    user_prompt = f"CONTEXTE COMPLET DU DOCUMENT:\n{context}\n\nFournis l'analyse détaillée demandée :"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        t0 = time.perf_counter()
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0.1,
                max_completion_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
                extra_body={"reasoning_effort": settings.LLM_REASONING_EFFORT},
            ),
            timeout=45.0
        )
        llm_ms = int((time.perf_counter() - t0) * 1000)
        answer = _strip_reasoning_markup(response.choices[0].message.content or "")
        logger.info(f"LLM Analyze ({analysis_type}) responded in {llm_ms}ms — {len(answer)} chars")
        return answer
    except asyncio.TimeoutError:
        logger.error("LLM Analyze timeout (> 45s)")
        raise HTTPException(status_code=504, detail="Timeout du LLM lors de l'analyse")
    except Exception as e:
        logger.error(f"LLM Analyze call failed: {e}")
        return f"⚠️ Erreur LLM lors de l'analyse: {str(e)}"
