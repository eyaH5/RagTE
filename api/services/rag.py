"""
Core RAG service — extracted from app.py for use by FastAPI backend.
Handles query enhancement, retrieval, reranking, and LLM generation.
"""
import re
import time
from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder
import httpx
from openai import AsyncOpenAI

import asyncio
from loguru import logger

from vector_store import AsyncVectorStore
from api.config import get_settings

settings = get_settings()

# ── Query Expansion (from app.py) ─────────────────────────────────────────

QUERY_EXPANSIONS = {
    "caution":  "caution provisoire caution définitive garantie bancaire ضمان وقتي كفالة بنكية",
    "délai":  "délai livraison installation période durée أجل تسليم مدة تنفيذ",
    "paiement":  "paiement modalité conditions échéance versement خلاص دفع أقساط فاتورة",
    "offre":  "offre soumission proposition prix montant عرض عروض مناقصة",
    "réception":  "réception provisoire définitive livraison validation استلام تسليم قبول",
    "garantie":  "garantie période maintenance SLA ضمان كفالة صلاحية",
    "document":  "document administratif technique registre attestation وثائق إدارية فنية",
    "cnss":  "CNSS affiliation attestation solde الضمان الاجتماعي",
    "registre":  "registre commerce RNE identification سجل تجاري المؤسسات",
    "variante":  "variante alternative offre multiple بديل",
    "ouverture":  "ouverture plis séance publique فتح العروض جلسة",
    "soumission":  "soumission offre plateforme envoi remise تقديم العروض تونبس",
    "validité":  "valable validité délai jours offre صلاحية مدة",
    "références":  "références projets installation justificatifs مراجع",
    "financier":  "financier prix bordereau soumission lettre عرض مالي أثمان أسعار",
    "constructeur":  "constructeur partenariat officielle lettre مصنع",
    "pénalité":  "pénalité retard sanction غرامة عقوبة تأخير",
    "objet":  "objet cahier charges acquisition fourniture projet موضوع اقتناء تزويد مشروع",
}

HALLUCINATION_SIGNALS = [
    "généralement", "habituellement", "en général", "il est courant",
    "typiquement", "on peut supposer", "il est probable", "dépend du contexte",
    "il faudrait consulter", "je recommande de consulter", "il est important de",
    "peut être requis", "peuvent être demandées", "certaines circonstances",
    "LinkedIn", "Indeed", "Glassdoor", "en France", "selon la nature",
    "il n'y a pas de mention claire", "il est donc recommandé",
]

ANALYSIS_QUESTIONS = [
    ("Objet du Cahier des Charges", "Quel est l'objet principal de ce cahier des charges ?"),
    ("Mode d'envoi de soumission", "Comment et où envoyer l'offre ? Quelle plateforme utiliser ?"),
    ("Date limite réelle", "Quelle est la date et heure limite de remise des offres ?"),
    ("Validité de l'offre", "Combien de temps l'offre reste-t-elle valable ?"),
    ("Ouverture des plis", "L'ouverture des offres est-elle publique ? Quand et où ?"),
    ("Variantes", "Les variantes ou offres alternatives sont-elles acceptées ?"),
    ("Caution provisoire", "Une caution provisoire est-elle exigée ? Quel montant et durée ?"),
    ("Fiche renseignement", "Y a-t-il une fiche de renseignements à remplir ?"),
    ("Affiliation CNSS", "L'affiliation CNSS est-elle mentionnée ou exigée ?"),
    ("Attestation de solde", "Une attestation de solde CNSS est-elle demandée ?"),
    ("Registre de commerce", "Le registre de commerce ou RNE est-il exigé ?"),
    ("Documents administratifs", "Quels documents administratifs doivent être fournis ?"),
    ("Documentation technique", "Quelle documentation technique doit accompagner l'offre ?"),
    ("Autorisation du constructeur", "Une autorisation du constructeur est-elle requise ?"),
    ("Liste des références", "Des références de projets similaires sont-elles demandées ?"),
    ("Documents financiers", "Quels documents financiers doivent être inclus ?"),
    ("Période de garantie", "Quelle durée de garantie est exigée ?"),
    ("Les Réceptions", "Quelles sont les étapes de livraison, installation et validation ?"),
    ("Caution définitive", "Une caution définitive est-elle prévue ? Conditions ?"),
    ("Pénalité de retard", "Des pénalités de retard sont-elles mentionnées ? Taux ?"),
    ("Modalités de paiement", "Quelles sont les conditions et délais de paiement ?"),
]


# ── Singletons ────────────────────────────────────────────────────────────

client = AsyncOpenAI(
    base_url=settings.VLLM_URL,
    api_key="EMPTY"
)

from fastapi import HTTPException

async def get_embedding(text: str | list[str]) -> list[float] | list[list[float]]:
    import asyncio
    from api.embeddings import get_embedder
    loop = asyncio.get_running_loop()
    embedder = get_embedder()
    if isinstance(text, str):
        result = await loop.run_in_executor(None, lambda: embedder.encode(text).tolist())
        return result
    else:
        result = await loop.run_in_executor(None, lambda: embedder.encode(text).tolist())
        return result

@lru_cache()
def _get_reranker():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return CrossEncoder("BAAI/bge-reranker-base", device=device)
    except Exception:
        return None

@lru_cache()
def _get_vector_store():
    return AsyncVectorStore(url=settings.QDRANT_URL, collection=settings.QDRANT_COLLECTION)


# ── Core Functions ────────────────────────────────────────────────────────

def enhance_query(query: str) -> str:
    enhanced = query.lower()
    expansions = []
    for key, expansion in QUERY_EXPANSIONS.items():
        if key in enhanced:
            expansions.append(expansion)
    if expansions:
        return query + " " + " ".join(expansions)
    return query


async def retrieve(
    query: str,
    k: int = 6,
    source_filter: list[str] | None = None,
    department_filter: list[str] | None = None,
    universe_id: str | None = None,
) -> tuple[list[str], list[dict]]:
    """
    Retrieve relevant chunks from Qdrant, then rerank.
    Returns (chunks, source_metas).
    
    department_filter is injected by the auth middleware — never by the user.
    """
    reranker = _get_reranker()
    vs = _get_vector_store()

    enhanced_query = enhance_query(query)

    t0 = time.perf_counter()
    query_vec = await get_embedding(enhanced_query)
    embed_ms = int((time.perf_counter() - t0) * 1000)
    logger.debug(f"Query embedding took {embed_ms}ms")

    # ── Qdrant dense search (with department isolation) ───────────────
    t0 = time.perf_counter()
    results = await vs.search(
        query_embedding=query_vec,
        k=25,
        source_filter=source_filter,
        department_filter=department_filter,
        universe_id=universe_id,
    )
    qdrant_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(f"Qdrant returned {len(results)} results in {qdrant_ms}ms (dept_filter={department_filter})")

    raw_chunks = [r["text"] for r in results]
    raw_metas = [{"source": r["source"], "page": r["page"], "section": r["section"], "score": r["score"]} for r in results]
    raw_scores = [r["score"] for r in results]

    if not raw_chunks:
        return [], []

    # ── Rerank ────────────────────────────────────────────────────────
    if reranker:
        t0 = time.perf_counter()
        pairs = [[query, chunk] for chunk in raw_chunks]
        scores = reranker.predict(pairs)
        rerank_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(f"Reranker scored {len(pairs)} pairs in {rerank_ms}ms")

        max_score = max(scores) if len(scores) > 0 else 0
        if max_score < 0.01:
            scored = sorted(zip(scores, raw_chunks, raw_metas),
                            key=lambda x: x[2].get("score", 0), reverse=True)
        else:
            scored = sorted(zip(scores, raw_chunks, raw_metas),
                            key=lambda x: x[0], reverse=True)

        top = scored[:k]
        return [c for _, c, _ in top], [m for _, _, m in top]
    else:
        logger.debug("No reranker available — returning raw ranked results")
        return raw_chunks[:k], raw_metas[:k]


async def ask_llm(question: str, chunks: list[str], metas: list[dict] | None = None,
                  system_prompt: str | None = None) -> str:
    """Generate answer using LLM with hallucination detection.
    
    When metas is provided, each chunk is prefixed with its source
    attribution so the LLM can cite specific documents and pages.
    """
    if not chunks:
        return "⚠️ Non mentionné dans ce document."

    # ── Contextual enrichment (Task 1.1) ──────────────────────────────
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        if metas and i < len(metas):
            m = metas[i]
            source = m.get("source", "inconnu")
            page = m.get("page", "?")
            section = m.get("section", "general")
            header = f"[Source: {source}, Page: {page}, Section: {section}]"
            enriched_chunks.append(f"{header}\n{chunk}")
        else:
            enriched_chunks.append(chunk)

    context = "\n\n---\n\n".join(enriched_chunks)

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

CONTEXTE:
{context}

QUESTION: {question}

RÉPONSE DIRECTE (extraite du contexte uniquement, avec citation de source):"""
        messages = [{"role": "user", "content": prompt}]

    try:
        t0 = time.perf_counter()
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.OLLAMA_MODEL,
                messages=messages,
                temperature=0.0
            ),
            timeout=45.0
        )
        llm_ms = int((time.perf_counter() - t0) * 1000)
        answer = response.choices[0].message.content.strip()
        logger.info(f"LLM ({settings.OLLAMA_MODEL}) responded in {llm_ms}ms — {len(answer)} chars")
    except asyncio.TimeoutError:
        logger.error("LLM timeout (> 45s)")
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

async def rag_query(
    question: str,
    source_filter: list[str] | None = None,
    department_filter: list[str] | None = None,
    universe_id: str | None = None,
    universe_department_id: str | None = None,
    universe_description: str = "",
    k: int = 6,
) -> tuple[str, list[dict]]:
    """
    Full RAG pipeline: retrieve → rerank → generate.
    Called by the query router.
    """
    pipeline_start = time.perf_counter()

    chunks, metas = await retrieve(
        query=question,
        k=k,
        source_filter=source_filter,
        department_filter=department_filter,
        universe_id=universe_id,
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
                model=settings.OLLAMA_MODEL,
                messages=messages,
                temperature=0.1
            ),
            timeout=45.0
        )
        llm_ms = int((time.perf_counter() - t0) * 1000)
        answer = response.choices[0].message.content.strip()
        logger.info(f"LLM Analyze ({analysis_type}) responded in {llm_ms}ms — {len(answer)} chars")
        return answer
    except asyncio.TimeoutError:
        logger.error("LLM Analyze timeout (> 45s)")
        raise HTTPException(status_code=504, detail="Timeout du LLM lors de l'analyse")
    except Exception as e:
        logger.error(f"LLM Analyze call failed: {e}")
        return f"⚠️ Erreur LLM lors de l'analyse: {str(e)}"
