"""
End-to-end RAG test — simulates the full pipeline using Qdrant VectorStore.
Tests 3 critical queries against CDC 01-2026.pdf.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from vector_store import VectorStore
import ollama

EMBEDDING_MODEL = "BAAI/bge-m3"
PDF_NAME = "CDC 01-2026.pdf"
K = 6

# Import query expansion from app.py
QUERY_EXPANSIONS = {
    "caution":  "caution provisoire caution definitive garantie bancaire ضمان وقتي كفالة بنكية",
    "delai":  "delai livraison installation periode duree أجل تسليم مدة تنفيذ",
    "paiement":  "paiement modalite conditions echeance versement خلاص دفع أقساط فاتورة",
    "garantie":  "garantie periode maintenance SLA ضمان كفالة صلاحية",
    "penalite":  "penalite retard sanction غرامة عقوبة تأخير",
    "reception": "reception provisoire definitive livraison validation استلام تسليم قبول",
}

def enhance_query(query):
    enhanced = query.lower()
    expansions = []
    for key, expansion in QUERY_EXPANSIONS.items():
        if key in enhanced:
            expansions.append(expansion)
    if expansions:
        return query + " " + " ".join(expansions)
    return query

def ask_llm(question, chunks):
    if not chunks:
        return "Non mentionne dans ce document."
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""Tu es un extracteur d'information specialise dans les cahiers des charges tunisiens.
TON SEUL TRAVAIL: extraire l'information EXACTE demandee depuis le contexte fourni.

REGLES ABSOLUES:
1. Si l'information est dans le contexte -> donne-la EXACTEMENT telle qu'elle est ecrite.
2. Si l'information N'EST PAS dans le contexte -> reponds UNIQUEMENT: "Non mentionne dans ce document."
3. FORMAT: 1 a 4 lignes maximum. Pas d'introduction. Pas de conclusion.

CONTEXTE:
{context}

QUESTION: {question}

REPONSE DIRECTE (extraite du contexte uniquement):"""

    response = ollama.chat(
        model="Qwen3-235B-A22B",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "top_p": 0.1}
    )
    return response["message"]["content"].strip()

# Load models
print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBEDDING_MODEL).to(device)
reranker = CrossEncoder('BAAI/bge-reranker-base', device=device)

# Connect to Qdrant via VectorStore
try:
    vs = VectorStore()
    print(f"✅ Connected to Qdrant — {vs.count()} chunks indexed")
except Exception as e:
    print(f"❌ Qdrant connection failed: {e}")
    print("   Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
    sys.exit(1)

# Test queries
TESTS = [
    ("Penalite de retard", "Quelle est la penalite de retard ?"),
    ("Duree de garantie", "Quelle est la duree de garantie ?"),
    ("Modalites de paiement", "Quelles sont les modalites de paiement ?"),
]

for label, question in TESTS:
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"Q: {question}")
    print(f"{'='*60}")

    # Step 1: Query expansion
    enhanced = enhance_query(question)
    print(f"  Enhanced query: {enhanced[:100]}...")

    # Step 2: Embed and retrieve via Qdrant
    query_vec = embedder.encode([enhanced]).tolist()[0]
    results = vs.search(
        query_embedding=query_vec,
        k=25,
        source_filter=[PDF_NAME],
    )

    if not results:
        print("  NO RESULTS!")
        continue

    raw_chunks = [r["text"] for r in results]
    raw_metas = [{"source": r["source"], "page": r["page"], "section": r["section"]} for r in results]
    raw_scores = [r["score"] for r in results]

    # Step 3: Rerank (with cross-lingual fallback)
    pairs = [[question, chunk] for chunk in raw_chunks]
    scores = reranker.predict(pairs)
    
    max_score = max(scores) if len(scores) > 0 else 0
    if max_score < 0.01:
        print(f"  [!] Reranker all ~0 (cross-lingual) -> using embedding score order")
        scored = sorted(zip(scores, raw_chunks, raw_metas, raw_scores),
                        key=lambda x: x[3], reverse=True)  # Sort by Qdrant score descending
    else:
        scored = sorted(zip(scores, raw_chunks, raw_metas, raw_scores),
                        key=lambda x: x[0], reverse=True)

    # Take top K
    top_chunks = []
    for i, (score, chunk, meta, qdrant_score) in enumerate(scored[:K]):
        top_chunks.append(chunk)
        preview = chunk[:120].replace('\n', ' ')
        print(f"  [{i+1}] RerankerScore={score:.3f} QdrantScore={qdrant_score:.3f} Page={meta.get('page')} | {preview}...")

    # Step 4: LLM
    print(f"\n  Calling Qwen3-235B-A22B...")
    answer = ask_llm(question, top_chunks)
    print(f"\n  ANSWER: {answer}")

print(f"\n{'='*60}")
print("ALL TESTS COMPLETE")
print(f"{'='*60}")
