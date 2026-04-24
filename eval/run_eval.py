"""
RAG Evaluation Suite — Automated accuracy scoring against ground truth.

Runs all 50 test pairs through the RAG pipeline and produces a scored report.
No external API keys needed — uses keyword overlap + hallucination detection.

Usage:
    uv run python eval/run_eval.py
    uv run python eval/run_eval.py --source "CDC 01-2026.pdf"   # test one doc
    uv run python eval/run_eval.py --category penalty           # test one category
"""
import json
import sys
import io
import os
import time
import argparse
import asyncio
from pathlib import Path
from datetime import datetime

# Fix encoding on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


# ── Scoring Functions ─────────────────────────────────────────────────────

HALLUCINATION_SIGNALS = [
    "généralement", "habituellement", "en général", "il est courant",
    "typiquement", "on peut supposer", "il est probable", "dépend du contexte",
    "il faudrait consulter", "je recommande de consulter", "il est important de",
    "peut être requis", "LinkedIn", "Indeed", "Glassdoor", "en France",
]

NOT_FOUND_SIGNALS = [
    "non mentionné", "non mentionee", "pas mentionné", "aucune mention",
    "aucune information", "pas dans le contexte", "non trouvé",
    "n'est pas mentionné", "ne mentionne pas",
]


def score_answer(answer: str, test_case: dict) -> dict:
    """
    Score a RAG answer against a ground-truth test case.
    
    Returns a dict with:
      - keyword_score: fraction of expected keywords found in answer (0.0-1.0)
      - is_hallucination: True if answer matches hallucination signals
      - is_not_found: True if answer says "not mentioned"
      - is_relevant: True if answer appears grounded in source context
      - contains_expected: True if expected_answer_contains text was found
    """
    answer_lower = answer.lower()
    
    # Check hallucination
    is_hallucination = any(sig.lower() in answer_lower for sig in HALLUCINATION_SIGNALS)
    
    # Check "not found" response
    is_not_found = any(sig.lower() in answer_lower for sig in NOT_FOUND_SIGNALS)
    
    # Keyword overlap
    keywords = test_case.get("expected_keywords", [])
    if keywords:
        found = sum(1 for kw in keywords if kw.lower() in answer_lower)
        keyword_score = found / len(keywords)
    else:
        keyword_score = 0.5  # neutral if no keywords specified
    
    # Expected content check
    expected = test_case.get("expected_answer_contains", "")
    contains_expected = expected.lower() in answer_lower if expected else True
    
    # Relevance: not hallucination AND (has keywords OR says not found genuinely)
    is_relevant = not is_hallucination and (keyword_score > 0.0 or is_not_found)
    
    return {
        "keyword_score": round(keyword_score, 2),
        "is_hallucination": is_hallucination,
        "is_not_found": is_not_found,
        "is_relevant": is_relevant,
        "contains_expected": contains_expected,
    }


def compute_aggregate(results: list[dict]) -> dict:
    """Compute aggregate metrics from individual test results."""
    total = len(results)
    if total == 0:
        return {}
    
    avg_keyword = sum(r["scores"]["keyword_score"] for r in results) / total
    hallucinations = sum(1 for r in results if r["scores"]["is_hallucination"])
    not_found = sum(1 for r in results if r["scores"]["is_not_found"])
    relevant = sum(1 for r in results if r["scores"]["is_relevant"])
    avg_latency = sum(r["latency_ms"] for r in results) / total
    
    return {
        "total_tests": total,
        "avg_keyword_score": round(avg_keyword, 3),
        "relevance_rate": round(relevant / total, 3),
        "hallucination_rate": round(hallucinations / total, 3),
        "not_found_rate": round(not_found / total, 3),
        "avg_latency_ms": round(avg_latency),
    }


# ── Main Evaluation ──────────────────────────────────────────────────────

async def _run_eval_async(
    source_filter: str | None = None,
    category_filter: str | None = None,
    output_dir: str = "eval/results",
):
    """Run the full evaluation suite."""
    
    # Load ground truth
    gt_path = Path(__file__).parent / "ground_truth.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    
    logger.info(f"Loaded {len(test_cases)} test cases from ground_truth.json")
    
    # Filter
    if source_filter:
        test_cases = [t for t in test_cases if t["source"] == source_filter]
        logger.info(f"Filtered to {len(test_cases)} tests for source={source_filter}")
    if category_filter:
        test_cases = [t for t in test_cases if t["category"] == category_filter]
        logger.info(f"Filtered to {len(test_cases)} tests for category={category_filter}")
    
    if not test_cases:
        logger.error("No test cases match the filters")
        return
    
    # Import RAG components
    logger.info("Loading RAG pipeline...")
    from api.services.rag import retrieve, ask_llm
    
    # Run tests
    results = []
    
    for i, tc in enumerate(test_cases):
        test_id = tc["id"]
        source = tc["source"]
        question = tc["question"]
        category = tc["category"]
        
        logger.info(f"[{i+1}/{len(test_cases)}] #{test_id} ({category}) — {question[:60]}")
        
        t0 = time.perf_counter()
        
        try:
            chunks, metas = await retrieve(
                query=question,
                k=6,
                source_filter=[source],
                department_filter=None,  # eval runs as admin
            )
            answer = await ask_llm(question, chunks, metas)
        except Exception as e:
            answer = f"ERROR: {e}"
            logger.error(f"Test #{test_id} failed: {e}")
        
        latency_ms = int((time.perf_counter() - t0) * 1000)
        
        # Score
        scores = score_answer(answer, tc)
        
        result = {
            "id": test_id,
            "source": source,
            "category": category,
            "question": question,
            "answer": answer[:500],
            "chunks_retrieved": len(chunks) if 'chunks' in dir() else 0,
            "latency_ms": latency_ms,
            "scores": scores,
        }
        results.append(result)
        
        # Print inline
        status = "✅" if scores["is_relevant"] else ("⚠️" if scores["is_not_found"] else "❌")
        kw = scores["keyword_score"]
        logger.info(f"  {status} kw={kw:.0%} latency={latency_ms}ms {'HALLUC' if scores['is_hallucination'] else ''}")
    
    # ── Aggregate ─────────────────────────────────────────────────────
    aggregate = compute_aggregate(results)
    
    # Per-category breakdown
    categories = set(r["category"] for r in results)
    category_scores = {}
    for cat in sorted(categories):
        cat_results = [r for r in results if r["category"] == cat]
        category_scores[cat] = compute_aggregate(cat_results)
    
    # Per-source breakdown
    sources = set(r["source"] for r in results)
    source_scores = {}
    for src in sorted(sources):
        src_results = [r for r in results if r["source"] == src]
        source_scores[src] = compute_aggregate(src_results)
    
    # ── Report ────────────────────────────────────────────────────────
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "source_filter": source_filter,
            "category_filter": category_filter,
            "total_tests": len(test_cases),
        },
        "aggregate": aggregate,
        "by_category": category_scores,
        "by_source": source_scores,
        "results": results,
    }
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"eval_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  RAG EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"  Tests run:          {aggregate['total_tests']}")
    print(f"  Avg keyword score:  {aggregate['avg_keyword_score']:.1%}")
    print(f"  Relevance rate:     {aggregate['relevance_rate']:.1%}")
    print(f"  Hallucination rate: {aggregate['hallucination_rate']:.1%}")
    print(f"  Not-found rate:     {aggregate['not_found_rate']:.1%}")
    print(f"  Avg latency:        {aggregate['avg_latency_ms']}ms")
    print(f"{'='*60}")
    print(f"\n  By Category:")
    for cat, scores in category_scores.items():
        print(f"    {cat:12s}  kw={scores['avg_keyword_score']:.0%}  rel={scores['relevance_rate']:.0%}  n={scores['total_tests']}")
    print(f"\n  By Source:")
    for src, scores in source_scores.items():
        print(f"    {src[:40]:40s}  kw={scores['avg_keyword_score']:.0%}  rel={scores['relevance_rate']:.0%}  n={scores['total_tests']}")
    print(f"\n  Full report: {report_path}")
    print(f"{'='*60}\n")
    
    return report


def run_eval(
    source_filter: str | None = None,
    category_filter: str | None = None,
    output_dir: str = "eval/results",
):
    return asyncio.run(_run_eval_async(
        source_filter=source_filter,
        category_filter=category_filter,
        output_dir=output_dir,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Evaluation Suite")
    parser.add_argument("--source", type=str, default=None, help="Filter by source PDF")
    parser.add_argument("--category", type=str, default=None, help="Filter by question category")
    parser.add_argument("--output", type=str, default="eval/results", help="Output directory")
    args = parser.parse_args()
    
    run_eval(
        source_filter=args.source,
        category_filter=args.category,
        output_dir=args.output,
    )
