# RAG System Enterprise Upgrade Documentation

## Overview
Upgrading from prototype RAG to enterprise-grade system for deployment on NVIDIA DGX Spark.

**Start Date:** April 17, 2026  
**Current Stack:** ChromaDB, PyMuPDF, Ollama (Qwen3:8b), BGE-M3, Streamlit  
**Target Stack:** Qdrant, Docling, vLLM (Qwen2.5-72B), BGE-M3/Qwen3-Embedding, FastAPI + React

---

## Priority Order (Revised)

| # | Step | Status | Impact |
|---|---|---|---|
| 1 | Migrate ChromaDB → Qdrant | ✅ Done | Infrastructure foundation |
| 2 | Docling for layout-aware parsing | ⏸️ DEFERRED | Better table/structure extraction |
| 3 | Fix Arabic OCR + extraction quality | ⬜ **NOW** | Fix root cause of 48% accuracy |
| 4 | Build accuracy test set + baseline | ⬜ NEXT | Measure before/after every change |
| 5 | Contextual Chunk Enrichment | ⬜ | +30-50% retrieval (on clean text) |
| 6 | Qdrant Sparse Vectors (replace BM25) | ⬜ | Drop SQLite, native hybrid search |
| 7 | API & Inference Migration (FastAPI + vLLM) | ⬜ | Production serving |
| 8 | Frontend Migration (Streamlit → React) | ⬜ | Production UI |

> **Lesson learned:** Steps 1-2 were done before fixing data quality. The correct order is: fix extraction → validate accuracy → then optimize infrastructure. We now prioritize Step 3.

---

## Step 1: Migrate ChromaDB → Qdrant ✅

### Why?
- Native hybrid search, payload filtering, RBAC, quantization, snapshots
- Production-ready (used by Mozilla, Deloitte, etc.)
- ChromaDB has no HA, weak filtering, no multi-tenancy

### Files Changed
| File | Change |
|---|---|
| `vector_store.py` | **NEW** — Qdrant abstraction layer |
| `migrate_chroma_to_qdrant.py` | **NEW** — One-time migration script |
| `ingest.py` | Removed chromadb import, uses VectorStore |
| `app.py` | Removed chromadb import, uses VectorStore |

### Setup
```bash
pip install qdrant-client
docker run -p 6333:6333 -v c:/Users/LOQ/my-rag/qdrant_data:/qdrant/storage qdrant/qdrant
python migrate_chroma_to_qdrant.py
streamlit run app.py
```

### Key Decisions
- **Abstraction layer** (`VectorStore` class): swap to Milvus/Pinecone by implementing same interface
- **UUID IDs**: Qdrant requires UUID/integer — we hash string IDs with `uuid5()` deterministically
- **BM25 kept temporarily**: will be replaced by Qdrant native Sparse Vectors (SPLADE/BM42) in Step 6
- **`page` stored as string**: Qdrant payload indexing works with keyword type

### TODO (from review)
- [ ] Add HNSW tuning parameters (`m`, `ef_construct`) for scale
- [ ] Document collection schema (vector dimensions, distance metric)

---

## Step 2: Docling for Layout-Aware PDF Parsing ⏸️ DEFERRED

### Why Deferred?
Docling uses deep learning models for layout analysis — too slow on CPU (~2+ hours for 46 PDFs).
On DGX Spark (GPU), the same workload takes ~10-20 minutes.
Get-Content ingest.py | Select-Object -First 30
### Current State
- Code is ready in `ingest.py` (`_extract_with_docling()` function)
- **Temporarily bypassed** — `get_text_with_pages()` routes directly to PyMuPDF
- Will re-enable when deployed on DGX Spark

### TODO (for DGX Spark deployment)
- [ ] Re-enable Docling in `get_text_with_pages()`
- [ ] Configure GPU acceleration (`DOCLING_DEVICE=cuda`)
- [ ] Configure Arabic OCR explicitly (`languages=["ara", "fra"]`)
- [ ] Benchmark Docling vs PyMuPDF extraction quality
- [ ] Re-ingest all PDFs with Docling on GPU

---

## Step 3: Fix Arabic OCR + Extraction Quality ⬜ **CURRENT**

### Why?
The accuracy test scored 48%. Root cause: Arabic pages extract as garbled text or empty.
Migrating broken embeddings to Qdrant gives the same 48% in a better database.

### Plan
1. Run `diagnose_extraction.py` to identify broken PDFs
2. Fix Arabic extraction pipeline (Tesseract config, reshaping, bidi)
3. Re-ingest broken PDFs only
4. Measure improvement

### Files
| File | Change |
|---|---|
| `diagnose_extraction.py` | **NEW** — Extraction quality diagnostic |

---

## Step 4: Build Accuracy Test Set + Baseline ⬜

### Why?
Without a ground truth test set, we can't measure whether any change improves or degrades quality.
Every step after this must show measurable improvement.

### Plan
1. Create 50+ question-answer pairs from known documents (manually verified)
2. Record baseline accuracy before each change
3. Automated eval script: `python eval.py` → scores retrieval + answer quality
4. Gate: no step proceeds without measured improvement

---

## Step 5: Contextual Chunk Enrichment ⬜

### Why?
Anthropic's research: prepending LLM-generated context to chunks before embedding improves retrieval.
**Caveat**: improvement estimate (30-50%) assumes clean extracted text. On garbled Arabic, gains will be minimal — hence this step comes after fixing OCR.

### Performance Note
- Requires 1 LLM call per chunk during ingestion
- For ~2000 chunks: ~30-60 min on CPU with Qwen3:8b
- **Plan**: Use Qwen3:8b for context generation (fast, cheap) while reserving Qwen2.5-72B for final RAG answers
- Batch processing with progress tracking

### Files
| File | Change |
|---|---|
| `contextual_enrichment.py` | **NEW** — LLM-based chunk enrichment |

---

## Step 6: Qdrant Sparse Vectors (Replace BM25) ⬜

### Why?
Current BM25 uses a separate SQLite FTS5 index — fragile, not scalable, requires manual sync.
Qdrant natively supports Sparse Vectors (SPLADE, BM42) for hybrid dense+sparse search in a single query.

### Plan
1. Add sparse vector encoder (SPLADE or BM42)
2. Store both dense + sparse vectors in Qdrant
3. Use Qdrant's `prefetch` + `fusion` for hybrid search
4. Remove `bm25.db` and `sync_bm25()` entirely

---

## Step 7: API & Inference Migration (FastAPI + vLLM) ⬜

### Why?
- Streamlit is not production-grade (no auth, no API, no concurrency)
- Ollama is single-request — vLLM gives continuous batching, 5-10x throughput
- FastAPI enables multi-client access, auth, rate limiting

### Plan
1. FastAPI backend with `/query`, `/upload`, `/analyze` endpoints
2. vLLM serving Qwen2.5-72B on DGX Spark GPU
3. Auth via JWT tokens
4. WebSocket for streaming responses

---

## Step 8: Frontend Migration (Streamlit → React) ⬜

### Why?
- Professional UI for client deployment
- Real-time streaming, better UX
- Multi-user support

### Plan
1. React + TypeScript frontend
2. Connects to FastAPI backend
3. Document management dashboard
4. Export reports as PDF

---

## Step 9: Advanced Retrieval (DGX Spark) ⬜ FUTURE

### 9a. ColPali — Vision-Based Retrieval
**Why**: Bypass OCR entirely. Embeds document pages as images. Solves Arabic OCR problem at the root.
- Works natively with Qdrant (multi-vector)
- Requires GPU (DGX Spark ready)
- Alternative path if Step 3 OCR fixes don't reach >85% accuracy

### 9b. ColBERT — Late Interaction
**Why**: Token-level embeddings preserve exact matches for product codes, article numbers, specific amounts.
- Triple hybrid: Dense (BGE-M3) + Sparse (SPLADE) + Late-Interaction (ColBERT)
- Qdrant supports multi-vector natively
- Major gain for technical tender documents

### 9c. TensorRT-LLM + FP8
**Why**: 2-3x throughput over vLLM on NVIDIA hardware.
- Only adopt if vLLM throughput insufficient
- **Risk**: FP8 may degrade Arabic generation quality — benchmark before committing
- Measure first, optimize second

### 9d. Document Hierarchy Chunking
**Why**: Chunk by Docling JSON structure (articles, table rows as key-value pairs).
- Already partially done in `semantic_chunk()`
- Extend to tables once Docling is enabled on GPU

### 9e. `uv` Package Manager
**Why**: 10-100x faster pip, deterministic lockfiles.
- Operational improvement, zero accuracy impact
- Adopt during DGX Spark environment setup

---

## Changelog

### 2026-04-17
- [x] Created `vector_store.py` — Qdrant abstraction (UUID-based IDs)
- [x] Created `migrate_chroma_to_qdrant.py`
- [x] Updated `ingest.py` — uses VectorStore, removed chromadb
- [x] Updated `app.py` — uses VectorStore, BM25 sync on upload/delete
- [x] Created `UPGRADE_DOCS.md`
- [x] **Step 2**: Added Docling parsing to `ingest.py` with PyMuPDF fallback
- [x] Separate Docling cache (`*.docling.txt`) to avoid conflicts
- [x] Created `contextual_enrichment.py` (Step 5, not yet integrated)
- [x] Created `diagnose_extraction.py` (Step 3)
- [x] Revised priority order based on review feedback
- [x] Added Steps 4-8 roadmap placeholders
- [x] Added performance notes for contextual enrichment
- [x] Added Qdrant sparse vectors plan (Step 6)