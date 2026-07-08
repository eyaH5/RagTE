# RAG Project Roadmap

Last updated: 2026-05-14

This file summarizes the current architecture, known bugs, pending work, and key engineering decisions for the Tunisie Electronique RAG project.

## 1. Current Goal

The current product focus is tender / cahier des charges analysis.

The application should let a user upload a procurement document, index it, and then:

- run an automatic 21-question "Analyse de consultation";
- chat with the document using cited answers;
- extract important tender facts such as object, deadline, submission method, validity, documents, guarantees, penalties, payment, and requested items.

General-purpose document QA is a later goal. For now, reliability on cahiers des charges is the priority.

## 2. Current Architecture

### Frontend

- Vite / React frontend.
- Main workflows:
  - login;
  - upload and list documents;
  - chat over selected sources;
  - run the "Analyse" checklist for a document.
- The frontend talks to the API through `/api/...` routes, usually proxied by the frontend container in Docker.
- There has been work/planning around upload progress and polling, because indexing can remain in `processing` until the page refreshes.

### API

- FastAPI application under `api/`.
- Main services:
  - `api/services/document_service.py`: document ingestion/reindexing orchestration.
  - `api/services/rag.py`: facts-first answering, retrieval, LLM answer generation, and checklist rendering.
  - `api/services/llm_fact_extractor.py`: newer hybrid LLM extraction service.
- `api/database.py` stores documents, users, departments, and extracted facts.
- Important `Document` fields include:
  - `status`;
  - `chunk_count`;
  - `extracted_facts`;
  - `filename`;
  - department/universe visibility metadata.

### Storage And Retrieval

- Postgres stores application data and `Document.extracted_facts`.
- Qdrant stores chunk vectors for retrieval.
- Embeddings are served by TEI / local embedding path depending on environment.
- The local Windows demo can run with degraded health if Qdrant is not available, but real RAG retrieval needs Qdrant.

### Ingestion Pipeline

The extraction pipeline is mainly in `ingest.py`.

Current extraction stack:

- PDF direct text via PyPDF.
- Docling layout/OCR fallback.
- RapidOCR through Docling, GPU-capable on DGX.
- Tesseract OCR fallback/reinforcement for difficult pages. Tesseract is CPU-based.
- OCR reinforcement when core facts look weak.
- DOCX parser support.
- Text cache written under `/data/text_cache`.
- Uploaded files stored under `/data/pdfs` in the Docker environment.

After text extraction, the pipeline:

- chunks text;
- assigns page/section metadata;
- extracts structured facts using deterministic rules;
- builds a `tender_profile`;
- stores chunks in Qdrant and facts in Postgres.

### Facts-First Answering

The system does not rely only on vector search.

For many checklist and direct questions, `api/services/rag.py` first checks `Document.extracted_facts`. If a field exists, the response can be returned instantly with a source. This is why good Analyse answers are often `1ms`.

If facts do not answer the question, the system falls back to retrieval and then LLM generation.

### Hybrid LLM Fact Extraction

A new hybrid layer exists in `api/services/llm_fact_extractor.py`.

Decision: LLM extraction should run at ingestion/reindex time, not when the user clicks Analyse. This keeps Analyse instant while using the LLM for hard fields.

Intended flow:

1. Regex / deterministic extraction produces `regex_facts`.
2. Weak or missing fields are detected.
3. Relevant evidence pages are selected from extracted page text.
4. The LLM extracts structured JSON facts.
5. Validators reject hallucinated or wrong-context answers.
6. Final merged facts are stored in `Document.extracted_facts`.
7. Debug metadata is stored under `_hybrid_extraction`.

Current configured target fields on DGX:

```text
subject,submission_method,opening,validity,administrative_documents,
technical_documents,financial_documents,manufacturer_authorization,
references,guarantee,reception,penalties,payment
```

This is still in active development.

## 3. What Works Today

- Uploading and indexing PDFs on DGX.
- Uploading and indexing DOCX files, with limitations described below.
- Facts-first Analyse for many French tender PDFs.
- Backfill/reindex scripts:
  - `backfill_document_facts.py`;
  - `reindex_documents.py`;
  - `import_legacy_pdfs.py`.
- Bad scan warning for documents where OCR quality is too poor.
- Requested item extraction for many tables, including:
  - licenses;
  - support IDs;
  - lots;
  - quantities;
  - pricing/bordereau rows.
- Improved Arabic handling compared to the first version, but Arabic scanned documents are still a weak area.
- Tests exist for many deterministic extraction and query paths.

## 4. Important Decisions Already Made

### Stop The Endless Regex Loop

Repeatedly uploading a new PDF, finding 10 missing fields, and adding 10 new regexes is not scalable.

Current decision:

- keep regex extraction as the fast deterministic baseline;
- use LLM extraction as semantic fallback for weak fields;
- add deterministic rules only when a failure repeats and is clearly structural.

### Analyse Must Stay Fast

The Analyse button should read precomputed facts. It should not run 21 LLM calls at click time.

LLM cost should be paid during ingestion/reindexing, where a 30-90 second delay is acceptable for hard documents.

### Bad OCR Should Be Visible

If the document is unreadable, rotated, or badly OCRized, the app should warn the user. Returning 21 times "Non mentionne" without warning is misleading.

### Refactor Later

`ingest.py` is large and should eventually be split, but not before reliability is stabilized. Refactoring it before the demo risks breaking extraction, Docker builds, and reindex scripts.

### Cahier De Charge First

Generic "ask anything about any document" is a later product mode. Current work should optimize tender/cahier analysis first.

## 5. Current Bugs And Weak Spots

### Hybrid LLM Extraction Is Not Finished

Observed issue on `SITEP-CC_225_L_2025_2026-03-05.pdf`:

- `manufacturer_authorization` and `references` remained `None`;
- but the information already existed inside `technical_documents`;
- `_hybrid_extraction.weak_fields` included these fields;
- `llm_fields` was empty for them.

Needed fix:

- derive `manufacturer_authorization` and `references` from strong list facts when technical/admin documents clearly contain constructor certification or reference requirements;
- improve LLM evidence selection and validators so these fields do not silently fail.

### Evidence Selection Still Needs Work

For LLM extraction, good evidence is more important than model size.

Current risk:

- the LLM may receive irrelevant pages;
- validators may reject useful output;
- some fields never reach the LLM because the system thinks regex is strong enough.

Fields most sensitive to evidence selection:

- administrative documents;
- technical documents;
- financial documents;
- references;
- manufacturer authorization;
- opening;
- penalties;
- reception.

### DOCX Support Is Functional But Not Clean

DOCX files can be parsed and indexed, but:

- displayed "page" numbers are artificial and misleading;
- source labels should use section/heading labels instead;
- Word heading styles should drive section splitting;
- annex/template pollution is still a problem;
- some annex content is incorrectly selected as real requirements.

Example: `CC_CNSS.docx` improved with LLM for admin/payment, but caution/reception/guarantee handling still needs refinement.

### Arabic Scanned Documents Are Still Hard

Arabic PDF extraction works sometimes but is not yet reliable enough.

Known constraints:

- Docling/RapidOCR can use GPU.
- Tesseract fallback is CPU-only.
- Some Arabic scans need rotation/quality detection.
- Some OCR output corrupts numbers and percentages.

Future options:

- better page quality scoring;
- automatic rotation correction;
- stronger Arabic OCR configuration;
- vision-based page fallback later.

### Reranker Is Not A Core Dependency Yet

The reranker exists, but Hugging Face downloads/caching previously caused delay and instability. It should only be enabled after:

- model is cached;
- environment variables are confirmed;
- startup latency is acceptable.

### Local Demo Is Degraded Without Qdrant

On Windows, the UI can be shown locally, but full retrieval/indexing requires Qdrant. API health may show:

```json
{"status":"degraded","qdrant":"disconnected"}
```

That is acceptable for interface demonstration, but not for full RAG behavior.

## 6. Pending Tasks

### P0 - Next Work Session

1. Finish hybrid LLM extraction for the 13 target fields.

   - Keep Analyse fast by storing final facts at ingestion time.
   - Ensure `_hybrid_extraction` stores:
     - `regex_facts`;
     - `llm_facts`;
     - `rejected_llm_facts`;
     - `weak_fields`;
     - final merged facts.

2. Improve grouped evidence selection.

   Use groups instead of sending unrelated fields together:

   - Submission: subject, submission_method, deadline, validity, opening, variants.
   - Documents: administrative_documents, technical_documents, financial_documents, CNSS, fiscal, RNE, information_sheet.
   - Guarantees: caution, definitive_caution, guarantee, manufacturer_authorization, references.
   - Execution: reception, penalties, payment.
   - Items: requested_items, lots, quantities, bordereau items.

3. Add derived-field logic.

   If strong list facts mention:

   - constructor certification / official partnership / manufacturer authorization, fill `manufacturer_authorization`;
   - similar projects / references / experience requirements, fill `references`.

   This should fix cases like SITEP without adding document-specific regex.

4. Reindex and compare problem documents.

   Priority documents:

   - `SITEP-CC_225_L_2025_2026-03-05.pdf`;
   - `BANQUE_DE_FINANCEMENT_DES_PETITES_ET_MOYENNES_ENTREPRISES.pdf`;
   - `CC_CNSS.docx`;
   - `Consultation_N_02_DCSI_2025_Acquisition_consommables_et_Accessoires__informatiques_2026-03-05.pdf`;
   - `CDC_Acquisition_Chassis_HPE_Synergy_12000.pdf`.

5. Create a small evaluation set.

   Start small:

   - 5 documents;
   - 21 fields each;
   - expected answer;
   - expected page/section;
   - status: present, absent, or OCR impossible.

   Expand later to 15-20 documents.

### P1 - After P0

1. Improve DOCX infrastructure.

   - Use section/heading labels instead of fake page labels.
   - Use Word heading styles for extraction boundaries.
   - Keep annex/template content separate from real clauses.

2. Improve bad scan handling.

   - Keep warning in Analyse output.
   - Add frontend indicator for low OCR quality.
   - Suggest re-uploading a better scan.

3. Improve upload/indexing UX.

   - Add progress/polling after upload.
   - Avoid the "processing forever until refresh" perception.
   - Show stages: uploaded, extracting, OCR, embedding, indexed/failed.

4. Expand tests.

   - Add regression tests for hybrid LLM helpers.
   - Add tests for DOCX section labels.
   - Add tests for bad OCR warnings.
   - Add tender evaluation runner.

### P2 - Later

1. General document QA mode.

   The system should eventually support non-tender files, but that needs a separate mode from the tender Analyse checklist.

2. Hybrid sparse/dense retrieval.

   Add BM25 or sparse vector retrieval for exact values, Arabic terms, and codes.

3. Vision/page-image fallback.

   Consider ColPali or a vision LLM only after the current text/OCR/LLM extraction path is stable.

4. Refactor `ingest.py`.

   Split into modules only after tests and evaluation set are strong enough.

   Candidate modules:

   - extraction backends;
   - page quality/OCR;
   - tender fact rules;
   - table/requested item mining;
   - polish/normalization;
   - profile builder.

## 7. Known Deployment Commands

### Copy Key Files To DGX

```powershell
scp C:\Users\LOQ\my-rag\ingest.py dgxtunelec@10.10.10.84:~/my-rag/
scp C:\Users\LOQ\my-rag\api\services\document_service.py dgxtunelec@10.10.10.84:~/my-rag/api/services/
scp C:\Users\LOQ\my-rag\api\services\llm_fact_extractor.py dgxtunelec@10.10.10.84:~/my-rag/api/services/
scp C:\Users\LOQ\my-rag\api\services\rag.py dgxtunelec@10.10.10.84:~/my-rag/api/services/
scp C:\Users\LOQ\my-rag\tests\test_llm_fact_extractor.py dgxtunelec@10.10.10.84:~/my-rag/tests/
```

### Rebuild API And Worker On DGX

```bash
cd ~/my-rag
docker compose --env-file .env.dgx build api ingestion-worker
docker compose --env-file .env.dgx up -d api ingestion-worker
```

### Reindex A Specific Document

```bash
docker compose --env-file .env.dgx exec -T api python reindex_documents.py --doc-id <doc_id>
```

### Backfill Indexed Documents

```bash
docker compose --env-file .env.dgx exec -T api python backfill_document_facts.py --status indexed --dry-run
docker compose --env-file .env.dgx exec -T api python backfill_document_facts.py --status indexed
```

### Inspect Extracted Facts

```bash
docker compose --env-file .env.dgx exec -T api python - <<'PY'
import asyncio
from sqlalchemy import select
from api.database import async_session, Document

DOC_ID = "<doc_id>"

async def main():
    async with async_session() as s:
        facts = (await s.execute(
            select(Document.extracted_facts).where(Document.id == DOC_ID)
        )).scalar_one()
        print(sorted((facts or {}).keys()))
        print((facts or {}).get("_hybrid_extraction"))

asyncio.run(main())
PY
```

## 8. Test Commands

Run targeted tests locally:

```powershell
cd C:\Users\LOQ\my-rag
uv --cache-dir .uv-cache run pytest tests/test_document_facts.py tests/test_query_stream_facts.py -q
uv --cache-dir .uv-cache run pytest tests/test_llm_fact_extractor.py -q
```

Run broader regression when time allows:

```powershell
uv --cache-dir .uv-cache run pytest tests/test_backfill_document_facts.py tests/test_document_facts.py tests/test_query_stream_facts.py
```

## 9. Definition Of Done For The Next Milestone

The next milestone is reached when:

- the 13-field hybrid LLM extraction path is enabled and stable;
- Analyse still returns instantly from stored facts;
- 5 representative documents have expected-answer evaluation;
- SITEP/BFPME/CC_CNSS/Consultation_N_02 show fewer missing fields without adding document-specific regex;
- bad scans show a clear warning;
- DOCX sources no longer pretend to be real PDF pages;
- deployment steps are repeatable on DGX.

