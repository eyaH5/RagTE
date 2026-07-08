# Tunisie Electronique RAG

This is an enterprise tender/procurement document analysis platform using FastAPI + React, Qdrant vector database, and vLLM/Ollama for LLM support. It supports PDF/DOCX ingestion with OCR.

## Setup Instructions

### 1. Environment Configuration
First, copy the environment template to create your local `.env.local` file:
```bash
cp .env.example .env.local
```
Edit `.env.local` to match your local setup, especially the `JWT_SECRET_KEY` if you plan to deploy this.

### 2. Running with Docker Compose
The easiest way to run the entire stack (API, Frontend, Database, Qdrant, TEI Embeddings) is via Docker Compose:

```bash
docker-compose up --build
```
This will start the full system. The API will be available at `http://localhost:8000` and the frontend at `http://localhost:3000`.

### 3. Running Locally (Python)

If you prefer to run the components directly:

1. **Install dependencies**:
   Ensure you have Python installed, then install the required packages. Note this project uses `uv` for fast package management.
   ```bash
   uv sync
   ```

2. **Start backing services**:
   You need Qdrant and Postgres running. You can start them using the provided compose file:
   ```bash
   docker-compose up qdrant postgres tei
   ```

3. **Start the API server**:
   ```bash
   python -m api.main
   ```

4. **Start the Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Architecture

- **Backend**: FastAPI
- **Vector DB**: Qdrant
- **Relational DB**: SQLite (local) / PostgreSQL (production)
- **Embeddings**: BGE-M3 (via Text Embeddings Inference)
- **LLM/VLM**: Qwen/Ollama integration
- **Frontend**: React + Vite

## Scripts
Maintenance and operational scripts are located in the `scripts/` directory.

- `scripts/import_legacy_pdfs.py`: Migrate old PDFs.
- `scripts/manage_users.py`: CLI for user management.
- `scripts/reindex_documents.py`: Re-index Qdrant documents.
- `scripts/backup.sh`: Backup routine script.
