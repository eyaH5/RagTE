# Tunisie Electronique RAG Platform

An enterprise document analysis and retrieval-augmented generation (RAG) platform designed for tender and procurement documents.

## Tech Stack

* **Backend:** FastAPI (Python)
* **Frontend:** React + Vite (TypeScript)
* **Vector Database:** Qdrant
* **Relational Database:** SQLite (local) / PostgreSQL (production)
* **Embeddings:** BGE-M3 (via Text Embeddings Inference)
* **LLM:** Qwen (via vLLM or Ollama)

## Prerequisites

- Docker and Docker Compose
- Python 3.10+ (if running locally without Docker)
- Node.js (if running the frontend locally)
- `uv` package manager (recommended for Python dependencies)

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/eyaH5/RagTE.git
   cd RagTE
   ```

2. **Configure environment variables:**
   Copy the example environment file and update it as needed.
   ```bash
   cp .env.example .env.local
   ```
   *Note: Ensure you update `JWT_SECRET_KEY` for production deployments.*

## Running the Application

### Option 1: Docker Compose (Recommended)

The easiest way to start the entire stack (API, Frontend, Qdrant, Postgres, TEI) is using Docker Compose:

```bash
docker-compose up --build
```

- **Frontend:** http://localhost:3000
- **API:** http://localhost:8000

### Option 2: Local Development

If you prefer to run the components independently for development:

1. **Start infrastructure services (Database & Vector Store):**
   ```bash
   docker-compose up qdrant postgres tei
   ```

2. **Start the API:**
   ```bash
   uv sync
   python -m api.main
   ```

3. **Start the Frontend:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Project Structure

- `api/`: FastAPI backend service, routers, and business logic.
- `frontend/`: React-based user interface.
- `scripts/`: Maintenance and administrative CLI scripts (e.g., user management, migrations).
- `vector_store.py`: Qdrant integration and search logic.
- `ingest.py`: Document processing, chunking, and OCR pipeline.
