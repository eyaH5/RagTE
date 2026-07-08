# Tunisie Electronique Procurement RAG

This repository houses the core Retrieval-Augmented Generation (RAG) platform developed for Tunisie Electronique. It is an enterprise-grade document analysis system specifically designed to ingest, process, and query complex tender, procurement, and financial documents.

## Platform Overview

Tender and procurement documents are traditionally dense, unstructured, and difficult to navigate. This platform solves that by acting as an intelligent, semantic search engine and automated fact-extractor. Instead of manually reading through hundreds of pages of technical specifications and financial requirements, users can query the platform in natural language to instantly retrieve exact clauses, deadlines, and requirements.

### Key Capabilities

* **Automated Fact Extraction**: The ingestion pipeline automatically reads incoming documents and uses Large Language Models (LLMs) to extract critical structured data (e.g., submission deadlines, financial guarantees required, technical criteria).
* **Advanced OCR & Vision Language Modeling (VLM)**: Many government and enterprise documents are scanned PDFs with low-quality text layers, often mixing Arabic and French. The platform utilizes VLM-driven OCR to transcribe and process these difficult formats accurately.
* **Semantic Retrieval**: Unlike traditional keyword search, the platform understands the context of a query. It uses high-dimensional vector embeddings to find the most conceptually relevant document chunks.
* **Departmental Isolation**: Built-in Role-Based Access Control (RBAC) ensures that sensitive procurement documents are only visible to authorized departments and users.

## How It Works

The platform operates through two primary lifecycles: **Ingestion** and **Retrieval**.

### 1. The Ingestion Pipeline
When a new tender document (PDF/DOCX) is uploaded, it does not just sit in a database. It goes through a rigorous processing pipeline:
1. **Parsing & OCR**: The document is parsed. If the text signal is low (e.g., a scanned document), it falls back to VLM-based OCR for high-fidelity transcription.
2. **Chunking**: The document is split into overlapping semantic chunks to ensure context is preserved without overwhelming the LLM later.
3. **Embedding**: Each chunk is passed through an embedding model (BGE-M3) to convert the text into mathematical vectors, which are then stored in a Qdrant vector database.
4. **Fact Extraction**: An LLM reviews the document to pull out standardized metadata and facts required by the Tunisie Electronique procurement team.

### 2. The Retrieval & Generation Pipeline
When a user asks a question (e.g., *"What is the required bank guarantee for the servers tender?"*):
1. **Query Embedding**: The user's question is converted into a vector.
2. **Vector Search**: The system searches the Qdrant database for the most mathematically similar document chunks.
3. **Reranking**: A cross-encoder reranks the retrieved chunks to ensure the most highly relevant pieces of context are prioritized.
4. **Generation**: The top chunks are injected into a prompt and sent to a local LLM (Qwen), which synthesizes a precise, cited answer based strictly on the provided documents.

## Core Technologies

* **Vector Engine**: Qdrant
* **Embeddings**: BGE-M3 via Text Embeddings Inference (TEI)
* **LLM / VLM**: Qwen model family (via vLLM / Ollama)
* **Backend**: FastAPI (Python)
* **Frontend**: React + Vite
