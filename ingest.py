import os
import re
import sys
import io
import unicodedata
import torch
from loguru import logger
from vector_store import VectorStore
from api.embeddings import get_embedder

# Force UTF-8 stdout to avoid cp1252 encoding errors on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============= Constants =============
PDFS_DIR = "pdfs"
CACHE_DIR = "text_cache"
CHUNK_SIZE = 180
OVERLAP = 30
EMBEDDING_MODEL = "BAAI/bge-m3"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PDFS_DIR, exist_ok=True)


# ── Section detection patterns ───────────────────────────────────────────
SECTION_PATTERNS = {
    "admin":     r"(document administratif|registre|cnss|attestation|déclaration|affiliation|matricule|légalis|وثائق إدارية|سجل تجاري|تصريح)",
    "technical": r"(document technique|spécification|caractéristique|installation|configuration|matériel|équipement|وثائق فنية|مواصفات)",
    "financial": r"(document financier|bordereau|prix|offre financière|soumission|lettre de soumission|montant|عرض مالي|أثمان|أسعار)",
    "guarantee": r"(caution|cautionnement|garantie bancaire|garantie définitive|garantie provisoire|ضمان|كفالة|ضمان وقتي|ضمان نهائي)",
    "deadline":  r"(date limite|délai de remise|validité|ouverture des plis|heure limite|تاريخ أقصى|آجال|فتح العروض)",
    "payment":   r"(modalit|paiement|règlement|versement|échéance|facture|خلاص|دفع|أقساط|فاتورة)",
    "penalty":   r"(pénalité|retard|sanction|indemnité|غرامة|عقوبة|تأخير)",
    "reception": r"(réception provisoire|réception définitive|réception quantitative|livraison|installation|استلام|تسليم|قبول)",
}

def detect_section(text: str) -> str:
    text_lower = text.lower()
    for section, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, text_lower):
            return section
    return "general"


# ============= Docling PDF Extraction =============

def extract_and_chunk(file_path: str, filename: str):
    """
    Extract text and structure using Docling, and chunk hierarchically.
    This replaces both PyMuPDF extraction and regex semantic chunking.
    """
    from docling.document_converter import DocumentConverter
    from docling.chunking import HierarchicalChunker

    logger.info(f"Extracting {filename} via Docling...")
    converter = DocumentConverter()
    res = converter.convert(file_path)
    
    chunker = HierarchicalChunker()
    doc_chunks = list(chunker.chunk(res.document))
    
    chunks = []
    metas = []
    ids_out = []
    
    for i, c in enumerate(doc_chunks):
        text = c.text
        
        # Apply Arabic Bidi fix if Arabic content is detected
        if _arabic_char_ratio(text) > 0.05:
            text = _fix_arabic_lines(text)
            
        page_num = 1
        if hasattr(c, "meta") and hasattr(c.meta, "doc_items") and c.meta.doc_items:
            for item in c.meta.doc_items:
                if hasattr(item, "prov") and item.prov:
                    page_num = item.prov[0].page_no
                    break
                    
        chunks.append(text)
        metas.append({
            "source": filename,
            "page": str(page_num),
            "section": detect_section(text),
        })
        ids_out.append(f"{filename}_c{i}")
        
    return chunks, metas, ids_out

# ============= Arabic text helpers =============

def _arabic_char_ratio(text: str) -> float:
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F"
                 or "\uFB50" <= c <= "\uFDFF" or "\uFE70" <= c <= "\uFEFF")
    alpha = sum(1 for c in text if c.isalpha())
    return arabic / alpha if alpha else 0.0



def _fix_arabic_lines(text: str) -> str:
    """Apply bidi per-line to preserve paragraph structure.
    
    Applying bidi to the entire text at once can scramble paragraph
    boundaries. Processing line-by-line keeps structure intact.
    """
    try:
        from bidi.algorithm import get_display
        lines = text.split("\n")
        fixed = [get_display(line) if _arabic_char_ratio(line) > 0.1 else line for line in lines]
        return "\n".join(fixed)
    except ImportError:
        return text


# ============= Ingest Pipeline =============

def ingest():
    vs = VectorStore()

    logger.info("Loading embedding model...")
    embedder = get_embedder()

    pdf_files = [f for f in os.listdir(PDFS_DIR) if f.endswith(".pdf")]
    logger.info(f"Found {len(pdf_files)} PDFs. Starting sync...")

    indexed_count = 0

    for pdf_file in pdf_files:
        if vs.has_source(pdf_file):
            logger.debug(f"Skipping {pdf_file} (already indexed)")
            continue

        # ── Extract and Chunk via Docling ────────────────────────────────────
        all_chunks, all_metas, all_ids = extract_and_chunk(os.path.join(PDFS_DIR, pdf_file), pdf_file)

        # ── Embed and store ───────────────────────────────────────────────
        if all_chunks:
            logger.info(f"Embedding {len(all_chunks)} chunks from {pdf_file}...")
            sys.stdout.flush()
            embeddings = embedder.encode(all_chunks, show_progress_bar=True).tolist()
            vs.add(chunks=all_chunks, embeddings=embeddings, metadatas=all_metas, ids=all_ids)
            indexed_count += 1

    logger.success(f"Ingestion complete — {indexed_count} new PDF(s) indexed")


if __name__ == "__main__":
    ingest()
