import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

PDFS_DIR = "pdfs"
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 200   # smaller chunks = more precise answers
OVERLAP = 40       # more overlap = less info lost at boundaries

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def chunk_text(text, filename):
    # Clean text
    text = " ".join(text.split())
    
    # Split into sentences first
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks, metas = [], []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) > CHUNK_SIZE:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                metas.append({"source": filename, "chunk_index": len(chunks)})
            # Keep overlap
            overlap_words = current_chunk[-OVERLAP:] if len(current_chunk) > OVERLAP else current_chunk
            current_chunk = overlap_words + words
            current_len = len(current_chunk)
        else:
            current_chunk.extend(words)
            current_len += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
        metas.append({"source": filename, "chunk_index": len(chunks)})

    return chunks, metas

def ingest():
    print("Loading embedding model (first run downloads ~90MB)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Clear existing collection to avoid duplicates on re-run
    try:
        client.delete_collection("rag_docs")
    except:
        pass
    collection = client.create_collection("rag_docs")

    pdf_files = [f for f in os.listdir(PDFS_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No PDFs found in /pdfs folder. Add some PDFs and try again.")
        return

    print(f"Found {len(pdf_files)} PDF(s)")
    total_chunks = 0

    for pdf_file in pdf_files:
        path = os.path.join(PDFS_DIR, pdf_file)
        print(f"  Processing: {pdf_file}...")
        try:
            text = load_pdf(path)
            if not text.strip():
                print(f"  WARNING: No text extracted from {pdf_file} (might be scanned)")
                continue
            chunks, metas = chunk_text(text, pdf_file)
            embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
            ids = [f"{pdf_file}_chunk_{i}" for i in range(len(chunks))]
            collection.add(documents=chunks, embeddings=embeddings,
                           metadatas=metas, ids=ids)
            total_chunks += len(chunks)
            print(f"  Done: {len(chunks)} chunks")
        except Exception as e:
            print(f"  ERROR on {pdf_file}: {e}")

    print(f"\nIndexing complete! {total_chunks} total chunks across {len(pdf_files)} PDF(s)")
    print("You can now run: streamlit run app.py")

if __name__ == "__main__":
    ingest()