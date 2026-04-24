import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker

converter = DocumentConverter()
result = converter.convert("pdfs/CDC 01-2026.pdf")
doc = result.document

chunker = HierarchicalChunker()
chunks = list(chunker.chunk(doc))

for i, chunk in enumerate(chunks[:5]):
    print(f"--- Chunk {i} ---")
    print(chunk.text)
    print("Meta:")
    if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items"):
        for item in chunk.meta.doc_items:
            for prov in item.prov:
                print(f"Page: {prov.page_no}")
