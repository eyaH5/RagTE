import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("pdfs/CDC 01-2026.pdf")
doc = result.document

print("Docling conversion successful")
print("Exported Markdown:")
print(doc.export_to_markdown()[:1000])

for i, page in enumerate(doc.pages.values()):
    print(f"Page {page.page_no}")
    if i > 2:
        break
