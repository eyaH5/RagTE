from __future__ import annotations

from collections import Counter

from reconcile_documents import DocumentState, build_report, compute_repair_updates


def test_build_report_flags_missing_mismatched_and_orphan_docs():
    documents = [
        DocumentState(id="doc-1", filename="one.pdf", status="indexed", chunk_count=3),
        DocumentState(id="doc-2", filename="two.pdf", status="processing", chunk_count=0),
        DocumentState(id="doc-3", filename="three.pdf", status="indexed", chunk_count=1),
    ]
    qdrant_counts = Counter({"doc-1": 3, "doc-2": 5, "orphan-doc": 2})

    report = build_report(documents, qdrant_counts, points_without_doc_id=4)

    assert report["document_count"] == 3
    assert report["indexed_in_qdrant"] == 2
    assert [doc.id for doc in report["missing_vectors"]] == ["doc-3"]
    assert [doc.id for doc in report["status_mismatches"]] == ["doc-2"]
    assert [(doc.id, count) for doc, count in report["chunk_mismatches"]] == [("doc-2", 5)]
    assert report["orphan_doc_ids"] == {"orphan-doc": 2}
    assert report["points_without_doc_id"] == 4


def test_compute_repair_updates_indexes_docs_with_vectors():
    documents = [
        DocumentState(id="doc-1", filename="one.pdf", status="processing", chunk_count=0),
        DocumentState(id="doc-2", filename="two.pdf", status="indexed", chunk_count=1),
        DocumentState(id="doc-3", filename="three.pdf", status="processing", chunk_count=0),
    ]
    qdrant_counts = Counter({"doc-1": 7, "doc-2": 1})

    updates = compute_repair_updates(documents, qdrant_counts, mark_missing_failed=False)

    assert updates == {"doc-1": ("indexed", 7)}


def test_compute_repair_updates_can_mark_missing_failed():
    documents = [
        DocumentState(id="doc-1", filename="one.pdf", status="processing", chunk_count=0),
        DocumentState(id="doc-2", filename="two.pdf", status="indexed", chunk_count=4),
    ]
    qdrant_counts = Counter()

    updates = compute_repair_updates(documents, qdrant_counts, mark_missing_failed=True)

    assert updates == {
        "doc-1": ("failed", 0),
        "doc-2": ("failed", 0),
    }
