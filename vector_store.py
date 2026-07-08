"""
Qdrant-backed vector store with abstraction layer.
"""

import os
import uuid
import hashlib
from qdrant_client import QdrantClient, AsyncQdrantClient, models
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchAny, MatchValue, MinShould,
)

from api.config import get_settings

_s = get_settings()
QDRANT_URL = _s.QDRANT_URL
COLLECTION_NAME = _s.QDRANT_COLLECTION
VECTOR_DIM = 1024  # BGE-M3
PAYLOAD_INDEX_FIELDS = [
    "source",
    "section",
    "page",
    "original_id",
    "doc_id",
    "department",
    "visibility",
    "uploaded_by",
    "universe_id",
]


def _to_uuid(string_id: str) -> str:
    """Convert any string ID to a deterministic UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))


def _build_search_filter(
    *,
    source_filter: list[str] | None = None,
    section_filter: str | None = None,
    department_filter: list[str] | None = None,
    universe_id: str | None = None,
    user_id: str | None = None,
    is_admin: bool = True,
) -> Filter | None:
    conditions = []
    if source_filter:
        conditions.append(FieldCondition(key="source", match=MatchAny(any=source_filter)))
    if section_filter:
        conditions.append(FieldCondition(key="section", match=MatchValue(value=section_filter)))
    if universe_id:
        conditions.append(FieldCondition(key="universe_id", match=MatchValue(value=universe_id)))

    if is_admin:
        return Filter(must=conditions) if conditions else None

    access_filters = []

    department_conditions = [
        FieldCondition(key="visibility", match=MatchValue(value="department")),
    ]
    if department_filter:
        department_conditions.append(
            FieldCondition(key="department", match=MatchAny(any=department_filter))
        )
    access_filters.append(Filter(must=department_conditions))

    if user_id:
        access_filters.append(
            Filter(
                must=[
                    FieldCondition(key="visibility", match=MatchValue(value="private")),
                    FieldCondition(key="uploaded_by", match=MatchValue(value=user_id)),
                ]
            )
        )

    return Filter(
        must=conditions,
        should=access_filters,
        min_should=MinShould(conditions=[], min_count=1),
    )


# VectorStore (sync) — CLI / ingest.py ONLY.
# All FastAPI routes must use AsyncVectorStore.
class VectorStore:
    def __init__(self, url: str | None = None, collection: str | None = None):
        self.url = url or QDRANT_URL
        self.collection = collection or COLLECTION_NAME
        self.client = QdrantClient(url=self.url, timeout=60)
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            from qdrant_client.models import SparseVectorParams, SparseIndexParams
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    ),
                },
            )
            for field in PAYLOAD_INDEX_FIELDS:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            print(f"Created Qdrant collection '{self.collection}'")

    def count(self) -> int:
        return self.client.get_collection(self.collection).points_count

    def add(self, chunks: list[str], embeddings: list[list[float]],
            metadatas: list[dict], ids: list[str]):
        points = []
        for chunk, emb, meta, doc_id in zip(chunks, embeddings, metadatas, ids):
            payload = {
                **meta,
                "text": chunk,
                "original_id": doc_id,
                "department": meta.get("department") or "unknown",
                "visibility": meta.get("visibility", "department"),
                "uploaded_by": meta.get("uploaded_by", "system"),
            }
            points.append(PointStruct(
                id=_to_uuid(doc_id),
                vector={"dense": emb},
                payload=payload,
            ))
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(collection_name=self.collection, points=points[i:i+batch_size])

    def search(self, query_embedding: list[float], k: int = 25,
               source_filter: list[str] | None = None,
               section_filter: str | None = None,
               department_filter: list[str] | None = None,
               universe_id: str | None = None,
               user_id: str | None = None,
               is_admin: bool = True) -> list[dict]:
        """Search vectors with optional filters.
        
        department_filter is injected server-side by the auth middleware.
        It ensures users only see documents from their department + shared.
        Admin users pass None (no filter = see everything).
        
        universe_id provides workspace-level isolation when querying within a Universe.
        """
        qf = _build_search_filter(
            source_filter=source_filter,
            section_filter=section_filter,
            department_filter=department_filter,
            universe_id=universe_id,
            user_id=user_id,
            is_admin=is_admin,
        )

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            using="dense",
            query_filter=qf, limit=k, with_payload=True,
        )
        return [
            {"text": p.payload.get("text", ""), "source": p.payload.get("source", "?"),
             "page": p.payload.get("page", "?"), "section": p.payload.get("section", "general"),
             "department": p.payload.get("department", "shared"),
             "score": p.score}
            for p in results.points
        ]

    def get_document_chunks(self, doc_id: str) -> list[dict]:
        """Fetch all chunks for a specific document, sorted by chunk_index."""
        all_points = []
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
                limit=1000,
                with_payload=True,
                offset=offset,
            )
            all_points.extend(results)
            if offset is None:
                break
        
        # Sort by chunk_index
        all_points.sort(key=lambda p: p.payload.get("chunk_index", 0))
        return [
            {
                "text": p.payload.get("text", ""),
                "page": p.payload.get("page", "?"),
                "section": p.payload.get("section", "general"),
                "chunk_index": p.payload.get("chunk_index", 0)
            }
            for p in all_points
        ]

    def get_all_sources(self) -> list[str]:
        sources = set()
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection, limit=1000,
                with_payload=["source"], offset=offset,
            )
            for p in results:
                src = p.payload.get("source")
                if src:
                    sources.add(src)
            if offset is None:
                break
        return sorted(sources)

    def has_source(self, source: str) -> bool:
        results, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
            limit=1, with_payload=False,
        )
        return len(results) > 0

    def delete_by_source(self, source: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
            ),
        )

    def delete_by_doc_id(self, doc_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
            ),
        )

    def get_all_data(self) -> tuple[list[str], list[str], list[dict]]:
        """Returns (ids, documents, metadatas) for BM25 sync."""
        all_ids, all_docs, all_metas = [], [], []
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection, limit=1000, with_payload=True, offset=offset,
            )
            for p in results:
                all_ids.append(p.payload.get("original_id", str(p.id)))
                all_docs.append(p.payload.get("text", ""))
                all_metas.append({k: v for k, v in p.payload.items() if k not in ("text", "original_id")})
            if offset is None:
                break
        return all_ids, all_docs, all_metas


class AsyncVectorStore:
    def __init__(self, url: str = QDRANT_URL, collection: str = COLLECTION_NAME):
        self.client = AsyncQdrantClient(url=url, timeout=60)
        self.collection = collection
        # We assume collection is already created by ingest.py or sync VectorStore
        # self._ensure_collection() is skipped to keep initialization fast and simple.

    async def search(self, query_embedding: list[float], k: int = 25,
               source_filter: list[str] | None = None,
               section_filter: str | None = None,
               department_filter: list[str] | None = None,
               universe_id: str | None = None,
               user_id: str | None = None,
               is_admin: bool = True) -> list[dict]:
        """Search vectors with optional filters using Async client."""
        qf = _build_search_filter(
            source_filter=source_filter,
            section_filter=section_filter,
            department_filter=department_filter,
            universe_id=universe_id,
            user_id=user_id,
            is_admin=is_admin,
        )

        results = await self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            using="dense",
            query_filter=qf, limit=k, with_payload=True,
        )
        return [
            {"text": p.payload.get("text", ""), "source": p.payload.get("source", "?"),
             "page": p.payload.get("page", "?"), "section": p.payload.get("section", "general"),
             "department": p.payload.get("department", "shared"),
             "score": p.score}
            for p in results.points
        ]

    async def get_document_chunks(self, doc_id: str) -> list[dict]:
        """Fetch all chunks for a specific document using Async client."""
        all_points = []
        offset = None
        while True:
            results, offset = await self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
                limit=1000,
                with_payload=True,
                offset=offset,
            )
            all_points.extend(results)
            if offset is None:
                break
        
        all_points.sort(key=lambda p: p.payload.get("chunk_index", 0))
        return [
            {
                "text": p.payload.get("text", ""),
                "page": p.payload.get("page", "?"),
                "section": p.payload.get("section", "general"),
                "chunk_index": p.payload.get("chunk_index", 0)
            }
            for p in all_points
        ]

    async def delete_by_source(self, source: str) -> None:
        await self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
            ),
        )

    async def delete_by_doc_id(self, doc_id: str) -> None:
        await self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
            ),
        )
