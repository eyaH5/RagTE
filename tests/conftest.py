from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from api.database import Base, Department, Document, Universe, User


@pytest.fixture
def temp_database_url(tmp_path: Path) -> str:
    db_path = tmp_path / "test.db"
    return f"sqlite+aiosqlite:///{db_path.as_posix()}"


@pytest.fixture
def temp_upload_dir(tmp_path: Path) -> Path:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


@pytest.fixture
def test_settings(temp_database_url: str, temp_upload_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        DATABASE_URL=temp_database_url,
        DEBUG=False,
        UPLOAD_DIR=str(temp_upload_dir),
        QDRANT_URL="http://test-qdrant:6333",
        QDRANT_COLLECTION="test_rag_docs",
        EMBEDDING_MODEL="test-embedding",
        TEI_URL="",
        LLM_BASE_URL="http://test-llm/v1",
        LLM_MODEL="test-model",
        CONTEXT_LIMIT=8192,
        LLM_TIMEOUT_SECONDS=5,
        MAX_UPLOAD_SIZE_MB=50,
    )


@pytest_asyncio.fixture
async def initialized_db(temp_database_url: str):
    engine = create_async_engine(temp_database_url, future=True)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    resources = SimpleNamespace(
        url=temp_database_url,
        engine=engine,
        session_factory=session_factory,
    )

    try:
        yield resources
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


@pytest.fixture
def settings_override(monkeypatch, test_settings: SimpleNamespace):
    def _apply(*modules):
        import api.config

        api.config.get_settings.cache_clear()
        monkeypatch.setattr(api.config, "get_settings", lambda: test_settings)
        for module in modules:
            monkeypatch.setattr(module, "settings", test_settings, raising=False)
        return test_settings

    return _apply


@pytest_asyncio.fixture
async def seed_department(initialized_db):
    async def _factory(
        *,
        department_id: str = "commerciale",
        name: str = "Direction Commerciale",
        description: str = "Test department",
    ) -> Department:
        async with initialized_db.session_factory() as session:
            department = Department(
                id=department_id,
                name=name,
                description=description,
                color="#3b82f6",
            )
            session.add(department)
            await session.commit()
            await session.refresh(department)
            return department

    return _factory


@pytest_asyncio.fixture
async def seed_user(initialized_db):
    async def _factory(
        *,
        user_id: str = "user-1",
        email: str = "analyst@example.com",
        name: str = "Analyst User",
        department_id: str = "commerciale",
        role: str = "analyst",
        is_active: bool = True,
    ) -> User:
        async with initialized_db.session_factory() as session:
            user = User(
                id=user_id,
                email=email,
                name=name,
                password_hash="not-a-real-hash",
                department_id=department_id,
                role=role,
                is_active=is_active,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    return _factory


@pytest_asyncio.fixture
async def seed_universe(initialized_db):
    async def _factory(
        *,
        universe_id: str = "universe-1",
        name: str = "Test Universe",
        department_id: str = "commerciale",
        created_by: str = "user-1",
        status: str = "active",
    ) -> Universe:
        async with initialized_db.session_factory() as session:
            universe = Universe(
                id=universe_id,
                name=name,
                description="Test universe",
                department_id=department_id,
                created_by=created_by,
                status=status,
            )
            session.add(universe)
            await session.commit()
            await session.refresh(universe)
            return universe

    return _factory


@pytest_asyncio.fixture
async def seed_document(initialized_db):
    async def _factory(
        *,
        doc_id: str = "doc-1",
        filename: str = "test.pdf",
        file_hash: str = "hash-1",
        department_id: str = "commerciale",
        uploaded_by: str = "user-1",
        universe_id: str | None = None,
        visibility: str = "department",
        doc_type: str = "cahier_charges",
        extracted_facts: dict | None = None,
        chunk_count: int = 0,
        status: str = "processing",
    ) -> Document:
        async with initialized_db.session_factory() as session:
            document = Document(
                id=doc_id,
                filename=filename,
                file_hash=file_hash,
                department_id=department_id,
                uploaded_by=uploaded_by,
                universe_id=universe_id,
                visibility=visibility,
                doc_type=doc_type,
                extracted_facts=extracted_facts,
                chunk_count=chunk_count,
                status=status,
            )
            session.add(document)
            await session.commit()
            await session.refresh(document)
            return document

    return _factory


@pytest.fixture
def stub_embedder():
    class DummyEmbedder:
        def encode(self, texts, **kwargs):
            if isinstance(texts, str):
                return [0.1] * 1024
            return [[0.1] * 1024 for _ in texts]

    return DummyEmbedder()


@pytest.fixture
def stub_extract_and_chunk():
    def _factory(
        *,
        chunks: list[str] | None = None,
        metas: list[dict] | None = None,
        ids: list[str] | None = None,
    ):
        resolved_chunks = chunks or ["chunk one", "chunk two"]
        resolved_metas = metas or [
            {"source": "test.pdf", "page": "1", "section": "general"},
            {"source": "test.pdf", "page": "2", "section": "deadline"},
        ]
        resolved_ids = ids or [f"test.pdf_c{i}" for i in range(len(resolved_chunks))]

        def _stub(file_path: str, filename: str):
            return resolved_chunks, [dict(meta) for meta in resolved_metas], list(resolved_ids)

        return _stub

    return _factory


@pytest.fixture
def fake_vector_store_factory():
    def _factory(
        *,
        delete_doc_exc: Exception | None = None,
        delete_source_exc: Exception | None = None,
        async_delete_doc_exc: Exception | None = None,
        async_delete_source_exc: Exception | None = None,
    ):
        state = {
            "sync_instances": [],
            "async_instances": [],
            "delete_by_doc_id": [],
            "delete_by_source": [],
            "async_delete_by_doc_id": [],
            "async_delete_by_source": [],
            "adds": [],
        }

        class FakeVectorStore:
            def __init__(self, url=None, collection=None):
                self.url = url
                self.collection = collection
                state["sync_instances"].append(self)

            def delete_by_doc_id(self, doc_id: str):
                state["delete_by_doc_id"].append(doc_id)
                if delete_doc_exc is not None:
                    raise delete_doc_exc

            def delete_by_source(self, source: str):
                state["delete_by_source"].append(source)
                if delete_source_exc is not None:
                    raise delete_source_exc

            def add(self, *, chunks, embeddings, metadatas, ids):
                state["adds"].append(
                    {
                        "chunks": list(chunks),
                        "embeddings": list(embeddings),
                        "metadatas": [dict(meta) for meta in metadatas],
                        "ids": list(ids),
                    }
                )

        class FakeAsyncVectorStore:
            def __init__(self, url=None, collection=None):
                self.url = url
                self.collection = collection
                state["async_instances"].append(self)

            async def delete_by_doc_id(self, doc_id: str):
                state["async_delete_by_doc_id"].append(doc_id)
                if async_delete_doc_exc is not None:
                    raise async_delete_doc_exc

            async def delete_by_source(self, source: str):
                state["async_delete_by_source"].append(source)
                if async_delete_source_exc is not None:
                    raise async_delete_source_exc

        return state, FakeVectorStore, FakeAsyncVectorStore

    return _factory
