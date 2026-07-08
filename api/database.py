"""
Database models and connection for the enterprise RAG platform.
Uses SQLAlchemy async with SQLite (local) or PostgreSQL (DGX Spark).
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Boolean, Integer, DateTime, ForeignKey, Text, JSON, text,
    create_engine, inspect,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from api.config import get_settings


class Base(DeclarativeBase):
    pass


# ── Department ────────────────────────────────────────────────────────────

class Department(Base):
    __tablename__ = "departments"

    id = Column(String, primary_key=True)                    # 'commercial', 'technique', etc.
    name = Column(String, nullable=False)                    # 'Direction Commerciale'
    description = Column(Text, default="")
    color = Column(String, default="#3b82f6")                # UI badge color
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("User", back_populates="department")
    documents = relationship("Document", back_populates="department")
    universes = relationship("Universe", back_populates="department")


# ── User ──────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    department_id = Column(String, ForeignKey("departments.id"), nullable=False)
    role = Column(String, nullable=False, default="viewer")  # admin, manager, analyst, viewer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    department = relationship("Department", back_populates="users")
    documents = relationship("Document", back_populates="uploaded_by_user")
    audit_entries = relationship("AuditLog", back_populates="user")
    universes_created = relationship("Universe", back_populates="creator")
    conversations = relationship("Conversation", back_populates="user")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")


# ── Refresh Token ─────────────────────────────────────────────────────────

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    token = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="refresh_tokens")


# ── Document Registry ─────────────────────────────────────────────────────
# Metadata only — vectors stay in Qdrant

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    file_hash = Column(String, nullable=False)               # SHA-256 for dedup
    department_id = Column(String, ForeignKey("departments.id"), nullable=False)
    uploaded_by = Column(String, ForeignKey("users.id"), nullable=False)
    universe_id = Column(String, ForeignKey("universes.id", ondelete="CASCADE"), nullable=True)  # NULL = legacy/unassigned
    visibility = Column(String, default="department")        # private/department/shared/restricted
    doc_type = Column(String, default="cahier_charges")      # cahier_charges, spec_technique, etc.
    extracted_facts = Column(JSON, nullable=True)            # structured facts extracted at ingestion time
    chunk_count = Column(Integer, default=0)
    status = Column(String, default="queued")                # queued/processing/indexed/failed
    created_at = Column(DateTime, default=datetime.utcnow)

    department = relationship("Department", back_populates="documents")
    uploaded_by_user = relationship("User", back_populates="documents")
    universe = relationship("Universe", back_populates="documents")


# ── Universe (Workspace) ──────────────────────────────────────────────────

class Universe(Base):
    __tablename__ = "universes"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    department_id = Column(String, ForeignKey("departments.id"), nullable=False)  # Determines AI persona
    created_by = Column(String, ForeignKey("users.id"), nullable=False)
    status = Column(String, default="active")               # active / deleting
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    department = relationship("Department", back_populates="universes")
    creator = relationship("User", back_populates="universes_created")
    documents = relationship("Document", back_populates="universe", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="universe", cascade="all, delete-orphan")


# ── Conversation ──────────────────────────────────────────────────────────

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    universe_id = Column(String, ForeignKey("universes.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)

    universe = relationship("Universe", back_populates="conversations")
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan",
                            order_by="Message.created_at")


# ── Message ───────────────────────────────────────────────────────────────

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)                    # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)                    # Source citations for assistant messages
    query_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


# ── Audit Log ─────────────────────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)                  # query, upload, delete, export, login
    resource = Column(Text, nullable=True)                   # document ID or query text
    department_id = Column(String, nullable=True)
    metadata_ = Column("metadata", JSON, nullable=True)      # extra context
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="audit_entries")


# ── Document Access (for 'restricted' visibility) ─────────────────────────

class DocumentAccess(Base):
    __tablename__ = "document_access"

    document_id = Column(String, ForeignKey("documents.id"), primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    granted_by = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Engine & Session ──────────────────────────────────────────────────────

settings = get_settings()

# Use aiosqlite for local dev, asyncpg for PostgreSQL
engine_kwargs = {"echo": settings.DEBUG}
if settings.DATABASE_URL.startswith("postgresql+asyncpg://"):
    engine_kwargs.update(
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=5,
        max_overflow=10,
    )

engine = create_async_engine(settings.DATABASE_URL, **engine_kwargs)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncSession:
    """Dependency: yields a database session."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Create all tables. Called on app startup."""
    def _ensure_schema(sync_conn):
        inspector = inspect(sync_conn)
        doc_columns = {col["name"] for col in inspector.get_columns("documents")}
        if "extracted_facts" not in doc_columns:
            sync_conn.execute(text("ALTER TABLE documents ADD COLUMN extracted_facts JSON"))

    async with engine.begin() as conn:
        # Uvicorn workers run startup independently. Serialize DDL on PostgreSQL
        # so concurrent create_all() calls do not race on table/type creation.
        if engine.url.get_backend_name() == "postgresql":
            await conn.execute(text("SELECT pg_advisory_xact_lock(:lock_key)"), {"lock_key": 824642913})
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_ensure_schema)
