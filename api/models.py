"""
Pydantic models (schemas) for API request/response validation.
"""
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional


# ── Auth ──────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: "UserResponse"

class RefreshRequest(BaseModel):
    refresh_token: str


# ── User ──────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: str
    name: str
    password: str = Field(min_length=8)
    department_id: str
    role: str = Field(default="viewer", pattern="^(admin|manager|analyst|viewer)$")

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    department_id: str
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    model_config = {"from_attributes": True}

class UserUpdate(BaseModel):
    name: Optional[str] = None
    department_id: Optional[str] = None
    role: Optional[str] = Field(default=None, pattern="^(admin|manager|analyst|viewer)$")
    is_active: Optional[bool] = None


# ── Department ────────────────────────────────────────────────────────────

class DepartmentCreate(BaseModel):
    id: str = Field(pattern="^[a-z_]+$")  # slug format: 'commercial', 'technique'
    name: str
    description: str = ""
    color: str = "#3b82f6"

class DepartmentResponse(BaseModel):
    id: str
    name: str
    description: str
    color: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Document ──────────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: str
    filename: str
    department_id: str
    uploaded_by: str
    universe_id: Optional[str] = None
    visibility: str
    doc_type: str
    chunk_count: int
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}

class DocumentVisibilityUpdate(BaseModel):
    visibility: str = Field(pattern="^(private|department|shared|restricted)$")


# ── Query ─────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    universe_id: Optional[str] = None                # Scope query to a specific universe
    source_filter: Optional[list[str]] = None        # Filter by specific documents
    k: int = Field(default=6, ge=1, le=20)           # Top-K results

class QueryResponse(BaseModel):
    answer: str
    sources: list["SourceCitation"]
    query_time_ms: int

class SourceCitation(BaseModel):
    source: str
    page: str
    score: float


# ── Analysis ──────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    analysis_type: str = Field(default="summary", pattern="^(summary|risks|deadlines|financials|action_items)$")
    prompt: Optional[str] = None

class AnalysisResult(BaseModel):
    label: str
    question: str
    answer: str

class AnalysisResponse(BaseModel):
    results: list[AnalysisResult]
    total_time_ms: int
    document_type: str


# ── Universe ──────────────────────────────────────────────────────────────

class UniverseCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    description: str = ""
    department_id: str

class UniverseUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=100)
    description: Optional[str] = None
    department_id: Optional[str] = None

class UniverseResponse(BaseModel):
    id: str
    name: str
    description: str
    department_id: str
    created_by: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    document_count: int = 0
    conversation_count: int = 0

    model_config = {"from_attributes": True}

class UniverseListResponse(BaseModel):
    universes: list[UniverseResponse]
    total: int


# ── Audit ─────────────────────────────────────────────────────────────────

class AuditLogResponse(BaseModel):
    id: int
    user_id: Optional[str]
    action: str
    resource: Optional[str]
    department_id: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Health ────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    qdrant: str           # connected / disconnected
    database: str         # connected / disconnected
    llm: str              # connected / disconnected
    chunks_indexed: int
    version: str
