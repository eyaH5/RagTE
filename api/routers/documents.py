"""
Documents router — upload, list, delete, and trigger background ingestion.

Allows authenticated users (admin, manager, analyst) to upload PDFs
which are then ingested asynchronously via BackgroundTasks.
"""
import hashlib
import time

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from api.policy import AccessPolicy, get_policy
from api.repositories import document_repo
from api.database import get_db
from api.config import get_settings
from api.models import DocumentResponse, DocumentVisibilityUpdate, AnalysisRequest
from api.services.audit import AuditService
from api.services.document_service import DocumentService
from api.services.rag import analyze_document as run_analysis
from api.repositories.universe import UniverseRepository

settings = get_settings()

router = APIRouter(prefix="/documents", tags=["Documents"])


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    universe_id: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    policy: AccessPolicy = Depends(get_policy),
):
    """
    Upload a PDF document for ingestion.
    
    - The file is saved to the upload directory
    - A document record is created with status='processing'
    - Background ingestion (OCR → chunk → embed → store) is triggered
    - The document status updates to 'indexed' or 'failed' when done
    
    Only admin, manager, and analyst roles can upload.
    """
    # ── Validate file ─────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Seuls les fichiers PDF sont acceptés",
        )

    # Read file content
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)

    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Fichier trop volumineux ({size_mb:.1f} MB). Maximum: {settings.MAX_UPLOAD_SIZE_MB} MB",
        )

    # ── Dedup check (SHA-256) ─────────────────────────────────────────
    file_hash = hashlib.sha256(content).hexdigest()

    existing = await document_repo.get_by_hash(db, file_hash)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ce document a déjà été importé (même empreinte SHA-256)",
        )

    # ── Save file to disk ─────────────────────────────────────────────
    file_path, safe_name = await DocumentService.save_upload_to_disk(content, file.filename, file_hash)

    logger.info(f"Saved upload: {safe_name} ({size_mb:.1f} MB) by {policy.user.email}")

    # ── Validate universe if provided ─────────────────────────────────
    if universe_id:
        repo = UniverseRepository(db, policy)
        universe = await repo.get_by_id(universe_id)
        if not universe or universe.status != "active":
            raise HTTPException(status_code=400, detail="Universe introuvable ou en suppression")
            
        if not policy.is_admin and universe.department_id != policy.department_id:
            raise HTTPException(status_code=403, detail="Accès refusé pour uploader dans cet universe")

    # ── Create document record ────────────────────────────────────────
    doc = await document_repo.create(
        db,
        filename=safe_name,
        file_hash=file_hash,
        department_id=policy.department_id,
        uploaded_by=policy.user.id,
        universe_id=universe_id,
        visibility="department",
        doc_type="cahier_charges",
        chunk_count=0,
        status="processing",
    )

    # Audit log
    await AuditService.log_action(
        db=db,
        user_id=policy.user.id,
        action="upload",
        resource=safe_name,
        department_id=policy.department_id,
        metadata={"size_mb": round(size_mb, 2), "hash": file_hash[:16]},
    )
    await db.commit()
    await db.refresh(doc)

    background_tasks.add_task(
        DocumentService.run_ingestion_background,
        doc_id=doc.id,
        file_path=str(file_path),
        department=policy.department_id,
        uploaded_by=policy.user.id,
        universe_id=universe_id,
    )

    logger.info(f"Background ingestion queued for doc_id={doc.id}")

    return DocumentResponse.model_validate(doc)


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    policy: AccessPolicy = Depends(get_policy),
):
    """List documents visible to the current user."""
    docs = await document_repo.list_for_user(db, policy.department_id, policy.is_admin)
    return [DocumentResponse.model_validate(d) for d in docs]


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    policy: AccessPolicy = Depends(get_policy),
):
    """Get a single document's metadata and ingestion status."""
    doc = await document_repo.get_by_id(db, doc_id, policy=policy)
    if not doc:
        raise HTTPException(status_code=404, detail="Document introuvable")
    return DocumentResponse.model_validate(doc)


@router.post("/{doc_id}/analyze")
async def analyze_document_endpoint(
    doc_id: str,
    request: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
    policy: AccessPolicy = Depends(get_policy),
):
    """Analyze an entire document holistically."""
    doc = await document_repo.get_by_id(db, doc_id, policy=policy)
    if not doc:
        raise HTTPException(status_code=404, detail="Document introuvable")

    start_time = time.time()
    answer = await run_analysis(doc_id=doc_id, analysis_type=request.analysis_type, prompt=request.prompt)
    elapsed_ms = int((time.time() - start_time) * 1000)

    await AuditService.log_action(
        db=db,
        user_id=policy.user.id,
        action="analyze_document",
        resource=doc.filename,
        department_id=policy.department_id,
        metadata={"analysis_type": request.analysis_type, "time_ms": elapsed_ms},
    )
    await db.commit()
    return {"answer": answer, "time_ms": elapsed_ms}


@router.patch("/{doc_id}/visibility", response_model=DocumentResponse)
async def update_visibility(
    doc_id: str,
    update: DocumentVisibilityUpdate,
    db: AsyncSession = Depends(get_db),
    policy: AccessPolicy = Depends(get_policy),
):
    """Update a document's visibility."""
    if policy.user.role not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Rôle requis: admin, manager")

    doc = await document_repo.get_by_id(db, doc_id, policy=policy)
    if not doc:
        raise HTTPException(status_code=404, detail="Document introuvable")

    old_visibility = doc.visibility
    await document_repo.update_visibility(db, doc_id, update.visibility, policy=policy)

    await AuditService.log_action(
        db=db,
        user_id=policy.user.id,
        action="update_visibility",
        resource=doc.filename,
        department_id=policy.department_id,
        metadata={"old": old_visibility, "new": update.visibility},
    )
    await db.commit()
    await db.refresh(doc)
    return DocumentResponse.model_validate(doc)


@router.delete("/{doc_id}", status_code=204)
async def delete_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    policy: AccessPolicy = Depends(get_policy),
):
    """
    Delete a document — removes from DB, disk, and Qdrant.
    Admin only.
    """
    if not policy.is_admin:
        raise HTTPException(status_code=403, detail="Rôle requis: admin")
    
    doc = await document_repo.get_by_id(db, doc_id, policy=policy)

    if not doc:
        raise HTTPException(status_code=404, detail="Document introuvable")

    # Remove file and vectors via service
    await DocumentService.delete_document_assets(doc.filename)

    # Audit
    await AuditService.log_action(
        db=db,
        user_id=policy.user.id,
        action="delete_document",
        resource=doc.filename,
        department_id=policy.department_id,
    )

    await document_repo.delete_by_id(db, doc_id, policy=policy)
    await db.commit()

    logger.info(f"Document {doc.filename} fully deleted by {policy.user.email}")
