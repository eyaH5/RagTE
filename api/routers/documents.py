"""
Documents router — upload, list, delete, and trigger background ingestion.

Allows authenticated users (admin, manager, analyst) to upload documents
which are then ingested asynchronously by the ingestion worker.
"""
import hashlib
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from api.policy import AccessPolicy, get_policy
from api.repositories import document_repo
from api.database import get_db
from api.config import get_settings
from api.models import DocumentResponse, DocumentVisibilityUpdate, AnalysisRequest
from api.services.audit import AuditService
from api.services.document_service import DocumentService
from api.services.rag import analyze_document as run_analysis, analyze_tender_checklist_document
from api.repositories.universe import UniverseRepository

settings = get_settings()
VISIBLE_UPLOAD_VALUES = {"private", "department"}
SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".xlsx"}
SUPPORTED_UPLOAD_LABEL = "PDF, DOCX, TXT, MD, CSV, JSON, XLSX"

router = APIRouter(prefix="/documents", tags=["Documents"])


def _visible_duplicate_for_policy(existing_docs, policy: AccessPolicy):
    return next((doc for doc in existing_docs if policy.can_view_document(doc)), None)


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    universe_id: str | None = Form(None),
    visibility: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    policy: AccessPolicy = Depends(get_policy),
):
    """
    Upload a document for ingestion.
    
    - The file is saved to the upload directory
    - A document record is created with status='processing'
    - Background ingestion (OCR → chunk → embed → store) is triggered
    - The document status updates to 'indexed' or 'failed' when done

    Operational note:
    - This endpoint queues in-process BackgroundTasks work for interactive
      uploads. It is not the right path for bulk historical imports or full
      corpus rebuilds; use the CLI import/reindex tooling for those flows.
    
    Only admin, manager, and analyst roles can upload.
    """
    policy.assert_can_upload()

    # ── Validate file ─────────────────────────────────────────────────
    suffix = Path(file.filename or "").suffix.lower()
    if not file.filename or suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Types acceptes: {SUPPORTED_UPLOAD_LABEL}",
        )

    selected_visibility = visibility or "department"
    if selected_visibility not in VISIBLE_UPLOAD_VALUES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Visibilité invalide. Valeurs autorisées: private, department",
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

    existing_docs = await document_repo.list_by_hash(db, file_hash)
    visible_existing = _visible_duplicate_for_policy(existing_docs, policy)
    if visible_existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Ce document est deja visible dans votre espace: {visible_existing.filename}",
        )
    if existing_docs:
        logger.info(
            "Allowing duplicate upload for hidden hash={} by user={}",
            file_hash[:16],
            policy.user.email,
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
        visibility=selected_visibility,
        doc_type="cahier_charges",
        chunk_count=0,
        status="queued",
    )

    # Audit log
    await AuditService.log_action(
        db=db,
        user_id=policy.user.id,
        action="upload",
        resource=safe_name,
        department_id=policy.department_id,
        metadata={
            "size_mb": round(size_mb, 2),
            "hash": file_hash[:16],
            "visibility": selected_visibility,
        },
    )
    await db.commit()
    await db.refresh(doc)

    logger.info(f"Ingestion queued for doc_id={doc.id}")

    return DocumentResponse.model_validate(doc)


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    policy: AccessPolicy = Depends(get_policy),
):
    """List documents visible to the current user."""
    docs = await document_repo.list_for_user(db, policy)
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
    if request.analysis_type == "tender_checklist":
        answer = analyze_tender_checklist_document(doc)
    else:
        if settings.CONTEXT_LIMIT < 32000:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Document analysis requires a large-context model. "
                    f"CONTEXT_LIMIT is {settings.CONTEXT_LIMIT}. "
                    f"Set CONTEXT_LIMIT >= 32000 in your env file."
                ),
            )
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
    if update.visibility not in VISIBLE_UPLOAD_VALUES:
        raise HTTPException(
            status_code=400,
            detail="Visibilité invalide. Valeurs autorisées: private, department",
        )

    doc = await document_repo.get_by_id(db, doc_id, policy=policy)
    if not doc:
        raise HTTPException(status_code=404, detail="Document introuvable")

    if not policy.can_change_document_visibility(doc):
        raise HTTPException(status_code=403, detail="Modification de visibilité non autorisée")

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
    await DocumentService.delete_document_assets(doc.filename, doc.id)

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
