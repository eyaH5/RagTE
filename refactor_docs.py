import re

with open("api/routers/documents.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace imports
content = content.replace("from api.auth import get_current_user, require_role, get_department_filter", "from api.auth import get_current_user, require_role\nfrom api.policy import AccessPolicy, get_policy\nfrom api.repositories import document_repo")

# Refactor upload_document
content = re.sub(
    r"async def upload_document\([\s\S]*?department_filter: list\[str\] \| None = Depends\(get_department_filter\),\n\):",
    "async def upload_document(\n    background_tasks: BackgroundTasks,\n    file: UploadFile = File(...),\n    universe_id: str | None = Form(None),\n    db: AsyncSession = Depends(get_db),\n    policy: AccessPolicy = Depends(get_policy),\n):",
    content
)
content = content.replace("current_user.id", "policy.user.id")
content = content.replace("current_user.email", "policy.user.email")
content = content.replace("current_user.department_id", "policy.department_id")
content = content.replace("current_user.role", "policy.user.role")
content = content.replace("department_filter and universe.department_id not in department_filter", "not policy.is_admin and universe.department_id != policy.department_id")

# Refactor list_documents
content = re.sub(
    r"@router\.get\(\"\", response_model=list\[DocumentResponse\]\)\nasync def list_documents\([\s\S]*?\n\):\n[\s\S]*?return \[DocumentResponse\.model_validate\(d\) for d in result\.scalars\(\)\.all\(\)\]",
    "@router.get(\"\", response_model=list[DocumentResponse])\nasync def list_documents(\n    db: AsyncSession = Depends(get_db),\n    policy: AccessPolicy = Depends(get_policy),\n):\n    \"\"\"List documents visible to the current user.\"\"\"\n    docs = await document_repo.list_for_user(db, policy.department_id, policy.is_admin)\n    return [DocumentResponse.model_validate(d) for d in docs]",
    content
)

# Refactor get_document
content = re.sub(
    r"@router\.get\(\"/\{doc_id\}\", response_model=DocumentResponse\)\nasync def get_document\([\s\S]*?\n\):\n[\s\S]*?return DocumentResponse\.model_validate\(doc\)",
    "@router.get(\"/{doc_id}\", response_model=DocumentResponse)\nasync def get_document(\n    doc_id: str,\n    db: AsyncSession = Depends(get_db),\n    policy: AccessPolicy = Depends(get_policy),\n):\n    \"\"\"Get a single document's metadata and ingestion status.\"\"\"\n    doc = await document_repo.get_by_id(db, doc_id, policy=policy)\n    if not doc:\n        raise HTTPException(status_code=404, detail=\"Document introuvable\")\n    return DocumentResponse.model_validate(doc)",
    content
)

# Refactor analyze_document_endpoint
content = re.sub(
    r"@router\.post\(\"/\{doc_id\}/analyze\"\)\nasync def analyze_document_endpoint\([\s\S]*?\n\):\n[\s\S]*?return \{\"answer\": answer, \"time_ms\": elapsed_ms\}",
    "@router.post(\"/{doc_id}/analyze\")\nasync def analyze_document_endpoint(\n    doc_id: str,\n    request: AnalysisRequest,\n    db: AsyncSession = Depends(get_db),\n    policy: AccessPolicy = Depends(get_policy),\n):\n    \"\"\"Analyze an entire document holistically.\"\"\"\n    doc = await document_repo.get_by_id(db, doc_id, policy=policy)\n    if not doc:\n        raise HTTPException(status_code=404, detail=\"Document introuvable\")\n\n    from api.services.rag import analyze_document as run_analysis\n    start_time = time.time()\n    answer = await run_analysis(doc_id=doc_id, analysis_type=request.analysis_type, prompt=request.prompt)\n    elapsed_ms = int((time.time() - start_time) * 1000)\n\n    db.add(AuditLog(\n        user_id=policy.user.id,\n        action=\"analyze_document\",\n        resource=doc.filename,\n        department_id=policy.department_id,\n        metadata_={\"analysis_type\": request.analysis_type, \"time_ms\": elapsed_ms},\n    ))\n    await db.commit()\n    return {\"answer\": answer, \"time_ms\": elapsed_ms}",
    content
)

# Refactor update_visibility
content = re.sub(
    r"@router\.patch\(\"/\{doc_id\}/visibility\", response_model=DocumentResponse\)\nasync def update_visibility\([\s\S]*?\n\):\n[\s\S]*?return DocumentResponse\.model_validate\(doc\)",
    "@router.patch(\"/{doc_id}/visibility\", response_model=DocumentResponse)\nasync def update_visibility(\n    doc_id: str,\n    update: DocumentVisibilityUpdate,\n    db: AsyncSession = Depends(get_db),\n    policy: AccessPolicy = Depends(get_policy),\n):\n    \"\"\"Update a document's visibility.\"\"\"\n    if policy.user.role not in [\"admin\", \"manager\"]:\n        raise HTTPException(status_code=403, detail=\"Rôle requis: admin, manager\")\n\n    doc = await document_repo.get_by_id(db, doc_id, policy=policy)\n    if not doc:\n        raise HTTPException(status_code=404, detail=\"Document introuvable\")\n\n    old_visibility = doc.visibility\n    await document_repo.update_visibility(db, doc_id, update.visibility, policy=policy)\n\n    db.add(AuditLog(\n        user_id=policy.user.id,\n        action=\"update_visibility\",\n        resource=doc.filename,\n        department_id=policy.department_id,\n        metadata_={\"old\": old_visibility, \"new\": update.visibility},\n    ))\n    await db.commit()\n    await db.refresh(doc)\n    return DocumentResponse.model_validate(doc)",
    content
)

# Refactor delete_document
content = re.sub(
    r"@router\.delete\(\"/\{doc_id\}\", status_code=204\)\nasync def delete_document\([\s\S]*?\n\):\n[\s\S]*?logger\.info\(f\"Document \{doc\.filename\} fully deleted by \{current_user\.email\}\"\)",
    "@router.delete(\"/{doc_id}\", status_code=204)\nasync def delete_document(\n    doc_id: str,\n    db: AsyncSession = Depends(get_db),\n    policy: AccessPolicy = Depends(get_policy),\n):\n    \"\"\"Delete a document.\"\"\"\n    if not policy.is_admin:\n        raise HTTPException(status_code=403, detail=\"Rôle requis: admin\")\n\n    doc = await document_repo.get_by_id(db, doc_id, policy=policy)\n    if not doc:\n        raise HTTPException(status_code=404, detail=\"Document introuvable\")\n\n    # Remove vectors from Qdrant\n    try:\n        from vector_store import AsyncVectorStore\n        vs = AsyncVectorStore(url=settings.QDRANT_URL, collection=settings.QDRANT_COLLECTION)\n        await vs.delete_by_source(doc.filename)\n        logger.info(f\"Deleted vectors for {doc.filename} from Qdrant\")\n    except Exception as e:\n        logger.warning(f\"Could not delete vectors for {doc.filename}: {e}\")\n\n    # Remove file from disk\n    file_path = Path(settings.UPLOAD_DIR) / doc.filename\n    if file_path.exists():\n        file_path.unlink()\n        logger.info(f\"Deleted file: {file_path}\")\n\n    # Audit\n    db.add(AuditLog(\n        user_id=policy.user.id,\n        action=\"delete_document\",\n        resource=doc.filename,\n        department_id=policy.department_id,\n    ))\n\n    await document_repo.delete_by_id(db, doc_id, policy=policy)\n    logger.info(f\"Document {doc.filename} fully deleted by {policy.user.email}\")",
    content
)

with open("api/routers/documents.py", "w", encoding="utf-8") as f:
    f.write(content)
