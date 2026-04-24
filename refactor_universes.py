import re

with open("api/routers/universes.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace imports
content = content.replace("from api.auth import get_current_user, require_role, get_department_filter", "from api.auth import get_current_user, require_role\nfrom api.policy import AccessPolicy, get_policy\nfrom api.repositories.universe import UniverseRepository")

# Refactor list_universes
content = re.sub(
    r"@router\.get\(\"\", response_model=UniverseListResponse\)\nasync def list_universes\([\s\S]*?\n\):\n[\s\S]*?return UniverseListResponse\(universes=enriched, total=total\)",
    "@router.get(\"\", response_model=UniverseListResponse)\nasync def list_universes(\n    page: int = Query(1, ge=1),\n    limit: int = Query(10, ge=1, le=100),\n    department_id: str | None = Query(None),\n    db: AsyncSession = Depends(get_db),\n    policy: AccessPolicy = Depends(get_policy),\n):\n    \"\"\"List universes visible to the current user.\"\"\"\n    repo = UniverseRepository(db, policy)\n    universes = await repo.list_all()\n    # Filtering and pagination logic can be optimized later\n    if department_id:\n        policy.assert_can_access_universe(department_id)\n        universes = [u for u in universes if u.department_id == department_id]\n    \n    total = len(universes)\n    paginated = universes[(page - 1) * limit : page * limit]\n    enriched = [await _enrich_universe_response(u, db) for u in paginated]\n    return UniverseListResponse(universes=enriched, total=total)",
    content
)

# Refactor get_universe
content = re.sub(
    r"@router\.get\(\"/\{universe_id\}\", response_model=UniverseResponse\)\nasync def get_universe\([\s\S]*?\n\):\n[\s\S]*?return await _enrich_universe_response\(universe, db\)",
    "@router.get(\"/{universe_id}\", response_model=UniverseResponse)\nasync def get_universe(\n    universe_id: str,\n    db: AsyncSession = Depends(get_db),\n    policy: AccessPolicy = Depends(get_policy),\n):\n    \"\"\"Get a single universe's details.\"\"\"\n    repo = UniverseRepository(db, policy)\n    universe = await repo.get_by_id(universe_id)\n    if not universe:\n        raise HTTPException(status_code=404, detail=\"Universe introuvable\")\n    if universe.status == \"deleting\":\n        raise HTTPException(status_code=410, detail=\"Universe en cours de suppression\")\n    return await _enrich_universe_response(universe, db)",
    content
)

# Refactor create_universe
content = re.sub(
    r"@router\.post\(\"\", response_model=UniverseResponse, status_code=201\)\nasync def create_universe\([\s\S]*?\n\):\n[\s\S]*?return await _enrich_universe_response\(universe, db\)",
    "@router.post(\"\", response_model=UniverseResponse, status_code=201)\nasync def create_universe(\n    data: UniverseCreate,\n    db: AsyncSession = Depends(get_db),\n    policy: AccessPolicy = Depends(get_policy),\n):\n    \"\"\"Create a new universe (workspace).\"\"\"\n    if policy.user.role not in [\"admin\", \"manager\", \"analyst\"]:\n        raise HTTPException(status_code=403, detail=\"Rôle requis: admin, manager, analyst\")\n    repo = UniverseRepository(db, policy)\n    universe = await repo.create(data.name, data.description, data.department_id)\n    db.add(AuditLog(\n        user_id=policy.user.id,\n        action=\"create_universe\",\n        resource=data.name,\n        department_id=data.department_id,\n    ))\n    await db.commit()\n    await db.refresh(universe)\n    logger.info(f\"Universe '{data.name}' created by {policy.user.email} (dept={data.department_id})\")\n    return await _enrich_universe_response(universe, db)",
    content
)

# Refactor update_universe
content = re.sub(
    r"@router\.put\(\"/\{universe_id\}\", response_model=UniverseResponse\)\nasync def update_universe\([\s\S]*?\n\):\n[\s\S]*?return await _enrich_universe_response\(universe, db\)",
    "@router.put(\"/{universe_id}\", response_model=UniverseResponse)\nasync def update_universe(\n    universe_id: str,\n    data: UniverseUpdate,\n    db: AsyncSession = Depends(get_db),\n    policy: AccessPolicy = Depends(get_policy),\n):\n    \"\"\"Update a universe's name, description, or department.\"\"\"\n    if policy.user.role not in [\"admin\", \"manager\"]:\n        raise HTTPException(status_code=403, detail=\"Rôle requis: admin, manager\")\n    repo = UniverseRepository(db, policy)\n    universe = await repo.get_by_id(universe_id)\n    if not universe:\n        raise HTTPException(status_code=404, detail=\"Universe introuvable\")\n    if universe.status == \"deleting\":\n        raise HTTPException(status_code=410, detail=\"Universe en cours de suppression\")\n    if data.department_id:\n        policy.assert_can_access_universe(data.department_id)\n        universe.department_id = data.department_id\n    if data.name is not None:\n        universe.name = data.name\n    if data.description is not None:\n        universe.description = data.description\n    db.add(AuditLog(\n        user_id=policy.user.id,\n        action=\"update_universe\",\n        resource=universe.name,\n        department_id=universe.department_id,\n    ))\n    await db.commit()\n    await db.refresh(universe)\n    logger.info(f\"Universe '{universe.name}' updated by {policy.user.email}\")\n    return await _enrich_universe_response(universe, db)",
    content
)

# Refactor delete_universe
content = re.sub(
    r"@router\.delete\(\"/\{universe_id\}\", status_code=202\)\nasync def delete_universe\([\s\S]*?\n\):\n[\s\S]*?return \{\"detail\": f\"Universe '\{universe\.name\}' en cours de suppression\"\}",
    "@router.delete(\"/{universe_id}\", status_code=202)\nasync def delete_universe(\n    universe_id: str,\n    background_tasks: BackgroundTasks,\n    db: AsyncSession = Depends(get_db),\n    policy: AccessPolicy = Depends(get_policy),\n):\n    \"\"\"Delete a universe — two-phase safe deletion.\"\"\"\n    if not policy.is_admin:\n        raise HTTPException(status_code=403, detail=\"Rôle requis: admin\")\n    repo = UniverseRepository(db, policy)\n    universe = await repo.get_by_id(universe_id)\n    if not universe:\n        raise HTTPException(status_code=404, detail=\"Universe introuvable\")\n    if universe.status == \"deleting\":\n        raise HTTPException(status_code=409, detail=\"Suppression déjà en cours\")\n    universe.status = \"deleting\"\n    db.add(AuditLog(\n        user_id=policy.user.id,\n        action=\"delete_universe\",\n        resource=universe.name,\n        department_id=universe.department_id,\n    ))\n    await db.commit()\n    background_tasks.add_task(_cascade_delete_universe, universe_id)\n    logger.info(f\"Universe '{universe.name}' marked for deletion by {policy.user.email}\")\n    return {\"detail\": f\"Universe '{universe.name}' en cours de suppression\"}",
    content
)

# Refactor list_universe_documents
content = re.sub(
    r"@router\.get\(\"/\{universe_id\}/documents\"\)\nasync def list_universe_documents\([\s\S]*?\n\):\n[\s\S]*?return \[DocumentResponse\.model_validate\(d\) for d in docs\.scalars\(\)\.all\(\)\]",
    "@router.get(\"/{universe_id}/documents\")\nasync def list_universe_documents(\n    universe_id: str,\n    db: AsyncSession = Depends(get_db),\n    policy: AccessPolicy = Depends(get_policy),\n):\n    \"\"\"List all documents within a specific universe.\"\"\"\n    repo = UniverseRepository(db, policy)\n    universe = await repo.get_by_id(universe_id)\n    if not universe or universe.status != \"active\":\n        raise HTTPException(status_code=404, detail=\"Universe introuvable ou en suppression\")\n    docs = await db.execute(\n        select(Document)\n        .where(Document.universe_id == universe_id)\n        .order_by(Document.created_at.desc())\n    )\n    from api.models import DocumentResponse\n    return [DocumentResponse.model_validate(d) for d in docs.scalars().all()]",
    content
)

with open("api/routers/universes.py", "w", encoding="utf-8") as f:
    f.write(content)
