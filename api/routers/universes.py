"""
Universes router — CRUD for workspaces with cascade deletion.

Universes are the core organizational unit. Each universe:
- Belongs to a department (which determines the AI persona)
- Contains documents and conversations
- Is accessible by any user (open company culture)
- Has a safe two-phase deletion process
"""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from api.policy import AccessPolicy, get_policy
from api.repositories.universe import UniverseRepository
from api.database import get_db
from api.config import get_settings
from api.models import UniverseCreate, UniverseUpdate, UniverseResponse, UniverseListResponse, DocumentResponse
from api.services.audit import AuditService
from api.services.universe_service import UniverseService
from api.repositories import document_repo

settings = get_settings()

router = APIRouter(prefix="/universes", tags=["Universes"])


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.get("", response_model=UniverseListResponse)
async def list_universes(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    department_id: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
    policy: AccessPolicy = Depends(get_policy),
):
    """List universes visible to the current user."""
    repo = UniverseRepository(db, policy)
    universes = await repo.list_all()
    # Filtering and pagination logic can be optimized later
    if department_id:
        policy.assert_can_access_universe(department_id)
        universes = [u for u in universes if u.department_id == department_id]
    
    total = len(universes)
    paginated = universes[(page - 1) * limit : page * limit]
    enriched = [await repo.enrich_universe_response(u) for u in paginated]
    return UniverseListResponse(universes=enriched, total=total)


@router.get("/{universe_id}", response_model=UniverseResponse)
async def get_universe(
    universe_id: str,
    db: AsyncSession = Depends(get_db),
    policy: AccessPolicy = Depends(get_policy),
):
    """Get a single universe's details."""
    repo = UniverseRepository(db, policy)
    universe = await repo.get_by_id(universe_id)
    if not universe:
        raise HTTPException(status_code=404, detail="Universe introuvable")
    if universe.status == "deleting":
        raise HTTPException(status_code=410, detail="Universe en cours de suppression")
    return await repo.enrich_universe_response(universe)


@router.post("", response_model=UniverseResponse, status_code=201)
async def create_universe(
    data: UniverseCreate,
    db: AsyncSession = Depends(get_db),
    policy: AccessPolicy = Depends(get_policy),
):
    """Create a new universe (workspace)."""
    if policy.user.role not in ["admin", "manager", "analyst"]:
        raise HTTPException(status_code=403, detail="Rôle requis: admin, manager, analyst")
    repo = UniverseRepository(db, policy)
    universe = await repo.create(data.name, data.description, data.department_id)
    await AuditService.log_action(
        db=db,
        user_id=policy.user.id,
        action="create_universe",
        resource=data.name,
        department_id=data.department_id,
    )
    await db.commit()
    await db.refresh(universe)
    logger.info(f"Universe '{data.name}' created by {policy.user.email} (dept={data.department_id})")
    return await repo.enrich_universe_response(universe)


@router.put("/{universe_id}", response_model=UniverseResponse)
async def update_universe(
    universe_id: str,
    data: UniverseUpdate,
    db: AsyncSession = Depends(get_db),
    policy: AccessPolicy = Depends(get_policy),
):
    """Update a universe's name, description, or department."""
    if policy.user.role not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Rôle requis: admin, manager")
    repo = UniverseRepository(db, policy)
    universe = await repo.get_by_id(universe_id)
    if not universe:
        raise HTTPException(status_code=404, detail="Universe introuvable")
    if universe.status == "deleting":
        raise HTTPException(status_code=410, detail="Universe en cours de suppression")
    if data.department_id:
        policy.assert_can_access_universe(data.department_id)
        universe.department_id = data.department_id
    if data.name is not None:
        universe.name = data.name
    if data.description is not None:
        universe.description = data.description
    await AuditService.log_action(
        db=db,
        user_id=policy.user.id,
        action="update_universe",
        resource=universe.name,
        department_id=universe.department_id,
    )
    await db.commit()
    await db.refresh(universe)
    logger.info(f"Universe '{universe.name}' updated by {policy.user.email}")
    return await repo.enrich_universe_response(universe)


@router.delete("/{universe_id}", status_code=202)
async def delete_universe(
    universe_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    policy: AccessPolicy = Depends(get_policy),
):
    """Delete a universe — two-phase safe deletion."""
    if not policy.is_admin:
        raise HTTPException(status_code=403, detail="Rôle requis: admin")
    repo = UniverseRepository(db, policy)
    universe = await repo.get_by_id(universe_id)
    if not universe:
        raise HTTPException(status_code=404, detail="Universe introuvable")
    if universe.status == "deleting":
        raise HTTPException(status_code=409, detail="Suppression déjà en cours")
    universe.status = "deleting"
    await AuditService.log_action(
        db=db,
        user_id=policy.user.id,
        action="delete_universe",
        resource=universe.name,
        department_id=universe.department_id,
    )
    await db.commit()
    background_tasks.add_task(UniverseService.cascade_delete_background, universe_id)
    logger.info(f"Universe '{universe.name}' marked for deletion by {policy.user.email}")
    return {"detail": f"Universe '{universe.name}' en cours de suppression"}


# ── Universe-scoped document listing ──────────────────────────────────────

@router.get("/{universe_id}/documents")
async def list_universe_documents(
    universe_id: str,
    db: AsyncSession = Depends(get_db),
    policy: AccessPolicy = Depends(get_policy),
):
    """List all documents within a specific universe."""
    repo = UniverseRepository(db, policy)
    universe = await repo.get_by_id(universe_id)
    if not universe or universe.status != "active":
        raise HTTPException(status_code=404, detail="Universe introuvable ou en suppression")
    docs = await document_repo.list_for_user(db, policy, universe_id=universe_id)
    return [DocumentResponse.model_validate(d) for d in docs]
