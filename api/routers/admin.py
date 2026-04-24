"""
Admin router — user management, department management, audit log.
Only accessible by admin role.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from api.repositories.user_repo import UserRepository
from api.repositories.department_repo import DepartmentRepository

from api.auth import get_current_user, require_role, hash_password
from api.database import get_db, User, Department, AuditLog
from api.models import (
    UserCreate, UserResponse, UserUpdate,
    DepartmentCreate, DepartmentResponse,
    AuditLogResponse,
)

router = APIRouter(prefix="/admin", tags=["Administration"])


# ── Department Management ─────────────────────────────────────────────────

@router.get("/departments", response_model=list[DepartmentResponse])
async def list_departments(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_role("admin", "manager")),
):
    """List all departments."""
    repo = DepartmentRepository(db)
    depts = await repo.list_all()
    return [DepartmentResponse.model_validate(d) for d in depts]


@router.post("/departments", response_model=DepartmentResponse, status_code=201)
async def create_department(
    dept: DepartmentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Create a new department."""
    repo = DepartmentRepository(db)
    existing = await repo.get_by_id(dept.id)
    if existing:
        raise HTTPException(status_code=400, detail=f"Département '{dept.id}' existe déjà")

    new_dept = Department(**dept.model_dump())
    await repo.create(new_dept)
    await db.commit()
    await db.refresh(new_dept)

    # Audit
    db.add(AuditLog(
        user_id=current_user.id,
        action="create_department",
        resource=dept.id,
        department_id=current_user.department_id,
    ))
    await db.commit()

    return DepartmentResponse.model_validate(new_dept)


# ── User Management ───────────────────────────────────────────────────────

@router.get("/users", response_model=list[UserResponse])
async def list_users(
    department_id: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """List all users, optionally filtered by department."""
    repo = UserRepository(db)
    users = await repo.list_all(department_id)
    return [UserResponse.model_validate(u) for u in users]


@router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Create a new user."""
    # Check email uniqueness
    repo = UserRepository(db)
    existing = await repo.get_by_email(user_data.email)
    if existing:
        raise HTTPException(status_code=400, detail=f"Email '{user_data.email}' déjà utilisé")

    # Verify department exists
    dept_repo = DepartmentRepository(db)
    dept = await dept_repo.get_by_id(user_data.department_id)
    if not dept:
        raise HTTPException(status_code=400, detail=f"Département '{user_data.department_id}' introuvable")

    new_user = User(
        email=user_data.email,
        name=user_data.name,
        password_hash=hash_password(user_data.password),
        department_id=user_data.department_id,
        role=user_data.role,
    )
    await repo.create(new_user)
    await db.commit()
    await db.refresh(new_user)

    # Audit
    db.add(AuditLog(
        user_id=current_user.id,
        action="create_user",
        resource=new_user.email,
        department_id=current_user.department_id,
    ))
    await db.commit()

    return UserResponse.model_validate(new_user)


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    updates: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Update user role, department, or active status."""
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")

    if updates.name is not None:
        user.name = updates.name
    if updates.department_id is not None:
        user.department_id = updates.department_id
    if updates.role is not None:
        user.role = updates.role
    if updates.is_active is not None:
        user.is_active = updates.is_active

    await db.commit()
    await db.refresh(user)

    # Audit
    db.add(AuditLog(
        user_id=current_user.id,
        action="update_user",
        resource=user.email,
        department_id=current_user.department_id,
        metadata_=updates.model_dump(exclude_none=True),
    ))
    await db.commit()

    return UserResponse.model_validate(user)


# ── Audit Log ─────────────────────────────────────────────────────────────

@router.get("/audit", response_model=list[AuditLogResponse])
async def get_audit_log(
    limit: int = 100,
    action: str | None = None,
    department_id: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin", "manager")),
):
    """
    View audit log entries.
    Managers can only see their department's logs.
    Admins see everything.
    """
    manager_dept_id = current_user.department_id if current_user.role == "manager" else None
    
    logs = await AuditService.get_logs(
        db, limit=limit, action=action, 
        department_id=department_id, manager_dept_id=manager_dept_id
    )
    return [AuditLogResponse.model_validate(entry) for entry in logs]
