from fastapi import Depends, HTTPException, status
from api.database import User, get_db
from api.auth import get_current_user
from api.config import get_settings
from sqlalchemy.ext.asyncio import AsyncSession

settings = get_settings()

class AccessPolicy:
    """
    Object-oriented Access Control layer injected per-request.
    """
    def __init__(self, user: User):
        self.user = user

    @property
    def is_admin(self) -> bool:
        return self.user.role == "admin"

    @property
    def department_id(self) -> str:
        return self.user.department_id

    def department_filter(self) -> list[str] | None:
        """
        Returns Qdrant/repo department filter.
        None = no filter (admin sees everything).
        Non-admin: own department only.
        """
        # Assuming ENABLE_DEPT_ISOLATION is in settings, fallback to True if not
        enable_isolation = getattr(settings, "ENABLE_DEPT_ISOLATION", True)
        if not enable_isolation or self.is_admin:
            return None
        return [self.user.department_id]

    def can_access_universe(self, universe_dept_id: str) -> bool:
        if self.is_admin:
            return True
        return universe_dept_id == self.user.department_id

    def assert_can_access_universe(self, universe_dept_id: str) -> None:
        if not self.can_access_universe(universe_dept_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Universe belongs to a different department",
            )

    def can_upload(self) -> bool:
        return self.user.role in ("admin", "manager", "analyst")

    def assert_can_upload(self) -> None:
        if not self.can_upload():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Upload requires analyst role or above",
            )

    def can_delete_document(self, doc_uploaded_by: str) -> bool:
        if self.is_admin or self.user.role == "manager":
            return True
        return doc_uploaded_by == self.user.id

    def can_manage_users(self) -> bool:
        return self.is_admin

async def get_policy(
    current_user: User = Depends(get_current_user),
) -> AccessPolicy:
    return AccessPolicy(current_user)
