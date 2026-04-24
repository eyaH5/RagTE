from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Any
from loguru import logger
from api.database import AuditLog

class AuditService:
    """
    Handles secure and reliable audit logging.
    Writes audit records in the critical path to ensure security compliance.
    """

    @staticmethod
    async def log_action(
        db: AsyncSession,
        user_id: str,
        action: str,
        resource: Optional[str] = None,
        department_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Record an action in the audit log. Does not commit the transaction,
        allowing the caller to group the audit with the primary transaction.
        """
        try:
            audit_entry = AuditLog(
                user_id=user_id,
                action=action,
                resource=resource,
                department_id=department_id,
                metadata_=metadata or {}
            )
            db.add(audit_entry)
            # We don't commit here so it happens atomically with the parent transaction,
            # or if it's a standalone read action, the caller can commit.
        except Exception as e:
            logger.error(f"Failed to write audit log for action {action}: {e}")
            raise

    @staticmethod
    async def get_logs(
        db: AsyncSession,
        limit: int = 100,
        action: Optional[str] = None,
        department_id: Optional[str] = None,
        manager_dept_id: Optional[str] = None,
    ):
        from sqlalchemy import select
        query = select(AuditLog).order_by(AuditLog.created_at.desc())

        if manager_dept_id:
            query = query.where(AuditLog.department_id == manager_dept_id)
        elif department_id:
            query = query.where(AuditLog.department_id == department_id)

        if action:
            query = query.where(AuditLog.action == action)

        query = query.limit(limit)
        result = await db.execute(query)
        return result.scalars().all()
