from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List

from api.database import Department

class DepartmentRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_all(self) -> List[Department]:
        result = await self.db.execute(select(Department).order_by(Department.name))
        return result.scalars().all()

    async def get_by_id(self, department_id: str) -> Optional[Department]:
        result = await self.db.execute(select(Department).where(Department.id == department_id))
        return result.scalar_one_or_none()

    async def create(self, department: Department) -> Department:
        self.db.add(department)
        return department
