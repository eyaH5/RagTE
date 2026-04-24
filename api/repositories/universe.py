from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from typing import Optional, List, Sequence
from fastapi import HTTPException

from api.database import Universe
from api.policy import AccessPolicy

class UniverseRepository:
    """
    Repository for Universe data access.
    """
    def __init__(self, db: AsyncSession, policy: AccessPolicy):
        self.db = db
        self.policy = policy

    async def get_by_id(self, universe_id: str) -> Optional[Universe]:
        """Fetch a universe and enforce read access."""
        result = await self.db.execute(select(Universe).where(Universe.id == universe_id))
        universe = result.scalar_one_or_none()
        
        if universe:
            self.policy.assert_can_access_universe(universe.department_id)
            
        return universe

    async def list_all(self) -> Sequence[Universe]:
        """List universes the user has access to."""
        query = select(Universe)
        
        dept_filter = self.policy.department_filter()
        if dept_filter is not None:
            query = query.where(Universe.department_id.in_(dept_filter))
            
        result = await self.db.execute(query)
        return result.scalars().all()

    async def create(self, name: str, description: str, department_id: str) -> Universe:
        """Create a new universe, ensuring it belongs to the creator's department (unless admin)."""
        self.policy.assert_can_access_universe(department_id)
        
        universe = Universe(
            name=name,
            description=description,
            department_id=department_id,
            created_by=self.policy.user.id
        )
        self.db.add(universe)
        return universe

    async def delete(self, universe_id: str) -> None:
        """Delete a universe, enforcing modification access."""
        universe = await self.get_by_id(universe_id)
        if not universe:
            raise HTTPException(status_code=404, detail="Universe introuvable")
            
        self.policy.assert_can_access_universe(universe.department_id)
        await self.db.delete(universe)

    async def enrich_universe_response(self, universe: Universe):
        from sqlalchemy import func
        from api.database import Document, Conversation
        from api.models import UniverseResponse

        doc_count = await self.db.scalar(
            select(func.count(Document.id)).where(Document.universe_id == universe.id)
        )
        conv_count = await self.db.scalar(
            select(func.count(Conversation.id)).where(Conversation.universe_id == universe.id)
        )
        resp = UniverseResponse.model_validate(universe)
        resp.document_count = doc_count or 0
        resp.conversation_count = conv_count or 0
        return resp
