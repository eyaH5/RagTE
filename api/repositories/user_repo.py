from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import Optional, List
from datetime import datetime, timezone

from api.database import User, RefreshToken

class UserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_by_id(self, user_id: str) -> Optional[User]:
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def list_all(self, department_id: Optional[str] = None) -> List[User]:
        query = select(User).order_by(User.name)
        if department_id:
            query = query.where(User.department_id == department_id)
        result = await self.db.execute(query)
        return result.scalars().all()

    async def create(self, user: User) -> User:
        self.db.add(user)
        return user

    async def update_last_login(self, user: User) -> None:
        user.last_login = datetime.utcnow()

    async def save_refresh_token(self, token: str, user_id: str, expires_at: datetime) -> None:
        self.db.add(RefreshToken(token=token, user_id=user_id, expires_at=expires_at))

    async def get_refresh_token(self, token: str) -> Optional[RefreshToken]:
        result = await self.db.execute(select(RefreshToken).where(RefreshToken.token == token))
        return result.scalar_one_or_none()

    async def delete_refresh_token(self, token_record: RefreshToken) -> None:
        await self.db.delete(token_record)
