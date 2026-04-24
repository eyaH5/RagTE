import asyncio
from api.database import async_session, Document
from sqlalchemy import update

async def fix():
    async with async_session() as s:
        await s.execute(update(Document).values(status='ready'))
        await s.commit()

asyncio.run(fix())
