from sqlalchemy import select, delete as sql_delete
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import Document
from api.policy import AccessPolicy

async def get_by_id(db: AsyncSession, doc_id: str, policy: AccessPolicy | None = None) -> Document | None:
    q = select(Document).where(Document.id == doc_id)
    if policy and not policy.is_admin:
        q = q.where(Document.department_id == policy.department_id)
    result = await db.execute(q)
    return result.scalar_one_or_none()

async def get_by_hash(db: AsyncSession, file_hash: str) -> Document | None:
    result = await db.execute(select(Document).where(Document.file_hash == file_hash))
    return result.scalar_one_or_none()

async def list_for_user(
    db: AsyncSession,
    department_id: str,
    is_admin: bool,
    universe_id: str | None = None,
) -> list[Document]:
    q = select(Document).order_by(Document.created_at.desc())
    if not is_admin:
        q = q.where(Document.department_id == department_id)
    if universe_id:
        q = q.where(Document.universe_id == universe_id)
    result = await db.execute(q)
    return list(result.scalars().all())

async def create(db: AsyncSession, **kwargs) -> Document:
    doc = Document(**kwargs)
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc

async def delete_by_id(db: AsyncSession, doc_id: str, policy: AccessPolicy | None = None) -> bool:
    # Get the document first to ensure we can delete it
    doc = await get_by_id(db, doc_id, policy=policy)
    if not doc:
        return False
    
    result = await db.execute(
        sql_delete(Document).where(Document.id == doc.id)
    )
    await db.commit()
    return result.rowcount > 0

async def update_status(db: AsyncSession, doc_id: str, status: str, policy: AccessPolicy | None = None) -> None:
    doc = await get_by_id(db, doc_id, policy=policy)
    if doc:
        doc.status = status
        await db.commit()

async def update_visibility(db: AsyncSession, doc_id: str, visibility: str, policy: AccessPolicy | None = None) -> None:
    doc = await get_by_id(db, doc_id, policy=policy)
    if doc:
        doc.visibility = visibility
        await db.commit()
