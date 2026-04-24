import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sqlalchemy import select
from api.database import async_session, Document
from api.config import get_settings

settings = get_settings()

async def main():
    print("Connecting to DB...")
    qdrant = QdrantClient(url=settings.QDRANT_URL)
    
    async with async_session() as session:
        result = await session.execute(select(Document))
        docs = result.scalars().all()
    
    print(f"Found {len(docs)} documents in DB. Updating Qdrant payloads...")
    
    for doc in docs:
        print(f"Updating Qdrant points for document {doc.id} (department: {doc.department_id})...")
        qdrant.set_payload(
            collection_name=settings.QDRANT_COLLECTION,
            payload={
                "department": doc.department_id
            },
            points=Filter(
                must=[
                    FieldCondition(
                        key="original_id",
                        match=MatchValue(value=doc.id)
                    )
                ]
            )
        )
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
