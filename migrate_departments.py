import asyncio
from sqlalchemy import select, update, delete
from api.database import async_session, Department, User, Universe, Document

OLD_TO_NEW = {
    "commercial": "commerciale",
    "technique": "software",
    "juridique": "backoffice",
    "finance": "backoffice",
    "direction": "admin",
}

async def migrate():
    async with async_session() as session:
        # First, ensure all new departments exist by calling the seed function or just assuming the app has been restarted
        # We will manually insert them to be safe if they don't exist
        new_depts = [
            ("backoffice", "Back Office", "#3b82f6"),
            ("software", "Développement Logiciel", "#10b981"),
            ("commerciale", "Direction Commerciale", "#f59e0b"),
            ("infrastructure", "Infrastructure IT", "#8b5cf6"),
            ("admin", "Administration Système", "#6b7280"),
        ]
        
        for d_id, d_name, d_color in new_depts:
            res = await session.execute(select(Department).where(Department.id == d_id))
            if not res.scalar_one_or_none():
                session.add(Department(id=d_id, name=d_name, color=d_color))
        await session.commit()

        # Update Users
        print("Migrating Users...")
        for old_id, new_id in OLD_TO_NEW.items():
            await session.execute(
                update(User).where(User.department_id == old_id).values(department_id=new_id)
            )

        # Update Universes
        print("Migrating Universes...")
        for old_id, new_id in OLD_TO_NEW.items():
            await session.execute(
                update(Universe).where(Universe.department_id == old_id).values(department_id=new_id)
            )

        # Update Documents
        print("Migrating Documents...")
        for old_id, new_id in OLD_TO_NEW.items():
            await session.execute(
                update(Document).where(Document.department_id == old_id).values(department_id=new_id)
            )

        await session.commit()
        
        # Delete old departments
        print("Deleting old departments...")
        for old_id in OLD_TO_NEW.keys():
            await session.execute(delete(Department).where(Department.id == old_id))
            
        await session.commit()
        print("Migration complete!")

if __name__ == "__main__":
    asyncio.run(migrate())
