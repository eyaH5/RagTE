import asyncio
import uuid
from api.database import async_session, User, Department, init_db
from api.auth import hash_password
from sqlalchemy import select

async def seed_admin():
    print("Initializing database...")
    await init_db()
    
    async with async_session() as session:
        # Ensure 'admin' department exists (main.py does this too, but just in case)
        result = await session.execute(select(Department).where(Department.id == "admin"))
        admin_dept = result.scalar_one_or_none()
        
        if not admin_dept:
            print("Creating admin department...")
            admin_dept = Department(
                id="admin",
                name="Administration",
                description="System Administration",
                color="#6b7280"
            )
            session.add(admin_dept)
            await session.commit()

        # Check if admin user already exists
        result = await session.execute(select(User).where(User.email == "eyahajali7@gmail.com"))
        admin_user = result.scalar_one_or_none()
        
        if not admin_user:
            print("Creating default admin user: eyahajali7@gmail.com")
            admin_user = User(
                id=str(uuid.uuid4()),
                email="eyahajali7@gmail.com",
                name="Eya Haj Ali",
                password_hash=hash_password("2x82vfk2da"),
                department_id="admin",
                role="admin",
                is_active=True
            )
            session.add(admin_user)
            await session.commit()
            print("Admin user created successfully!")
        else:
            print("Admin user already exists.")

if __name__ == "__main__":
    asyncio.run(seed_admin())
