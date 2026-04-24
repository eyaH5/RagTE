"""
CLI tool to create the first admin user.
Run: python -m api.seed_admin
"""
import asyncio
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


from api.auth import hash_password
from api.database import init_db, async_session, User, Department
from sqlalchemy import select


async def seed():
    await init_db()

    async with async_session() as session:
        # Check if admin already exists
        result = await session.execute(select(User).where(User.role == "admin"))
        if result.scalar_one_or_none():
            print("[!] Admin user already exists. Skipping.")
            return

        # Ensure 'admin' department exists
        dept = await session.execute(select(Department).where(Department.id == "admin"))
        if not dept.scalar_one_or_none():
            session.add(Department(
                id="admin",
                name="Administration Système",
                description="Administration système",
                color="#6b7280",
            ))

        # Create admin
        admin = User(
            email="admin@tunisie-electronique.com",
            name="Administrateur Système",
            password_hash=hash_password("changeme2026"),
            department_id="admin",
            role="admin",
        )
        session.add(admin)
        await session.commit()

        print("[OK] Admin user created:")
        print(f"   Email:    admin@tunisie-electronique.com")
        print(f"   Password: changeme2026")
        print(f"   [!] Change this password immediately!")


if __name__ == "__main__":
    asyncio.run(seed())
