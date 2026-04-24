"""Quick validation of all API modules."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import asyncio

# 1. Config
from api.config import get_settings
settings = get_settings()
print(f"✅ Config: {settings.APP_NAME} v{settings.APP_VERSION}")

# 2. Database models
from api.database import Base
print(f"✅ Database: {len(Base.metadata.tables)} tables — {list(Base.metadata.tables.keys())}")

# 3. Auth
from api.auth import hash_password, verify_password, create_access_token, decode_token
h = hash_password("test123")
assert verify_password("test123", h), "Password verification failed"
token = create_access_token({"sub": "test-user", "email": "test@test.com", "department": "commercial", "role": "analyst"})
payload = decode_token(token)
assert payload["sub"] == "test-user", "Token decode failed"
print(f"✅ Auth: bcrypt hashing + JWT creation + decode all working")

# 4. Pydantic models
from api.models import LoginRequest, QueryRequest, UserCreate, HealthResponse
lr = LoginRequest(email="test@test.com", password="test")
qr = QueryRequest(question="What is the deadline?")
print(f"✅ Models: Pydantic validation working")

# 5. Database init + seed
async def test_db():
    from api.database import init_db, async_session, Department
    from sqlalchemy import select
    
    await init_db()
    
    async with async_session() as session:
        # Seed a test department
        existing = await session.execute(select(Department).where(Department.id == "commercial"))
        if not existing.scalar_one_or_none():
            session.add(Department(id="commercial", name="Direction Commerciale", description="Test", color="#3b82f6"))
            await session.commit()
        
        result = await session.execute(select(Department))
        depts = result.scalars().all()
        print(f"✅ Database init: {len(depts)} department(s) in SQLite")

asyncio.run(test_db())

# 6. FastAPI app import
from api.main import app
routes = [r.path for r in app.routes if hasattr(r, 'path')]
print(f"✅ FastAPI app: {len(routes)} routes registered")
for r in sorted(routes):
    print(f"   {r}")

print(f"\n{'='*50}")
print(f"ALL MODULES VALIDATED SUCCESSFULLY")
print(f"{'='*50}")
