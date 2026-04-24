"""
FastAPI application — Enterprise RAG Platform for Tunisie Electronique.

Multi-department document analysis with JWT auth, RBAC, and audit logging.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.config import get_settings
from api.database import init_db, async_session, Department
from api.logging_config import setup_logging
from api.routers import auth, admin, query, health, documents, universes

settings = get_settings()


# ── Startup / Shutdown ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and seed default departments on startup."""
    setup_logging(log_dir="logs", debug=settings.DEBUG)
    await init_db()
    await _seed_departments()
    logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION} ready")
    yield
    logger.info("Shutting down...")


async def _seed_departments():
    """Create default departments if they don't exist."""
    defaults = [
        ("backoffice",     "Back Office",                "Support administratif et gestion",                        "#3b82f6"),
        ("software",       "Développement Logiciel",     "Conception, développement et maintenance",                "#10b981"),
        ("commerciale",    "Direction Commerciale",      "Analyse d'appels d'offres et préparation des soumissions", "#f59e0b"),
        ("infrastructure", "Infrastructure IT",          "Réseaux, serveurs, sécurité et systèmes",                 "#8b5cf6"),
        ("admin",          "Administration Système",     "Supervision et administration de la plateforme",          "#6b7280"),
    ]

    async with async_session() as session:
        from sqlalchemy import select
        for dept_id, name, desc, color in defaults:
            existing = await session.execute(
                select(Department).where(Department.id == dept_id)
            )
            if not existing.scalar_one_or_none():
                session.add(Department(id=dept_id, name=name, description=desc, color=color))
                logger.debug(f"Seeded department: {dept_id}")
        await session.commit()


# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Système intelligent d'analyse des cahiers des charges — multi-départements avec authentification",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────

app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(universes.router)
app.include_router(health.router)


# ── Root ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }

