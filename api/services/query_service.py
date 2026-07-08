import time
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException
from api.models import QueryRequest, QueryResponse, SourceCitation
from api.policy import AccessPolicy
from api.repositories.universe import UniverseRepository
from api.services.rag import rag_query
from api.services.audit import AuditService

class QueryService:
    @staticmethod
    async def execute_query(
        req: QueryRequest,
        policy: AccessPolicy,
        db: AsyncSession
    ) -> QueryResponse:
        # 1. Universe access check (delegated to repo + policy)
        universe_description = ""
        universe_department_id = None
        
        if req.universe_id:
            repo = UniverseRepository(db, policy)
            universe = await repo.get_by_id(req.universe_id)
            if not universe:
                raise HTTPException(status_code=404, detail="Universe introuvable")
            universe_description = universe.description or ""
            universe_department_id = universe.department_id

        # 2. RAG — policy carries department filter
        start = time.time()
        answer, sources = await rag_query(
            question=req.question,
            source_filter=req.source_filter,
            department_filter=policy.department_filter(),
            universe_id=req.universe_id,
            user_id=policy.user.id,
            is_admin=policy.is_admin,
            db=db,
            universe_department_id=universe_department_id,
            universe_description=universe_description,
            k=req.k,
        )
        elapsed_ms = int((time.time() - start) * 1000)

        # 3. Audit — synchronous, in request flow (security record)
        await AuditService.log_action(
            db=db,
            user_id=policy.user.id,
            action="query", 
            resource=req.question[:200],
            department_id=policy.department_id,
            metadata={
                "k": req.k,
                "time_ms": elapsed_ms,
                "source_count": len(sources),
                "universe_id": req.universe_id,
            }
        )
        await db.commit()

        return QueryResponse(
            answer=answer,
            sources=[SourceCitation(source=s["source"], page=s["page"], score=s.get("score", 0.0)) for s in sources],
            query_time_ms=elapsed_ms,
        )
