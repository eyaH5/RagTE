from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db
from api.models import QueryRequest, QueryResponse
from api.policy import AccessPolicy, get_policy
from api.services.query_service import QueryService

router = APIRouter(prefix="/query", tags=["RAG Query"])

@router.post("", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    policy: AccessPolicy = Depends(get_policy),
    db: AsyncSession = Depends(get_db),
):
    """
    Query the RAG system.
    Strictly delegates logic to the Query service and access checks to Policy.
    """
    return await QueryService.execute_query(req, policy, db)
