import asyncio
import contextlib
import json
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models import QueryRequest, QueryResponse
from api.policy import AccessPolicy, get_policy
from api.repositories.universe import UniverseRepository
from api.services.audit import AuditService
from api.services.query_service import QueryService
from api.services.rag import (
    _missing_answer_text,
    answer_chatbot_identity,
    answer_from_document_facts,
    retrieve,
    stream_llm_answer,
)

router = APIRouter(prefix="/query", tags=["RAG Query"])


def _single_retrieved_source(metas: list[dict]) -> str | None:
    sources = []
    for meta in metas:
        source = meta.get("source")
        if source and source not in sources:
            sources.append(source)
        if len(sources) > 1:
            return None
    return sources[0] if sources else None


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


@router.post("/stream")
async def query_stream(
    req: QueryRequest,
    policy: AccessPolicy = Depends(get_policy),
    db: AsyncSession = Depends(get_db),
):
    """Stream query output as server-sent events for responsive chat UIs."""
    universe_description = ""
    universe_department_id = None

    if req.universe_id:
        repo = UniverseRepository(db, policy)
        universe = await repo.get_by_id(req.universe_id)
        if not universe:
            raise HTTPException(status_code=404, detail="Universe introuvable")
        universe_description = universe.description or ""
        universe_department_id = universe.department_id

    async def event_generator():
        start = time.time()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def worker() -> None:
            logger.info(
                "Stream query received source_filter={} universe_id={} user={} role={}",
                req.source_filter,
                req.universe_id,
                policy.user.email,
                policy.user.role,
            )
            chatbot_answer = answer_chatbot_identity(req.question)
            if chatbot_answer:
                full_answer, metas = chatbot_answer
                await queue.put("event: meta\ndata: " + json.dumps({"sources": metas}) + "\n\n")
                await queue.put("event: token\ndata: " + json.dumps({"text": full_answer}) + "\n\n")
                elapsed_ms = int((time.time() - start) * 1000)
                await queue.put(
                    "event: done\ndata: "
                    + json.dumps({"query_time_ms": elapsed_ms, "answer": full_answer})
                    + "\n\n"
                )
                await AuditService.log_action(
                    db=db,
                    user_id=policy.user.id,
                    action="query",
                    resource=req.question[:200],
                    department_id=policy.department_id,
                    metadata={
                        "k": req.k,
                        "time_ms": elapsed_ms,
                        "source_count": len(metas),
                        "universe_id": req.universe_id,
                        "static_intent": "chatbot_identity",
                    },
                )
                await db.commit()
                await queue.put(None)
                return

            fact_answer = await answer_from_document_facts(
                db=db,
                question=req.question,
                source_filter=req.source_filter,
                department_filter=policy.department_filter(),
                universe_id=req.universe_id,
                user_id=policy.user.id,
                is_admin=policy.is_admin,
                strict_missing=len(req.source_filter or []) == 1,
            )
            if fact_answer:
                full_answer, metas = fact_answer
                await queue.put("event: meta\ndata: " + json.dumps({"sources": metas}) + "\n\n")
                await queue.put("event: token\ndata: " + json.dumps({"text": full_answer}) + "\n\n")
                elapsed_ms = int((time.time() - start) * 1000)
                await queue.put(
                    "event: done\ndata: "
                    + json.dumps({"query_time_ms": elapsed_ms, "answer": full_answer})
                    + "\n\n"
                )
                await AuditService.log_action(
                    db=db,
                    user_id=policy.user.id,
                    action="query",
                    resource=req.question[:200],
                    department_id=policy.department_id,
                    metadata={
                        "k": req.k,
                        "time_ms": elapsed_ms,
                        "source_count": len(metas),
                        "universe_id": req.universe_id,
                    },
                )
                await db.commit()
                await queue.put(None)
                return

            chunks, metas = await retrieve(
                query=req.question,
                k=req.k,
                source_filter=req.source_filter,
                department_filter=policy.department_filter(),
                universe_id=req.universe_id,
                user_id=policy.user.id,
                is_admin=policy.is_admin,
            )

            inferred_source = _single_retrieved_source(metas) if not req.source_filter else None
            if inferred_source:
                inferred_fact_answer = await answer_from_document_facts(
                    db=db,
                    question=req.question,
                    source_filter=[inferred_source],
                    department_filter=policy.department_filter(),
                    universe_id=req.universe_id,
                    user_id=policy.user.id,
                    is_admin=policy.is_admin,
                )
                if inferred_fact_answer:
                    full_answer, fact_metas = inferred_fact_answer
                    logger.info(
                        "Facts-after-retrieval answer resolved source={} question={}",
                        inferred_source,
                        req.question[:80],
                    )
                    await queue.put("event: meta\ndata: " + json.dumps({"sources": fact_metas}) + "\n\n")
                    await queue.put("event: token\ndata: " + json.dumps({"text": full_answer}) + "\n\n")
                    elapsed_ms = int((time.time() - start) * 1000)
                    await queue.put(
                        "event: done\ndata: "
                        + json.dumps({"query_time_ms": elapsed_ms, "answer": full_answer})
                        + "\n\n"
                    )
                    await AuditService.log_action(
                        db=db,
                        user_id=policy.user.id,
                        action="query",
                        resource=req.question[:200],
                        department_id=policy.department_id,
                        metadata={
                            "k": req.k,
                            "time_ms": elapsed_ms,
                            "source_count": len(fact_metas),
                            "universe_id": req.universe_id,
                            "inferred_source": inferred_source,
                        },
                    )
                    await db.commit()
                    await queue.put(None)
                    return

            await queue.put("event: meta\ndata: " + json.dumps({"sources": metas}) + "\n\n")


            if not chunks:
                full_answer = "Aucune information trouvee dans les documents autorises."
                await queue.put("event: token\ndata: " + json.dumps({"text": full_answer}) + "\n\n")
                elapsed_ms = int((time.time() - start) * 1000)
                await queue.put(
                    "event: done\ndata: "
                    + json.dumps({"query_time_ms": elapsed_ms, "answer": full_answer})
                    + "\n\n"
                )
                await AuditService.log_action(
                    db=db,
                    user_id=policy.user.id,
                    action="query",
                    resource=req.question[:200],
                    department_id=policy.department_id,
                    metadata={
                        "k": req.k,
                        "time_ms": elapsed_ms,
                        "source_count": 0,
                        "universe_id": req.universe_id,
                    },
                )
                await db.commit()
                await queue.put(None)
                return

            system_prompt = None
            if universe_department_id:
                from api.services.prompts import get_system_prompt

                system_prompt = get_system_prompt(
                    universe_department_id,
                    universe_description,
                )

            full_answer = ""
            async for token in stream_llm_answer(
                req.question,
                chunks,
                metas,
                system_prompt=system_prompt,
            ):
                full_answer += token
                await queue.put("event: token\ndata: " + json.dumps({"text": token}) + "\n\n")

            full_answer = full_answer.strip()
            if not full_answer:
                full_answer = _missing_answer_text(req.question)
                await queue.put("event: token\ndata: " + json.dumps({"text": full_answer}) + "\n\n")

            elapsed_ms = int((time.time() - start) * 1000)
            await queue.put(
                "event: done\ndata: "
                + json.dumps({"query_time_ms": elapsed_ms, "answer": full_answer})
                + "\n\n"
            )

            await AuditService.log_action(
                db=db,
                user_id=policy.user.id,
                action="query",
                resource=req.question[:200],
                department_id=policy.department_id,
                metadata={
                    "k": req.k,
                    "time_ms": elapsed_ms,
                    "source_count": len(metas),
                    "universe_id": req.universe_id,
                },
            )
            await db.commit()
            await queue.put(None)

        task = asyncio.create_task(worker())

        yield "event: start\ndata: " + json.dumps({"status": "started"}) + "\n\n"

        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=10)
                except asyncio.TimeoutError:
                    yield "event: ping\ndata: " + json.dumps({"status": "working"}) + "\n\n"
                    continue

                if item is None:
                    break
                yield item
        finally:
            if not task.done():
                task.cancel()

        if task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
