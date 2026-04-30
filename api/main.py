from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable when running from any CWD
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

import store
from agent.graph import graph
from agent.state import ResearchState
from models import (
    DraftResponse,
    FeedbackRequest,
    HealthResponse,
    ReportResponse,
    ResearchRequest,
    TaskResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("api")

# In-process background tasks — keyed by task_id
_running_tasks: dict[str, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    try:
        await store.connect(redis_url)
        logger.info("Startup complete")
    except Exception as exc:
        logger.error("Redis connection failed: %s — tasks will not persist", exc)
    yield
    await store.disconnect()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Research Agent Crew",
    description="LangGraph multi-agent research system with human-in-the-loop review",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Graph execution helpers ────────────────────────────────────────────────────

async def _run_graph_initial(task_id: str, query: str) -> None:
    """Run graph from START. Pauses automatically before human_review."""
    config = {"configurable": {"thread_id": task_id}}
    initial_state: ResearchState = {
        "task_id": task_id,
        "query": query,
        "plan": [],
        "research_results": {},
        "summary": "",
        "fact_check_notes": "",
        "draft": "",
        "human_feedback": "",
        "revision_count": 0,
        "approved": False,
        "final_report": "",
    }
    try:
        await store.set_status(task_id, "running", "Starting pipeline")
        async for chunk in graph.astream(initial_state, config):
            node_name = next(iter(chunk), "")
            if node_name and node_name != "__end__":
                progress = f"Completed: {node_name}"
                await store.set_status(task_id, "running", progress)
                logger.info("Task %s: node %s complete", task_id, node_name)

        # Stream ends when graph is interrupted before human_review
        state_snapshot = graph.get_state(config)
        draft = state_snapshot.values.get("draft", "")
        await store.set_draft(task_id, draft)
        await store.set_status(task_id, "awaiting_approval", "Draft ready for review")
        logger.info("Task %s: draft ready, awaiting human approval", task_id)

    except asyncio.CancelledError:
        await store.set_status(task_id, "cancelled")
    except Exception as exc:
        logger.exception("Graph failed for task %s", task_id)
        await store.set_error(task_id, str(exc))


async def _resume_graph(task_id: str) -> None:
    """Resume graph after human_review (approval or rejection). May pause again on rejection."""
    config = {"configurable": {"thread_id": task_id}}
    try:
        async for chunk in graph.astream(None, config):
            node_name = next(iter(chunk), "")
            if node_name and node_name != "__end__":
                await store.set_status(task_id, "running", f"Completed: {node_name}")
                logger.info("Task %s: resumed node %s", task_id, node_name)

        # Check if stream ended at completion or at another interrupt
        state_snapshot = graph.get_state(config)
        final = state_snapshot.values.get("final_report", "")
        if final:
            await store.set_report(task_id, final)
            logger.info("Task %s: complete", task_id)
        else:
            # Rejected → editor revised → interrupted again before human_review
            draft = state_snapshot.values.get("draft", "")
            await store.set_draft(task_id, draft)
            await store.set_status(task_id, "awaiting_approval", "Revised draft ready")
            logger.info("Task %s: revised draft ready", task_id)

    except asyncio.CancelledError:
        await store.set_status(task_id, "cancelled")
    except Exception as exc:
        logger.exception("Graph resume failed for task %s", task_id)
        await store.set_error(task_id, str(exc))


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    redis_ok = False
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        await r.ping()
        await r.aclose()
        redis_ok = True
    except Exception:
        pass
    return HealthResponse(status="ok", redis=redis_ok)


@app.post("/research", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_research(req: ResearchRequest):
    task_id = str(uuid4())
    await store.create_task(task_id, req.query)
    _running_tasks[task_id] = asyncio.create_task(_run_graph_initial(task_id, req.query))
    logger.info("Started task %s: %s", task_id, req.query[:80])
    return TaskResponse(
        task_id=task_id,
        status="running",
        query=req.query,
        progress="Pipeline started",
    )


@app.get("/status/{task_id}", response_model=TaskResponse)
async def get_status(task_id: str):
    task = await store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return TaskResponse(
        task_id=task_id,
        status=task.get("status", "unknown"),
        query=task.get("query", ""),
        progress=task.get("progress", ""),
        created_at=task.get("created_at", ""),
    )


@app.get("/draft/{task_id}", response_model=DraftResponse)
async def get_draft(task_id: str):
    task = await store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    task_status = task.get("status", "")
    if task_status not in ("awaiting_approval", "revising"):
        raise HTTPException(
            status_code=400,
            detail=f"Draft not available in status '{task_status}'. Wait for 'awaiting_approval'.",
        )
    return DraftResponse(
        task_id=task_id,
        status=task_status,
        draft=task.get("draft", ""),
        revision_count=int(task.get("revision_count", 0)),
    )


@app.post("/approve/{task_id}", response_model=TaskResponse)
async def approve(task_id: str):
    task = await store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    if task.get("status") != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot approve task in status '{task.get('status')}'. Requires 'awaiting_approval'.",
        )
    config = {"configurable": {"thread_id": task_id}}
    await graph.aupdate_state(config, {"approved": True})
    _running_tasks[task_id] = asyncio.create_task(_resume_graph(task_id))
    await store.set_status(task_id, "finalising", "Approved — generating final report")
    logger.info("Task %s: approved by user", task_id)
    return TaskResponse(task_id=task_id, status="finalising", query=task.get("query", ""))


@app.post("/reject/{task_id}", response_model=TaskResponse)
async def reject(task_id: str, body: FeedbackRequest):
    task = await store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    if task.get("status") != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot reject task in status '{task.get('status')}'. Requires 'awaiting_approval'.",
        )
    revision_count = await store.increment_revision(task_id)
    config = {"configurable": {"thread_id": task_id}}
    await graph.aupdate_state(config, {
        "approved": False,
        "human_feedback": body.feedback,
        "revision_count": revision_count,
    })
    _running_tasks[task_id] = asyncio.create_task(_resume_graph(task_id))
    await store.set_status(task_id, "revising", "Revising based on feedback")
    logger.info("Task %s: rejected (revision %d). Feedback: %s", task_id, revision_count, body.feedback[:80])
    return TaskResponse(task_id=task_id, status="revising", query=task.get("query", ""))


@app.get("/report/{task_id}", response_model=ReportResponse)
async def get_report(task_id: str):
    task = await store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    if task.get("status") != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Report not ready. Current status: '{task.get('status')}'.",
        )
    return ReportResponse(
        task_id=task_id,
        status="complete",
        report=task.get("report", ""),
        query=task.get("query", ""),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("Unhandled exception: %s", exc)
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
