from __future__ import annotations

import logging
from datetime import datetime, timezone

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None
_memory: dict[str, dict] = {}  # in-memory fallback when Redis is unavailable


async def connect(url: str) -> None:
    global _redis
    client = aioredis.from_url(url, decode_responses=True)
    await client.ping()  # raises if unreachable — _redis stays None on failure
    _redis = client
    logger.info("Redis connected: %s", url)


async def disconnect() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


def _key(task_id: str) -> str:
    return f"task:{task_id}"


async def create_task(task_id: str, query: str) -> None:
    data = {
        "task_id": task_id,
        "query": query,
        "status": "queued",
        "progress": "",
        "draft": "",
        "report": "",
        "error": "",
        "revision_count": "0",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if _redis is not None:
        await _redis.hset(_key(task_id), mapping=data)
        await _redis.expire(_key(task_id), 86400 * 7)
    else:
        _memory[task_id] = data
        logger.debug("Task %s stored in memory (Redis unavailable)", task_id)


async def get_task(task_id: str) -> dict | None:
    if _redis is not None:
        data = await _redis.hgetall(_key(task_id))
        return data if data else None
    return _memory.get(task_id)


async def set_status(task_id: str, status: str, progress: str = "") -> None:
    if _redis is not None:
        update: dict = {"status": status}
        if progress:
            update["progress"] = progress
        await _redis.hset(_key(task_id), mapping=update)
    elif task_id in _memory:
        _memory[task_id]["status"] = status
        if progress:
            _memory[task_id]["progress"] = progress


async def set_draft(task_id: str, draft: str) -> None:
    if _redis is not None:
        await _redis.hset(_key(task_id), mapping={"draft": draft})
    elif task_id in _memory:
        _memory[task_id]["draft"] = draft


async def set_report(task_id: str, report: str) -> None:
    if _redis is not None:
        await _redis.hset(_key(task_id), mapping={"report": report, "status": "complete"})
    elif task_id in _memory:
        _memory[task_id]["report"] = report
        _memory[task_id]["status"] = "complete"


async def set_error(task_id: str, error: str) -> None:
    if _redis is not None:
        await _redis.hset(_key(task_id), mapping={"error": error, "status": "failed"})
    elif task_id in _memory:
        _memory[task_id]["error"] = error
        _memory[task_id]["status"] = "failed"


async def increment_revision(task_id: str) -> int:
    if _redis is not None:
        count = await _redis.hincrby(_key(task_id), "revision_count", 1)
        return int(count)
    if task_id in _memory:
        current = int(_memory[task_id].get("revision_count", 0))
        _memory[task_id]["revision_count"] = str(current + 1)
        return current + 1
    return 0
