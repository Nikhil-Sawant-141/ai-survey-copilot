import json
from typing import Any

import redis.asyncio as aioredis

from app.config import settings

_pool: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _pool
    if _pool is None:
        _pool = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    return _pool


# ─── Session helpers ──────────────────────────────────────────────────────────

async def set_session(session_id: str, data: dict, ttl: int = 7200) -> None:
    r = get_redis()
    await r.setex(f"session:{session_id}", ttl, json.dumps(data))


async def get_session(session_id: str) -> dict | None:
    r = get_redis()
    raw = await r.get(f"session:{session_id}")
    return json.loads(raw) if raw else None


async def delete_session(session_id: str) -> None:
    r = get_redis()
    await r.delete(f"session:{session_id}")


# ─── Rate limiter ─────────────────────────────────────────────────────────────

async def check_rate_limit(key: str, limit: int, window_seconds: int = 3600) -> bool:
    """Returns True if within limit, False if exceeded."""
    r = get_redis()
    full_key = f"rate_limit:{key}"
    count = await r.incr(full_key)
    if count == 1:
        await r.expire(full_key, window_seconds)
    return count <= limit


# ─── Generic cache ────────────────────────────────────────────────────────────

async def cache_set(key: str, value: Any, ttl: int = 3600) -> None:
    r = get_redis()
    await r.setex(key, ttl, json.dumps(value))


async def cache_get(key: str) -> Any | None:
    r = get_redis()
    raw = await r.get(key)
    return json.loads(raw) if raw else None
