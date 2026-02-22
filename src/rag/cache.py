"""
Session-isolated dual-pool cache store for AgenticRAGService.

Architecture (v2):
  InterviewCacheStore (singleton)
    ├── _topic_cache:   dict[session_id → OrderedDict[topic:difficulty → CacheEntry]]
    ├── _concept_cache: dict[session_id → OrderedDict[concept_name → ConceptEntry]]
    ├── _session_locks: defaultdict[session_id → asyncio.Lock]
    └── _global_lock:   asyncio.Lock  (session creation / cleanup only)

Design constraints:
  - Two separate pools evict independently (no cross-eviction)
  - Per-session asyncio.Lock — concurrent interviews never block each other
  - Atomic select_and_mark() eliminates TOCTOU race on question selection
  - Grade-based TTL: HIGH=30 min, MEDIUM=15 min, LOW=never cached
  - Background pre-warming via retrieve_batch (full CRAG → real grades)
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from src.rag.models import RetrievalResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ─── Pool limits ───────────────────────────────────────────────
MAX_TOPICS_PER_SESSION = 10
MAX_CONCEPTS_PER_SESSION = 30

# ─── Grade-based TTL (seconds) ────────────────────────────────
# Architecture v2 values
TTL_HIGH = 1800      # 30 minutes
TTL_MEDIUM = 900     # 15 minutes
TTL_LOW = 300        # 5 minutes — but LOW is never cached in put()

# ─── Session cleanup ──────────────────────────────────────────
ABANDONED_SESSION_THRESHOLD_MINUTES = 90


# ═══════════════════════════════════════════════════════════════
# Enums & shared types
# ═══════════════════════════════════════════════════════════════

class RelevanceGrade(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CacheInvalidationReason(str, Enum):
    TTL_EXPIRED = "ttl_expired"
    QUALITY_TOO_LOW = "quality_too_low"
    POOL_FULL = "pool_full"
    PARTIAL_EXHAUSTED = "partial_exhausted"
    SESSION_CLEARED = "session_cleared"
    SESSION_ABANDONED = "session_abandoned"


TTL_BY_GRADE: Dict[RelevanceGrade, int] = {
    RelevanceGrade.HIGH: TTL_HIGH,
    RelevanceGrade.MEDIUM: TTL_MEDIUM,
    RelevanceGrade.LOW: TTL_LOW,
}


# ═══════════════════════════════════════════════════════════════
# Cache Entries
# ═══════════════════════════════════════════════════════════════

@dataclass
class CacheEntry:
    """A cached batch of topic questions for one session + topic:difficulty."""

    documents: List[RetrievalResult]
    grade: RelevanceGrade
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    hit_count: int = 0
    used_ids: Set[str] = field(default_factory=set)

    # ── TTL / reusability ──────────────────────────────────
    def is_expired(self) -> bool:
        ttl = TTL_BY_GRADE[self.grade]
        return (time.time() - self.created_at) > ttl

    def is_reusable(self) -> bool:
        """LOW results are never reusable."""
        return self.grade in (RelevanceGrade.HIGH, RelevanceGrade.MEDIUM)

    # ── Partial-reuse helpers ──────────────────────────────
    def get_unused(self, exclude_ids: Set[str]) -> List[RetrievalResult]:
        """Return documents not yet served and not in caller's exclude set."""
        combined = self.used_ids | exclude_ids
        return [d for d in self.documents if d.id not in combined]

    def mark_used(self, doc_ids: List[str]) -> None:
        self.used_ids.update(doc_ids)

    def touch(self) -> None:
        """Update LRU timestamp and hit counter."""
        self.last_accessed_at = time.time()
        self.hit_count += 1


@dataclass
class ConceptEntry:
    """A cached concept lookup result."""

    concept_name: str
    data: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > 3600  # 60 min (stable data)

    def touch(self) -> None:
        self.last_accessed_at = time.time()


# ═══════════════════════════════════════════════════════════════
# Observability
# ═══════════════════════════════════════════════════════════════

@dataclass
class CacheMetrics:
    """Observability counters for the cache store."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    invalidations: Dict[str, int] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.total_requests if self.total_requests else 0.0

    def record_hit(self, pool: str, interview_id: str, key: str) -> None:
        self.total_requests += 1
        self.cache_hits += 1
        logger.debug(
            "CACHE_HIT | pool=%s",
            pool,
            extra={"interview_id": interview_id, "key": key,
                    "hit_rate": f"{self.hit_rate:.2%}"},
        )

    def record_miss(self, pool: str, interview_id: str, reason: str) -> None:
        self.total_requests += 1
        self.cache_misses += 1
        logger.debug(
            "CACHE_MISS | pool=%s reason=%s",
            pool, reason,
            extra={"interview_id": interview_id,
                    "hit_rate": f"{self.hit_rate:.2%}"},
        )

    def record_invalidation(self, reason: CacheInvalidationReason) -> None:
        key = reason.value
        self.invalidations[key] = self.invalidations.get(key, 0) + 1


# ═══════════════════════════════════════════════════════════════
# InterviewCacheStore — Dual-pool, per-session-locked singleton
# ═══════════════════════════════════════════════════════════════

class InterviewCacheStore:
    """Singleton holding dual cache pools for ALL concurrent interviews.

    Topic Pool: Stores batches of retrieved questions per topic:difficulty.
                Max 10 entries per session, grade-based TTL.
    Concept Pool: Stores concept lookup results for feedback enrichment.
                  Max 30 entries per session, 60 min TTL.

    Thread safety:
      - Each session gets its own asyncio.Lock (via defaultdict).
      - A global lock protects session creation and cleanup only.
      - Concurrent interviews never block each other.
    """

    def __init__(self) -> None:
        # Per-session pools
        self._topic_cache: Dict[str, OrderedDict[str, CacheEntry]] = {}
        self._concept_cache: Dict[str, OrderedDict[str, ConceptEntry]] = {}
        self._session_created_at: Dict[str, datetime] = {}

        # Locking
        self._session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._global_lock = asyncio.Lock()

        # Observability
        self.metrics = CacheMetrics()

    # ═══════════════════════════════════════════════════════
    #  TOPIC POOL — Questions
    # ═══════════════════════════════════════════════════════

    async def set_topic_questions(
        self,
        session_id: str,
        topic: str,
        difficulty: str,
        questions: List[RetrievalResult],
        crag_grade: RelevanceGrade,
    ) -> None:
        """Store a batch of questions for a topic:difficulty pair.

        LOW-grade batches are rejected — no point caching bad results.
        """
        if crag_grade == RelevanceGrade.LOW:
            logger.debug(
                "Topic cache put() rejected LOW grade | session=%s topic=%s",
                session_id, topic,
            )
            return

        key = f"{topic}:{difficulty}"
        async with self._session_locks[session_id]:
            pool = self._ensure_topic_pool(session_id)
            entry = CacheEntry(documents=questions, grade=crag_grade)
            pool[key] = entry

            # LRU eviction if pool exceeds limit
            while len(pool) > MAX_TOPICS_PER_SESSION:
                oldest_key, _ = pool.popitem(last=False)
                self.metrics.record_invalidation(CacheInvalidationReason.POOL_FULL)
                logger.debug(
                    "Topic pool LRU eviction | session=%s key=%s",
                    session_id, oldest_key,
                )

        logger.debug(
            "Topic cache stored | session=%s key=%s grade=%s docs=%d",
            session_id, key, crag_grade.value, len(questions),
        )

    async def get_topic_questions(
        self,
        session_id: str,
        topic: str,
        difficulty: str,
        exclude_ids: Set[str],
        n_results: int = 5,
    ) -> Optional[List[RetrievalResult]]:
        """Try to serve questions from the topic cache.

        Returns None on miss (expired, exhausted, not found).
        """
        key = f"{topic}:{difficulty}"
        async with self._session_locks[session_id]:
            pool = self._topic_cache.get(session_id, OrderedDict())
            entry = pool.get(key)

            if entry is None:
                self.metrics.record_miss("topic", session_id, "not_found")
                return None

            if not entry.is_reusable():
                pool.pop(key, None)
                self.metrics.record_miss("topic", session_id,
                                         CacheInvalidationReason.QUALITY_TOO_LOW.value)
                return None

            if entry.is_expired():
                pool.pop(key, None)
                self.metrics.record_miss("topic", session_id,
                                         CacheInvalidationReason.TTL_EXPIRED.value)
                return None

            available = entry.get_unused(exclude_ids)
            min_needed = max(1, int(n_results * 0.5))

            if len(available) < min_needed:
                pool.pop(key, None)
                self.metrics.record_miss("topic", session_id,
                                         CacheInvalidationReason.PARTIAL_EXHAUSTED.value)
                return None

            # Hit — move to end for LRU
            pool.move_to_end(key)
            entry.touch()
            self.metrics.record_hit("topic", session_id, key)
            return available[:n_results]

    async def select_and_mark(
        self,
        session_id: str,
        topic: str,
        difficulty: str,
        selector_fn: Callable[[List[RetrievalResult]], Awaitable[RetrievalResult]],
    ) -> Optional[RetrievalResult]:
        """Atomically select a question and mark it as used.

        Runs selector_fn inside the session lock so no TOCTOU race:
        another coroutine can't grab the same question between get and mark.

        Returns None if cache miss (caller should CRAG then retry).
        """
        key = f"{topic}:{difficulty}"
        async with self._session_locks[session_id]:
            pool = self._topic_cache.get(session_id, OrderedDict())
            entry = pool.get(key)

            if entry is None or entry.is_expired() or not entry.is_reusable():
                return None

            available = entry.get_unused(set())
            if not available:
                return None

            selected = await selector_fn(available)
            entry.mark_used([selected.id])
            entry.touch()
            pool.move_to_end(key)
            return selected

    # ═══════════════════════════════════════════════════════
    #  CONCEPT POOL — Feedback enrichment
    # ═══════════════════════════════════════════════════════

    async def get_concept(
        self,
        session_id: str,
        concept_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Look up a cached concept."""
        async with self._session_locks[session_id]:
            pool = self._concept_cache.get(session_id, OrderedDict())
            entry = pool.get(concept_name)

            if entry is None:
                self.metrics.record_miss("concept", session_id, "not_found")
                return None

            if entry.is_expired():
                pool.pop(concept_name, None)
                self.metrics.record_miss("concept", session_id,
                                         CacheInvalidationReason.TTL_EXPIRED.value)
                return None

            pool.move_to_end(concept_name)
            entry.touch()
            self.metrics.record_hit("concept", session_id, concept_name)
            return entry.data

    async def set_concept(
        self,
        session_id: str,
        concept_name: str,
        data: Dict[str, Any],
    ) -> None:
        """Cache a concept lookup result."""
        async with self._session_locks[session_id]:
            pool = self._ensure_concept_pool(session_id)
            pool[concept_name] = ConceptEntry(
                concept_name=concept_name, data=data,
            )
            # LRU eviction
            while len(pool) > MAX_CONCEPTS_PER_SESSION:
                oldest_key, _ = pool.popitem(last=False)
                self.metrics.record_invalidation(CacheInvalidationReason.POOL_FULL)
                logger.debug(
                    "Concept pool LRU eviction | session=%s key=%s",
                    session_id, oldest_key,
                )

    # ═══════════════════════════════════════════════════════
    #  SESSION LIFECYCLE
    # ═══════════════════════════════════════════════════════

    async def clear_session(self, session_id: str) -> int:
        """Remove ALL cache entries for a completed / abandoned session.

        Returns total entries removed.
        """
        removed = 0
        async with self._global_lock:
            if session_id in self._topic_cache:
                removed += len(self._topic_cache.pop(session_id))
            if session_id in self._concept_cache:
                removed += len(self._concept_cache.pop(session_id))
            self._session_created_at.pop(session_id, None)
            # Don't remove the lock — defaultdict re-creates cheaply

        if removed:
            self.metrics.record_invalidation(CacheInvalidationReason.SESSION_CLEARED)
            logger.info(
                "Session cache cleared | session=%s entries_removed=%d",
                session_id, removed,
            )
        return removed

    async def cleanup_abandoned_sessions(self) -> int:
        """Periodic sweep: remove sessions older than threshold.

        Should be called from a background task (e.g., every 15 min).
        """
        now = datetime.now()
        to_remove = []

        async with self._global_lock:
            for sid, created_at in self._session_created_at.items():
                age_minutes = (now - created_at).total_seconds() / 60
                if age_minutes > ABANDONED_SESSION_THRESHOLD_MINUTES:
                    to_remove.append(sid)

        cleaned = 0
        for sid in to_remove:
            cleaned += await self.clear_session(sid)
            self.metrics.record_invalidation(
                CacheInvalidationReason.SESSION_ABANDONED
            )

        if cleaned:
            logger.info(
                "Abandoned session cleanup | sessions=%d entries=%d",
                len(to_remove), cleaned,
            )
        return cleaned

    async def pre_warm_topics_background(
        self,
        session_id: str,
        rag_service: Any,
        topics: List[str],
        difficulty: str,
    ) -> None:
        """Pre-warm cache with upcoming topics via full CRAG.

        Called from FastAPI BackgroundTasks after /start.
        Each topic gets real CRAG grades → accurate TTLs.
        """
        for topic in topics:
            try:
                result = await rag_service.retrieve(
                    topic=topic,
                    difficulty=difficulty,
                    exclude_ids=[],
                    n_results=5,
                )
                if result.documents:
                    await self.set_topic_questions(
                        session_id=session_id,
                        topic=topic,
                        difficulty=difficulty,
                        questions=result.documents,
                        crag_grade=result.grade,
                    )
                    logger.debug(
                        "Pre-warmed topic | session=%s topic=%s",
                        session_id, topic,
                    )
            except Exception as e:
                logger.warning(
                    "Pre-warm failed for topic=%s: %s",
                    topic, str(e),
                )

    # ═══════════════════════════════════════════════════════
    #  INTERNALS
    # ═══════════════════════════════════════════════════════

    def _ensure_topic_pool(self, session_id: str) -> OrderedDict:
        if session_id not in self._topic_cache:
            self._topic_cache[session_id] = OrderedDict()
            self._session_created_at[session_id] = datetime.now()
        return self._topic_cache[session_id]

    def _ensure_concept_pool(self, session_id: str) -> OrderedDict:
        if session_id not in self._concept_cache:
            self._concept_cache[session_id] = OrderedDict()
            if session_id not in self._session_created_at:
                self._session_created_at[session_id] = datetime.now()
        return self._concept_cache[session_id]

    # ── Diagnostics ────────────────────────────────────────

    @property
    def active_sessions(self) -> int:
        return len(
            set(self._topic_cache.keys()) | set(self._concept_cache.keys())
        )

    def topic_pool_size(self, session_id: str) -> int:
        return len(self._topic_cache.get(session_id, {}))

    def concept_pool_size(self, session_id: str) -> int:
        return len(self._concept_cache.get(session_id, {}))


# ═══════════════════════════════════════════════════════════════
# Singleton accessor
# ═══════════════════════════════════════════════════════════════

_cache_store: Optional[InterviewCacheStore] = None


def get_cache_store() -> InterviewCacheStore:
    """Returns the process-wide singleton InterviewCacheStore.

    Call once at app startup; inject via dependency.
    """
    global _cache_store
    if _cache_store is None:
        _cache_store = InterviewCacheStore()
        logger.info("InterviewCacheStore initialized (dual-pool)")
    return _cache_store
"""
Backward compatibility: CacheKey is no longer used by the new dual-pool
design, but existing code may import it. Export a simple NamedTuple
so imports don't break until all callers are updated.
"""
from collections import namedtuple  # noqa: E402

CacheKey = namedtuple("CacheKey", ["interview_id", "difficulty", "topic_intent", "stage"])
"""Deprecated: CacheKey is no longer used by InterviewCacheStore.
The dual-pool design uses simple string keys (topic:difficulty, concept_name).
Kept for backward compatibility during migration.
"""
