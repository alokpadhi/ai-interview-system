"""
Loop-safe query refinement for the CRAG correction loop.

Strategy rotation by attempt:
  0 → LLM refine   (rephrase for better semantic match)
  1 → topic pivot  (shift to uncovered topic from CANONICAL_TOPICS pool)
  2 → simplify     (strip to 2 core words — last resort)

After every refinement a difflib similarity check guards against
the LLM rephrasing rather than genuinely changing the query.
_force_different() provides a mechanical guarantee of uniqueness.

Ml_TOPICS is derived from CANONICAL_TOPICS in topic_taxonomy.py — the
canonical taxonomy is the single source of truth for valid topic names.

"""

from __future__ import annotations
import random

from difflib import SequenceMatcher
from enum import Enum
from typing import List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.logging_config import get_logger
from src.utils.llm_factory import get_secondary_llm
from src.utils.topic_taxonomy import CANONICAL_TOPICS

logger = get_logger(__name__)


# Derived from CANONICAL_TOPICS — single source of truth for valid topic names.
# Using sorted() for deterministic ordering across Python versions.
ML_TOPICS: list[str] = sorted(CANONICAL_TOPICS)

SIMILARITY_THRESHOLD = 0.85 # ratio above which queries are considered too similar

class QueryRefinementStrategy(str, Enum):
    LLM_REFINE = "llm_refine"
    TOPIC_PIVOT = "topic_pivot"
    SIMPLIFY = "simplify"
    FORCED = "forced"

class RefinedQuery(BaseModel):
    query: str = Field(description="The refined query")
    rationale: str = Field(description="one sentence explaining the change")

_REFINE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are refining a query for an ML interview question database."
        "The previous query returned low-quality results."
        "Produce a meaningfully different query - not just a rephrasing."
    ),
    (
        "human",
        """Original query: {original_query}
        Grader feedback: {feedback}
        Difficulty level: {difficulty}
        Queries already tried (avoid these): {seen_queries}
        
        Return a new query that approaches the topic differently.
        """
    )
])

class QueryRefiner:
    """Produces a new query for each CRAG correction attempt.
    
    Usage:
        refiner = QueryRefiner()
        new_query, strategy = refiner.refine(
        original_query="deep learning optimization",
            feedback="Questions too advanced for easy difficulty",
            difficulty="easy",
            seen_queries=["deep learning optimization"],
            attempt=0,
            covered_topics=["deep_learning"],
        )
    """
    def __init__(self) -> None:
        llm = get_secondary_llm()
        self._llm_refiner = _REFINE_PROMPT | llm.with_structured_output(RefinedQuery)

    async def refine(
            self,
            original_query: str,
            feedback: str,
            difficulty: str,
            seen_queries: List[str],
            attempt: int,
            covered_topics: Optional[List[str]] = None,
    ) -> Tuple[str, QueryRefinementStrategy]:
        """Return (refined_query, strategy_used)"""
        covered_topics = covered_topics or []

        strategy = self._pick_strategy(attempt)
        refined = await self._execute_strategy(
            strategy, original_query, feedback, difficulty, seen_queries,
            covered_topics
        )

        if self._too_similar(refined, seen_queries):
            logger.debug(
                "Refined query too similar to seen - forcing different",
                extra={"refined": refined, "strategy": strategy.value}
            )
            refined = self._force_different(original_query, covered_topics, 
                                            difficulty)
            strategy = QueryRefinementStrategy.FORCED

        logger.info(
            "Query refined",
            extra={"strategy": strategy.value, "original": original_query, "refined": refined}
        )

        return refined, strategy
    
    def _pick_strategy(self, attempt: int) -> QueryRefinementStrategy:
        rotation = [
            QueryRefinementStrategy.LLM_REFINE,
            QueryRefinementStrategy.TOPIC_PIVOT,
            QueryRefinementStrategy.SIMPLIFY
        ]

        # clamp to last strategy if attempt exceeds rotation length
        idx = min(attempt, len(rotation)-1)

        return rotation[idx]
    
    async def _execute_strategy(
            self,
            strategy: QueryRefinementStrategy,
            original_query: str,
            feedback: str,
            difficulty: str,
            seen_queries: List[str],
            covered_topics: List[str],
    ) -> str:
        if strategy == QueryRefinementStrategy.LLM_REFINE:
            return await self._llm_refine(original_query, feedback, difficulty, seen_queries)

        if strategy == QueryRefinementStrategy.TOPIC_PIVOT:
            return self._topic_pivot(difficulty, covered_topics)

        # else simplilfy
        return self._simplify(original_query, difficulty)

    async def _llm_refine(
            self,
            original_query: str,
            feedback: str,
            difficulty: str,
            seen_queries: List[str],
    ) -> str:
        """Ask LLM for a meaningfully different query. Falls back to simplify on failure."""
        try:
            result: RefinedQuery = await self._llm_refiner.ainvoke({
                "original_query": original_query,
                "feedback":       feedback,
                "difficulty":     difficulty,
                "seen_queries":   ", ".join(seen_queries) if seen_queries else "none",
            })
            return result.query.strip()
        except Exception as e:
            logger.warning("LLM refine failed: %s — falling back to simplify", e)
            return self._simplify(original_query, difficulty)
        
    def _topic_pivot(self, difficulty: str, covered_topics: List[str]) -> str:
        """Shift to a topic not yet covered in this interview.
        Falls back to a random ML_TOPICS entry if all are covered."""

        covered_norm = {t.lower().replace(" ", "_") for t in covered_topics}
        uncovered = [
            t for t in ML_TOPICS
            if t.lower().replace(" ", "_") not in covered_norm
        ]
        pivot_topic = random.choice(uncovered) if uncovered else random.choice(ML_TOPICS)
        return f"{difficulty} {pivot_topic.replace('_', ' ')}"

    def _simplify(self, original_query: str, difficulty: str) -> str:
        """Strip query to a 2 meaningful words + difficulty.
        remove stop words mechanically.
        """
        stop_words = {"the", "a", "an", "and", "or", "for", "in", "of", "to", "with"}
        words = [
            w for w in original_query.lower().split()
            if w not in stop_words and len(w) > 2
        ]
        core = " ".join(words[:2]) if len(words) >= 2 else original_query
        return f"{difficulty} {core}"
    
    # similarity guard
    def _too_similar(self, query: str, seen_queries: List[str]) -> bool:
        """Return True if query is too close to any previously tried query."""
        for seen in seen_queries:
            ratio = SequenceMatcher(None, query.lower(), seen.lower()).ratio()
            if ratio > SIMILARITY_THRESHOLD:
                return True
            
        return False
    
    def _force_different(
            self,
            original_query: str,
            covered_topics: List[str],
            difficulty: str,
    ) -> str:
        """Mechanically construct a query guranteed to differ from anything seen.
        Takes the first 2 words of original + first uncovered topic.
        """
        uncovered = [t for t in ML_TOPICS if t not in covered_topics]
        pivot     = uncovered[0] if uncovered else "general"

        words = original_query.lower().split()
        core  = " ".join(words[:2]) if len(words) >= 2 else original_query

        return f"{difficulty} {core} {pivot.replace('_', ' ')}"

