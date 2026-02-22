"""
Agentic RAG Service — CRAG via LangGraph subgraph.

Architecture:
  AgenticRAGService (stateless facade)
      │
      ├─ InterviewCacheStore  (dual-pool, session-isolated singleton)
      ├─ crag_graph           (compiled ONCE at init, invoked per call)
      │      retrieve_node → grade_node → route_after_grade
      │                                       ├─ HIGH/MED → package_results_node
      │                                       └─ LOW      → refine_query_node → retrieve_node
      ├─ DocumentGrader       (hybrid: score-fast-path + LLM borderline)
      └─ QueryRefiner         (loop-safe: LLM → pivot → simplify)

Public API (agents call ONLY this):
  rag.retrieve(topic, difficulty, exclude_ids, n_results) → RAGResult
  rag.end_interview(session_id)

Design principles:
  - AgenticRAGService is stateless; all session state lives in InterviewCacheStore
  - CRAG is a LangGraph StateGraph subgraph for tracing, visualization, composability
  - Always retrieves — no Self-RAG gate (interview system always needs fresh questions)
  - Fallback: hardcoded ML questions if all retrieval paths fail
"""
from __future__ import annotations

import time
import asyncio
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple

from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict

from src.rag.cache import (
    InterviewCacheStore,
    RelevanceGrade,
    get_cache_store,
)
from src.rag.grader import DocumentGrader
from src.rag.models import RetrievalContext, RetrievalResult
from src.rag.retriever import VectorRetriever
from src.rag.query_refiner import QueryRefiner
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

MAX_CORRECTION_ATTEMPTS = 2
MEDIUM_FILTER_THRESHOLD = 0.55




# ═══════════════════════════════════════════════════════════════
# Fallback questions — last resort if CRAG exhausts
# ═══════════════════════════════════════════════════════════════

FALLBACK_QUESTIONS: List[RetrievalResult] = [
    RetrievalResult(
        id="fallback_001",
        text="Explain the bias-variance tradeoff in machine learning.",
        relevance_score=0.5,
        metadata={"difficulty": "medium", "topic": "ml_fundamentals",
                  "source": "fallback"},
    ),
    RetrievalResult(
        id="fallback_002",
        text="What is gradient descent and how does the learning rate "
             "affect convergence?",
        relevance_score=0.5,
        metadata={"difficulty": "medium", "topic": "optimization",
                  "source": "fallback"},
    ),
    RetrievalResult(
        id="fallback_003",
        text="Describe the difference between precision and recall. "
             "When would you prioritise each?",
        relevance_score=0.5,
        metadata={"difficulty": "medium", "topic": "evaluation_metrics",
                  "source": "fallback"},
    ),
    RetrievalResult(
        id="fallback_004",
        text="What is regularisation? Compare L1 and L2 penalties.",
        relevance_score=0.5,
        metadata={"difficulty": "medium", "topic": "regularization",
                  "source": "fallback"},
    ),
    RetrievalResult(
        id="fallback_005",
        text="Explain how a transformer's self-attention mechanism works.",
        relevance_score=0.5,
        metadata={"difficulty": "hard", "topic": "transformers",
                  "source": "fallback"},
    ),
]


# ═══════════════════════════════════════════════════════════════
# RAGResult — what agents receive from AgenticRAGService
# ═══════════════════════════════════════════════════════════════

@dataclass
class RAGResult:
    """Result from AgenticRAGService.retrieve_with_crag().

    Superset of v2's CRAGResult — adds observability fields.
    """
    candidates: List[RetrievalResult]
    grade: RelevanceGrade
    attempts: int = 1
    refined_query: Optional[str] = None
    served_from_cache: bool = False
    corrective_applied: bool = False
    queries_used: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    is_fallback: bool = False


# ═══════════════════════════════════════════════════════════════
# CRAG LangGraph Subgraph — State + Nodes + Graph builder
# ═══════════════════════════════════════════════════════════════

class CRAGState(TypedDict):
    """Internal state for the CRAG correction subgraph."""
    query: str
    documents: List[RetrievalResult]
    grade: Optional[str]
    grading_feedback: str
    correction_count: int
    seen_queries: List[str]
    # Retrieval context fields (TypedDicts must be JSON-serializable)
    difficulty: str
    topic_intent: str
    n_results: int
    exclude_ids: List[str]
    final_documents: List[RetrievalResult]


async def retrieve_node(
    state: CRAGState,
    retriever: VectorRetriever,
) -> Dict[str, Any]:
    """Query ChromaDB for candidate questions.

    ChromaDB is sync — wrapped in asyncio.to_thread() to avoid
    blocking the event loop.
    """
    docs: List[RetrievalResult] = await asyncio.to_thread(
        retriever.retrieve_questions,
        query=state["query"],
        difficulty=state["difficulty"] or None,
        topic=state["topic_intent"] or None,
        exclude_ids=set(state["exclude_ids"]),
        n_results=state["n_results"],
    )
    return {
        "documents": docs,
        "seen_queries": state["seen_queries"] + [state["query"]],
    }


async def grade_node(
    state: CRAGState,
    grader: DocumentGrader,
) -> Dict[str, Any]:
    """Grade retrieved documents.

    topic_intent passed explicitly; fast paths skip LLM;
    borderline invokes structured-output chain.
    """
    context = RetrievalContext(difficulty_level=state["difficulty"])
    result = await grader.grade(
        documents=state["documents"],
        context=context,
        topic_intent=state["topic_intent"],
    )
    return {
        "grade": result.grade.value,
        "grading_feedback": result.feedback,
    }


async def refine_query_node(
    state: CRAGState,
    refiner: QueryRefiner,
) -> Dict[str, Any]:
    """Refine query for next retrieval attempt.

    Strategy rotation: LLM refine → topic pivot → simplify.
    """
    new_query, strategy = await refiner.refine(
        original_query=state["query"],
        feedback=state["grading_feedback"],
        difficulty=state["difficulty"],
        seen_queries=state["seen_queries"],
        attempt=state["correction_count"],
    )
    logger.info(
        "Query refined | strategy=%s | query=%s",
        strategy.value, new_query,
    )
    return {
        "query": new_query,
        "correction_count": state["correction_count"] + 1,
    }


def package_results_node(state: CRAGState) -> Dict[str, Any]:
    """Finalize documents based on grade.

    HIGH   — all documents
    MEDIUM — filter by relevance_score >= threshold, fallback to all
    LOW    — best available sorted by score (correction exhausted)
    """
    grade = RelevanceGrade(state["grade"]) if state["grade"] else RelevanceGrade.LOW
    docs = state["documents"]

    if grade == RelevanceGrade.HIGH:
        final = docs
    elif grade == RelevanceGrade.MEDIUM:
        filtered = [d for d in docs if d.relevance_score >= MEDIUM_FILTER_THRESHOLD]
        final = filtered if filtered else docs
    else:
        final = sorted(docs, key=lambda d: d.relevance_score, reverse=True)

    return {"final_documents": final}


def route_after_grade(state: CRAGState) -> str:
    """Conditional edge: route to correction or packaging."""
    grade = state.get("grade")
    attempts = state.get("correction_count", 0)

    if grade in (RelevanceGrade.HIGH.value, RelevanceGrade.MEDIUM.value):
        return "package_results"

    if attempts >= MAX_CORRECTION_ATTEMPTS:
        logger.info(
            "CRAG correction exhausted | attempts=%s grade=%s — "
            "packaging best available",
            attempts, grade,
        )
        return "package_results"

    return "refine_query"


def build_crag_graph(
    retriever: VectorRetriever,
    grader: DocumentGrader,
    refiner: QueryRefiner,
) -> Any:
    """Build and compile the CRAG StateGraph.

    Compiled once at AgenticRAGService init.
    Each ainvoke() gets an independent execution context — concurrent-safe.
    """
    workflow = StateGraph(CRAGState)

    # functools.partial injects dependencies (DI without classes)
    workflow.add_node("retrieve", partial(retrieve_node, retriever=retriever))
    workflow.add_node("grade", partial(grade_node, grader=grader))
    workflow.add_node("refine_query", partial(refine_query_node, refiner=refiner))
    workflow.add_node("package_results", package_results_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "refine_query": "refine_query",
            "package_results": "package_results",
        },
    )
    workflow.add_edge("refine_query", "retrieve")   # correction loop
    workflow.add_edge("package_results", END)

    return workflow.compile()


# ═══════════════════════════════════════════════════════════════
# AgenticRAGService — Stateless facade
# ═══════════════════════════════════════════════════════════════

class AgenticRAGService:
    """Stateless RAG facade. All session state lives in InterviewCacheStore.

    Flow:
      1. Check topic cache → serve if hit
      2. Run CRAG subgraph → retrieve + grade + correct
      3. Store in cache
      4. Fallback to hardcoded questions if all else fails

    Usage:
        rag = AgenticRAGService(retriever, grader, refiner, cache_store)
        result = await rag.retrieve_with_crag(
            topic="ml_fundamentals", difficulty="medium",
            exclude_ids=[], n_results=5,
        )
    """

    def __init__(
        self,
        retriever: VectorRetriever,
        grader: DocumentGrader,
        refiner: QueryRefiner,
        cache_store: Optional[InterviewCacheStore] = None,
    ) -> None:
        self.retriever = retriever
        self.grader = grader
        self.refiner = refiner
        self.cache_store = cache_store or get_cache_store()
        self.crag_graph = build_crag_graph(retriever, grader, refiner)

        logger.info("AgenticRAGService initialized")

    async def retrieve_with_crag(
        self,
        topic: str,
        difficulty: str,
        exclude_ids: List[str],
        remaining_time: Optional[float] = None,
        n_results: int = 5,
        session_id: Optional[str] = None,
    ) -> RAGResult:
        """Primary retrieval API — called by QuestionSelectorAgent.

        Tries cache → CRAG → fallback in order.
        If session_id is provided, results are cached for reuse.

        Args:
            topic: ML topic to retrieve questions for
            difficulty: Target difficulty level
            exclude_ids: Already-asked question IDs
            remaining_time: Minutes left in interview (for time-aware filtering)
            n_results: Number of candidates to retrieve
            session_id: Interview session ID (enables caching)
        """
        start = time.time()

        # 1. Try topic cache
        if session_id:
            cached = await self.cache_store.get_topic_questions(
                session_id=session_id,
                topic=topic,
                difficulty=difficulty,
                exclude_ids=set(exclude_ids),
                n_results=n_results,
            )
            if cached:
                return RAGResult(
                    candidates=cached,
                    grade=RelevanceGrade.HIGH,  # Cache only stores HIGH/MEDIUM
                    served_from_cache=True,
                    latency_ms=(time.time() - start) * 1000,
                )

        # 2. CRAG subgraph
        try:
            crag_state: CRAGState = {
                "query": topic,
                "documents": [],
                "grade": None,
                "grading_feedback": "",
                "correction_count": 0,
                "seen_queries": [],
                "difficulty": difficulty,
                "topic_intent": topic,
                "n_results": n_results,
                "exclude_ids": exclude_ids,
                "final_documents": [],
            }

            result_state = await self.crag_graph.ainvoke(crag_state)

            final_docs = result_state.get("final_documents", [])
            grade_str = result_state.get("grade", "LOW")
            grade = RelevanceGrade(grade_str) if grade_str else RelevanceGrade.LOW
            queries_used = result_state.get("seen_queries", [])
            correction_count = result_state.get("correction_count", 0)
            corrective = correction_count > 0

            if final_docs:
                # 3. Cache results for reuse
                if session_id:
                    await self.cache_store.set_topic_questions(
                        session_id=session_id,
                        topic=topic,
                        difficulty=difficulty,
                        questions=final_docs,
                        crag_grade=grade,
                    )

                return RAGResult(
                    candidates=final_docs,
                    grade=grade,
                    attempts=correction_count + 1,
                    refined_query=queries_used[-1] if corrective else None,
                    corrective_applied=corrective,
                    queries_used=queries_used,
                    latency_ms=(time.time() - start) * 1000,
                )

        except Exception as e:
            logger.error("CRAG subgraph failed: %s", str(e), exc_info=True)

        # 4. Fallback
        logger.warning(
            "All retrieval paths exhausted — serving fallback | "
            "topic=%s difficulty=%s",
            topic, difficulty,
        )
        return RAGResult(
            candidates=FALLBACK_QUESTIONS,
            grade=RelevanceGrade.LOW,
            is_fallback=True,
            latency_ms=(time.time() - start) * 1000,
        )

    async def retrieve_batch(
        self,
        topic: str,
        difficulty: str,
        n_results: int = 5,
    ) -> RAGResult:
        """Retrieve a batch with no exclusions — used for cache pre-warming.

        Thin wrapper over retrieve_with_crag() with empty exclude_ids.
        """
        return await self.retrieve_with_crag(
            topic=topic,
            difficulty=difficulty,
            exclude_ids=[],
            n_results=n_results,
        )

    async def end_interview(self, session_id: str) -> int:
        """Cleanup: remove all cached data for a finished session."""
        removed = await self.cache_store.clear_session(session_id)
        logger.info(
            "Interview ended — cache cleared | session=%s entries=%d",
            session_id, removed,
        )
        return removed
