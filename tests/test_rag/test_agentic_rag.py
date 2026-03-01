import pytest
from unittest.mock import AsyncMock, MagicMock

from src.rag.agentic_rag import (
    AgenticRAGService,
    RAGResult,
    FALLBACK_QUESTIONS,
    MEDIUM_FILTER_THRESHOLD,
)
from src.rag.cache import RelevanceGrade
from src.rag.grader import GradingResult
from src.rag.models import RetrievalResult
from src.rag.query_refiner import QueryRefinementStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def retriever():
    mock = AsyncMock()
    # retrieve_questions is sync (wrapped in asyncio.to_thread inside the graph)
    mock.retrieve_questions = MagicMock()
    return mock


@pytest.fixture
def grader():
    return AsyncMock()


@pytest.fixture
def refiner():
    return AsyncMock()


@pytest.fixture
def cache_store():
    store = AsyncMock()
    store.get_topic_questions.return_value = None  # cache miss by default
    return store


@pytest.fixture
def rag_service(retriever, grader, refiner, cache_store):
    return AgenticRAGService(retriever, grader, refiner, cache_store)


@pytest.fixture
def sample_docs():
    return [
        RetrievalResult(
            id="1", text="q1", relevance_score=0.9,
            metadata={"difficulty": "medium", "topic": "ml"},
        ),
        RetrievalResult(
            id="2", text="q2", relevance_score=0.8,
            metadata={"difficulty": "medium", "topic": "ml"},
        ),
    ]


# ---------------------------------------------------------------------------
# Cache hit path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_with_crag_cache_hit(rag_service, cache_store, sample_docs):
    """Cache hit bypasses CRAG entirely and returns served_from_cache=True."""
    cache_store.get_topic_questions.return_value = sample_docs

    result = await rag_service.retrieve_with_crag(
        topic="ml", difficulty="medium", exclude_ids=[], session_id="session_123"
    )

    assert result.served_from_cache is True
    assert result.candidates == sample_docs
    assert result.grade == RelevanceGrade.HIGH

    rag_service.retriever.retrieve_questions.assert_not_called()
    rag_service.grader.grade.assert_not_called()


# ---------------------------------------------------------------------------
# HIGH grade — straightforward retrieval
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_with_crag_high_grade(rag_service, retriever, grader, sample_docs):
    """HIGH grade: no correction loop, all documents returned, attempts=1."""
    retriever.retrieve_questions.return_value = sample_docs
    grader.grade.return_value = GradingResult(
        grade=RelevanceGrade.HIGH, feedback="Great", avg_score=0.85
    )

    result = await rag_service.retrieve_with_crag(
        topic="ml", difficulty="medium", exclude_ids=[]
    )

    assert result.served_from_cache is False
    assert result.corrective_applied is False
    assert result.candidates == sample_docs
    assert result.grade == RelevanceGrade.HIGH
    assert result.attempts == 1

    retriever.retrieve_questions.assert_called_once()
    grader.grade.assert_called_once()


# ---------------------------------------------------------------------------
# MEDIUM grade — package_results_node filtering behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_with_crag_medium_grade_filters_low_score_docs(
    rag_service, retriever, grader
):
    """MEDIUM grade: docs with relevance_score < MEDIUM_FILTER_THRESHOLD are removed."""
    mixed_docs = [
        RetrievalResult(
            id="1", text="q1", relevance_score=MEDIUM_FILTER_THRESHOLD + 0.15,  # above
            metadata={"difficulty": "medium", "topic": "ml"},
        ),
        RetrievalResult(
            id="2", text="q2", relevance_score=MEDIUM_FILTER_THRESHOLD - 0.15,  # below
            metadata={"difficulty": "medium", "topic": "ml"},
        ),
    ]
    retriever.retrieve_questions.return_value = mixed_docs
    grader.grade.return_value = GradingResult(
        grade=RelevanceGrade.MEDIUM, feedback="Mixed quality", avg_score=0.55
    )

    result = await rag_service.retrieve_with_crag(
        topic="ml", difficulty="medium", exclude_ids=[]
    )

    assert result.grade == RelevanceGrade.MEDIUM
    assert result.corrective_applied is False
    # Only the doc with score >= MEDIUM_FILTER_THRESHOLD survives
    assert len(result.candidates) == 1
    assert result.candidates[0].id == "1"


@pytest.mark.asyncio
async def test_retrieve_with_crag_medium_grade_all_below_threshold_returns_all(
    rag_service, retriever, grader
):
    """MEDIUM grade: when ALL docs fall below threshold, all are returned as-is."""
    low_score_docs = [
        RetrievalResult(
            id="1", text="q1", relevance_score=MEDIUM_FILTER_THRESHOLD - 0.25,  # below
            metadata={"difficulty": "medium", "topic": "ml"},
        ),
        RetrievalResult(
            id="2", text="q2", relevance_score=MEDIUM_FILTER_THRESHOLD - 0.15,  # below
            metadata={"difficulty": "medium", "topic": "ml"},
        ),
    ]
    retriever.retrieve_questions.return_value = low_score_docs
    grader.grade.return_value = GradingResult(
        grade=RelevanceGrade.MEDIUM, feedback="Low across the board", avg_score=0.35
    )

    result = await rag_service.retrieve_with_crag(
        topic="ml", difficulty="medium", exclude_ids=[]
    )

    assert result.grade == RelevanceGrade.MEDIUM
    # fallback: nothing passes filter → return all docs unchanged
    assert result.candidates == low_score_docs


# ---------------------------------------------------------------------------
# Correction loop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_with_crag_correction_loop(
    rag_service, retriever, grader, refiner, sample_docs
):
    """LOW grade triggers query refinement; second retrieval succeeds with HIGH."""
    bad_docs = [RetrievalResult(id="0", text="bad", relevance_score=0.2)]
    retriever.retrieve_questions.side_effect = [bad_docs, sample_docs]

    grader.grade.side_effect = [
        GradingResult(grade=RelevanceGrade.LOW, feedback="Bad", avg_score=0.2),
        GradingResult(grade=RelevanceGrade.HIGH, feedback="Good", avg_score=0.85),
    ]

    refiner.refine.return_value = ("new refined query", QueryRefinementStrategy.LLM_REFINE)

    result = await rag_service.retrieve_with_crag(
        topic="ml", difficulty="medium", exclude_ids=[]
    )

    assert result.corrective_applied is True
    assert result.attempts == 2
    assert result.grade == RelevanceGrade.HIGH
    assert result.candidates == sample_docs
    assert result.refined_query == "new refined query"

    assert retriever.retrieve_questions.call_count == 2
    assert grader.grade.call_count == 2
    refiner.refine.assert_called_once()


@pytest.mark.asyncio
async def test_retrieve_with_crag_exhausted_fallback(
    rag_service, retriever, grader, refiner
):
    """MAX_CORRECTION_ATTEMPTS reached with persistent LOW grades → best available returned."""
    bad_docs = [RetrievalResult(id="0", text="bad", relevance_score=0.2)]
    retriever.retrieve_questions.return_value = bad_docs
    grader.grade.return_value = GradingResult(
        grade=RelevanceGrade.LOW, feedback="Bad", avg_score=0.2
    )
    refiner.refine.return_value = ("another query", QueryRefinementStrategy.LLM_REFINE)

    result = await rag_service.retrieve_with_crag(
        topic="ml", difficulty="medium", exclude_ids=[]
    )

    # initial + 2 correction loops = 3 retrieval attempts total
    assert result.grade == RelevanceGrade.LOW
    assert result.corrective_applied is True
    assert result.attempts == 3
    assert result.candidates == bad_docs

    assert retriever.retrieve_questions.call_count == 3


@pytest.mark.asyncio
async def test_retrieve_with_crag_refined_query_is_last_seen_query(
    rag_service, retriever, grader, refiner, sample_docs
):
    """refined_query on the result is the last query attempted during correction."""
    bad_docs = [RetrievalResult(id="0", text="bad", relevance_score=0.2)]
    retriever.retrieve_questions.side_effect = [bad_docs, sample_docs]
    grader.grade.side_effect = [
        GradingResult(grade=RelevanceGrade.LOW, feedback="Bad", avg_score=0.2),
        GradingResult(grade=RelevanceGrade.HIGH, feedback="Good", avg_score=0.85),
    ]
    refiner.refine.return_value = (
        "gradient descent optimization", QueryRefinementStrategy.TOPIC_PIVOT
    )

    result = await rag_service.retrieve_with_crag(
        topic="ml", difficulty="medium", exclude_ids=[]
    )

    assert result.corrective_applied is True
    assert result.refined_query == "gradient descent optimization"


# ---------------------------------------------------------------------------
# Exception / catastrophic fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_with_crag_exception_fallback(rag_service, retriever):
    """Catastrophic retrieval exception falls back to hardcoded FALLBACK_QUESTIONS."""
    retriever.retrieve_questions.side_effect = Exception("ChromaDB blew up")

    result = await rag_service.retrieve_with_crag(
        topic="ml", difficulty="medium", exclude_ids=[]
    )

    assert result.is_fallback is True
    assert result.grade == RelevanceGrade.LOW
    assert result.candidates == FALLBACK_QUESTIONS


# ---------------------------------------------------------------------------
# Session-aware cache storage
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_with_crag_stores_result_in_cache_on_success(
    rag_service, retriever, grader, cache_store, sample_docs
):
    """Successful CRAG retrieval with session_id writes results to the topic cache."""
    retriever.retrieve_questions.return_value = sample_docs
    grader.grade.return_value = GradingResult(
        grade=RelevanceGrade.HIGH, feedback="Great", avg_score=0.85
    )

    result = await rag_service.retrieve_with_crag(
        topic="ml", difficulty="medium", exclude_ids=[], session_id="session_abc"
    )

    assert result.served_from_cache is False
    assert result.grade == RelevanceGrade.HIGH

    cache_store.set_topic_questions.assert_called_once_with(
        session_id="session_abc",
        topic="ml",
        difficulty="medium",
        questions=sample_docs,
        crag_grade=RelevanceGrade.HIGH,
    )


@pytest.mark.asyncio
async def test_retrieve_with_crag_no_session_id_skips_cache_entirely(
    rag_service, retriever, grader, cache_store, sample_docs
):
    """Without session_id the cache is never read or written."""
    retriever.retrieve_questions.return_value = sample_docs
    grader.grade.return_value = GradingResult(
        grade=RelevanceGrade.HIGH, feedback="Great", avg_score=0.85
    )

    await rag_service.retrieve_with_crag(
        topic="ml", difficulty="medium", exclude_ids=[]  # no session_id
    )

    cache_store.get_topic_questions.assert_not_called()
    cache_store.set_topic_questions.assert_not_called()


# ---------------------------------------------------------------------------
# retrieve_batch convenience wrapper
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_batch_calls_crag_with_no_exclusions(
    rag_service, retriever, grader, sample_docs
):
    """retrieve_batch is a thin wrapper that runs CRAG with an empty exclude_ids list."""
    retriever.retrieve_questions.return_value = sample_docs
    grader.grade.return_value = GradingResult(
        grade=RelevanceGrade.HIGH, feedback="Great", avg_score=0.85
    )

    result = await rag_service.retrieve_batch(topic="ml", difficulty="medium")

    assert isinstance(result, RAGResult)
    assert result.candidates == sample_docs
    assert result.grade == RelevanceGrade.HIGH
    retriever.retrieve_questions.assert_called_once()


# ---------------------------------------------------------------------------
# end_interview session cleanup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_end_interview_clears_session_cache(rag_service, cache_store):
    """end_interview delegates to cache_store.clear_session and returns removed count."""
    cache_store.clear_session.return_value = 7

    removed = await rag_service.end_interview("session_xyz")

    assert removed == 7
    cache_store.clear_session.assert_called_once_with(session_id="session_xyz")


# ---------------------------------------------------------------------------
# Structural / sanity checks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_with_crag_latency_always_populated(
    rag_service, retriever, grader, sample_docs
):
    """latency_ms is always a non-negative float, even on the happy path."""
    retriever.retrieve_questions.return_value = sample_docs
    grader.grade.return_value = GradingResult(
        grade=RelevanceGrade.HIGH, feedback="Great", avg_score=0.85
    )

    result = await rag_service.retrieve_with_crag(
        topic="ml", difficulty="medium", exclude_ids=[]
    )

    assert isinstance(result.latency_ms, float)
    assert result.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_retrieve_with_crag_latency_populated_on_fallback(rag_service, retriever):
    """latency_ms is set even when the hardcoded fallback path is taken."""
    retriever.retrieve_questions.side_effect = Exception("boom")

    result = await rag_service.retrieve_with_crag(
        topic="ml", difficulty="medium", exclude_ids=[]
    )

    assert result.is_fallback is True
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms >= 0.0
