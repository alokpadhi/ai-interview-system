import pytest
from unittest.mock import AsyncMock, patch

from src.rag.query_refiner import QueryRefiner, QueryRefinementStrategy, RefinedQuery


@pytest.fixture
def mock_llm_refiner():
    return AsyncMock()


@pytest.fixture
def query_refiner(mock_llm_refiner):
    with patch("src.rag.query_refiner.get_secondary_llm"):
        qr = QueryRefiner()
        qr._llm_refiner = mock_llm_refiner
        return qr


def test_pick_strategy(query_refiner):
    """Test strategy rotation based on attempt count."""
    assert query_refiner._pick_strategy(0) == QueryRefinementStrategy.LLM_REFINE
    assert query_refiner._pick_strategy(1) == QueryRefinementStrategy.TOPIC_PIVOT
    assert query_refiner._pick_strategy(2) == QueryRefinementStrategy.SIMPLIFY
    # Should clamp at SIMPLIFY
    assert query_refiner._pick_strategy(5) == QueryRefinementStrategy.SIMPLIFY


def test_simplify(query_refiner):
    """Test simplification which mechanically strips noise words."""
    q = "what is the best activation function for convolutional neural networks"
    # Stop words matched: "is", "the", "for" -> removed 
    # Words left: "what", "best", "activation", "function", "convolutional", "neural", "networks"
    # Takes first 2 -> "what best" + difficulty
    res = query_refiner._simplify(q, "hard")
    assert res == "hard what best"


def test_topic_pivot(query_refiner):
    """Test topic pivot avoids covered topics."""
    from src.rag.query_refiner import ML_TOPICS
    res = query_refiner._topic_pivot("medium", ["deep_learning_fundamentals", "machine_learning_fundamentals"])
    
    assert res.startswith("medium ")
    # The pivoted topic is what follows "medium "
    topic_words = res.split(" ", 1)[1]
    # Reconstruct snake_case to check if it's in ML_TOPICS
    topic_snake = topic_words.replace(" ", "_")
    
    assert topic_snake in ML_TOPICS
    assert topic_snake != "deep_learning_fundamentals"
    assert topic_snake != "machine_learning_fundamentals"


def test_too_similar(query_refiner):
    """Test the difflib similarity guard."""
    # Exact match is 1.0 > 0.85
    assert query_refiner._too_similar("deep learning", ["deep learning"]) is True
    # Completely different
    assert query_refiner._too_similar("deep learning", ["random forest", "svm"]) is False
    # Very minor rephrase -> ratio > 0.85
    assert query_refiner._too_similar("how to do deep learning well", ["how to do deep learning very well"]) is True


def test_force_different(query_refiner):
    """Test mechanical fallback when similarity check fails."""
    # Takes "easy", first 2 words of original ("machine", "learning") + first uncovered topic
    res = query_refiner._force_different("machine learning basics", ["ml_fundamentals"], "easy")
    assert res.startswith("easy machine learning ")


@pytest.mark.asyncio
async def test_refine_llm_strategy(query_refiner, mock_llm_refiner):
    """Test standard LLM refinement."""
    mock_llm_refiner.ainvoke.return_value = RefinedQuery(query="new query", rationale="test")
    
    new_q, strategy = await query_refiner.refine(
        original_query="old query",
        feedback="too basic",
        difficulty="hard",
        seen_queries=[],
        attempt=0,
        covered_topics=[]
    )
    
    assert new_q == "new query"
    assert strategy == QueryRefinementStrategy.LLM_REFINE
    mock_llm_refiner.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_refine_too_similar_forces_change(query_refiner, mock_llm_refiner):
    """Test that LLM hallucinating a slightly different wording gets caught and forced."""
    # LLM returns something too similar to "old query"
    mock_llm_refiner.ainvoke.return_value = RefinedQuery(query="an old query", rationale="slight tweak")
    
    new_q, strategy = await query_refiner.refine(
        original_query="old query",
        feedback="too basic",
        difficulty="hard",
        seen_queries=["old query", "an old query"],
        attempt=0,
        covered_topics=[]
    )
    
    assert strategy == QueryRefinementStrategy.FORCED
    assert new_q != "an old query"
    assert new_q != "old query"
