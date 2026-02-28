import pytest
from unittest.mock import AsyncMock, patch

from src.rag.cache import RelevanceGrade
from src.rag.grader import DocumentGrader, DocumentGrade, _FallbackGrade
from src.rag.models import RetrievalContext, RetrievalResult


@pytest.fixture
def mock_chain():
    return AsyncMock()


@pytest.fixture
def grader(mock_chain):
    with patch("src.rag.grader.get_secondary_llm"):
        # We patch get_secondary_llm so it doesn't try to instantiate a real LLM for the chain
        # But we override _chain immediately anyway
        grader = DocumentGrader()
        grader._chain = mock_chain
        return grader


@pytest.fixture
def base_docs():
    return [
        RetrievalResult(
            id="1",
            text="doc1",
            relevance_score=0.85,
            metadata={"difficulty": "medium", "topic": "ml_fundamentals"}
        ),
        RetrievalResult(
            id="2",
            text="doc2",
            relevance_score=0.80,
            metadata={"difficulty": "medium", "topic": "ml_fundamentals"}
        ),
    ]


@pytest.fixture
def context():
    return RetrievalContext(difficulty_level="medium")


@pytest.mark.asyncio
async def test_grader_fast_path_high(grader, base_docs, context):
    """Test that high average score bypasses LLM and returns HIGH."""
    result = await grader.grade(base_docs, context, topic_intent="ml_fundamentals")
    
    assert result.grade == RelevanceGrade.HIGH
    assert not result.used_llm
    # Scores are 0.85 and 0.80 -> avg 0.825
    assert result.avg_score == 0.825
    assert grader._chain.ainvoke.call_count == 0


@pytest.mark.asyncio
async def test_grader_fast_path_low(grader, context):
    """Test that low average score bypasses LLM and returns LOW."""
    docs = [
        RetrievalResult(id="1", text="bad", relevance_score=0.3, metadata={"difficulty": "medium"}),
        RetrievalResult(id="2", text="worse", relevance_score=0.2, metadata={"difficulty": "medium"}),
    ]
    result = await grader.grade(docs, context, topic_intent="ml_fundamentals")
    
    assert result.grade == RelevanceGrade.LOW
    assert not result.used_llm
    assert grader._chain.ainvoke.call_count == 0


@pytest.mark.asyncio
async def test_grader_borderline_llm(grader, context, mock_chain):
    """Test that borderline average score triggers LLM."""
    docs = [
        RetrievalResult(id="1", text="ok", relevance_score=0.6, metadata={"difficulty": "medium", "topic": "ml_fundamentals"}),
        RetrievalResult(id="2", text="okish", relevance_score=0.55, metadata={"difficulty": "medium", "topic": "ml_fundamentals"}),
    ]
    
    # Mock LLM returning MEDIUM
    mock_chain.ainvoke.return_value = DocumentGrade(grade="MEDIUM", feedback="It is ok")
    
    result = await grader.grade(docs, context, topic_intent="ml_fundamentals")
    
    assert result.grade == RelevanceGrade.MEDIUM
    assert result.used_llm
    assert result.feedback == "It is ok"
    mock_chain.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_grader_llm_fallback(grader, context, mock_chain):
    """Test that grader gracefully handles complete LLM failure by receiving _FallbackGrade."""
    docs = [
        RetrievalResult(id="1", text="ok", relevance_score=0.6, metadata={"difficulty": "medium", "topic": "ml_fundamentals"}),
    ]
    
    # Mock LLM failing completely (returns _FallbackGrade)
    mock_chain.ainvoke.return_value = _FallbackGrade()
    
    result = await grader.grade(docs, context, topic_intent="ml_fundamentals")
    
    # Fallback grade is always MEDIUM
    assert result.grade == RelevanceGrade.MEDIUM
    assert not result.used_llm  # Because LLM failed
    assert "LLM grading unavailable" in result.feedback


@pytest.mark.asyncio
async def test_grader_penalties(grader, context):
    """Test that difficulty and topic mismatches penalise the score and force LOW."""
    # Base score is 0.70 (which would be borderline LLM), but penalties should push it to fast-path LOW
    docs = [
        RetrievalResult(
            id="1", 
            text="doc1", 
            relevance_score=0.70, 
            metadata={
                "difficulty": "hard",  # mismatch (context is medium) -> -0.10 penalty
                "topic": "deep_learning"  # mismatch (intent is ml_fundamentals) -> -0.15 penalty
            }
        ),
    ]
    
    result = await grader.grade(docs, context, topic_intent="ml_fundamentals")
    
    # 0.70 - 0.10 - 0.15 = 0.45 (which hits LOW_SCORE_THRESHOLD)
    assert result.penalised_score == pytest.approx(0.45)
    assert result.avg_score == 0.70
    assert result.grade == RelevanceGrade.LOW
    assert not result.used_llm


@pytest.mark.asyncio
async def test_grader_empty_documents(grader, context):
    """Test behaviour with empty documents list."""
    result = await grader.grade([], context)
    assert result.grade == RelevanceGrade.LOW
    assert result.avg_score == 0.0
