import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.agents.supervisor import (
    SupervisorAgent, PlanOutput, Observation, Analysis,
    DIFFICULTY_ORDER, DIFFICULTY_FROM_ORDER, NEUTRAL_EMA
)
from src.services.trend_analyzer import TrendAnalyzer
from src.services.validation import ValidationGateRegistry, CircuitBreaker


# ─────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def trend_analyzer():
    return TrendAnalyzer(alpha=0.3)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=llm)
    llm.with_retry = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def supervisor(mock_llm, trend_analyzer):
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.complex_llm = mock_llm
    agent.trend_analyzer = trend_analyzer
    agent.validation_gates = ValidationGateRegistry()
    agent.circuit_breaker = CircuitBreaker(max_retries=1)
    agent.plan_chain = AsyncMock()
    return agent


@pytest.fixture
def base_state():
    """Minimal valid state for validate_and_decide."""
    return {
        "interview_id": "test-session-123",
        "question_count": 3,
        "difficulty_level": "medium",
        "original_difficulty": "medium",
        "difficulty_reduced_due_to_performance": False,
        "time_budget_minutes": 30,
        "interview_start_time": datetime.now(),
        "performance_trajectory": [6.0, 7.0, 6.5],
        "ema_trajectory": [6.0, 6.3, 6.4],
        "difficulty_history": ["medium", "medium", "medium"],
        "all_evaluations": [],
        "topics_covered": ["gradient_descent", "backprop"],
        "question_mode": "retrieve",
        "interview_plan": {
            "topic_sequence": [
                "gradient_descent", "backprop", "regularization",
                "transformers", "attention", "rag", "evaluation"
            ],
            "difficulty_curve": [
                "easy", "medium", "medium", "hard", "hard", "hard", "hard"
            ],
            "time_allocation": {"gradient_descent": 5.0},
            "focus_areas": []
        },
        "current_evaluation": {
            "overall_score": 7.5,
            "is_fallback": False,
            "topic": "backprop",
            "reasoning": "Good understanding demonstrated with clear explanation",
            "key_points_covered": ["chain rule"],
            "key_points_missed": [],
            "misconceptions": []
        },
        "focus_topics": []
    }


# ─────────────────────────────────────────────────────────────────
# 1. create_interview_plan
# ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_interview_plan_returns_correct_keys(supervisor, base_state):
    """Plan creation returns all required state keys."""
    plan_output = PlanOutput(
        topic_sequence=["gradient_descent", "backprop", "regularization"],
        difficulty_curve=["easy", "medium", "hard"],
        time_allocation={"gradient_descent": 10.0, "backprop": 10.0, "regularization": 10.0},
        focus_areas=[]
    )
    supervisor.plan_chain.ainvoke = AsyncMock(return_value=plan_output)

    result = await supervisor.create_interview_plan(base_state, config=MagicMock())

    assert "interview_plan" in result
    assert "difficulty_level" in result
    assert "original_difficulty" in result
    assert "difficulty_reduced_due_to_performance" in result
    assert "stage" in result


@pytest.mark.asyncio
async def test_create_interview_plan_seeds_difficulty_from_curve(supervisor, base_state):
    """difficulty_level is seeded from curve[0], not from request difficulty."""
    plan_output = PlanOutput(
        topic_sequence=["gradient_descent", "backprop"],
        difficulty_curve=["easy", "hard"],  # curve[0] = "easy"
        time_allocation={"gradient_descent": 15.0, "backprop": 15.0},
        focus_areas=[]
    )
    supervisor.plan_chain.ainvoke = AsyncMock(return_value=plan_output)
    base_state["difficulty_level"] = "hard"  # request says hard

    result = await supervisor.create_interview_plan(base_state, config=MagicMock())

    assert result["difficulty_level"] == "easy"  # curve[0] wins
    assert result["original_difficulty"] == "hard"  # original preserved


@pytest.mark.asyncio
async def test_create_interview_plan_stores_as_dict(supervisor, base_state):
    """interview_plan stored as dict not Pydantic model."""
    plan_output = PlanOutput(
        topic_sequence=["gradient_descent"],
        difficulty_curve=["medium"],
        time_allocation={"gradient_descent": 30.0},
        focus_areas=[]
    )
    supervisor.plan_chain.ainvoke = AsyncMock(return_value=plan_output)

    result = await supervisor.create_interview_plan(base_state, config=MagicMock())

    assert isinstance(result["interview_plan"], dict)
    assert result["difficulty_reduced_due_to_performance"] is False


# ─────────────────────────────────────────────────────────────────
# 2. validate_and_decide — reducer semantics
# ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fallback_excluded_from_trajectory(supervisor, base_state):
    """Fallback evaluation returns empty trajectory — not added to EMA."""
    base_state["current_evaluation"]["is_fallback"] = True

    result = await supervisor.validate_and_decide(base_state, config=MagicMock())

    assert result["performance_trajectory"] == []


@pytest.mark.asyncio
async def test_non_fallback_returns_only_new_score(supervisor, base_state):
    """operator.add reducer — only new score returned, not full history."""
    base_state["current_evaluation"]["overall_score"] = 8.0
    base_state["current_evaluation"]["is_fallback"] = False

    result = await supervisor.validate_and_decide(base_state, config=MagicMock())

    assert result["performance_trajectory"] == [8.0]  # only new item


@pytest.mark.asyncio
async def test_question_count_always_incremented(supervisor, base_state):
    """question_count increments regardless of should_continue."""
    base_state["question_count"] = 3

    result = await supervisor.validate_and_decide(base_state, config=MagicMock())

    assert result["question_count"] == 4


# ─────────────────────────────────────────────────────────────────
# 3. _decide_continuation
# ─────────────────────────────────────────────────────────────────

def test_decide_continuation_time_critical(supervisor, base_state):
    """time_critical terminates with time_up."""
    analysis = Analysis(
        time_critical=True, time_pressure=True,
        should_adjust_difficulty=False, adjustment_direction="stable",
        performance_trend="stable", questions_remaining=3, avg_ema=6.0
    )

    should_continue, reason = supervisor._decide_continuation(analysis, base_state)

    assert should_continue is False
    assert reason == "time_up"


def test_decide_continuation_completed(supervisor, base_state):
    """Reaching target question count terminates with completed."""
    analysis = Analysis(
        time_critical=False, time_pressure=False,
        should_adjust_difficulty=False, adjustment_direction="stable",
        performance_trend="stable", questions_remaining=0, avg_ema=6.0
    )
    # question_count + 1 >= len(topic_sequence) = 7
    base_state["question_count"] = 6

    should_continue, reason = supervisor._decide_continuation(analysis, base_state)

    assert should_continue is False
    assert reason == "completed"


def test_decide_continuation_continues(supervisor, base_state):
    """Neither condition met — interview continues."""
    analysis = Analysis(
        time_critical=False, time_pressure=False,
        should_adjust_difficulty=False, adjustment_direction="stable",
        performance_trend="stable", questions_remaining=3, avg_ema=6.0
    )

    should_continue, reason = supervisor._decide_continuation(analysis, base_state)

    assert should_continue is True
    assert reason is None


# ─────────────────────────────────────────────────────────────────
# 4. _resolve_difficulty
# ─────────────────────────────────────────────────────────────────

def make_analysis(
    time_pressure=False, time_critical=False,
    should_adjust=False, direction="stable", trend="stable"
) -> Analysis:
    return Analysis(
        time_critical=time_critical,
        time_pressure=time_pressure,
        should_adjust_difficulty=should_adjust,
        adjustment_direction=direction,
        performance_trend=trend,
        questions_remaining=3,
        avg_ema=6.0
    )


def test_resolve_difficulty_time_pressure(supervisor, base_state):
    """Time pressure — keep current difficulty."""
    analysis = make_analysis(time_pressure=True)

    difficulty, reduced = supervisor._resolve_difficulty(analysis, base_state)

    assert difficulty == base_state["difficulty_level"]
    assert reduced is False


def test_resolve_difficulty_follow_up_mode(supervisor, base_state):
    """Follow-up mode — keep current difficulty."""
    base_state["question_mode"] = "follow_up"
    analysis = make_analysis()

    difficulty, reduced = supervisor._resolve_difficulty(analysis, base_state)

    assert difficulty == base_state["difficulty_level"]
    assert reduced is False


def test_resolve_difficulty_no_adjustment(supervisor, base_state):
    """EMA says stable — return plan difficulty."""
    analysis = make_analysis(should_adjust=False)
    # topics_covered has 2 items → index 2 → "medium" from curve
    difficulty, reduced = supervisor._resolve_difficulty(analysis, base_state)

    assert difficulty == "medium"
    assert reduced is False


def test_resolve_difficulty_increase(supervisor, base_state):
    """EMA says increase — take harder of plan vs ema_adjusted."""
    analysis = make_analysis(should_adjust=True, direction="increase")
    base_state["difficulty_level"] = "medium"

    difficulty, reduced = supervisor._resolve_difficulty(analysis, base_state)

    assert difficulty in ("medium", "hard")
    assert reduced is False


def test_resolve_difficulty_decrease(supervisor, base_state):
    """EMA says decrease — take easier, reduced=True."""
    analysis = make_analysis(should_adjust=True, direction="decrease")
    base_state["difficulty_level"] = "hard"

    difficulty, reduced = supervisor._resolve_difficulty(analysis, base_state)

    assert difficulty in ("easy", "medium")
    assert reduced is True


# ─────────────────────────────────────────────────────────────────
# 5. _get_plan_difficulty_for_next_topic
# ─────────────────────────────────────────────────────────────────

def test_plan_difficulty_uses_topic_index(supervisor, base_state):
    """Indexes by len(topics_covered), not question_count."""
    # topics_covered has 2 items → index 2 → "medium"
    base_state["question_count"] = 5  # deliberately different from topic index

    result = supervisor._get_plan_difficulty_for_next_topic(base_state)

    assert result == "medium"  # curve[2] = "medium"


def test_plan_difficulty_beyond_curve(supervisor, base_state):
    """Beyond curve length — falls back to current difficulty."""
    base_state["topics_covered"] = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]

    result = supervisor._get_plan_difficulty_for_next_topic(base_state)

    assert result == base_state["difficulty_level"]


def test_plan_difficulty_empty_plan(supervisor, base_state):
    """Missing plan — falls back to current difficulty."""
    base_state["interview_plan"] = {}

    result = supervisor._get_plan_difficulty_for_next_topic(base_state)

    assert result == base_state["difficulty_level"]


# ─────────────────────────────────────────────────────────────────
# 6. _harder_of / _easier_of
# ─────────────────────────────────────────────────────────────────

def test_harder_of(supervisor):
    assert supervisor._harder_of("easy", "hard") == "hard"
    assert supervisor._harder_of("medium", "medium") == "medium"


def test_easier_of(supervisor):
    assert supervisor._easier_of("hard", "medium") == "medium"
    assert supervisor._easier_of("easy", "easy") == "easy"