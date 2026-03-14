"""
Unit tests for QuestionSelectorAgent — pure methods only (no real LLM calls).

Covers:
- _get_remaining_minutes       (time budget calculation)
- _determine_question_mode     (routing rules)
- _get_next_topic_from_plan    (topic sequencing + fundamentals priority)
- _select_weakest_topic        (lowest-avg topic selection)
- _get_performance_trend       (EMA-based trend detection)
- _react_select                (LLM selection with edge cases)
- _generate_follow_up          (follow-up question structure)
- _generate_clarification      (clarification question structure)
- _get_fallback_question       (fallback shape)
- execute                      (dispatch: retrieve / follow_up / clarify)
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.agents.question_selector import QuestionSelectorAgent
from src.services.validation import ValidationGateRegistry, CircuitBreaker


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_agent() -> QuestionSelectorAgent:
    """
    Bypass QuestionSelectorAgent.__init__.
    All chains and dependencies are MagicMock / AsyncMock.
    """
    a = QuestionSelectorAgent.__new__(QuestionSelectorAgent)
    a.rag = MagicMock()
    a.fast_llm = MagicMock()
    a.complex_llm = MagicMock()

    cache = MagicMock()
    cache.get_concept = AsyncMock(return_value=None)
    cache.set_concept = AsyncMock()
    cache.select_and_mark = AsyncMock(return_value=None)
    cache.set_topic_questions = AsyncMock()
    a.cache_store = cache

    a.follow_up_chain = AsyncMock()
    a.clarify_chain = AsyncMock()
    a.react_select_chain = AsyncMock()
    a.validation_gates = ValidationGateRegistry().get("question_selector")
    a.circuit_breaker = CircuitBreaker(max_retries=1)
    return a


def make_state(
    question_count: int = 1,
    time_budget_minutes: int = 30,
    interview_start_time=None,
    difficulty_level: str = "medium",
    follow_up_count: int = 0,
    topics_covered: "list | None" = None,
    overall_score: float = 5.0,
    key_points_missed: "list | None" = None,
    misconceptions: "list | None" = None,
    performance_trajectory: "list | None" = None,
    ema_trajectory: "list | None" = None,
    all_evaluations: "list | None" = None,
    difficulty_reduced: bool = False,
    interview_plan: "dict | None" = None,
) -> dict:
    return {
        "interview_id": "session_001",
        "question_count": question_count,
        "time_budget_minutes": time_budget_minutes,
        "interview_start_time": interview_start_time,
        "difficulty_level": difficulty_level,
        "follow_up_count": follow_up_count,
        "topics_covered": topics_covered or [],
        "current_evaluation": {
            "overall_score": overall_score,
            "key_points_missed": key_points_missed or [],
            "misconceptions": misconceptions or [],
        },
        "current_question": {
            "id": "q_001",
            "text": "Explain gradient descent and its variants.",
            "topic": "optimization",
            "difficulty": "medium",
        },
        "candidate_response": "Gradient descent iteratively minimises the loss function.",
        "performance_trajectory": performance_trajectory or [],
        "ema_trajectory": ema_trajectory or [],
        "all_evaluations": all_evaluations or [],
        "difficulty_reduced_due_to_performance": difficulty_reduced,
        "interview_plan": interview_plan or {
            "topic_sequence": ["machine_learning_fundamentals", "deep_learning", "optimization"]
        },
    }


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    return make_agent()


# ── _get_remaining_minutes ────────────────────────────────────────────────────

class TestGetRemainingMinutes:

    def test_no_start_time_returns_full_budget(self, agent):
        state = make_state(time_budget_minutes=30, interview_start_time=None)
        assert agent._get_remaining_minutes(state) == 30

    def test_elapsed_subtracted_from_budget(self, agent):
        start = datetime.now() - timedelta(minutes=10)
        state = make_state(time_budget_minutes=30, interview_start_time=start)
        remaining = agent._get_remaining_minutes(state)
        assert 19.0 <= remaining <= 21.0  # allow ±1 min for execution time

    def test_remaining_clamped_to_zero_when_overrun(self, agent):
        start = datetime.now() - timedelta(minutes=35)
        state = make_state(time_budget_minutes=30, interview_start_time=start)
        assert agent._get_remaining_minutes(state) == 0


# ── _determine_question_mode ──────────────────────────────────────────────────

class TestDetermineQuestionMode:

    def test_first_question_always_retrieve(self, agent):
        state = make_state(question_count=0)
        assert agent._determine_question_mode(state, 25.0) == "retrieve"

    def test_low_remaining_time_forces_retrieve(self, agent):
        state = make_state(question_count=3, overall_score=5.0, key_points_missed=["bias"])
        assert agent._determine_question_mode(state, 3.0) == "retrieve"

    def test_misconception_triggers_clarify(self, agent):
        state = make_state(
            question_count=2,
            overall_score=5.0,
            misconceptions=["Confused regularization with normalization"],
            follow_up_count=0,
        )
        assert agent._determine_question_mode(state, 20.0) == "clarify"

    def test_low_score_with_missed_points_triggers_follow_up(self, agent):
        state = make_state(
            question_count=2,
            overall_score=5.0,
            key_points_missed=["tradeoff mechanism"],
            follow_up_count=0,
        )
        assert agent._determine_question_mode(state, 20.0) == "follow_up"

    def test_mid_score_with_missed_points_and_no_follow_ups_triggers_follow_up(self, agent):
        """Score 7.0–8.0 with missed points and follow_up_count < 1 → follow_up."""
        state = make_state(
            question_count=2,
            overall_score=7.5,
            key_points_missed=["cross-entropy loss"],
            follow_up_count=0,
        )
        assert agent._determine_question_mode(state, 20.0) == "follow_up"

    def test_mid_score_follow_up_count_1_retrieves(self, agent):
        """Score 7.0–8.0 but follow_up_count >= 1 → skip follow-up."""
        state = make_state(
            question_count=3,
            overall_score=7.5,
            key_points_missed=["cross-entropy loss"],
            follow_up_count=1,
        )
        assert agent._determine_question_mode(state, 20.0) == "retrieve"

    def test_max_follow_ups_reached_returns_retrieve(self, agent):
        state = make_state(
            question_count=3,
            overall_score=5.0,
            key_points_missed=["tradeoff"],
            misconceptions=["wrong claim"],
            follow_up_count=QuestionSelectorAgent.MAX_FOLLOW_UPS,
        )
        assert agent._determine_question_mode(state, 20.0) == "retrieve"

    def test_high_score_no_missed_returns_retrieve(self, agent):
        state = make_state(
            question_count=2,
            overall_score=9.0,
            key_points_missed=[],
            misconceptions=[],
            follow_up_count=0,
        )
        assert agent._determine_question_mode(state, 20.0) == "retrieve"


# ── _get_next_topic_from_plan ─────────────────────────────────────────────────

class TestGetNextTopicFromPlan:

    def test_returns_first_uncovered_topic(self, agent):
        state = make_state(
            topics_covered=["machine_learning_fundamentals"],
            interview_plan={"topic_sequence": ["machine_learning_fundamentals", "deep_learning", "optimization"]},
        )
        assert agent._get_next_topic_from_plan(state) == "deep_learning"

    def test_returns_first_topic_when_nothing_covered(self, agent):
        state = make_state(
            topics_covered=[],
            interview_plan={"topic_sequence": ["machine_learning_fundamentals", "deep_learning"]},
        )
        assert agent._get_next_topic_from_plan(state) == "machine_learning_fundamentals"

    def test_all_covered_delegates_to_select_weakest(self, agent):
        state = make_state(
            topics_covered=["machine_learning_fundamentals", "deep_learning"],
            interview_plan={"topic_sequence": ["machine_learning_fundamentals", "deep_learning"]},
            all_evaluations=[
                {"topic": "machine_learning_fundamentals", "overall_score": 4.0},
                {"topic": "deep_learning", "overall_score": 7.0},
            ],
        )
        # machine_learning_fundamentals has lowest avg → selected
        result = agent._get_next_topic_from_plan(state)
        assert result == "machine_learning_fundamentals"

    def test_difficulty_reduced_reprioritises_fundamentals(self, agent):
        """When difficulty_reduced=True, fundamental topics bubble to the front."""
        state = make_state(
            topics_covered=[],
            difficulty_reduced=True,
            interview_plan={
                "topic_sequence": [
                    "mlops",
                    "machine_learning_fundamentals",
                    "deep_learning_fundamentals",
                ]
            },
        )
        result = agent._get_next_topic_from_plan(state)
        assert result in QuestionSelectorAgent.FUNDAMENTAL_TOPICS


# ── _select_weakest_topic ─────────────────────────────────────────────────────

class TestSelectWeakestTopic:

    def test_returns_topic_with_lowest_avg_score(self, agent):
        state = make_state(
            all_evaluations=[
                {"topic": "optimization", "overall_score": 8.0},
                {"topic": "deep_learning", "overall_score": 4.0},
                {"topic": "deep_learning", "overall_score": 5.0},
            ]
        )
        assert agent._select_weakest_topic(state) == "deep_learning"

    def test_skips_unknown_topic(self, agent):
        state = make_state(
            all_evaluations=[
                {"topic": "unknown", "overall_score": 1.0},
                {"topic": "optimization", "overall_score": 7.0},
            ]
        )
        # "unknown" filtered out; "optimization" is the only real topic
        assert agent._select_weakest_topic(state) == "optimization"

    def test_no_evaluations_returns_default(self, agent):
        state = make_state(all_evaluations=[])
        assert agent._select_weakest_topic(state) == "machine_learning_fundamentals"

    def test_single_topic_returns_it(self, agent):
        state = make_state(
            all_evaluations=[{"topic": "mlops", "overall_score": 6.0}]
        )
        assert agent._select_weakest_topic(state) == "mlops"


# ── _get_performance_trend ────────────────────────────────────────────────────

class TestGetPerformanceTrend:

    def test_fewer_than_3_trajectory_points_is_stable(self, agent):
        state = make_state(performance_trajectory=[7.0, 6.5])
        assert agent._get_performance_trend(state) == "stable"

    def test_improving_trend_detected(self, agent):
        # ema[-1] - ema[-4] > 0.8
        ema = [5.0, 5.3, 5.7, 6.2, 6.9]
        state = make_state(performance_trajectory=[5.0, 5.3, 5.7, 6.2, 6.9], ema_trajectory=ema)
        assert agent._get_performance_trend(state) == "improving"

    def test_declining_trend_detected(self, agent):
        # ema[-1] - ema[-4] < -0.8
        ema = [8.0, 7.5, 7.0, 6.5, 6.0]
        state = make_state(performance_trajectory=[8.0, 7.5, 7.0, 6.5, 6.0], ema_trajectory=ema)
        assert agent._get_performance_trend(state) == "declining"

    def test_flat_ema_returns_stable(self, agent):
        ema = [6.0, 6.1, 6.0, 6.1, 6.0]
        state = make_state(performance_trajectory=ema, ema_trajectory=ema)
        assert agent._get_performance_trend(state) == "stable"

    def test_short_ema_with_3_trajectory_returns_stable(self, agent):
        """Trajectory has 3 items but EMA also has 3 — too short for trend."""
        ema = [5.0, 6.0, 7.5]
        state = make_state(performance_trajectory=ema, ema_trajectory=ema)
        assert agent._get_performance_trend(state) == "stable"


# ── _react_select ─────────────────────────────────────────────────────────────

class TestReactSelect:

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_fallback(self, agent):
        state = make_state()
        result = await agent._react_select([], state, MagicMock())
        assert result["id"].startswith("fallback_")

    @pytest.mark.asyncio
    async def test_single_candidate_returned_without_llm_call(self, agent):
        state = make_state()
        candidate = {"id": "q_only", "text": "Describe dropout.", "difficulty": "medium"}
        result = await agent._react_select([candidate], state, MagicMock())
        assert result == candidate
        agent.react_select_chain.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_matching_id_returns_correct_candidate(self, agent):
        state = make_state()
        candidates = [
            {"id": "q_001", "text": "Question A.", "difficulty": "medium"},
            {"id": "q_002", "text": "Question B.", "difficulty": "hard"},
        ]
        selection = MagicMock()
        selection.selected_id = "q_002"
        agent.react_select_chain.ainvoke = AsyncMock(return_value=selection)

        result = await agent._react_select(candidates, state, MagicMock())
        assert result["id"] == "q_002"

    @pytest.mark.asyncio
    async def test_unmatched_id_falls_back_to_first_candidate(self, agent):
        state = make_state()
        candidates = [
            {"id": "q_001", "text": "Question A.", "difficulty": "medium"},
            {"id": "q_002", "text": "Question B.", "difficulty": "hard"},
        ]
        selection = MagicMock()
        selection.selected_id = "q_999"  # doesn't match anything
        agent.react_select_chain.ainvoke = AsyncMock(return_value=selection)

        result = await agent._react_select(candidates, state, MagicMock())
        assert result == candidates[0]


# ── _generate_follow_up ───────────────────────────────────────────────────────

class TestGenerateFollowUp:

    @pytest.mark.asyncio
    async def test_follow_up_has_correct_structure(self, agent):
        state = make_state(
            follow_up_count=0,
            key_points_missed=["L2 penalty", "weight decay", "dropout"],
        )
        agent.follow_up_chain.ainvoke = AsyncMock(
            return_value="Can you expand on how weight decay affects generalisation?"
        )
        result = await agent._generate_follow_up(state, MagicMock())

        assert result["question_type"] == "follow_up"
        assert "q_001_followup_1" in result["id"]
        assert result["topic"] == "optimization"
        assert result["parent_question_id"] == "q_001"

    @pytest.mark.asyncio
    async def test_target_concepts_limited_to_two_missed_points(self, agent):
        state = make_state(
            key_points_missed=["L2 penalty", "weight decay", "dropout"],
        )
        agent.follow_up_chain.ainvoke = AsyncMock(return_value="Question text.")
        result = await agent._generate_follow_up(state, MagicMock())

        assert result["target_concepts"] == ["L2 penalty", "weight decay"]

    @pytest.mark.asyncio
    async def test_topic_defaults_to_general_when_absent(self, agent):
        state = make_state(key_points_missed=["concept"])
        state["current_question"].pop("topic")
        agent.follow_up_chain.ainvoke = AsyncMock(return_value="Follow up?")
        result = await agent._generate_follow_up(state, MagicMock())
        assert result["topic"] == "general"

    @pytest.mark.asyncio
    async def test_follow_up_text_stripped(self, agent):
        state = make_state(key_points_missed=["bias"])
        agent.follow_up_chain.ainvoke = AsyncMock(return_value="  Follow up question?  ")
        result = await agent._generate_follow_up(state, MagicMock())
        assert result["text"] == "Follow up question?"


# ── _generate_clarification ───────────────────────────────────────────────────

class TestGenerateClarification:

    @pytest.mark.asyncio
    async def test_clarification_has_correct_structure(self, agent):
        state = make_state(
            misconceptions=["Confused normalization with regularization"],
        )
        agent.clarify_chain.ainvoke = AsyncMock(
            return_value="Can you elaborate on what regularization does?"
        )
        result = await agent._generate_clarification(state, MagicMock())

        assert result["question_type"] == "clarification"
        assert "q_001_clarify" in result["id"]
        assert result["topic"] == "optimization"
        assert result["parent_question_id"] == "q_001"

    @pytest.mark.asyncio
    async def test_target_misconception_set_to_first_misconception(self, agent):
        misconception = "Confused normalization with regularization"
        state = make_state(misconceptions=[misconception, "another misconception"])
        agent.clarify_chain.ainvoke = AsyncMock(return_value="Clarification question?")
        result = await agent._generate_clarification(state, MagicMock())
        assert result["target_misconception"] == misconception

    @pytest.mark.asyncio
    async def test_target_concepts_contains_misconception(self, agent):
        misconception = "Confused normalization with regularization"
        state = make_state(misconceptions=[misconception])
        agent.clarify_chain.ainvoke = AsyncMock(return_value="Clarification question?")
        result = await agent._generate_clarification(state, MagicMock())
        assert result["target_concepts"] == [misconception]

    @pytest.mark.asyncio
    async def test_clarification_text_stripped(self, agent):
        state = make_state(misconceptions=["wrong concept"])
        agent.clarify_chain.ainvoke = AsyncMock(return_value="  Clarification?  ")
        result = await agent._generate_clarification(state, MagicMock())
        assert result["text"] == "Clarification?"

    @pytest.mark.asyncio
    async def test_topic_injected_from_current_question(self, agent):
        state = make_state(misconceptions=["wrong concept"])
        state["current_question"]["topic"] = "deep_learning"
        agent.clarify_chain.ainvoke = AsyncMock(return_value="Clarification?")
        result = await agent._generate_clarification(state, MagicMock())
        assert result["topic"] == "deep_learning"


# ── _get_fallback_question ────────────────────────────────────────────────────

class TestGetFallbackQuestion:

    def test_id_starts_with_fallback(self, agent):
        q = agent._get_fallback_question()
        assert q["id"].startswith("fallback_")

    def test_topic_is_ml_fundamentals(self, agent):
        q = agent._get_fallback_question()
        assert q["topic"] == "machine_learning_fundamentals"

    def test_question_type_is_retrieved(self, agent):
        q = agent._get_fallback_question()
        assert q["question_type"] == "retrieved"

    def test_target_concepts_non_empty(self, agent):
        """Drift detection must work on fallback questions too."""
        q = agent._get_fallback_question()
        assert len(q["target_concepts"]) > 0

    def test_two_calls_produce_unique_ids(self, agent):
        q1 = agent._get_fallback_question()
        q2 = agent._get_fallback_question()
        assert q1["id"] != q2["id"]


# ── execute (dispatch) ────────────────────────────────────────────────────────

class TestExecuteDispatch:

    @pytest.mark.asyncio
    async def test_retrieve_mode_resets_follow_up_count(self, agent):
        """retrieve mode → follow_up_count=0 and topics_covered updated."""
        state = make_state(question_count=0)
        question = {
            "id": "q_new",
            "text": "What is the bias-variance tradeoff?",
            "topic": "machine_learning_fundamentals",
            "difficulty": "medium",
            "question_type": "retrieved",
            "estimated_time_minutes": 4,
        }
        agent._retrieve_question = AsyncMock(return_value=(question, "machine_learning_fundamentals"))
        config = MagicMock()

        result = await agent.execute(state, config)

        assert result["question_mode"] == "retrieve"
        assert result["follow_up_count"] == 0
        assert "machine_learning_fundamentals" in result["topics_covered"]
        assert result["current_question"] == question

    @pytest.mark.asyncio
    async def test_follow_up_mode_increments_follow_up_count(self, agent):
        state = make_state(
            question_count=2,
            follow_up_count=0,
            overall_score=5.0,
            key_points_missed=["tradeoff mechanism"],
        )
        follow_up_q = {
            "id": "q_001_followup_1",
            "text": "Can you explain the tradeoff?",
            "topic": "optimization",
            "question_type": "follow_up",
            "difficulty": "medium",
            "estimated_time_minutes": 3,
            "target_concepts": ["tradeoff mechanism"],
        }
        agent._generate_follow_up = AsyncMock(return_value=follow_up_q)
        config = MagicMock()

        result = await agent.execute(state, config)

        assert result["question_mode"] == "follow_up"
        assert result["follow_up_count"] == 1
        assert result["topics_covered"] == []

    @pytest.mark.asyncio
    async def test_clarify_mode_increments_follow_up_count(self, agent):
        state = make_state(
            question_count=2,
            follow_up_count=0,
            overall_score=5.0,
            misconceptions=["Confused bias with variance"],
        )
        clarify_q = {
            "id": "q_001_clarify",
            "text": "Can you clarify what you mean by bias?",
            "topic": "optimization",
            "question_type": "clarification",
            "difficulty": "medium",
            "estimated_time_minutes": 3,
            "target_concepts": ["Confused bias with variance"],
        }
        agent._generate_clarification = AsyncMock(return_value=clarify_q)
        config = MagicMock()

        result = await agent.execute(state, config)

        assert result["question_mode"] == "clarify"
        assert result["follow_up_count"] == 1
        assert result["topics_covered"] == []

    @pytest.mark.asyncio
    async def test_retrieve_mode_sets_conversation_thread(self, agent):
        state = make_state(question_count=0)
        question = {
            "id": "q_retrieved_01",
            "text": "Explain attention mechanisms.",
            "topic": "deep_learning",
            "difficulty": "hard",
            "question_type": "retrieved",
            "estimated_time_minutes": 5,
        }
        agent._retrieve_question = AsyncMock(return_value=(question, "deep_learning"))
        config = MagicMock()

        result = await agent.execute(state, config)

        assert result["conversation_thread"] == ["q_retrieved_01"]

    @pytest.mark.asyncio
    async def test_follow_up_mode_sets_conversation_thread(self, agent):
        state = make_state(
            question_count=2,
            overall_score=5.0,
            key_points_missed=["attention heads"],
            follow_up_count=0,
        )
        follow_up_q = {
            "id": "q_001_followup_1",
            "text": "How many attention heads are typically used?",
            "topic": "deep_learning",
            "question_type": "follow_up",
            "difficulty": "medium",
            "estimated_time_minutes": 3,
            "target_concepts": ["attention heads"],
        }
        agent._generate_follow_up = AsyncMock(return_value=follow_up_q)
        config = MagicMock()

        result = await agent.execute(state, config)

        assert result["conversation_thread"] == ["q_001_followup_1"]
