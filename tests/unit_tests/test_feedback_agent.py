"""
Unit tests for FeedbackAgent — pure methods only (no real LLM calls).

Covers:
- FeedbackComposer  (_get_score_band, _select_transition, _select_structure, compose)
- FeedbackAgent     (_get_tone_guidance, _get_concept_context, execute)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.feedback import FeedbackAgent, FeedbackComposer, FeedbackComponents
from src.services.validation import ValidationGateRegistry, CircuitBreaker


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_concept_lookup_patch(return_value: dict):
    """Replace concept_lookup in the feedback module namespace."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=return_value)
    return patch("src.agents.feedback.concept_lookup", mock)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def composer():
    return FeedbackComposer()


@pytest.fixture
def agent():
    """
    Bypass FeedbackAgent.__init__ — unit tests target pure / async methods
    and must not trigger LangChain chain construction.
    """
    a = FeedbackAgent.__new__(FeedbackAgent)
    a.fast_llm = MagicMock()
    a.feedback_chain = AsyncMock()
    a.repetition_chain = AsyncMock()
    a.composer = FeedbackComposer()
    a.validation_gate = ValidationGateRegistry().get("feedback")
    a.circuit_breaker = CircuitBreaker(max_retries=1)

    cache = MagicMock()
    cache.get_concept = AsyncMock(return_value=None)
    cache.set_concept = AsyncMock()
    a.cache_store = cache
    return a


@pytest.fixture
def base_components():
    return FeedbackComponents(
        strength_acknowledgment="You demonstrated solid intuition about gradient flow.",
        gap_hint="It is worth exploring when depth matters more than breadth.",
        transition_phrase="Building on that...",
    )


@pytest.fixture
def base_state():
    return {
        "interview_id": "session_001",
        "current_question": {
            "id": "q_001",
            "text": "Explain the bias-variance tradeoff.",
            "topic": "ml_fundamentals",
        },
        "candidate_response": "Bias is underfitting and variance is overfitting.",
        "current_evaluation": {
            "overall_score": 6.0,
            "key_points_missed": ["tradeoff mechanism", "regularization link"],
            "misconceptions": [],
        },
        "question_count": 2,
        "previous_feedback_structures": [],
        "recent_feedbacks": [],
    }


# ── FeedbackComposer._get_score_band ─────────────────────────────────────────

class TestGetScoreBand:

    def test_score_8_is_high(self, composer):
        assert composer._get_score_band(8.0) == "high"

    def test_score_above_8_is_high(self, composer):
        assert composer._get_score_band(9.5) == "high"

    def test_score_exactly_5_is_medium(self, composer):
        assert composer._get_score_band(5.0) == "medium"

    def test_score_between_5_and_8_is_medium(self, composer):
        assert composer._get_score_band(6.5) == "medium"

    def test_score_just_below_8_is_medium(self, composer):
        assert composer._get_score_band(7.9) == "medium"

    def test_score_just_below_5_is_low(self, composer):
        assert composer._get_score_band(4.9) == "low"

    def test_score_0_is_low(self, composer):
        assert composer._get_score_band(0.0) == "low"


# ── FeedbackComposer._select_transition ──────────────────────────────────────

class TestSelectTransition:

    def test_turn_0_returns_empty_string(self, composer):
        assert composer._select_transition(0) == ""

    def test_turn_3_returns_empty_string(self, composer):
        assert composer._select_transition(3) == ""

    def test_turn_6_returns_empty_string(self, composer):
        assert composer._select_transition(6) == ""

    def test_turn_1_returns_non_empty(self, composer):
        assert composer._select_transition(1) != ""

    def test_turn_2_returns_valid_transition(self, composer):
        result = composer._select_transition(2)
        assert result in FeedbackComposer.TRANSITIONS

    def test_turn_4_returns_valid_transition(self, composer):
        result = composer._select_transition(4)
        assert result in FeedbackComposer.TRANSITIONS


# ── FeedbackComposer._select_structure ───────────────────────────────────────

class TestSelectStructure:

    def test_excludes_last_two_previous_structures(self, composer):
        structures = ["{strength}", "{gap_hint}", "{strength} {gap_hint}"]
        previous = ["{strength}", "{gap_hint}"]
        result = composer._select_structure(structures, turn_number=0, previous_structures=previous)
        assert result == "{strength} {gap_hint}"

    def test_falls_back_to_all_structures_when_all_excluded(self, composer):
        structures = ["{strength}", "{gap_hint}"]
        previous = ["{strength}", "{gap_hint}"]
        result = composer._select_structure(structures, turn_number=0, previous_structures=previous)
        assert result in structures

    def test_turn_number_drives_rotation_among_available(self, composer):
        structures = ["A", "B", "C"]
        previous = []
        r0 = composer._select_structure(structures, 0, previous)
        r1 = composer._select_structure(structures, 1, previous)
        assert r0 in structures
        assert r1 in structures

    def test_empty_previous_uses_any_structure(self, composer):
        structures = ["{strength}", "{gap_hint}"]
        result = composer._select_structure(structures, 0, [])
        assert result in structures


# ── FeedbackComposer.compose ──────────────────────────────────────────────────

class TestCompose:

    def test_returns_tuple_of_text_and_template(self, composer, base_components):
        text, template = composer.compose(base_components, score=7.0, turn_number=1, previous_structures=[])
        assert isinstance(text, str)
        assert isinstance(template, str)

    def test_high_score_does_not_include_gap_hint_in_all_structures(self, composer):
        for template in FeedbackComposer.STRUCTURES["high"]:
            assert "{gap_hint}" not in template

    def test_low_score_does_not_include_strength_in_all_structures(self, composer):
        for template in FeedbackComposer.STRUCTURES["low"]:
            assert "{strength}" not in template

    def test_whitespace_collapsed(self, composer):
        components = FeedbackComponents(
            strength_acknowledgment="Strength here.",
            gap_hint="Gap hint here.",
            transition_phrase="",
        )
        text, _ = composer.compose(components, score=6.0, turn_number=0, previous_structures=[])
        assert "  " not in text

    def test_medium_score_selects_medium_band(self, composer, base_components):
        _, template = composer.compose(base_components, score=6.0, turn_number=0, previous_structures=[])
        assert template in FeedbackComposer.STRUCTURES["medium"]

    def test_high_score_selects_high_band(self, composer, base_components):
        _, template = composer.compose(base_components, score=9.0, turn_number=0, previous_structures=[])
        assert template in FeedbackComposer.STRUCTURES["high"]

    def test_low_score_selects_low_band(self, composer, base_components):
        _, template = composer.compose(base_components, score=3.0, turn_number=0, previous_structures=[])
        assert template in FeedbackComposer.STRUCTURES["low"]


# ── FeedbackAgent._get_tone_guidance ─────────────────────────────────────────

class TestGetToneGuidance:

    def test_score_8_returns_brief_acknowledgment(self, agent):
        result = agent._get_tone_guidance(8.0)
        assert "Brief" in result or "genuine" in result.lower()

    def test_score_9_returns_brief_acknowledgment(self, agent):
        result = agent._get_tone_guidance(9.0)
        assert result == agent._get_tone_guidance(8.0)

    def test_score_7_returns_encouraging(self, agent):
        result = agent._get_tone_guidance(7.0)
        assert "Encouraging" in result or "depth" in result.lower()

    def test_score_6_returns_encouraging(self, agent):
        assert agent._get_tone_guidance(6.0) == agent._get_tone_guidance(6.5)

    def test_score_5_returns_supportive(self, agent):
        result = agent._get_tone_guidance(5.0)
        assert "Supportive" in result or "gently" in result.lower()

    def test_score_4_returns_supportive(self, agent):
        assert agent._get_tone_guidance(4.0) == agent._get_tone_guidance(4.5)

    def test_score_3_returns_patient(self, agent):
        result = agent._get_tone_guidance(3.0)
        assert "Patient" in result or "patient" in result.lower()

    def test_score_0_returns_patient(self, agent):
        result = agent._get_tone_guidance(0.0)
        assert "Patient" in result or "patient" in result.lower()


# ── FeedbackAgent._get_concept_context ───────────────────────────────────────

class TestGetConceptContext:

    @pytest.mark.asyncio
    async def test_high_score_skips_enrichment(self, agent, base_state):
        base_state["current_evaluation"]["overall_score"] = 7.0
        result = await agent._get_concept_context(
            base_state,
            {"key_points_missed": ["regularization"]},
            7.0
        )
        assert result == ""
        agent.cache_store.get_concept.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_missed_points_returns_empty(self, agent, base_state):
        result = await agent._get_concept_context(base_state, {"key_points_missed": []}, 5.0)
        assert result == ""
        agent.cache_store.get_concept.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_hit_returns_simple_explanation(self, agent, base_state):
        agent.cache_store.get_concept = AsyncMock(
            return_value={"simple_explanation": "Cached: regularization prevents overfitting."}
        )
        result = await agent._get_concept_context(
            base_state,
            {"key_points_missed": ["regularization"]},
            5.0
        )
        assert result == "Cached: regularization prevents overfitting."
        agent.cache_store.set_concept.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_tool_found_stores_and_returns(self, agent, base_state):
        agent.cache_store.get_concept = AsyncMock(return_value=None)
        tool_result = {
            "found": True,
            "simple_explanation": "Regularization adds a penalty term to the loss.",
        }
        with make_concept_lookup_patch(tool_result):
            result = await agent._get_concept_context(
                base_state,
                {"key_points_missed": ["regularization"]},
                5.0
            )
        assert result == "Regularization adds a penalty term to the loss."
        agent.cache_store.set_concept.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cache_miss_tool_not_found_returns_empty(self, agent, base_state):
        agent.cache_store.get_concept = AsyncMock(return_value=None)
        tool_result = {"found": False}
        with make_concept_lookup_patch(tool_result):
            result = await agent._get_concept_context(
                base_state,
                {"key_points_missed": ["unknown_concept"]},
                5.0
            )
        assert result == ""

    @pytest.mark.asyncio
    async def test_only_first_missed_concept_used(self, agent, base_state):
        agent.cache_store.get_concept = AsyncMock(return_value=None)
        tool_result = {"found": False}
        with make_concept_lookup_patch(tool_result) as mock_lookup:
            await agent._get_concept_context(
                base_state,
                {"key_points_missed": ["first_concept", "second_concept"]},
                5.0
            )
            call_args = mock_lookup.ainvoke.call_args
            assert call_args[0][0]["concept_name"] == "first_concept"


# ── FeedbackAgent.execute ─────────────────────────────────────────────────────

class TestFeedbackAgentExecute:

    def _make_components(
        self,
        strength=(
            "You demonstrated a solid foundational understanding of the core concept "
            "and applied it meaningfully across multiple practical real-world scenarios."
        ),
        gap=(
            "It would be beneficial to explore the deeper implications of this tradeoff "
            "and to consider carefully how it typically manifests in production systems."
        ),
        transition="Building on that...",
    ):
        return FeedbackComponents(
            strength_acknowledgment=strength,
            gap_hint=gap,
            transition_phrase=transition,
        )

    @pytest.mark.asyncio
    async def test_happy_path_returns_correct_state_keys(self, agent, base_state):
        components = self._make_components()
        agent.feedback_chain.ainvoke = AsyncMock(return_value=components)
        config = MagicMock()

        result = await agent.execute(base_state, config)

        assert "current_feedback" in result
        assert "previous_feedback_structures" in result
        assert "recent_feedbacks" in result
        assert isinstance(result["current_feedback"], str)
        assert len(result["previous_feedback_structures"]) == 1
        assert len(result["recent_feedbacks"]) == 1

    @pytest.mark.asyncio
    async def test_repetition_check_skipped_with_fewer_than_two_feedbacks(self, agent, base_state):
        base_state["recent_feedbacks"] = ["previous feedback text here"]
        components = self._make_components()
        agent.feedback_chain.ainvoke = AsyncMock(return_value=components)
        config = MagicMock()

        await agent.execute(base_state, config)

        agent.repetition_chain.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_repetition_check_triggered_with_two_recent_feedbacks(self, agent, base_state):
        base_state["recent_feedbacks"] = ["feedback one here", "feedback two here"]
        components = self._make_components()
        agent.feedback_chain.ainvoke = AsyncMock(return_value=components)

        rep_response = MagicMock()
        rep_response.content = "different - the feedback uses a new angle."
        agent.repetition_chain.ainvoke = AsyncMock(return_value=rep_response)
        config = MagicMock()

        await agent.execute(base_state, config)

        agent.repetition_chain.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_similar_feedback_triggers_regeneration(self, agent, base_state):
        """When repetition chain says 'similar', feedback_chain is invoked again."""
        base_state["recent_feedbacks"] = ["feedback one long enough", "feedback two long enough"]

        # Both calls return long-enough components to pass the validation gate
        agent.feedback_chain.ainvoke = AsyncMock(side_effect=[
            self._make_components(),   # initial generation
            self._make_components(),   # regeneration — must also pass gate
        ])

        rep_response = MagicMock()
        rep_response.content = "similar - same angle used again."
        agent.repetition_chain.ainvoke = AsyncMock(return_value=rep_response)
        config = MagicMock()

        await agent.execute(base_state, config)

        assert agent.feedback_chain.ainvoke.await_count == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_two_failures(self, agent, base_state):
        """
        Gate fails twice → circuit breaker opens → fallback returned.
        Short components ensure gate always rejects.
        """
        short_components = FeedbackComponents(
            strength_acknowledgment="OK.",
            gap_hint="More depth.",
            transition_phrase="",
        )
        agent.feedback_chain.ainvoke = AsyncMock(return_value=short_components)
        config = MagicMock()

        result = await agent.execute(base_state, config)

        assert isinstance(result, dict)
        assert "current_feedback" in result
        assert isinstance(result["previous_feedback_structures"], list)
        assert isinstance(result["recent_feedbacks"], list)