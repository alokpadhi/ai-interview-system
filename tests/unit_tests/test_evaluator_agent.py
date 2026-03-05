"""
Unit tests for EvaluatorAgent — pure methods only (no LLM calls).

Covers:
- _build_rubric_context   (rubric priority chain)
- _response_contains_code (delegation to _contains_code)
- _apply_reflection       (score clamping, field merging)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.evaluator import EvaluatorAgent
from src.services.validation import ValidationGateRegistry, CircuitBreaker


def make_rubric_patch(return_value: dict):
    """
    StructuredTool is a Pydantic model — patch() cannot set attributes on it
    directly via dotted path. Replace the whole rubric_lookup object in the
    evaluator module namespace with a MagicMock that has ainvoke wired up.
    """
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=return_value)
    return patch("src.agents.evaluator.rubric_lookup", mock)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    """
    Bypass EvaluatorAgent.__init__ entirely — unit tests target pure methods
    (_build_rubric_context, _response_contains_code, _apply_reflection) and
    must not trigger LangChain chain construction.

    AsyncMock().with_structured_output() returns a coroutine, which LangChain's
    | operator rejects with 'Expected a Runnable'. __new__ avoids this entirely.
    """
    a = EvaluatorAgent.__new__(EvaluatorAgent)
    a.eval_chain = AsyncMock()
    a.reflect_chain = AsyncMock()
    a.consistency_samples = 1
    a.validation_gates = ValidationGateRegistry()
    a.complex_llm = MagicMock()
    a.fast_llm = MagicMock()
    a.circuit_breaker = CircuitBreaker(max_retries=1)
    return a


@pytest.fixture
def base_question():
    return {
        "id": "q_001",
        "text": "Explain the bias-variance tradeoff.",
        "topic": "ml_fundamentals",
        "difficulty": "medium",
        "question_type": "retrieved",
        "target_concepts": [],
    }


@pytest.fixture
def base_eval_dict():
    return {
        "overall_score": 7.0,
        "technical_accuracy": {"score": 7.0, "reasoning": "Mostly correct."},
        "completeness": {"score": 6.5, "reasoning": "Some gaps."},
        "depth": {"score": 7.5, "reasoning": "Good depth."},
        "clarity": {"score": 8.0, "reasoning": "Clear."},
        "evaluation_reasoning": "Candidate demonstrated solid understanding.",
        "key_points_covered": ["bias definition", "variance definition"],
        "key_points_missed": ["tradeoff mechanism"],
        "misconceptions": [],
        "is_fallback": False,
        "topic": "ml_fundamentals",
        "question_id": "q_001",
    }


# ── _build_rubric_context ─────────────────────────────────────────────────────

class TestBuildRubricContext:

    @pytest.mark.asyncio
    async def test_rubric_found_returns_key_points_and_common_mistakes(
        self, agent, base_question
    ):
        rubric_return = {
            "found": True,
            "key_points": ["Define bias", "Define variance"],
            "common_mistakes": ["Confusing bias with error"],
        }
        with make_rubric_patch(rubric_return):
            ctx = await agent._build_rubric_context(base_question)

        assert "Key Points:" in ctx
        assert "Define bias" in ctx
        assert "Define variance" in ctx
        assert "Common Mistakes" in ctx
        assert "Confusing bias with error" in ctx

    @pytest.mark.asyncio
    async def test_rubric_found_no_common_mistakes(self, agent, base_question):
        rubric_return = {
            "found": True,
            "key_points": ["Define bias"],
            "common_mistakes": [],
        }
        with make_rubric_patch(rubric_return):
            ctx = await agent._build_rubric_context(base_question)

        assert "Key Points:" in ctx
        assert "Common Mistakes" not in ctx

    @pytest.mark.asyncio
    async def test_rubric_not_found_falls_back_to_target_concepts(
        self, agent
    ):
        question = {
            "id": "q_001_followup_1",
            "text": "Can you elaborate on regularization?",
            "topic": "ml_fundamentals",
            "question_type": "follow_up",
            "target_concepts": ["L2 penalty", "weight decay"],
        }
        rubric_return = {"found": False, "key_points": [], "common_mistakes": []}
        with make_rubric_patch(rubric_return):
            ctx = await agent._build_rubric_context(question)

        assert "Expected concepts:" in ctx
        assert "L2 penalty" in ctx
        assert "weight decay" in ctx

    @pytest.mark.asyncio
    async def test_rubric_not_found_no_target_concepts_falls_back_to_misconception(
        self, agent
    ):
        question = {
            "id": "q_001_clarify",
            "text": "You mentioned X — can you clarify?",
            "topic": "ml_fundamentals",
            "question_type": "clarification",
            "target_misconception": "Candidate confused regularization with normalization",
            "target_concepts": [],
        }
        rubric_return = {"found": False, "key_points": [], "common_mistakes": []}
        with make_rubric_patch(rubric_return):
            ctx = await agent._build_rubric_context(question)

        assert "Misconception to address:" in ctx
        assert "Candidate confused regularization with normalization" in ctx

    @pytest.mark.asyncio
    async def test_no_rubric_no_dynamic_fields_returns_graceful_fallback(
        self, agent
    ):
        question = {
            "id": "",
            "text": "Explain gradient descent.",
            "topic": "ml_fundamentals",
            "question_type": "retrieved",
            "target_concepts": [],
        }
        # No question_id → rubric_lookup never called
        ctx = await agent._build_rubric_context(question)
        assert "No rubric available" in ctx

    @pytest.mark.asyncio
    async def test_empty_question_id_skips_rubric_lookup(self, agent):
        """No question_id → rubric_lookup.ainvoke should not be called."""
        question = {
            "id": "",
            "text": "Explain dropout.",
            "topic": "deep_learning",
            "question_type": "retrieved",
            "target_concepts": ["dropout probability", "inference mode"],
        }
        with make_rubric_patch(AsyncMock()) as mock_lookup:
            await agent._build_rubric_context(question)
            mock_lookup.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_rubric_found_early_return_ignores_target_concepts(
        self, agent
    ):
        """When rubric is found, target_concepts must NOT appear in context."""
        question = {
            "id": "q_001",
            "text": "Explain bias-variance tradeoff.",
            "topic": "ml_fundamentals",
            "question_type": "retrieved",
            "target_concepts": ["should not appear"],
        }
        rubric_return = {
            "found": True,
            "key_points": ["rubric point"],
            "common_mistakes": [],
        }
        with make_rubric_patch(rubric_return):
            ctx = await agent._build_rubric_context(question)

        assert "should not appear" not in ctx
        assert "rubric point" in ctx


# ── _response_contains_code ───────────────────────────────────────────────────

class TestResponseContainsCode:

    def test_fenced_code_block_detected(self, agent):
        assert agent._response_contains_code("```python\nprint('hi')\n```") is True

    def test_plain_fence_detected(self, agent):
        assert agent._response_contains_code("```\nx = 1\n```") is True

    def test_self_dot_syntax_detected(self, agent):
        assert agent._response_contains_code("You can use self.model to access it.") is True

    def test_dunder_init_detected(self, agent):
        assert agent._response_contains_code("The __init__ method initializes.") is True

    def test_def_with_colon_detected(self, agent):
        assert agent._response_contains_code("def fit(self, X):\n    pass") is True

    def test_import_statement_detected(self, agent):
        assert agent._response_contains_code("import numpy as np") is True

    def test_from_import_detected(self, agent):
        assert agent._response_contains_code("from sklearn.linear_model import Ridge") is True

    def test_control_flow_with_indent_detected(self, agent):
        text = "if x > 0:\n    return x"
        assert agent._response_contains_code(text) is True

    def test_plain_text_not_detected(self, agent):
        text = (
            "Gradient descent is an optimization algorithm that minimizes "
            "a loss function by iteratively updating model parameters in the "
            "direction of the negative gradient."
        )
        assert agent._response_contains_code(text) is False

    def test_empty_string_not_detected(self, agent):
        assert agent._response_contains_code("") is False


# ── _apply_reflection ─────────────────────────────────────────────────────────

class TestApplyReflection:

    def test_no_adjustment_returns_dict_unchanged(self, agent, base_eval_dict):
        reflection = {
            "adjustment_needed": False,
            "reason": "Evaluation is consistent.",
            "score_adjustment": 0,
            "missed_misconceptions": [],
            "additional_key_points_missed": [],
        }
        result = agent._apply_reflection(base_eval_dict.copy(), reflection)
        assert result["overall_score"] == 7.0
        assert result["misconceptions"] == []
        assert result["key_points_missed"] == ["tradeoff mechanism"]

    def test_positive_score_adjustment_applied(self, agent, base_eval_dict):
        reflection = {
            "adjustment_needed": True,
            "reason": "Score was too low.",
            "score_adjustment": 1.5,
            "missed_misconceptions": [],
            "additional_key_points_missed": [],
        }
        result = agent._apply_reflection(base_eval_dict.copy(), reflection)
        assert result["overall_score"] == 8.5

    def test_negative_score_adjustment_applied(self, agent, base_eval_dict):
        reflection = {
            "adjustment_needed": True,
            "reason": "Score was too high.",
            "score_adjustment": -2.0,
            "missed_misconceptions": [],
            "additional_key_points_missed": [],
        }
        result = agent._apply_reflection(base_eval_dict.copy(), reflection)
        assert result["overall_score"] == 5.0

    def test_score_clamped_at_ceiling_10(self, agent, base_eval_dict):
        base_eval_dict["overall_score"] = 9.5
        reflection = {
            "adjustment_needed": True,
            "reason": "Should be higher.",
            "score_adjustment": 2.0,
            "missed_misconceptions": [],
            "additional_key_points_missed": [],
        }
        result = agent._apply_reflection(base_eval_dict, reflection)
        assert result["overall_score"] == 10.0

    def test_score_clamped_at_floor_0(self, agent, base_eval_dict):
        base_eval_dict["overall_score"] = 0.5
        reflection = {
            "adjustment_needed": True,
            "reason": "Should be lower.",
            "score_adjustment": -2.0,
            "missed_misconceptions": [],
            "additional_key_points_missed": [],
        }
        result = agent._apply_reflection(base_eval_dict, reflection)
        assert result["overall_score"] == 0.0

    def test_missed_misconceptions_merged(self, agent, base_eval_dict):
        base_eval_dict["misconceptions"] = ["existing misconception"]
        reflection = {
            "adjustment_needed": True,
            "reason": "Missed a misconception.",
            "score_adjustment": 0,
            "missed_misconceptions": ["new misconception"],
            "additional_key_points_missed": [],
        }
        result = agent._apply_reflection(base_eval_dict, reflection)
        assert "existing misconception" in result["misconceptions"]
        assert "new misconception" in result["misconceptions"]

    def test_additional_key_points_missed_merged(self, agent, base_eval_dict):
        reflection = {
            "adjustment_needed": True,
            "reason": "Key point missed.",
            "score_adjustment": 0,
            "missed_misconceptions": [],
            "additional_key_points_missed": ["bias-complexity relationship"],
        }
        result = agent._apply_reflection(base_eval_dict.copy(), reflection)
        assert "tradeoff mechanism" in result["key_points_missed"]
        assert "bias-complexity relationship" in result["key_points_missed"]

    def test_all_adjustments_applied_together(self, agent, base_eval_dict):
        reflection = {
            "adjustment_needed": True,
            "reason": "Multiple issues found.",
            "score_adjustment": -1.0,
            "missed_misconceptions": ["wrong claim about bias"],
            "additional_key_points_missed": ["regularization link"],
        }
        result = agent._apply_reflection(base_eval_dict.copy(), reflection)
        assert result["overall_score"] == 6.0
        assert "wrong claim about bias" in result["misconceptions"]
        assert "regularization link" in result["key_points_missed"]

    def test_non_dict_reflection_returns_eval_dict_unchanged(
        self, agent, base_eval_dict
    ):
        original_score = base_eval_dict["overall_score"]
        result = agent._apply_reflection(base_eval_dict.copy(), "not a dict")
        assert result["overall_score"] == original_score

    def test_score_rounded_to_one_decimal(self, agent, base_eval_dict):
        base_eval_dict["overall_score"] = 6.0
        reflection = {
            "adjustment_needed": True,
            "reason": "Slight upward adjustment.",
            "score_adjustment": 1.15,
            "missed_misconceptions": [],
            "additional_key_points_missed": [],
        }
        result = agent._apply_reflection(base_eval_dict, reflection)
        # 6.0 + 1.15 = 7.15 → rounded to 7.2
        assert result["overall_score"] == 7.2