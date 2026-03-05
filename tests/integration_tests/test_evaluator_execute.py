"""
Integration-level tests for EvaluatorAgent.execute() and _single_evaluate().

LLM chains are mocked — no Ollama dependency.
Tests verify the full orchestration path including:
- Chain invocation and normalisation
- Topic + question_id injection
- Code validation integration
- Reflection application
- Validation gate + circuit breaker
- Self-consistency: median selection and divergence flagging
- Fallback on all-exception gather

Key mocking pattern:
    agent.eval_chain = MagicMock()
    agent.eval_chain.ainvoke = AsyncMock(return_value=...)

    NOT: agent.eval_chain = AsyncMock(return_value=...)

Why: The code calls `eval_chain.ainvoke(...)`, not `eval_chain(...)`.
Setting AsyncMock(return_value=X) only controls `await eval_chain(...)`.
Accessing `.ainvoke` on an AsyncMock produces a *child* AsyncMock whose
return value is another AsyncMock (not X). dict(AsyncMock()) then calls
.keys(), which returns a coroutine — hence the TypeError.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.contracts import EvaluationOutput
from src.agents.evaluator import EvaluatorAgent
from src.services.validation import (
    ValidationResult, Check, ValidationGateRegistry, CircuitBreaker
)


# ── Patching helpers ──────────────────────────────────────────────────────────

def make_rubric_patch(return_value: dict):
    """StructuredTool is a Pydantic model — patch dotted ainvoke path fails.
    Replace the whole object in the evaluator module namespace instead."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=return_value)
    return patch("src.agents.evaluator.rubric_lookup", mock)


def make_code_validator_patch(return_value: dict):
    """Same StructuredTool issue — replace whole object."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=return_value)
    return patch("src.agents.evaluator.code_validator", mock)


def wire_eval_chain(agent, return_value):
    """
    Correctly wire eval_chain so eval_chain.ainvoke returns the given value.
    AsyncMock(return_value=X) only affects await agent.eval_chain(...),
    not await agent.eval_chain.ainvoke(...) which the real code calls.
    """
    agent.eval_chain = MagicMock()
    agent.eval_chain.ainvoke = AsyncMock(return_value=return_value)


def wire_reflect_chain(agent, return_value: dict):
    """Same pattern for reflect_chain."""
    agent.reflect_chain = MagicMock()
    agent.reflect_chain.ainvoke = AsyncMock(return_value=return_value)


# ── Data helpers ──────────────────────────────────────────────────────────────

def make_evaluation_output(**overrides) -> EvaluationOutput:
    defaults = dict(
        overall_score=7.0,
        technical_accuracy={"score": 7.0, "reasoning": "Correct."},
        completeness={"score": 6.5, "reasoning": "Mostly complete."},
        depth={"score": 7.5, "reasoning": "Good depth."},
        clarity={"score": 8.0, "reasoning": "Clear."},
        evaluation_reasoning="Solid understanding demonstrated.",
        key_points_covered=["bias", "variance"],
        key_points_missed=["tradeoff curve"],
        misconceptions=[],
        is_fallback=False,
        needs_human_review=False,
        consistency_divergence=None,
        topic="",
        question_id="",
    )
    defaults.update(overrides)
    return EvaluationOutput(**defaults)


def make_state(**overrides) -> dict:
    defaults = dict(
        current_question={
            "id": "q_001",
            "text": "Explain the bias-variance tradeoff.",
            "topic": "ml_fundamentals",
            "difficulty": "medium",
            "question_type": "retrieved",
            "target_concepts": [],
        },
        candidate_response=(
            "Bias is the error from wrong assumptions. "
            "Variance is sensitivity to training data."
        ),
        question_count=1,
        performance_trajectory=[],
        all_evaluations=[],
    )
    defaults.update(overrides)
    return defaults


def passing_validation() -> ValidationResult:
    return ValidationResult(failed_checks=[])


def failing_validation(msg: str = "Gate failed") -> ValidationResult:
    return ValidationResult(failed_checks=[Check(passed=False, message=msg)])


def no_adjustment_reflection() -> dict:
    return {
        "adjustment_needed": False,
        "reason": "Consistent.",
        "score_adjustment": 0,
        "missed_misconceptions": [],
        "additional_key_points_missed": [],
    }


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    """
    Bypass EvaluatorAgent.__init__ — build_eval_chain() uses the | operator
    on LangChain runnables which MagicMock can't satisfy. __new__ lets tests
    control exactly what each chain returns.
    """
    a = EvaluatorAgent.__new__(EvaluatorAgent)
    a.eval_chain = MagicMock()
    a.reflect_chain = MagicMock()
    a.consistency_samples = 1
    a.validation_gates = ValidationGateRegistry()
    a.complex_llm = MagicMock()
    a.fast_llm = MagicMock()
    a.circuit_breaker = CircuitBreaker(max_retries=1)
    return a


@pytest.fixture
def config():
    return MagicMock()


# ── _single_evaluate ──────────────────────────────────────────────────────────

class TestSingleEvaluate:

    @pytest.mark.asyncio
    async def test_returns_current_evaluation_key(self, agent, config):
        wire_eval_chain(agent, make_evaluation_output())
        wire_reflect_chain(agent, no_adjustment_reflection())

        with make_rubric_patch({"found": False, "key_points": [], "common_mistakes": []}):
            result = await agent._single_evaluate(make_state(), config)

        assert "current_evaluation" in result
        assert isinstance(result["current_evaluation"], dict)

    @pytest.mark.asyncio
    async def test_topic_injected_from_question(self, agent, config):
        wire_eval_chain(agent, make_evaluation_output())
        wire_reflect_chain(agent, no_adjustment_reflection())

        with make_rubric_patch({"found": False, "key_points": [], "common_mistakes": []}):
            result = await agent._single_evaluate(make_state(), config)

        assert result["current_evaluation"]["topic"] == "ml_fundamentals"

    @pytest.mark.asyncio
    async def test_question_id_injected(self, agent, config):
        wire_eval_chain(agent, make_evaluation_output())
        wire_reflect_chain(agent, no_adjustment_reflection())

        with make_rubric_patch({"found": False, "key_points": [], "common_mistakes": []}):
            result = await agent._single_evaluate(make_state(), config)

        assert result["current_evaluation"]["question_id"] == "q_001"

    @pytest.mark.asyncio
    async def test_plain_dict_from_lenient_chain_normalised(self, agent, config):
        """Lenient chain (JsonOutputParser) returns plain dict — must be handled."""
        plain_dict = make_evaluation_output().model_dump()
        wire_eval_chain(agent, plain_dict)
        wire_reflect_chain(agent, no_adjustment_reflection())

        with make_rubric_patch({"found": False, "key_points": [], "common_mistakes": []}):
            result = await agent._single_evaluate(make_state(), config)

        assert isinstance(result["current_evaluation"], dict)

    @pytest.mark.asyncio
    async def test_code_validator_not_called_for_text_response(self, agent, config):
        wire_eval_chain(agent, make_evaluation_output())
        wire_reflect_chain(agent, no_adjustment_reflection())

        mock_cv = MagicMock()
        mock_cv.ainvoke = AsyncMock()
        with make_rubric_patch({"found": False, "key_points": [], "common_mistakes": []}), \
             patch("src.agents.evaluator.code_validator", mock_cv):
            await agent._single_evaluate(make_state(), config)
            mock_cv.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_code_syntax_error_appended_to_misconceptions(self, agent, config):
        state = make_state(candidate_response="```python\ndef foo(\n```")
        wire_eval_chain(agent, make_evaluation_output())
        wire_reflect_chain(agent, no_adjustment_reflection())

        code_result = {
            "code_detected": True,
            "is_valid": False,
            "errors": ["Line 2: invalid syntax"],
        }
        with make_rubric_patch({"found": False, "key_points": [], "common_mistakes": []}), \
             make_code_validator_patch(code_result):
            result = await agent._single_evaluate(state, config)

        misconceptions = result["current_evaluation"]["misconceptions"]
        assert any("syntax error" in m.lower() for m in misconceptions)

    @pytest.mark.asyncio
    async def test_valid_code_does_not_add_misconceptions(self, agent, config):
        state = make_state(candidate_response="```python\nx = 1\n```")
        wire_eval_chain(agent, make_evaluation_output())
        wire_reflect_chain(agent, no_adjustment_reflection())

        code_result = {"code_detected": True, "is_valid": True, "errors": []}
        with make_rubric_patch({"found": False, "key_points": [], "common_mistakes": []}), \
             make_code_validator_patch(code_result):
            result = await agent._single_evaluate(state, config)

        assert result["current_evaluation"]["misconceptions"] == []

    @pytest.mark.asyncio
    async def test_reflection_adjustment_applied(self, agent, config):
        wire_eval_chain(agent, make_evaluation_output(overall_score=7.0))
        wire_reflect_chain(agent, {
            "adjustment_needed": True,
            "reason": "Too low.",
            "score_adjustment": 1.0,
            "missed_misconceptions": [],
            "additional_key_points_missed": [],
        })

        with make_rubric_patch({"found": False, "key_points": [], "common_mistakes": []}):
            result = await agent._single_evaluate(make_state(), config)

        assert result["current_evaluation"]["overall_score"] == 8.0


# ── execute — standard path ───────────────────────────────────────────────────

class TestExecuteStandardPath:

    @pytest.mark.asyncio
    async def test_returns_current_evaluation_on_success(self, agent, config):
        eval_result = {"current_evaluation": make_evaluation_output().model_dump()}
        eval_result["current_evaluation"]["topic"] = "ml_fundamentals"

        agent._single_evaluate = AsyncMock(return_value=eval_result)
        agent.validation_gates.get("evaluator").validate = MagicMock(
            return_value=passing_validation()
        )

        result = await agent.execute(make_state(), config)
        assert "current_evaluation" in result

    @pytest.mark.asyncio
    async def test_gate_passes_no_retry(self, agent, config):
        eval_dict = make_evaluation_output().model_dump()
        eval_dict["topic"] = "ml_fundamentals"
        agent._single_evaluate = AsyncMock(return_value={"current_evaluation": eval_dict})
        agent.validation_gates.get("evaluator").validate = MagicMock(
            return_value=passing_validation()
        )
        agent.circuit_breaker.should_retry = MagicMock(return_value=False)

        await agent.execute(make_state(), config)
        agent.circuit_breaker.should_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_gate_fails_circuit_breaker_retries_once(self, agent, config):
        eval_dict = make_evaluation_output().model_dump()
        eval_dict["topic"] = "ml_fundamentals"

        call_count = 0

        async def single_eval_side_effect(state, cfg):
            nonlocal call_count
            call_count += 1
            return {"current_evaluation": eval_dict}

        agent._single_evaluate = single_eval_side_effect

        gate = agent.validation_gates.get("evaluator")
        # First call: fail. Second call (retry): pass.
        gate.validate = MagicMock(
            side_effect=[failing_validation(), passing_validation()]
        )
        agent.circuit_breaker.should_retry = MagicMock(return_value=True)

        await agent.execute(make_state(), config)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_gate_fails_circuit_breaker_exhausted_uses_fallback(
        self, agent, config
    ):
        eval_dict = make_evaluation_output().model_dump()
        eval_dict["topic"] = "ml_fundamentals"
        agent._single_evaluate = AsyncMock(return_value={"current_evaluation": eval_dict})

        gate = agent.validation_gates.get("evaluator")
        gate.validate = MagicMock(return_value=failing_validation())
        agent.circuit_breaker.should_retry = MagicMock(return_value=False)

        fallback = gate.get_fallback()
        gate.get_fallback = MagicMock(return_value=fallback)

        result = await agent.execute(make_state(), config)
        assert result["current_evaluation"]["is_fallback"] is True

    @pytest.mark.asyncio
    async def test_outer_exception_uses_fallback(self, agent, config):
        agent._single_evaluate = AsyncMock(side_effect=RuntimeError("LLM crashed"))

        result = await agent.execute(make_state(), config)
        assert result["current_evaluation"]["is_fallback"] is True

    @pytest.mark.asyncio
    async def test_fallback_always_has_topic_from_question(self, agent, config):
        agent._single_evaluate = AsyncMock(side_effect=RuntimeError("crash"))

        result = await agent.execute(make_state(), config)
        assert result["current_evaluation"]["topic"] == "ml_fundamentals"

    @pytest.mark.asyncio
    async def test_fallback_topic_unknown_when_question_has_no_topic(
        self, agent, config
    ):
        state = make_state()
        state["current_question"].pop("topic")
        agent._single_evaluate = AsyncMock(side_effect=RuntimeError("crash"))

        result = await agent.execute(state, config)
        assert result["current_evaluation"]["topic"] == "unknown"


# ── execute — self-consistency path ──────────────────────────────────────────

class TestExecuteSelfConsistency:

    @pytest.fixture
    def sc_agent(self):
        """
        Same __new__ pattern as `agent` but with consistency_samples=2.
        Avoids calling __init__ which tries to build LCEL chains using
        the | operator on MagicMocks (rejected by LangChain's Runnable type check).
        """
        a = EvaluatorAgent.__new__(EvaluatorAgent)
        a.eval_chain = MagicMock()
        a.reflect_chain = MagicMock()
        a.consistency_samples = 2
        a.validation_gates = ValidationGateRegistry()
        a.complex_llm = MagicMock()
        a.fast_llm = MagicMock()
        a.circuit_breaker = CircuitBreaker(max_retries=1)
        return a

    def make_sc_result(self, score: float) -> dict:
        d = make_evaluation_output(overall_score=score).model_dump()
        d["topic"] = "ml_fundamentals"
        return {"current_evaluation": d}

    @pytest.mark.asyncio
    async def test_median_result_selected(self, sc_agent, config):
        """With scores [6.0, 8.0], sorted median index 1 → score 8.0."""
        results = [self.make_sc_result(6.0), self.make_sc_result(8.0)]
        call_idx = 0

        async def side_effect(state, cfg):
            nonlocal call_idx
            r = results[call_idx % len(results)]
            call_idx += 1
            return r

        sc_agent._single_evaluate = side_effect
        gate = sc_agent.validation_gates.get("evaluator")
        gate.validate = MagicMock(return_value=passing_validation())

        result = await sc_agent.execute(make_state(), config)
        assert result["current_evaluation"]["overall_score"] == 8.0

    @pytest.mark.asyncio
    async def test_divergence_above_threshold_flags_review(self, sc_agent, config):
        results = [self.make_sc_result(4.0), self.make_sc_result(7.0)]
        call_idx = 0

        async def side_effect(state, cfg):
            nonlocal call_idx
            r = results[call_idx % len(results)]
            call_idx += 1
            return r

        sc_agent._single_evaluate = side_effect
        gate = sc_agent.validation_gates.get("evaluator")
        gate.validate = MagicMock(return_value=passing_validation())

        result = await sc_agent.execute(make_state(), config)
        assert result["current_evaluation"]["needs_human_review"] is True
        assert result["current_evaluation"]["consistency_divergence"] == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_divergence_below_threshold_no_review_flag(self, sc_agent, config):
        results = [self.make_sc_result(7.0), self.make_sc_result(8.5)]
        call_idx = 0

        async def side_effect(state, cfg):
            nonlocal call_idx
            r = results[call_idx % len(results)]
            call_idx += 1
            return r

        sc_agent._single_evaluate = side_effect
        gate = sc_agent.validation_gates.get("evaluator")
        gate.validate = MagicMock(return_value=passing_validation())

        result = await sc_agent.execute(make_state(), config)
        assert not result["current_evaluation"].get("needs_human_review", False)

    @pytest.mark.asyncio
    async def test_all_gather_results_exceptions_uses_fallback(
        self, sc_agent, config
    ):
        sc_agent._single_evaluate = AsyncMock(side_effect=RuntimeError("crash"))

        result = await sc_agent.execute(make_state(), config)
        assert result["current_evaluation"]["is_fallback"] is True

    @pytest.mark.asyncio
    async def test_partial_gather_failure_uses_valid_results(
        self, sc_agent, config
    ):
        """One eval fails, one succeeds — should use the valid one."""
        good_result = self.make_sc_result(7.5)
        call_idx = 0

        async def side_effect(state, cfg):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                raise RuntimeError("one eval failed")
            return good_result

        sc_agent._single_evaluate = side_effect
        gate = sc_agent.validation_gates.get("evaluator")
        gate.validate = MagicMock(return_value=passing_validation())

        result = await sc_agent.execute(make_state(), config)
        assert result["current_evaluation"]["is_fallback"] is False
        assert result["current_evaluation"]["overall_score"] == 7.5