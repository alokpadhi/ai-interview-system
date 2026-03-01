"""
Tests for ValidationGates — non-trivial checks only.

Skipped intentionally:
  - Pydantic field validation (Pydantic's responsibility)
  - Trivial presence checks (not worth the maintenance cost)

Focus:
  EvaluatorGate   → drift detection, score range, dynamic rubric fallback
  FeedbackGate    → sycophancy threshold, score leakage patterns, length bounds
  QSGate          → time budget with buffer
  Registry        → correct gate dispatch, KeyError on unknown agent
"""

import pytest
from src.services.validation import (
    EvaluatorValidationGate,
    FeedbackValidationGate,
    QuestionSelectorValidationGate,
    ValidationGateRegistry,
    ValidationResult,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def make_valid_evaluation(
    overall_score: float = 7.0,
    reasoning: str = "x" * 60,
    key_points_covered: list = None,
) -> dict:
    """Minimal valid EvaluationOutput dict."""
    return {
        "overall_score": overall_score,
        "technical_accuracy": {"score": 7.0, "reasoning": "Good"},
        "completeness": {"score": 7.0, "reasoning": "Good"},
        "depth": {"score": 7.0, "reasoning": "Good"},
        "clarity": {"score": 7.0, "reasoning": "Good"},
        "evaluation_reasoning": reasoning,
        "key_points_covered": key_points_covered if key_points_covered is not None else ["point_a", "point_b"],
        "key_points_missed": [],
        "misconceptions": [],
        "topic": "ml_fundamentals",
        "question_id": "q_001",
        "is_fallback": False,
        "needs_human_review": False,
    }


def make_valid_question(
    question_type: str = "retrieved",
    key_points: list = None,
    target_concepts: list = None,
    target_misconception: str = None,
) -> dict:
    """Minimal valid question dict."""
    q = {
        "id": "q_001",
        "text": "Explain gradient descent.",
        "question_type": question_type,
        "topic": "ml_fundamentals",
        "difficulty": "medium",
    }
    if key_points:
        q["rubric"] = {
            "criteria": {
                "technical_accuracy": {"key_points": key_points}
            }
        }
    if target_concepts:
        q["target_concepts"] = target_concepts
    if target_misconception:
        q["target_misconception"] = target_misconception
    return q


def make_valid_feedback(text: str = None) -> dict:
    """Minimal valid FeedbackOutput dict."""
    return {
        "feedback_text": text or (
            "Your explanation covered the core idea well. "
            "Consider exploring how learning rate affects convergence speed. "
            "Building on that, let's look at a related concept."
        ),
        "strength_acknowledgment": "Good understanding of the core concept.",
        "gap_hint": "Consider the impact of learning rate more carefully.",
        "transition_phrase": "Building on that...",
        "structure_template": "strength_gap",
    }


# ── ValidationResult ───────────────────────────────────────────────────────

class TestValidationResult:

    def test_is_valid_true_when_no_failures(self):
        result = ValidationResult()
        assert result.is_valid is True

    def test_is_valid_false_when_failures_present(self):
        from src.services.validation import Check
        result = ValidationResult(failed_checks=[Check(passed=False, message="bad")])
        assert result.is_valid is False

    def test_is_valid_is_property_not_field(self):
        """is_valid derived from failed_checks — impossible state unrepresentable."""
        result = ValidationResult()
        # Cannot set is_valid directly on dataclass without failed_checks
        assert result.is_valid is True


# ── EvaluatorValidationGate ────────────────────────────────────────────────

class TestEvaluatorValidationGate:

    @pytest.fixture
    def gate(self):
        return EvaluatorValidationGate()

    # ── Drift detection ───────────────────────────────────────────────────

    def test_high_score_low_coverage_fails(self, gate):
        """Score >= 8.0 but coverage < 50% → drift detected."""
        question = make_valid_question(key_points=["a", "b", "c", "d"])
        output = make_valid_evaluation(
            overall_score=8.5,
            key_points_covered=["a"],  # 1/4 = 25% coverage
        )
        result = gate.validate(output, question)
        assert result.is_valid is False
        assert any("drift" in msg.lower() or "coverage" in msg.lower()
                   for msg in result.feedback)

    def test_low_score_high_coverage_fails(self, gate):
        """Score <= 4.0 but coverage > 70% → drift detected."""
        question = make_valid_question(key_points=["a", "b", "c", "d"])
        output = make_valid_evaluation(
            overall_score=3.5,
            key_points_covered=["a", "b", "c", "d"],  # 100% coverage
        )
        result = gate.validate(output, question)
        assert result.is_valid is False

    def test_high_score_high_coverage_passes(self, gate):
        """Score >= 8.0 and coverage >= 50% → no drift."""
        question = make_valid_question(key_points=["a", "b", "c", "d"])
        output = make_valid_evaluation(
            overall_score=8.5,
            key_points_covered=["a", "b", "c"],  # 75% coverage
        )
        result = gate.validate(output, question)
        assert result.is_valid is True

    def test_no_key_points_drift_check_skipped(self, gate):
        """No key points available → drift check silently skipped, gate passes."""
        question = make_valid_question()  # no rubric, no target_concepts
        output = make_valid_evaluation(overall_score=9.0, key_points_covered=[])
        result = gate.validate(output, question)
        assert result.is_valid is True

    # ── Dynamic rubric fallback ───────────────────────────────────────────

    def test_uses_target_concepts_when_no_static_rubric(self, gate):
        """Follow-up question uses target_concepts as rubric."""
        question = make_valid_question(
            question_type="follow_up",
            target_concepts=["learning_rate", "convergence", "momentum", "batch_size"]
        )
        # High score but only covers 1 of 4 concepts → drift
        output = make_valid_evaluation(
            overall_score=8.5,
            key_points_covered=["learning_rate"],
        )
        result = gate.validate(output, question)
        assert result.is_valid is False

    def test_uses_target_misconception_for_clarify(self, gate):
        """Clarify question wraps target_misconception in list for drift check."""
        question = make_valid_question(
            question_type="clarification",
            target_misconception="gradient descent always finds global minimum"
        )
        # High score but covered nothing → drift
        output = make_valid_evaluation(
            overall_score=8.5,
            key_points_covered=[],
        )
        result = gate.validate(output, question)
        assert result.is_valid is False

    def test_static_rubric_takes_priority_over_target_concepts(self, gate):
        """When both static rubric and target_concepts exist, static wins."""
        question = make_valid_question(
            key_points=["a", "b", "c", "d"],
            target_concepts=["x", "y"]  # should be ignored
        )
        # Low coverage of static rubric points → drift (4 static points used, not 2 concepts)
        output = make_valid_evaluation(
            overall_score=8.5,
            key_points_covered=["a"],  # 1/4 of static points = 25%
        )
        result = gate.validate(output, question)
        assert result.is_valid is False

    # ── Score range ───────────────────────────────────────────────────────

    def test_score_above_10_fails(self, gate):
        question = make_valid_question()
        output = make_valid_evaluation(overall_score=11.0)
        result = gate.validate(output, question)
        assert result.is_valid is False

    def test_score_below_0_fails(self, gate):
        question = make_valid_question()
        output = make_valid_evaluation(overall_score=-1.0)
        result = gate.validate(output, question)
        assert result.is_valid is False

    def test_subscore_above_10_fails(self, gate):
        question = make_valid_question()
        output = make_valid_evaluation()
        output["technical_accuracy"] = {"score": 11.0, "reasoning": "too high"}
        result = gate.validate(output, question)
        assert result.is_valid is False

    # ── Reasoning length ─────────────────────────────────────────────────

    def test_reasoning_too_short_fails(self, gate):
        question = make_valid_question()
        output = make_valid_evaluation(reasoning="too short")
        result = gate.validate(output, question)
        assert result.is_valid is False
        assert any("reasoning" in msg.lower() for msg in result.feedback)

    def test_reasoning_exactly_50_chars_passes(self, gate):
        question = make_valid_question()
        output = make_valid_evaluation(reasoning="x" * 50)
        result = gate.validate(output, question)
        assert result.is_valid is True

    # ── Required fields ───────────────────────────────────────────────────

    def test_missing_topic_fails(self, gate):
        question = make_valid_question()
        output = make_valid_evaluation()
        del output["topic"]
        result = gate.validate(output, question)
        assert result.is_valid is False
        assert any("topic" in msg.lower() for msg in result.feedback)

    def test_missing_overall_score_fails(self, gate):
        question = make_valid_question()
        output = make_valid_evaluation()
        del output["overall_score"]
        result = gate.validate(output, question)
        assert result.is_valid is False

    # ── Score consistency ─────────────────────────────────────────────────

    def test_high_variance_sub_scores_fails(self, gate):
        """Sub-score variance > 5.0 is suspicious."""
        question = make_valid_question()
        output = make_valid_evaluation()
        output["technical_accuracy"] = {"score": 1.0, "reasoning": "very low"}
        output["completeness"] = {"score": 9.5, "reasoning": "very high"}
        result = gate.validate(output, question)
        assert result.is_valid is False

    # ── Fallback ──────────────────────────────────────────────────────────

    def test_fallback_has_required_fields(self, gate):
        fallback = gate.get_fallback()
        assert fallback["is_fallback"] is True
        assert fallback["needs_human_review"] is True
        assert fallback["topic"] == "unknown"
        assert fallback["overall_score"] == 5.0

    def test_fallback_passes_its_own_gate(self, gate):
        """Fallback output should pass validation (it's always safe)."""
        question = make_valid_question()
        result = gate.validate(gate.get_fallback(), question)
        assert result.is_valid is True


# ── FeedbackValidationGate ─────────────────────────────────────────────────

class TestFeedbackValidationGate:

    @pytest.fixture
    def gate(self):
        return FeedbackValidationGate()

    # ── Length ────────────────────────────────────────────────────────────

    def test_too_short_fails(self, gate):
        output = make_valid_feedback("Too short.")
        result = gate.validate(output, evaluation_score=7.0)
        assert result.is_valid is False
        assert any("short" in msg.lower() for msg in result.feedback)

    def test_too_long_fails(self, gate):
        output = make_valid_feedback(" ".join(["word"] * 250))
        result = gate.validate(output, evaluation_score=7.0)
        assert result.is_valid is False
        assert any("long" in msg.lower() for msg in result.feedback)

    def test_valid_length_passes(self, gate):
        output = make_valid_feedback()
        result = gate.validate(output, evaluation_score=7.0)
        assert result.is_valid is True

    # ── Sycophancy ────────────────────────────────────────────────────────

    def test_sycophancy_at_low_score_fails(self, gate):
        """Sycophantic opener at score < 7.0 → fail."""
        output = make_valid_feedback(
            "Great job on that explanation! There are some areas "
            "where you could dig a bit deeper into the mechanics. "
            "Consider thinking about how regularization affects the loss landscape."
        )
        result = gate.validate(output, evaluation_score=4.5)
        assert result.is_valid is False
        assert any("sycophant" in msg.lower() for msg in result.feedback)

    def test_sycophancy_at_high_score_passes(self, gate):
        """Sycophantic opener at score >= 7.0 → allowed."""
        output = make_valid_feedback(
            "Excellent explanation of the concept! You covered the key "
            "points clearly and demonstrated strong understanding. "
            "Building on that, let us explore a related nuance."
        )
        result = gate.validate(output, evaluation_score=8.5)
        assert result.is_valid is True

    def test_sycophancy_check_threshold_boundary(self, gate):
        """Score exactly at 7.0 → sycophancy check disabled (>= threshold)."""
        output = make_valid_feedback(
            "Outstanding work on that response! You touched on the key "
            "aspects well. Let us explore a follow-up area together now."
        )
        result = gate.validate(output, evaluation_score=7.0)
        assert result.is_valid is True

    def test_sycophancy_buried_in_text_passes(self, gate):
        """
        Sycophantic phrase buried beyond first 150 chars should not trigger.
        Sycophancy check only covers opening window.
        """
        prefix = "Consider how the learning rate affects convergence. " * 3
        output = make_valid_feedback(prefix + " That was an excellent point.")
        result = gate.validate(output, evaluation_score=4.0)
        # Should pass — "excellent" is beyond the 150-char window
        # (prefix is ~150 chars, "excellent" is after)
        assert result.is_valid is True

    # ── Forbidden phrases ─────────────────────────────────────────────────

    def test_forbidden_phrase_fails(self, gate):
        output = make_valid_feedback(
            "That was actually an incorrect answer to the question asked. "
            "You should revisit the fundamentals before attempting this level. "
            "Consider reviewing gradient descent more carefully next time."
        )
        result = gate.validate(output, evaluation_score=3.0)
        assert result.is_valid is False
        assert any("forbidden" in msg.lower() for msg in result.feedback)

    def test_forbidden_phrase_case_insensitive(self, gate):
        output = make_valid_feedback(
            "WRONG ANSWER provided here unfortunately for the question. "
            "Let us take a step back and reconsider the fundamentals together. "
            "Building on your attempt, here is how to approach this better."
        )
        result = gate.validate(output, evaluation_score=3.0)
        assert result.is_valid is False

    # ── Score leakage ─────────────────────────────────────────────────────

    @pytest.mark.parametrize("leaky_text", [
        "You scored 7 out of 10 on this question overall today.",
        "Your rating: 6 for this response was noted by the system.",
        "That response was worth about 5.5/10 in our rubric scale.",
        "You scored a 8 on technical accuracy in my evaluation.",
    ])
    def test_score_leakage_detected(self, gate, leaky_text):
        output = make_valid_feedback(leaky_text * 2)  # ensure length
        result = gate.validate(output, evaluation_score=7.0)
        assert result.is_valid is False
        assert any("leakage" in msg.lower() or "score" in msg.lower()
                   for msg in result.feedback)

    def test_no_score_leakage_passes(self, gate):
        output = make_valid_feedback()
        result = gate.validate(output, evaluation_score=7.0)
        assert result.is_valid is True

    # ── Fallback ──────────────────────────────────────────────────────────

    def test_fallback_passes_its_own_gate(self, gate):
        result = gate.validate(gate.get_fallback(), evaluation_score=5.0)
        assert result.is_valid is True


# ── QuestionSelectorValidationGate ────────────────────────────────────────

class TestQuestionSelectorValidationGate:

    @pytest.fixture
    def gate(self):
        return QuestionSelectorValidationGate()

    def make_valid_output(self, q_type="retrieved", est_time=4.0):
        return {
            "id": "q_001",
            "text": "Explain the bias-variance tradeoff.",
            "question_type": q_type,
            "topic": "ml_fundamentals",
            "difficulty": "medium",
            "estimated_time_minutes": est_time,
            "target_concepts": ["bias", "variance"],
        }

    # ── Question presence ─────────────────────────────────────────────────

    def test_missing_text_fails(self, gate):
        output = self.make_valid_output()
        output["text"] = ""
        result = gate.validate(output, remaining_minutes=20.0)
        assert result.is_valid is False

    def test_whitespace_only_text_fails(self, gate):
        output = self.make_valid_output()
        output["text"] = "   "
        result = gate.validate(output, remaining_minutes=20.0)
        assert result.is_valid is False

    # ── Question type ─────────────────────────────────────────────────────

    def test_valid_types_pass(self, gate):
        for q_type in ["retrieved", "follow_up", "clarification"]:
            output = self.make_valid_output(q_type=q_type)
            result = gate.validate(output, remaining_minutes=20.0)
            assert result.is_valid is True, f"Type '{q_type}' should pass"

    def test_invalid_type_fails(self, gate):
        output = self.make_valid_output(q_type="unknown_type")
        result = gate.validate(output, remaining_minutes=20.0)
        assert result.is_valid is False
        assert any("invalid" in msg.lower() or "question_type" in msg.lower()
                   for msg in result.feedback)

    # ── Time budget ───────────────────────────────────────────────────────

    def test_question_fits_remaining_time_passes(self, gate):
        """est_time=4, remaining=10 → fits with buffer."""
        output = self.make_valid_output(est_time=4.0)
        result = gate.validate(output, remaining_minutes=10.0)
        assert result.is_valid is True

    def test_question_exceeds_remaining_time_fails(self, gate):
        """est_time=10, remaining=5 → exceeds budget + buffer."""
        output = self.make_valid_output(est_time=10.0)
        result = gate.validate(output, remaining_minutes=5.0)
        assert result.is_valid is False
        assert any("time" in msg.lower() for msg in result.feedback)

    def test_time_buffer_allows_slight_overage(self, gate):
        """
        est_time = remaining + 1 → still passes (buffer is 2 min).
        est_time <= remaining + 2 should pass.
        """
        output = self.make_valid_output(est_time=6.0)
        result = gate.validate(output, remaining_minutes=5.0)
        assert result.is_valid is True

    def test_exactly_at_buffer_boundary_passes(self, gate):
        """est_time == remaining + 2 → exactly at boundary, should pass."""
        output = self.make_valid_output(est_time=7.0)
        result = gate.validate(output, remaining_minutes=5.0)
        assert result.is_valid is True

    def test_one_above_buffer_fails(self, gate):
        """est_time == remaining + 2.1 → just over boundary, should fail."""
        output = self.make_valid_output(est_time=7.1)
        result = gate.validate(output, remaining_minutes=5.0)
        assert result.is_valid is False

    def test_missing_time_uses_default(self, gate):
        """
        Missing estimated_time_minutes → defaults to 5.0.
        Should not false-fail with sufficient remaining time.
        """
        output = self.make_valid_output()
        del output["estimated_time_minutes"]
        result = gate.validate(output, remaining_minutes=20.0)
        assert result.is_valid is True

    # ── Fallback ──────────────────────────────────────────────────────────

    def test_fallback_passes_its_own_gate(self, gate):
        result = gate.validate(gate.get_fallback(), remaining_minutes=20.0)
        assert result.is_valid is True

    def test_fallback_has_target_concepts(self, gate):
        """Fallback must have target_concepts so drift detection works."""
        fallback = gate.get_fallback()
        assert len(fallback["target_concepts"]) > 0


# ── ValidationGateRegistry ────────────────────────────────────────────────

class TestValidationGateRegistry:

    @pytest.fixture
    def registry(self):
        return ValidationGateRegistry()

    def test_get_evaluator_gate(self, registry):
        gate = registry.get("evaluator")
        assert isinstance(gate, EvaluatorValidationGate)

    def test_get_feedback_gate(self, registry):
        gate = registry.get("feedback")
        assert isinstance(gate, FeedbackValidationGate)

    def test_get_question_selector_gate(self, registry):
        gate = registry.get("question_selector")
        assert isinstance(gate, QuestionSelectorValidationGate)

    def test_unknown_agent_raises_key_error(self, registry):
        with pytest.raises(KeyError, match="supervisor"):
            registry.get("supervisor")

    def test_error_message_lists_available_gates(self, registry):
        """Error message should tell you what IS available."""
        with pytest.raises(KeyError) as exc_info:
            registry.get("nonexistent")
        assert "evaluator" in str(exc_info.value)