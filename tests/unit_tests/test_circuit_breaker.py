"""
Tests for CircuitBreaker — per-turn retry budget management.

Focus: retry budget logic, per-agent independence, reset behavior.
Small surface area — verify completely.
"""

import pytest
from src.services.validation import CircuitBreaker


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def breaker():
    """Default CircuitBreaker with max_retries=1."""
    return CircuitBreaker(max_retries=1)


# ── Retry budget ───────────────────────────────────────────────────────────

class TestShouldRetry:

    def test_first_call_returns_true(self, breaker):
        assert breaker.should_retry("evaluator") is True

    def test_second_call_returns_false(self, breaker):
        breaker.should_retry("evaluator")
        assert breaker.should_retry("evaluator") is False

    def test_third_call_still_false(self, breaker):
        """Budget exhausted — subsequent calls stay False."""
        breaker.should_retry("evaluator")
        breaker.should_retry("evaluator")
        assert breaker.should_retry("evaluator") is False

    def test_agents_are_independent(self, breaker):
        """
        Exhausting evaluator budget should not affect feedback budget.
        Each agent gets its own independent retry counter.
        """
        breaker.should_retry("evaluator")
        breaker.should_retry("evaluator")  # evaluator exhausted

        # feedback should still have full budget
        assert breaker.should_retry("feedback") is True

    def test_new_agent_always_starts_fresh(self, breaker):
        """An agent that has never failed starts with full budget."""
        assert breaker.should_retry("question_selector") is True

    def test_max_retries_two(self):
        """With max_retries=2, agent gets two retries before False."""
        b = CircuitBreaker(max_retries=2)
        assert b.should_retry("evaluator") is True
        assert b.should_retry("evaluator") is True
        assert b.should_retry("evaluator") is False


# ── Reset behavior ─────────────────────────────────────────────────────────

class TestReset:

    def test_reset_all_restores_all_agents(self, breaker):
        """reset() with no args clears ALL agent counts."""
        breaker.should_retry("evaluator")
        breaker.should_retry("evaluator")   # exhausted
        breaker.should_retry("feedback")
        breaker.should_retry("feedback")    # exhausted

        breaker.reset()

        # Both should have fresh budget after reset
        assert breaker.should_retry("evaluator") is True
        assert breaker.should_retry("feedback") is True

    def test_reset_specific_agent_only(self, breaker):
        """reset("evaluator") clears only evaluator, not feedback."""
        breaker.should_retry("evaluator")
        breaker.should_retry("evaluator")   # evaluator exhausted
        breaker.should_retry("feedback")
        breaker.should_retry("feedback")    # feedback exhausted

        breaker.reset("evaluator")

        assert breaker.should_retry("evaluator") is True   # reset
        assert breaker.should_retry("feedback") is False   # still exhausted

    def test_reset_empty_state_no_error(self, breaker):
        """reset() on a fresh breaker should not raise."""
        breaker.reset()  # should not raise

    def test_reset_unknown_agent_no_error(self, breaker):
        """reset("unknown") on agent that never failed should not raise."""
        breaker.reset("unknown_agent")  # should not raise

    def test_per_turn_pattern(self, breaker):
        """
        Simulate the per-turn usage pattern from architecture:
          Turn N: reset() → agent fails → should_retry=True → retry fails → should_retry=False
          Turn N+1: reset() → fresh budget again
        """
        # Turn 1
        breaker.reset()
        assert breaker.should_retry("evaluator") is True   # first failure
        assert breaker.should_retry("evaluator") is False  # budget exhausted

        # Turn 2 — reset gives fresh budget
        breaker.reset()
        assert breaker.should_retry("evaluator") is True   # fresh budget
        assert breaker.should_retry("evaluator") is False  # exhausted again