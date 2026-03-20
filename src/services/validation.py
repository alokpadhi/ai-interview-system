"""
Validation gates and circuit breaker for the AI Interview System.

Every agent output passes through a gate before touching InterviewState.
Bad outputs trigger retry (via CircuitBreaker) or fallback — never silent corruption.

Gate responsibilities:
  EvaluatorValidationGate    → scores in range, reasoning length, drift detection
  FeedbackValidationGate     → length, forbidden phrases, sycophancy, score leakage
  QuestionSelectorGate       → question present, valid type, time budget fit

Circuit breaker:
  Max 1 retry per agent per turn.
  reset() called at start of each validate_and_decide turn.
  Prevents infinite retry loops on degraded LLM outputs.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Check:
    """
    Result of a single validation check
    passed=False must always include a message explaining why.
    """
    passed: bool
    message: str = ""

@dataclass
class ValidationResult:
    """
    Aggregate result from a full gate validation run.
    is_valid=True only when ALL checks pass.
    """
    failed_checks: list[Check] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return (len(self.failed_checks)) == 0
    
# Evaluator Gate
class EvaluatorValidationGate:
    """
    alidates EvaluatorAgent output.

    Checks:
      1. All scores in 0-10 range
      2. evaluation_reasoning >= 50 chars
      3. Score variance < 5.0 (consistency)
      4. Required fields present
      5. Key-point coverage alignment (drift detection)
         — skipped silently when no key points available
         — covers BOTH static rubric and dynamic target_concepts

    Dynamic rubric support:
      Retrieved questions  → question["rubric"]["key_points"]
      Follow-up questions  → question["target_concepts"]
      Clarify questions    → question["target_misconception"]
      No rubric available  → skip drift check, log it

    Without dynamic rubric support, drift detection silently
    disables for ~30-40% of questions (all follow-ups/clarifications).
    """
    # sub score fields to validate - not overall _score (its the weighted)
    SUB_SCORE_FIELDS = ["technical_accuracy", "completeness", "depth", "clarity"]

    REQUIRED_FIELDS = ["overall_score", "evaluation_reasoning", "key_points_covered", "topic"]

    # Drift detection thresholds
    HIGH_SCORE_THRESHOLD = 8.0
    LOW_SCORE_THRESHOLD = 4.0
    HIGH_COVERAGE_RATIO = 0.7
    LOW_COVERAGE_RATIO = 0.5

    # maximum allowed variance to check consistency
    MAX_SCORE_VARIANCE = 6.0

    # minimum reasoning length
    MIN_REASONING_CHARS = 50

    def validate(self, output: dict, question: dict) -> ValidationResult:
        """_summary_

        Args:
            output (dict): EvaluationOutput dict from EvaluatorAgent
            question (dict): current_question dict — contains rubric OR target_concepts

        Returns:
            ValidationResult: _description_
        """
        rubric_key_points = self._extract_key_points(question)

        checks = [
            self._scores_in_range(output),
            self._reasoning_provided(output),
            self._scores_consistent(output),
            self._required_fields_present(output),
            self._key_point_coverage_alignment(output, rubric_key_points),
            self._overall_aligns_with_subscores(output),
        ]

        failed = [c for c in checks if not c.passed]

        return ValidationResult(
            failed_checks=failed,
            feedback=[c.message for c in failed]
        )
    
    def _extract_key_points(self, question: dict) -> list[str]:
        """
        Extract key points from either static rubric or dynamic target_concepts.

        Priority:
         1. Static rubric — flattens key_points from ALL criteria, not just
            technical_accuracy. Consistent with rubric_tool._format_rubric().
         2. target_concepts (follow-up questions)
         3. target_misconception (clarify questions)
         4. [] — drift check skipped, logged at debug level
        """
        # 1. Static rubric — flatten all criteria
        rubric = question.get("rubric", {})
        criteria = rubric.get("criteria", {})
        static_points: list[str] = []
        for criterion in criteria.values():
            if isinstance(criterion, dict):
                static_points.extend(criterion.get("key_points", []))

        if static_points:
            return static_points

        # 2. Dynamic rubric (follow-up questions)
        target_concepts = question.get("target_concepts", [])
        if target_concepts:
            return target_concepts

        # 3. Clarification target
        misconception = question.get("target_misconception")
        if misconception:
            return [str(misconception)]

        # 4. Nothing available — drift check skipped
        logger.debug(
            "No key points available for question_id=%s (type=%s) — "
            "drift detection skipped for this turn.",
            question.get("id", "unknown"),
            question.get("question_type", "unknown"),
        )
        return []
    
    def _scores_in_range(self, output: dict) -> Check:
        """
        All score fields must be between 0 and 10.
        """
        for f in self.SUB_SCORE_FIELDS + ["overall_score"]:
            val = output.get(f)
            if val is None:
                continue
            score = val.get("score") if isinstance(val, dict) else val
            if score is None:
                return Check(passed=False, message=f"{f} score is missing. {val!r}")
            try:
                score = float(score)
            except (TypeError, ValueError):
                return Check(passed=False, message=f"{f} score is not numeric. {val!r}")
            if not 0.0 <= score <= 10.0:
                return Check(passed=False, message=f"{f}={score} is outside range 0-10")
        return Check(passed=True)

    def _reasoning_provided(self, output: dict) -> Check:
        """Minium MIN_REASONING_CHARS of reasoning required."""
        reasoning = output.get("evaluation_reasoning", "")
        if len(reasoning) < self.MIN_REASONING_CHARS:
            return Check(
                passed=False,
                message=(
                    f"evaluation_reasoning is {len(reasoning)} chars — "
                    f"minimum {self.MIN_REASONING_CHARS} required"
                )
            )
        return Check(passed=True)
    def _scores_consistent(self, output: dict) -> Check:
        """
        Variance across sub-scores must be < MAX_SCORE_VARIANCE.
        Only checks SUB_SCORE_FIELDS - not overall_score.
        """
        scores = []
        for f in self.SUB_SCORE_FIELDS:
            val = output.get(f)
            if val is None:
                continue
            score = val.get("score") if isinstance(val, dict) else val
            if score is None:
                continue
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                continue

        if len(scores) >= 2:
            variance = max(scores) - min(scores)
            if variance > self.MAX_SCORE_VARIANCE:
                return Check(
                    passed=False,
                    message=(
                        f"Sub-score variance={variance:.1f} exceeds "
                        f"maximum {self.MAX_SCORE_VARIANCE} "
                        f"(scores: {[round(s, 1) for s in scores]})"
                    )
                )
        return Check(passed=True)

    def _overall_aligns_with_subscores(self, output: dict) -> "Check":
        """
        overall_score should be within 2.0 points of sub-score average.
        Catches cases where LLM assigns mismatched overall vs sub-scores.
        """
        sub_fields = ["technical_accuracy", "completeness", "depth", "clarity"]
        scores = []
        for f in sub_fields:
            val = output.get(f)
            score = val.get("score") if isinstance(val, dict) else val
            if score is not None:
                scores.append(float(score))
        
        if not scores:
            return Check(passed=True)  # can't validate without sub-scores
        
        avg = sum(scores) / len(scores)
        overall = float(output.get("overall_score", 5.0))
        
        if abs(overall - avg) > 2.0:
            return Check(
                passed=False,
                message=f"overall_score {overall} misaligned with sub-score avg {avg:.1f}"
            )
        return Check(passed=True)
    
    def _required_fields_present(self, output: dict) -> Check:
        """
        All REQUIRED_FIELDS must be present.

        Why is topic required: Supervisor._select_weakest_topic() reads
        evaluation["topic"] to build per-topic scoring. Missing topic
        means that evaluation is silently excluded from topic analysis.
        """
        missing = [f for f in self.REQUIRED_FIELDS if f not in output]
        if missing:
            return Check(passed=False, message=f"Missing required fields: {missing}")
        return Check(passed=True)
    
    def _key_point_coverage_alignment(
        self, output: dict, key_points: list[str]
    ) -> Check:
        """
        Drift detection: score must align with coverage ratio.

        Rules:
          score >= 8.0 AND coverage < 50% → score too high for coverage
          score <= 4.0 AND coverage > 70% → score too low for coverage

        Skipped silently when key_points=[] — logged in _extract_key_points.

        Thresholds chosen to catch genuine drift (not tight scoring variation):
          8.0+ is "strong" — should cover at least half the key points
          4.0- is "weak"  — shouldn't cover most of the key points
        """
        if not key_points:
            return Check(passed=True)

        covered = output.get("key_points_covered", [])
        score = output.get("overall_score", 5.0)
        coverage_ratio = len(covered) / len(key_points)

        if score >= self.HIGH_SCORE_THRESHOLD and coverage_ratio < self.LOW_COVERAGE_RATIO:
            return Check(
                passed=False,
                message=(
                    f"Score {score} but only {coverage_ratio:.0%} key points covered "
                    f"({len(covered)}/{len(key_points)}) — possible drift"
                )
            )

        if score <= self.LOW_SCORE_THRESHOLD and coverage_ratio > self.HIGH_COVERAGE_RATIO:
            return Check(
                passed=False,
                message=(
                    f"Score {score} but {coverage_ratio:.0%} key points covered "
                    f"({len(covered)}/{len(key_points)}) — possible drift"
                )
            )

        return Check(passed=True)
    
    def get_fallback(self) -> dict:
        """
        Neutral fallback when circuit breaker fires.

        is_fallback=True  → Supervisor excludes from EMA trajectory
        needs_human_review=True → flagged for manual inspection
        topic="unknown"   → detectable by _select_weakest_topic (skipped)
        All scores=5.0    → neutral midpoint, no distortion to final report
        """
        return {
            "overall_score": 5.0,
            "technical_accuracy": {"score": 5.0, "reasoning": "Unable to evaluate"},
            "completeness": {"score": 5.0, "reasoning": "Unable to evaluate"},
            "depth": {"score": 5.0, "reasoning": "Unable to evaluate"},
            "clarity": {"score": 5.0, "reasoning": "Unable to evaluate"},
            "evaluation_reasoning": "Evaluation could not be completed. Neutral scores have been assigned for this turn.",
            "key_points_covered": [],
            "key_points_missed": [],
            "misconceptions": [],
            "topic": "unknown",
            "question_id": "",
            "is_fallback": True,
            "needs_human_review": True,
            "consistency_divergence": None,
        }
    
class FeedbackValidationGate:
    """
    Validates FeedbackAgent Output.

    Checks:
      1. Length 20-200 words
      2. No forbidden phrases
      3. No sycophancy at low scores (score < SYCOPHANCY_SCORE_THRESHOLD)
      4. No score leakage
    """
    MIN_WORDS = 15
    MAX_WORDS = 200

    # check sycophancy only if score below 7.0
    SYCOPHANCY_SCORE_THRESHOLD = 7.0

    # check sycophancy for opening of feedback 
    SYCOPHANCY_CHECK_WINDOW = 150

    FORBIDDEN_PHRASES = [
        "you failed",
        "wrong answer",
        "incorrect",
        "you don't understand",
        "completely wrong",
    ]

    SYCOPHANTIC_PHRASES = [
        "great job",
        "excellent",
        "perfect",
        "well done",
        "amazing",
        "fantastic",
        "wonderful",
        "brilliant",
        "impressive",
        "outstanding",
    ]

    # Regex patterns for score leakage detection
    SCORE_LEAK_PATTERNS = [
        r'\b\d+\.?\d*/10\b',             # "8/10", "7.5/10"
        r'\bscored?\s+(?:\w+\s+)?\d+',   # "score 8", "scored 7", "scored a 8"
        r'\b\d+\.?\d*\s*out of',         # "8 out of 10"
        r'rating[:\s]+\d+',              # "rating: 8"
    ]
    def validate(self, output: dict, evaluation_score: float) -> ValidationResult:
        """
        Args:
            output: FeedbackOutput dict from FeedbackAgent
            evaluation_score: overall_score from EvaluationOutput
                              — needed for sycophancy threshold check
        """
        text = output.get("feedback_text", "")

        checks = [
            self._appropriate_length(text),
            self._no_forbidden_phrases(text),
            self._no_questions(text),
            self._no_sycophancy_at_low_scores(text, evaluation_score),
            self._no_score_leakage(text),
        ]

        failed = [c for c in checks if not c.passed]
        return ValidationResult(
            failed_checks=failed,
            feedback=[c.message for c in failed]
        )
    
    def _appropriate_length(self, text: str) -> Check:
        """
        Word count between MIN_WORDS and MAX_WORDS.

        Why words not chars: feedback quality is about density of content,
        not character count. Short words vs long words shouldn't affect
        whether feedback is "long enough".
        """
        words = len(text.split())
        if words < self.MIN_WORDS:
            return Check(passed=False, message=f"Feedback too short: {words} words (minimum {self.MIN_WORDS})")
        if words > self.MAX_WORDS:
            return Check(passed=False, message=f"Feedback too long: {words} words (maximum {self.MAX_WORDS})")
        return Check(passed=True)
    
    def _no_questions(self, text: str) -> Check:
        """
        Feedback must never contain questions directed at the candidate.
        Questions belong exclusively to the question selector — if feedback
        contains one, the candidate doesn't know which to answer.
        A '?' anywhere in the text is sufficient signal.
        """
        if "?" in text:
            return Check(
                passed=False,
                message="Feedback contains a question mark — feedback must be observational statements only"
            )
        return Check(passed=True)

    def _no_forbidden_phrases(self, text: str) -> Check:
        """Case-insensitive check across full feedback text."""
        text_lower = text.lower()
        for phrase in self.FORBIDDEN_PHRASES:
            if phrase in text_lower:
                return Check(passed=False, message=f"Forbidden phrase detected: '{phrase}'")
        return Check(passed=True)
    
    def _no_sycophancy_at_low_scores(self, text: str, score: float) -> Check:
        """
        Only fires when score < SYCOPHANCY_SCORE_THRESHOLD.
        Checks opening SYCOPHANCY_CHECK_WINDOW chars only —
        sycophantic openers appear at the start, not buried in the text.
        """
        if score >= self.SYCOPHANCY_SCORE_THRESHOLD:
            return Check(passed=True)

        opening = text.lower()[:self.SYCOPHANCY_CHECK_WINDOW]
        for phrase in self.SYCOPHANTIC_PHRASES:
            if phrase in opening:
                return Check(
                    passed=False,
                    message=(
                        f"Sycophantic opener '{phrase}' detected for "
                        f"score {score:.1f} (threshold {self.SYCOPHANCY_SCORE_THRESHOLD})"
                    )
                )
        return Check(passed=True)
    
    def _no_score_leakage(self, text: str) -> Check:
        """
        Regex scan for patterns that would reveal numeric scores to candidate.
        re imported at module level — compiled once, not per call.
        """
        text_lower = text.lower()
        for pattern in self.SCORE_LEAK_PATTERNS:
            if re.search(pattern, text_lower):
                return Check(passed=False, message=f"Score leakage detected (pattern: '{pattern}')")
        return Check(passed=True)
    
    def get_fallback(self) -> dict:
        """Minimal safe feedback that keeps the interview moving."""
        return {
            "feedback_text": "Thank you for your response. Let us take a moment and "
                "move on to the next question in our interview session.",
            "strength_acknowledgment": "",
            "gap_hint": "",
            "transition_phrase": "Moving on",
            "structure_template": "fallback",
        }
    
class QuestionSelectorValidationGate:
    """
    Validates QuestionSelectorAgent output.

    Checks:
      1. Question text present
      2. Valid question type
      3. Estimated time fits remaining budget (with 2-min buffer)
    """
    VALID_QUESTION_TYPES = ["retrieved", "follow_up", "clarification"]

    TIME_BUFFER_MINUTES = 2.0

    # fallback if estimated_time is missing
    DEFAULT_ESTIMATED_TIME = 5.0

    def validate(self, output: dict, remaining_minutes: float) -> ValidationResult:
        """
        Args:
            output:            QuestionOutput dict from QS Agent
            remaining_minutes: time remaining in interview session
        """
        checks = [
            self._question_present(output),
            self._valid_question_type(output),
            self._time_appropriate(output, remaining_minutes),
        ]

        failed = [c for c in checks if not c.passed]
        return ValidationResult(
            failed_checks=failed,
            feedback=[c.message for c in failed]
        )
    
    def _question_present(self, output: dict) -> Check:
        if not output.get("text", "").strip():
            return Check(passed=False, message="Question text is missing.")
        
        return Check(passed=True)
    
    def _valid_question_type(self, output: dict) -> Check:
        """
        Uses VALID_QUESTION_TYPES class constant — not hardcoded inline.
        Gate owns this constraint; QuestionOutput.question_type is str (not Literal)
        precisely because the gate validates it here.
        """
        q_type = output.get("question_type")
        if q_type not in self.VALID_QUESTION_TYPES:
            return Check(
                passed=False,
                message=f"Invalid question_type '{q_type}' — \
                must be one of {self.VALID_QUESTION_TYPES}"
            )
        return Check(passed=True)
    
    def _time_appropriate(self, output: dict, remaining_minutes: float) -> Check:
        """
        est_time > remaining_minutes + TIME_BUFFER_MINUTES → fail.
        +2 buffer accounts for transition time between questions.
        Default est_time matches QuestionOutput default (5.0) — no false-fails
        until real time metadata is ingested.
        """
        est_time = output.get("estimated_time_minutes", self.DEFAULT_ESTIMATED_TIME)
        if est_time > remaining_minutes + self.TIME_BUFFER_MINUTES:
            return Check(
                passed=False,
                message=(
                    f"Question estimated time ({est_time}min) exceeds "
                    f"remaining time ({remaining_minutes:.1f}min) + buffer ({self.TIME_BUFFER_MINUTES}min)"
                )
            )
        return Check(passed=True)
    
    def get_fallback(self) -> dict:
        """
        Safe fallback question.
        Bias-variance tradeoff chosen because:
          - Medium difficulty (safe default for any performance level)
          - ML fundamentals (always a valid topic)
          - 4 min (conservative estimate)
          - target_concepts populated (drift detection works on fallback)
        """
        return {
            "id": "fallback_001",
            "text": "Can you explain the bias-variance tradeoff?",
            "question_type": "retrieved",
            "topic": "ml_fundamentals",
            "difficulty": "medium",
            "estimated_time_minutes": 4.0,
            "target_concepts": ["bias", "variance", "tradeoff"],
            "parent_question_id": None,
            "target_misconception": None,
        }
    
class CircuitBreaker:
    """
    Per-turn retry budget for agent validation failures.

    Pattern:
      Turn starts  -> reset() clears all counts
      Agent fails  -> should_retry("evaluator") -> True  (first failure)
      Agent fails  -> should_retry("evaluator") -> False (budget exhausted)
      -> caller uses gate.get_fallback() instead of retrying

    Why per-turn reset:
      Without reset(), a failure on turn 2 permanently exhausts the retry
      budget for that agent for the rest of the interview. Each turn
      deserves a fresh budget — a transient LLM hiccup on turn 2 shouldn't
      prevent recovery on turn 5.
    """
    def __init__(self, max_retries: int = 1):
        self.max_retries = max_retries
        # per agent tracking
        self.retry_counts: dict[str, int] = {}

    def should_retry(self, agent_name: str) -> bool:
        """
        Returns True if agent has retries remaining, increments count.
        Returns False if retry budget exhausted - caller should use fallback.
        """
        count = self.retry_counts.get(agent_name, 0)
        if count < self.max_retries:
            self.retry_counts[agent_name] = count + 1
            return True
        return False
    def reset(self, agent_name: Optional[str] = None) -> None:
        """
        Reset retry counts.
            No args -> reset all agents (called at turn start)
            with agent_name -> reset specific agent only
        """
        if agent_name:
            self.retry_counts[agent_name] =0
        else:
            self.retry_counts = {}

class ValidationGateRegistry:
    """
    Single access point for all validation gates

    Agents call registry.get("evaluator") rather than instantiating
    gates directly — makes gate swapping testable and keeps agent
    constructors free of gate instantiation logic.
    """
    def __init__(self):
        self.gates = {
            "evaluator": EvaluatorValidationGate(),
            "feedback": FeedbackValidationGate(),
            "question_selector": QuestionSelectorValidationGate(),
        }

    def get(self, agent_name: str):
        """
        Args:
            agent_name (str): _descrione of "evaluator" | "feedback" | "question_selector"

        """
        if agent_name not in self.gates:
            raise KeyError(
                f"No validation gate registered for '{agent_name}'. "
                f"Available: {list(self.gates.keys())}"
            )
        return self.gates[agent_name]