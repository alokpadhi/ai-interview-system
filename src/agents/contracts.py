"""
Inter-agent contracts for the AI Interview System.

These Pydantic models define the typed interfaces between agents.
Validation gates use these to catch schema violations at boundary time,
not when a downstream agent tries to read a missing field.

Ownership:
  EvaluationOutput  → written by EvaluatorAgent, read by Supervisor + QS
  FeedbackOutput    → written by FeedbackAgent, read by API layer
  QuestionOutput    → written by QS, read by EvaluatorAgent (next turn) + API layer
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# Sub Models
class SubScore(BaseModel):
    """
    Nested score model for individual evaluation dimensions.
    The validation gate does: val.get("score") if isinstance(val, dict) else val
    This is why sub-scores are dicts, not raw floats.
    """
    score: float = Field(..., ge=0.0, le=10.0, description="score between 0 and 10")
    reasoning: str = Field(..., description="Explanation for the given score")

class EvaluationOutput(BaseModel):
    """
    Contract: Evaluator -> Supervisor + QS.
    QS reads overall_score, key_points_missed, misconceptions.
    Supervisor reads overall_score, is_fallback.
    """
    overall_score: float = Field(..., ge=0.0, le=10.0, 
                                 description="aggregate score across all dimenions")
    technical_accuracy: SubScore = Field(...,
                                         description="correctness and precision of technical content")
    completeness: SubScore = Field(..., 
                                   description="coverage of how complete the candidate response is")
    depth: SubScore = Field(...,
                            description="Depth of understanding")
    clarity: SubScore = Field(...,
                              description="Clarity and structure of the response.")
    
    # Reasoning — validation gate enforces minimum 50 chars
    evaluation_reasoning: str = Field(...,
                                      description="Ful CoT reasoning behind the score")

    # Key point tracking - drives QS follow-up decisions and drift detection
    key_points_covered: list[str] = Field(default_factory=list)
    key_points_missed: list[str] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list,
                                      description="Factual errors or conceptual misunderstanding detected")
    
    # Topic injection - set by EvaluatorAgent from current_question["topic"]
    topic: str = Field(default="", description="Topic injected from current_question - " \
                                    "empty string signals injection failure.")
    question_id: str = Field(default="", description="ID of the question being evaluated")

    # circuit breaker signal - Supervisor excludes is_fallback=True from EMA trajectory
    is_fallback: bool = Field(default=False, description="True if circuit breaker fired" \
                                " and neutral scores were assigned")
    
    # Self-consistency divergence flag — set when max-min score spread > 2.0
    needs_human_review: bool = Field(default=False, description="True if self-consistency divergence exceeded 2.0")
    consistency_divergence: Optional[float] = Field(default=None, description="Score spread across self-consistency samples — set when needs_human_review=True")

    @field_validator("overall_score")
    @classmethod
    def check_overall_score(cls, v: float) -> float:
        if not 0.0 <= v <= 10.0:
            raise ValueError(f"overall_score must be between 0 and 10, got {v}")
        return v

class FeedbackOutput(BaseModel):
    """
    Contract: FeedbackAgent -> API Layer

    API layer only exposes feedback text to the user.
    All other fields are internal - used for variation tracking and composition.
    """
    # for the user
    feedback_text: str = Field(..., description="Final composed feedback shown to candidate.")

    # FeedbackComposer components
    # Using empty string not None, for structure.format()
    strength_acknowledgment: str = Field(default="", description="Acknowledgment of" \
                                    "what candidate did well - empty score when" \
                                    " score < 5")
    gap_hint: str = Field(default="", description="Implicit hint about gaps without" \
                                        " stating 'you misses x'")
    transition_phrase: str = Field(default="", description="Natural trasition to " \
                                    "the next question")
    
    # Variation tracking - Written to state via operator.add
    # Feedback Agent reads previous_feedback_structures from state to avoid repitition
    structure_template: str = Field(..., description="Template key used this turn -" \
                                            " tracked to prevent Mad libs repetition " \
                                            "across turns")

class QuestionOutput(BaseModel):
    """
    Contract: QuestionSelectorAgent -> EvaluatorAgent (next turn) + API layer
    
    EvaluatorAgent reads: text, rubric, target_concepts, topic
    API layer reads: text, topic, estimated_time_minutes
    ValidationGate reads: text, question_type, estimated_time_minutes
    """
    id: str = Field(..., description="Unique question identifier - used for deduplication" \
                        "and used_ids tracking in cache")
    text: str = Field(..., description="The question text shown to the candidate")

    # str not Literal — validation gate owns the constraint check:
    #   output.get("question_type") not in ["retrieved", "follow_up", "clarification"]
    question_type: str = Field(..., description="One of: retrieved | follow_up | clarification")

    topic: str = Field(..., description="Topic area — injected by QS, read by Evaluator for per-topic scoring")
    difficulty: str = Field(..., description="One of: easy | medium | hard")

    # Dynamic rubric — QS generates this for follow_up and clarify modes
    # EvaluatorAgent uses target_concepts when no static rubric exists in DB
    # ValidationGate uses this for drift detection on non-retrieved questions
    target_concepts: list[str] = Field(default_factory=list, description="Expected concepts for follow-up/clarify questions — acts as dynamic rubric")

    estimated_time_minutes: float = Field(default=5.0, description="Estimated time to answer in minutes")

    parent_question_id: Optional[str] = Field(default=None, description="ID of the question this follows up on")
    target_misconception: Optional[str] = Field(default=None, description="Specific misconception this clarification question targets")


