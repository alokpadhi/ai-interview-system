"""
InterviewState — shared state for the LangGraph interview graph.

Design notes:
  - Every field has an explicit reducer (no implicit defaults)
  - operator.add fields: agents return ONLY new items, reducer appends
  - last_value fields: agents return the new value, reducer replaces
  - Parallel fan-out safety: Feedback and QS write to non-overlapping keys
"""
from typing import TypedDict, Annotated, Literal, Optional
from datetime import datetime
from langgraph.graph.message import add_messages
import operator
import uuid


def last_value(existing, new):
    """Explicit last-write-winds reducer - makes the policy visible.
    Used for scalar fields owned by a single agent."""
    return new if new is not None else existing


class InterviewState(TypedDict):
    # Conversation Manager
    messages: Annotated[list, add_messages]
    conversation_summary: Annotated[Optional[str], last_value]
    summary_turn_count: Annotated[int, last_value]

    # Current Turn Data (each owned by exactly one parallel branch)
    current_question: Annotated[Optional[dict], last_value] # QS Agent
    candidate_response: Annotated[Optional[str], last_value] # API Layer
    current_evaluation: Annotated[Optional[dict], last_value] # Evaluator
    current_feedback: Annotated[Optional[str], last_value] # Feedback

    # Interview metadata (immutable after /start)
    interview_id: str
    interview_plan: Annotated[Optional[dict], last_value]
    interview_start_time: datetime
    time_budget_minutes: int
    focus_topics: Annotated[list[str], last_value]

    # Mutable metadata
    stage: Annotated[Literal["init", "planning", "questioning", "complete"], 
                     last_value]
    question_count: Annotated[int, last_value] # Supervisor increments after fan-in

    # difficulty tracking (owned by Supervisor)
    difficulty_level: Annotated[Literal["easy", "medium", "hard"], last_value]
    original_difficulty: Annotated[Optional[str], last_value]
    difficulty_reduced_due_to_performance: Annotated[bool, last_value]

    # Performance tracking (Internal, Owned by Supervisor)
    performance_trajectory: Annotated[list[float], operator.add]
    ema_trajectory: Annotated[list[float], last_value]
    difficulty_history: Annotated[list[str], operator.add]
    all_evaluations: Annotated[list[dict], operator.add]

    # Topic tracking (QS)
    topics_covered: Annotated[list[str], operator.add]

    # Question flow (QS)
    question_mode: Annotated[Literal["retrieve", "follow_up", "clarify"], 
                             last_value]
    follow_up_count: Annotated[int, last_value]
    conversation_thread: Annotated[list[str], operator.add]

    # Feedback variation tracking (Feedback Agent)
    previous_feedback_structures: Annotated[list[str], operator.add]
    recent_feedbacks: Annotated[list[str], operator.add]

    # control flags (Supervisor owns)
    should_continue: Annotated[bool, last_value]
    needs_human_review: Annotated[bool, last_value]
    error_state: Annotated[Optional[dict], last_value]
    end_reason: Annotated[Optional[str], last_value]

def initialize_state(
    difficulty: str,
    time_budget_minutes: int,
    focus_topics: list[str],
) -> InterviewState:
    """
    State initialization at the very start.
    called from FastAPI main.py
    """
    return {
        "difficulty_level": difficulty,
        "time_budget_minutes": time_budget_minutes,
        "focus_topics": focus_topics,
        "interview_id": str(uuid.uuid4()),
        "interview_start_time": datetime.now(),
        "stage":"init",
        "question_count":0,
        "should_continue": True,
        "difficulty_reduced_due_to_performance": False,
        "original_difficulty": difficulty,
        "messages": [],
        "performance_trajectory": [],
        "ema_trajectory": [],
        "all_evaluations": [],
        "difficulty_history": [],
        "topics_covered": [],
        "conversation_thread": [],
        "previous_feedback_structures": [],
        "recent_feedbacks": [],
        "conversation_summary" : None,
        "summary_turn_count": 0,
        "current_question": None,
        "candidate_response": None,
        "current_evaluation": None,
        "current_feedback": None,
        "interview_plan": None,
        "question_mode": "retrieve",
        "follow_up_count": 0,
        "needs_human_review": False,
        "error_state": None,
        "end_reason": None
    }
