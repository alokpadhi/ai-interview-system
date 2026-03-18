from typing import Literal
from pydantic import BaseModel, Field

class StartRequest(BaseModel):
    user_id: str = Field(description="The user id")
    time_budget_minutes: int = Field(description="total time allocated for the interview", 
                                     default=30)
    difficulty: Literal["easy", "medium", "hard"] = Field(description="initial dificulty level", 
                                                          default="medium")
    focus_topics: list[str] = Field(description="The focused topics in the interview", 
                                    default_factory=list)

class QuestionInfo(BaseModel):
    text: str
    topic: str
    estimated_time_minutes: float

class StartResponse(BaseModel):
    session_id: str
    question: QuestionInfo
    time_budget_minutes: int
    target_questions: int

class SubmitRequest(BaseModel):
    session_id: str
    response: str

class ProgressInfo(BaseModel):
    questions_completed: int
    time_elapsed_minutes: float
    time_remaining_minutes: float

class SubmitResponse(BaseModel):
    feedback: str
    next_question: QuestionInfo | None = None
    progress: ProgressInfo
    continue_interview: bool

class FinalReport(BaseModel):
    overall_score: float
    adjusted_score: float
    questions_asked: int
    time_taken_minutes: float
    difficulty_progression: list[str]
    topic_scores: dict[str, float]
    strengths: list[str]
    areas_for_improvement: list[str]
    performance_notes: list[str]
    fallback_count: int
    detailed_evaluations: list[dict]