from dataclasses import dataclass
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing import Literal, Optional

from src.utils.logging_config import get_logger
from src.services.trend_analyzer import TrendAnalyzer
from src.services.validation import ValidationGateRegistry, CircuitBreaker
from src.graph.state import InterviewState

logger = get_logger(__name__)

DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
DIFFICULTY_FROM_ORDER = {0: "easy", 1: "medium", 2: "hard"}
NEUTRAL_EMA = 5.0

@dataclass
class Observation:
    question_count: int
    topics_covered: list[str]
    difficulty_level: str
    elapsed_minutes: float
    remaining_minutes: float
    current_ema: float
    target_questions: int

@dataclass
class Analysis:
    time_critical: bool
    time_pressure: bool
    should_adjust_difficulty: bool
    adjustment_direction: str
    performance_trend: str
    questions_remaining: int
    avg_ema: float

class PlanOutput(BaseModel):
    topic_sequence: list[str] = Field(
        description="Ordered list of technical topics to cover, from foundational to advanced. "
                    "Use specific names e.g. 'gradient_descent', not vague ones like 'ML basics'."
    )
    difficulty_curve: list[Literal["easy", "medium", "hard"]] = Field(
        description="Difficulty level for each topic in topic_sequence, same order, same length. "
                    "e.g. ['easy', 'medium', 'hard']. One entry per topic, never per question."
    )
    time_allocation: dict[str, float] = Field(
        description="Minutes allocated per topic. Keys must match topic_sequence exactly. "
                    "Values must sum to the total time budget. e.g. {'gradient_descent': 8.0}"
    )
    focus_areas: list[str] = Field(
        description="Subset of topics needing extra depth based on candidate's stated interests. "
                    "May be empty. Must be topics that exist in topic_sequence."
    )


INTERVIEW_PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI interview planner. Create a structured interview plan.\n"
     "Return valid JSON with exactly these keys: "
     "topic_sequence, difficulty_curve, time_allocation, focus_areas.\n\n"
     "Rules:\n"
     "- topic_sequence MUST only contain topics from the available_topics list. "
     "Do not invent new topic names.\n"
     "- difficulty_curve has ONE entry per topic (not per question). "
     "Values must be exactly one of: \"easy\", \"medium\", \"hard\".\n"
     "  Example: if topic_sequence is [\"transformers\", \"RAG\"], "
     "difficulty_curve must be [\"medium\", \"hard\"]\n"
     "- time_allocation must map each topic name to minutes as a flat dict. "
     "Example: {{\"transformers\": 8, \"RAG\": 7}}. "
     "Every topic in topic_sequence must have an entry. "
     "Total must equal time_budget_minutes exactly.\n"
     "- focus_areas is a list of strings describing key concepts to emphasize."),
    ("human",
     "Difficulty: {difficulty}\n"
     "Focus topics: {focus_topics}\n"
     "Time budget: {time_budget} minutes\n"
     "Target questions: {target_questions}\n"
     "Available topics: {available_topics}")
])
def _build_plan_chain(complex_llm):
    return (INTERVIEW_PLAN_PROMPT
            | complex_llm.with_structured_output(PlanOutput)).with_retry(
                stop_after_attempt=2, wait_exponential_jitter=True
            )

class SupervisorAgent:
    """
    Orchestrates the interview flow.
    
    Key responsibilities:
    - Plan-and-Execute at /start (1 LLM call, 14B)
    - Rule-based OODA routing per turn (0 LLM calls)
    - EMA-smoothed trend detection for difficulty adaptation
    - Owns question_count: INCREMENTS in validate_and_decide after fan-in
    - Indexes difficulty_curve by TOPIC INDEX (len(topics_covered)), not question_count
    - NO early termination for performance
    
    Decoupled from RAG at /start:
    - Supervisor ONLY creates the plan
    - First topic retrieval is handled by QS in the start_graph's first_question node
    - Background pre-warming of remaining topics triggered by API layer
    """
    def __init__(self,
                 complex_llm: BaseChatModel,
                 trend_analyzer: TrendAnalyzer,
                 available_topics: list[str]):
        self.complex_llm = complex_llm
        self.trend_analyzer = trend_analyzer
        self.validation_gates = ValidationGateRegistry()
        self.circuit_breaker = CircuitBreaker(max_retries=1)
        self.plan_chain = _build_plan_chain(complex_llm)
        self.available_topics = available_topics

    async def create_interview_plan(self, 
                              state: InterviewState, 
                              config: RunnableConfig) -> dict:
        """
        Create plan only. First topic retrieval handled by QS.
        Background pre-warming triggered by API layer after graph completes.
        """
        target_questions = self._calculate_target_questions(state["time_budget_minutes"])

        plan = await self.plan_chain.ainvoke({
            "difficulty": state["difficulty_level"],
            "focus_topics": state.get("focus_topics", []),
            "time_budget": state["time_budget_minutes"],
            "target_questions": target_questions,
            "available_topics": self.available_topics,
        }, config=config)

        difficulty_level = plan.difficulty_curve[0]

        return {
            "interview_plan": plan.model_dump(),
            "original_difficulty": state["difficulty_level"],
            "difficulty_level": difficulty_level,
            "difficulty_reduced_due_to_performance": False,
            "stage": "questioning"
        }
    
    def _calculate_target_questions(self, time_budget: int) -> int:
        return max(5, min(12, time_budget // 4)) # 30min -> 7-8 questions; 60min -> 12 questions
    
    async def validate_and_decide(
            self,
            state: InterviewState,
            config: RunnableConfig
    ) -> dict:
        """
        Rule-based OODA with EMA authority and fallback protection.
        Supervisor is the SOLE owner of question_count — increments here.
        """
        self.circuit_breaker.reset()
        evaluation = state["current_evaluation"]
        is_fallback = evaluation.get("is_fallback", False)

        if is_fallback:
            new_trajectory = []
        else:
            new_score = evaluation["overall_score"]
            new_trajectory = [new_score]

        full_trajectory = state["performance_trajectory"] + new_trajectory
        new_ema = self.trend_analyzer.calculate_ema(full_trajectory)

        new_all_evaluations = [evaluation]
        new_difficulty_history = [state["difficulty_level"]]

        observation = self._observe(state, new_ema)
        analysis = self._orient(observation, full_trajectory)
        should_continue, end_reason = self._decide_continuation(analysis, state)
        new_difficulty, reduced = self._resolve_difficulty(analysis, state)

        logger.info(f"Interview {state['interview_id']} | Q{state['question_count']} | "
                f"EMA: {new_ema[-1] if new_ema else NEUTRAL_EMA:.2f} | difficulty: {new_difficulty} | "
                f"continue: {should_continue}")
        
        if not should_continue:
            logger.info(f"Interview ending — reason: {end_reason}")

        return {
            # lists; return only new items (operator.add reducer)
            "performance_trajectory": new_trajectory,
            "ema_trajectory": new_ema, # full recalculation
            "difficulty_history": new_difficulty_history,
            "all_evaluations": new_all_evaluations,

            # scalars: return new value(last_value reducer)
            "should_continue": should_continue,
            "end_reason": end_reason,
            "difficulty_level": new_difficulty,
            "difficulty_reduced_due_to_performance": (
                state["difficulty_reduced_due_to_performance"] or reduced
            ),
            "stage": "questioning",

            # increment
            "question_count": state['question_count'] + 1
        }
    
    def _observe(self, state: InterviewState, ema: list[float]) -> Observation:
        elapsed_time = self._get_elapsed_minutes(state)
        remaining_time = state["time_budget_minutes"] - elapsed_time
        question_count = state["question_count"]
        return Observation(
            question_count=question_count,
            topics_covered=state["topics_covered"],
            difficulty_level=state["difficulty_level"],
            elapsed_minutes=elapsed_time,
            remaining_minutes=remaining_time,
            current_ema=ema[-1] if ema else NEUTRAL_EMA,
            target_questions=len(state.get("interview_plan", {}).get("topic_sequence", []))
        )
    def _get_elapsed_minutes(self, state: InterviewState) -> float:
        if not state.get("interview_start_time"):
            return 0
        return (datetime.now() - state["interview_start_time"]).total_seconds() / 60
    
    def _orient(self, obs: Observation, trajectory: list[float]) -> Analysis:
        trend = self.trend_analyzer.get_trend(trajectory)
        should_adjust, direction = self.trend_analyzer.should_adjust_difficulty(
            trajectory)
        return Analysis(
            time_critical=obs.remaining_minutes < 2,
            time_pressure=obs.remaining_minutes < 5,
            should_adjust_difficulty=should_adjust,
            adjustment_direction=direction,
            performance_trend=trend,
            questions_remaining=obs.target_questions - obs.question_count,
            avg_ema=obs.current_ema
        )
    
    def _decide_continuation(self,
                             analysis: Analysis,
                             state: InterviewState) -> tuple[bool, Optional[str]]:
        if analysis.time_critical:
            return False, "time_up"
        
        target = len(state["interview_plan"]["topic_sequence"])
        if state["question_count"] + 1 >= target:
            return False, "completed"
        
        return True, None
    
    def _resolve_difficulty(self,
                            analysis: Analysis,
                            state: InterviewState) -> tuple[str, bool]:
        current_difficulty = state["difficulty_level"]
        # time pressure
        if analysis.time_pressure:
            return current_difficulty, False
        
        # question mode
        if state["question_mode"] in ("follow_up", "clarify"):
            return current_difficulty, False
        
        # plan's difficulty is indexed by topic position
        plan_difficulty = self._get_plan_difficulty_for_next_topic(state)
        if not analysis.should_adjust_difficulty:
            return plan_difficulty, False
        
        if analysis.adjustment_direction == "increase": # ema says increase
            ema_adjusted = {"easy": "medium", "medium": "hard", "hard": "hard"}[current_difficulty]
            return self._harder_of(plan_difficulty, ema_adjusted), False #take the HARDER of (plan, ema_adjusted)
        
        if analysis.adjustment_direction == "decrease": # ema says decrease
            ema_adjusted = {"easy": "easy", "medium": "easy", "hard": "medium"}[current_difficulty]
            return self._easier_of(plan_difficulty, ema_adjusted), True # take the EASIER of (plan, ema_adjusted)
        
        return plan_difficulty, False

    def _get_plan_difficulty_for_next_topic(self, state: InterviewState) -> str:
        plan = state.get("interview_plan", {})
        difficulty_curve = plan.get("difficulty_curve", [])
        topic_index = len(state.get('topics_covered', []))

        if topic_index < len(difficulty_curve):
            return difficulty_curve[topic_index]
        
        return state["difficulty_level"] # current difficulty
    
    def _harder_of(self, plan_difficulty: str, ema_adjusted: str) -> str:
        return DIFFICULTY_FROM_ORDER[max(DIFFICULTY_ORDER[plan_difficulty],
                                         DIFFICULTY_ORDER[ema_adjusted])]
    
    def _easier_of(self, plan_difficulty: str, ema_adjusted: str) -> str:
        return DIFFICULTY_FROM_ORDER[min(DIFFICULTY_ORDER[plan_difficulty],
                                         DIFFICULTY_ORDER[ema_adjusted])]



        

