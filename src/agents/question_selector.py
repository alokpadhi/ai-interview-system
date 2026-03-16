from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

from src.rag.agentic_rag import AgenticRAGService
from src.graph.state import InterviewState
from src.utils.logging_config import get_logger
from src.rag.cache import InterviewCacheStore
from src.services.validation import ValidationGateRegistry, CircuitBreaker


logger = get_logger(__name__)

class QuestionSelection(BaseModel):
    """Structured output (pydantic) for question selection"""
    selected_id: str = Field(description="ID of the best candidate question")
    reasoning: str = Field(description="One sentence explaining the selection")

FOLLOW_UP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an experienced technical interviewer.\n\n"
     "Your task is to generate a targeted follow-up question based on a candidate's response.\n\n"
     "Goals:\n"
     "- Probe the candidate's understanding of concepts they missed.\n"
     "- Encourage deeper reasoning.\n"
     "- Maintain a professional interviewer tone.\n\n"
     "Constraints:\n"
     "- Ask ONLY ONE question. One sentence. No sub-questions. No 'and also...' constructions.\n"
     "- Do NOT reveal the correct answer.\n"
     "- Do NOT explicitly mention the missed points.\n"
     "- The tone should be probing and evaluative, NOT teaching or explanatory.\n"
     "- The question should naturally guide the candidate toward the missing concepts.\n"
     "- If missed_points is empty, generate a depth-probing question about the core "
     "concept in the original question.\n\n"
     "Return ONLY the question text."),
    ("human",
     "Original Question: {original_question}\n"
     "Candidate Response: {candidate_response}\n"
     "Concepts the candidate missed: {missed_points}\n"
     "Topic: {topic}")
])

CLARIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a technical interviewer conducting a conceptual interview.\n\n"
     "Your goal is to ask a clarification question that helps the candidate "
     "re-examine their reasoning and recognize a potential misconception.\n\n"
     "Guidelines:\n"
     "- Ask ONE concise clarification question. One sentence.\n"
     "- Ground your question specifically in the misconception provided. "
     "The question should make the candidate reconsider that specific claim, "
     "not the topic generally.\n"
     "- Encourage the candidate to rethink their explanation.\n"
     "- Do NOT provide hints, corrections, or teaching.\n"
     "- Maintain an interviewer tone rather than a tutor tone.\n"
     "- The question should guide the candidate toward identifying the issue themselves.\n\n"
     "Return ONLY the clarification question."),
    ("human",
     "Original Question: {original_question}\n"
     "Candidate Response: {candidate_response}\n"
     "Detected Misconception: {misconception}")
])

RE_ENGAGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an experienced technical interviewer.\n\n"
     "The candidate did not address the question that was asked.\n"
     "Rephrase the original question from a slightly different angle or a simpler "
     "entry point to help the candidate understand what you are looking for.\n\n"
     "Constraints:\n"
     "- ONE question only. One sentence.\n"
     "- Must be a genuine rephrasing or simpler version of the original — not a new topic.\n"
     "- Do NOT give hints about the answer or partial solutions.\n"
     "- Maintain an evaluative interviewer tone, not a tutoring tone.\n\n"
     "Return ONLY the question text."),
    ("human",
     "Original Question: {original_question}\n"
     "Topic: {topic}")
])

REACT_SELECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are managing the flow of a technical interview.\n\n"
     "Your task is to select the most appropriate next question from a list "
     "of candidate questions.\n\n"
     "Each candidate question has: id, text, difficulty, topic. "
     "Base your selection on text and difficulty primarily.\n\n"
     "Selection criteria:\n"
     "1. The question should match the target difficulty level.\n"
     "2. Avoid repeating recently covered topics.\n"
     "3. Adapt to the candidate's performance trend.\n"
     "4. Prefer questions that best evaluate conceptual understanding.\n\n"
     "Return ONLY a JSON object. No preamble. No explanation outside the JSON.\n"
     "{{\n"
     "  \"selected_id\": \"<id of chosen question>\",\n"
     "  \"reasoning\": \"<one sentence>\"\n"
     "}}"),
    ("human",
     "Candidate Questions: {candidates}\n"
     "Target Difficulty Level: {difficulty_level}\n"
     "Recently Covered Topics: {topics_covered}\n"
     "Candidate Performance Trend: {performance_trend}")
])

def _build_followup_chain(llm):
    parser = StrOutputParser()
    followup_chain = FOLLOW_UP_PROMPT | llm | parser

    return followup_chain.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=True
    )

def _build_clarify_chain(llm):
    parser = StrOutputParser()
    clarify_chain = CLARIFICATION_PROMPT | llm | parser

    return clarify_chain.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=True
    )

def _build_reengage_chain(llm):
    parser = StrOutputParser()
    reengage_chain = RE_ENGAGE_PROMPT | llm | parser
    return reengage_chain.with_retry(stop_after_attempt=2, wait_exponential_jitter=True)

def _build_selection_chain(llm):
    chain = REACT_SELECTION_PROMPT | llm.with_structured_output(QuestionSelection)
    return chain.with_retry(stop_after_attempt=2, wait_exponential_jitter=True)

class QuestionSelectorAgent:
    """Owns ALL question decisions and topic tracking.
    Delegates CRAG to AgenticRAGService.
    
    Key fixes:
    - Injects topic into every question dict (so Evaluator and _select_weakest_topic work)
    - Generates target_concepts alongside follow-up/clarify Qs (dynamic rubric)
    - Uses atomic select_and_mark via cache_store.select_and_mark()
    """
    MAX_FOLLOW_UPS = 2
    FUNDAMENTAL_TOPICS = ["machine_learning_fundamentals", "deep_learning_fundamentals", "large_language_models"]

    def __init__(
            self, rag_service: AgenticRAGService,
            fast_llm: BaseChatModel,
            complex_llm: BaseChatModel,
            cache_store: InterviewCacheStore,
            circuit_breaker: CircuitBreaker
    ):
        self.rag = rag_service
        self.fast_llm = fast_llm
        self.complex_llm = complex_llm
        self.cache_store = cache_store
        self.follow_up_chain = _build_followup_chain(complex_llm)
        self.clarify_chain = _build_clarify_chain(complex_llm)
        self.reengage_chain = _build_reengage_chain(complex_llm)
        self.react_select_chain = _build_selection_chain(fast_llm)
        self.validation_gates = ValidationGateRegistry().get("question_selector")
        self.circuit_breaker = circuit_breaker

    async def execute(
            self,
            state: InterviewState,
            config: RunnableConfig
    ) -> dict:
        remaining_time = self._get_remaining_minutes(state)
        mode = self._determine_question_mode(state, remaining_time)

        if mode == "retrieve":
            question, selected_topic = await self._retrieve_question(
                state, remaining_time, config
            )
            return {
                "current_question": question,
                "question_mode": mode,
                "follow_up_count": 0,
                "conversation_thread": [question["id"]],
                "topics_covered": [selected_topic]
            }
        
        elif mode == "follow_up":
            question = await self._generate_follow_up(state, config)
            return {
                "current_question": question,
                "question_mode": mode,
                "follow_up_count": state.get("follow_up_count", 0) + 1,
                "conversation_thread": [question["id"]],
                "topics_covered": [],
            }
        
        elif mode == "clarify":
            question = await self._generate_clarification(state, config)
            return {
                "current_question": question,
                "question_mode": mode,
                "follow_up_count": state.get("follow_up_count", 0) + 1,
                "conversation_thread": [question["id"]],
                "topics_covered": [],
            }
        
    def _get_remaining_minutes(self, state: InterviewState) -> float:
        if not state.get("interview_start_time"):
            return state.get("time_budget_minutes", 30)
        elapsed = (datetime.now() - state["interview_start_time"]).total_seconds() / 60

        return max(0, state["time_budget_minutes"] - elapsed)
    
    def _determine_question_mode(
            self, state: InterviewState, remaining_time: float
    ) -> str:
        if not state.get("current_evaluation"):
            return "retrieve"
        if remaining_time < 5:
            return "retrieve"

        eval_data = state["current_evaluation"]
        score = eval_data["overall_score"]
        missed = eval_data.get("key_points_missed", [])
        covered = eval_data.get("key_points_covered", [])
        misconceptions = eval_data.get("misconceptions", [])
        follow_ups = state.get("follow_up_count", 0)

        # Off-topic: candidate answered a completely different question.
        # A real interviewer rephrases rather than abandoning the topic.
        # _generate_follow_up detects off-topic internally and routes to re-engagement.
        #
        # NOTE: covered=[] can occur when the evaluator has no rubric for the question
        # and lists no specific key points, even for a good on-topic response.
        # Trust the score: score >= 5.0 signals an on-topic response regardless of
        # whether covered is populated. Only flag off-topic when score is genuinely low.
        is_off_topic = score < 3.0 or (not bool(covered) and score < 5.0)
        if is_off_topic:
            if follow_ups < self.MAX_FOLLOW_UPS:
                return "follow_up"   # _generate_follow_up → _generate_reengagement
            return "retrieve"        # Give up after MAX attempts, move to next topic

        # On-topic paths below — candidate addressed the question.

        # clarify takes priority over follow_up.
        # A misconception left unchallenged will distort every subsequent answer.
        # Probing gaps when the candidate holds a wrong belief is counterproductive.
        if misconceptions and follow_ups < self.MAX_FOLLOW_UPS:
            return "clarify"

        # follow_up → on-topic but incomplete (no misconceptions, just gaps)
        if score < 7.0 and missed and follow_ups < self.MAX_FOLLOW_UPS:
            return "follow_up"
        if 7.0 <= score < 8.0 and missed and follow_ups < 1:
            return "follow_up"

        return "retrieve"
    
    async def _retrieve_question(
            self,
            state: InterviewState,
            remaining_time: float,
            config: RunnableConfig
    ) -> tuple[dict, str]:
        # Note: difficulty_level reflects current turn's value — Supervisor
        # updates difficulty after fan-in, so first retrieval for a new topic
        # uses previous turn's difficulty. Accepted tradeoff — takes effect
        # on the following turn.
        topic = self._get_next_topic_from_plan(state)
        difficulty = state["difficulty_level"]
        session_id = state["interview_id"]

        # atomic select + mark: eliminates TOCTOU race
        async def _select_fn(candidates: list[dict]) -> dict:
            return await self._react_select(candidates, state, config)
        
        selected = await self.cache_store.select_and_mark(
            session_id=session_id,
            topic=topic,
            difficulty=difficulty,
            selector_fn=_select_fn
        )

        if selected is None:
            # cache missed - retrive via crag
            crag_result = await self.rag.retrieve_with_crag(
                topic=topic,
                difficulty=difficulty,
                exclude_ids=self._get_used_question_ids(state),
                remaining_time=remaining_time
            )

            await self.cache_store.set_topic_questions(
                session_id=session_id,
                topic=topic,
                difficulty=difficulty,
                questions=[c.to_question_dict() for c in crag_result.candidates],
                crag_grade=crag_result.grade
            )

            selected = await self.cache_store.select_and_mark(
                session_id=session_id,
                topic=topic,
                difficulty=difficulty,
                selector_fn=_select_fn
            )

            if selected is None and crag_result.candidates:
                selected = crag_result.candidates[0].to_question_dict()
            elif selected is None:
                selected = self._get_fallback_question()

        selected["question_type"] = "retrieved"
        selected["topic"] = topic
        return selected, topic
    
    async def _react_select(
            self,
            candidates: list[dict],
            state: InterviewState,
            config: RunnableConfig
    ) -> dict:
        if not candidates:
            return self._get_fallback_question()
        
        if len(candidates) == 1:
            return candidates[0]
        
        selection: QuestionSelection = await self.react_select_chain.ainvoke({
            "candidates": [
                {"id": c["id"], "text": c["text"], "difficulty": c.get("difficulty", "")}
                for c in candidates[:5]
            ],
            "difficulty_level": state["difficulty_level"],
            "topics_covered": state["topics_covered"][-3:],
            "performance_trend": self._get_performance_trend(state)
        }, config=config)

        for c in candidates:
            if c["id"] == selection.selected_id:
                return c
        return candidates[0]
    
    def _get_next_topic_from_plan(self, state: InterviewState) -> str:
        plan = state["interview_plan"]
        topic_sequence = plan.get("topic_sequence", [])
        topics_covered = state.get("topics_covered", [])

        # remaining = [t for t in topic_sequence if t not in topics_covered]

        uncovered = [t for t in topic_sequence if t not in topics_covered]
        remaining = uncovered if uncovered else topic_sequence

        if not remaining:
            return self._select_weakest_topic(state)
        
        if state.get("difficulty_reduced_due_to_performance"):
            fundamentals = [t for t in remaining if t in self.FUNDAMENTAL_TOPICS]
            others = [t for t in remaining if t not in self.FUNDAMENTAL_TOPICS]
            remaining = fundamentals + others if fundamentals else remaining

        return remaining[0]
    
    def _select_weakest_topic(self, state: InterviewState) -> str:
        """
        Select topic where candidate struggled most.
        Uses evaluation["topic"] field — Evaluator injects this from current_question.
        """
        performance_by_topic = {}
        for ev in state.get("all_evaluations", []):
            topic = ev.get("topic", "general")
            if topic == "unknown":
                continue
            if topic not in performance_by_topic:
                performance_by_topic[topic] = []

            performance_by_topic[topic].append(ev["overall_score"])

        topic_avgs = {t: sum(s)/len(s) for t, s in performance_by_topic.items()}
        if topic_avgs:
            return min(topic_avgs.keys(), key=lambda t: topic_avgs[t])
        return "machine_learning_fundamentals"
    
    def _get_performance_trend(self, state: InterviewState):
        trajectory = state.get("performance_trajectory", [])
        if len(trajectory) < 3:
            return "stable"
        ema = state.get("ema_trajectory", trajectory)
        if len(ema) >=4 and ema[-1] - ema[-4] > 0.8:
            return "improving"
        elif len(ema) >= 4 and ema[-1] - ema[-4] < -0.8:
            return "declining"
        
        return "stable"
    
    async def _generate_reengagement(
            self,
            state: InterviewState,
            config: RunnableConfig
    ) -> dict:
        """Rephrases the current question when the candidate answered off-topic."""
        original = state["current_question"]

        response = await self.reengage_chain.ainvoke({
            "original_question": original["text"],
            "topic": original.get("topic", "general"),
        }, config=config)

        return {
            "id": f"{original['id']}_reengage_{state.get('follow_up_count', 0) + 1}",
            "text": response.strip(),
            "question_type": "follow_up",
            "topic": original.get("topic", "general"),
            "difficulty": original.get("difficulty", "medium"),
            "parent_question_id": original.get("parent_question_id") or original["id"],
            "target_concepts": original.get("target_concepts", []),
            "estimated_time_minutes": 3,
        }

    async def _generate_follow_up(
            self,
            state: InterviewState,
            config: RunnableConfig
    ) -> dict:
        eval_data = state["current_evaluation"]
        original = state["current_question"]

        # Off-topic response: rephrase the question instead of probing missed points.
        # Probing missed points for an off-topic response would directly reveal the answer.
        covered = eval_data.get("key_points_covered", [])
        score = eval_data["overall_score"]
        if not bool(covered) or score < 3.0:
            return await self._generate_reengagement(state, config)

        missed = eval_data.get("key_points_missed", [])[:2]

        response = await self.follow_up_chain.ainvoke(
            {
                "original_question": original["text"],
                "candidate_response": state["candidate_response"],
                "missed_points": missed,
                "topic": original.get("topic", "general")
            }, config=config
        )
        return {
            "id": f"{original['id']}_followup_{state.get('follow_up_count', 0) + 1}",
            "text": response.strip(),
            "question_type": "follow_up",
            "topic": original.get("topic", "general"),
            "difficulty": original.get("difficulty", "medium"),
            "parent_question_id": original["id"],
            "target_concepts": missed,
            "estimated_time_minutes": 3
        }
    
    async def _generate_clarification(
            self,
            state: InterviewState, 
            config: RunnableConfig
    ) -> dict:
        eval_data = state["current_evaluation"]
        misconception = eval_data["misconceptions"][0]
        original = state["current_question"]

        response = await self.clarify_chain.ainvoke({
            "original_question": original["text"],
            "candidate_response": state["candidate_response"],
            "misconception": misconception,
        }, config=config)

        return {
            "id": f"{original['id']}_clarify",
            "text": response.strip(),
            "question_type": "clarification",
            "topic": original.get("topic", "general"),  # Always inject topic
            "difficulty": original.get("difficulty", "medium"),
            "parent_question_id": original["id"],
            "target_misconception": misconception,
            "target_concepts": [misconception],  # Dynamic rubric
            "estimated_time_minutes": 3
        }
    
    def _get_used_question_ids(self, state: InterviewState) -> list[str]:
        return [
            ev.get("question_id", "")
            for ev in state.get("all_evaluations", [])
            if ev.get("question_id")
        ]
    
    def _get_fallback_question(self) -> dict:
        fallback_id = str(uuid.uuid4())
        return {
            "id": "fallback_" + fallback_id,
            "text": "Can you explain the bias-variance tradeoff?",
            "question_type": "retrieved",
            "topic": "machine_learning_fundamentals",
            "difficulty": "medium",
            "estimated_time_minutes": 4,
            "target_concepts": ["bias", "variance", "tradeoff"]
        }