from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from src.agents.contracts import FeedbackOutput
from src.utils.logging_config import get_logger
from src.services.validation import ValidationGateRegistry, CircuitBreaker, ValidationResult
from src.rag.cache import InterviewCacheStore
from src.graph.state import InterviewState
from src.tools.concept_lookup import concept_lookup

logger = get_logger(__name__)

class FeedbackComponents(BaseModel):
    """
    LLM-generated semantic parts. Internal to FeedbackAgent.
    Not an interagent contract.
    """
    strength_acknowledgment: str = Field(
        description="One sentence explaining what they did well. Empty when score < 5"
    )
    gap_hint: str = Field(
        description="Implicit hints about gaps without stating 'you missed X'"
    )
    transition_phrase: str = Field(
        description="Natural transition; e.g building on that or can be empty"
    )

OFF_TOPIC_FEEDBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior technical interviewer giving brief, conversational feedback.\n"
     "The candidate's response went in a different direction from what was asked.\n\n"
     "Write 1-3 sentences that:\n"
     "1. Briefly acknowledge what the candidate actually talked about (be specific to their words)\n"
     "2. Note warmly but directly that it didn't address the question\n\n"
     "Tone: warm, direct, human — like a senior colleague, not a grading system.\n\n"
     "HARD RULES:\n"
     "- Do NOT mention, name, or hint at what the question was asking about\n"
     "- Do NOT reveal any concept, term, or domain from the correct answer\n"
     "- Do NOT ask a question — end with a statement\n"
     "- Do NOT be robotic. Reference what the candidate actually said.\n"
     "- Between 20 and 50 words"),
    ("human",
     "Candidate's response: {candidate_response}")
])


class FeedbackComposer:
    """
    Composes FeedbackComponents into a varied feedback string.
    Pure Python — no LLM involved.
    
    Variation is controlled by:
    - score_band: determines which component slots are used
    - turn_number: drives modulo rotation through templates
    - previous_structures: exclusion window of last 2 (prevents structural repeat)
    
    Transitions rotate independently on a 3-turn cycle.
    """
    STRUCTURES = {
        "high": [
            "{strength}",
            "{strength} {transition}",
            "{transition}",                    # Skip strength sometimes
            "{strength}",                      # Added: avoids 2-template repetition
        ],
        "medium": [
            "{strength} {gap_hint}",
            "{gap_hint} {strength}",
            "{strength} {transition}",
            "{gap_hint} {transition}",
        ],
        "low": [
            "{gap_hint}",
            "{gap_hint} {transition}",
            "{transition} {gap_hint}",         # Added for variety
        ]
    }
    TRANSITIONS = [
        "Building on that...",
        "I'm curious...",
        "Let me ask you this...",
        "Thinking about that...",
        "Related to what you mentioned...",
        "On that note...",
        "Following up...",
    ]

    # Score band boundaries — deliberate asymmetry (see architecture)
    _HIGH_THRESHOLD = 8.0
    _MEDIUM_THRESHOLD = 5.0

    def _get_score_band(self, score: float) -> str:
        if score >= self._HIGH_THRESHOLD:
            return "high"
        elif score >= self._MEDIUM_THRESHOLD:
            return "medium"
        return "low"

    def _select_structure(
        self,
        structures: list[str],
        turn_number: int,
        previous_structures: list[str]
    ) -> str:
        available = [s for s in structures if s not in previous_structures[-2:]]
        if not available:
            available = structures
        return available[turn_number % len(available)]

    def _select_transition(self, turn_number: int) -> str:
        if turn_number % 3 == 0:
            return ""
        return self.TRANSITIONS[turn_number % len(self.TRANSITIONS)]

    def compose(
        self,
        components: FeedbackComponents,
        score: float,
        turn_number: int,
        previous_structures: list[str]
    ) -> tuple[str, str]:
        """
        Entry point. Returns a single clean string.
        
        Args:
            components:           LLM-generated semantic parts
            score:                raw evaluation score (drives band selection)
            turn_number:          current question_count (drives rotation)
            previous_structures:  last N used templates (drives exclusion)
        """
        score_band = self._get_score_band(score)
        structures = self.STRUCTURES[score_band]
        structure = self._select_structure(structures, turn_number, previous_structures)
        transition = self._select_transition(turn_number)
        
        result = structure.format(
            strength=components.strength_acknowledgment,
            gap_hint=components.gap_hint,
            transition=transition
        )
        return " ".join(result.split()), structure
    
FEEDBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an interview feedback generator. Generate constructive feedback components.\n"
     "NEVER reveal scores. NEVER say 'you missed X'. NEVER use forbidden phrases.\n"
     "NEVER ask questions of any kind — no clarifying questions, no rhetorical questions, "
     "no 'can you explain...' constructions. Questions are the exclusive responsibility "
     "of the question selector. Feedback is observational and directional only.\n\n"
     "CRITICAL: Do NOT give away the direct answer, complete definitions, or full solutions "
     "to the question. Write as a thoughtful senior colleague giving honest, warm feedback — "
     "not as a grading system producing an evaluation. "
     "Your tone should feel conversational and human, like feedback after a "
     "real interview exchange, not a performance report.\n\n"
     "Tone guidance: {tone_guidance}\n\n"
     "If concept_context is provided below, use it ONLY as an internal check that your feedback "
     "is pointing in the right direction. "
     "Do NOT paraphrase, quote, echo, or derive any term, mechanism, process, or example from it. "
     "The gap_hint text must come entirely from how the question itself is framed — "
     "not from concept_context content. Treat concept_context as a compass, not as material to write from. "
     "If concept_context is empty, generate feedback based solely on the question and response. "
     "Do not reference or acknowledge the absence of concept context.\n"
     "Concept context: {concept_context}\n\n"
     "Generate feedback between 15-200 words. "
     "Never generate fewer than 15 words regardless of score band.\n"
     "Feedback must address the QUESTION that was asked, not just what the candidate said. "
     "Never introduce concepts from the candidate's response if they are irrelevant to the question.\n\n"
     "ABSOLUTE RULE: Do NOT ask questions of any kind. No clarifying questions, "
     "no rhetorical questions, no 'can you explain...', no 'can you think about...' "
     "constructions. End every feedback with a statement, never a question mark."),
    ("human",
     "Score band: {score_band}\n"
     "Question asked: {question}\n"
     "Candidate response: {response}")
])

REPETITION_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Check if new feedback is semantically similar to recent feedback.\n\n"
     "'Similar' means the feedback conveys the same core message or uses the same framing "
     "even if worded differently.\n"
     "'Different' means it takes a meaningfully different angle.\n\n"
     "Respond with exactly two lines:\n"
     "Line 1: similar OR different\n"
     "Line 2: One sentence explanation."),
    ("human",
     "New feedback:\n{new_feedback}\n\n"
     "Recent feedback (last 2 turns):\n{recent_feedbacks}")
])

def build_feedback_chain(fast_llm: BaseChatModel):
    """
    strict chain: structured_output + retry
    """
    feedback_chain = ( FEEDBACK_PROMPT 
                    | fast_llm.with_structured_output(FeedbackComponents)
                    ).with_retry(
                        stop_after_attempt=2,
                        wait_exponential_jitter=True
                    )
    return feedback_chain

class FeedbackAgent:
    def __init__(
            self,
            fast_llm: BaseChatModel,
            cache_store: InterviewCacheStore,
        ):
        self.fast_llm = fast_llm
        self.cache_store = cache_store
        self.feedback_chain = build_feedback_chain(fast_llm)
        self.repetition_chain = REPETITION_CHECK_PROMPT | fast_llm
        self.off_topic_chain = (
            OFF_TOPIC_FEEDBACK_PROMPT | fast_llm | StrOutputParser()
        )
        self.composer = FeedbackComposer()
        self.validation_gate = ValidationGateRegistry().get("feedback")
        self.circuit_breaker = CircuitBreaker(max_retries=1)

    def _get_tone_guidance(self, score: float) -> str:
        if score >= 8.0:
            return (
                "Genuine and brief. Acknowledge what landed well without overdoing it. "
                "One natural observation — no elaboration needed."
            )
        elif score >= 6.0:
            return (
                "Encouraging but honest. Acknowledge the solid foundation, "
                "then steer toward the depth that's missing. "
                "Sound like someone who wants them to succeed."
            )
        elif score >= 4.0:
            return (
                "Supportive and patient. Acknowledge the attempt genuinely. "
                "Guide them toward the right framing without revealing it. "
                "Avoid any tone that feels like a correction."
            )
        return (
            "Calm and grounding. No praise openers. "
            "Gently redirect toward the actual question's territory. "
            "Sound like a patient mentor, not a disappointed evaluator."
        ) 
    # async def _get_concept_context(
    #         self,
    #         state: InterviewState,
    #         eval_data: dict,
    #         score: float
    # ) -> str:
    #     # high score; no enrichment needed
    #     if score >= 7.0:
    #         return ""
        
    #     missed = eval_data.get("key_points_missed", [])
    #     if not missed:
    #         return ""
        
    #     # take only the first missed concept
    #     concept = missed[0]

    #     # check the cache first
    #     cached = await self.cache_store.get_concept(state["interview_id"], concept)
    #     if cached:
    #         return cached.get("simple_explanation", "")
        
    #     # cache miss - call the tool
    #     result = await concept_lookup.ainvoke({"concept_name": concept})

    #     if result and result.get("found"):
    #         await self.cache_store.set_concept(state["interview_id"], concept, result)
    #         return result.get("simple_explanation", "")
        
    #     return ""
    async def _get_concept_context(
        self,
        state: InterviewState,
        eval_data: dict,
        score: float
        ) -> str:
        # High score — no enrichment needed
        if score >= 7.0:
            return ""

        missed = eval_data.get("key_points_missed", [])
        if not missed:
            return ""

        # Candidate is completely off-topic — concept context won't help redirect.
        # Let feedback handle redirection on its own.
        covered = eval_data.get("key_points_covered", [])
        if score < 3.0 and not covered:
            return ""

        # Take only the first missed concept
        concept = missed[0]

        # Check cache first
        cached = await self.cache_store.get_concept(state["interview_id"], concept)
        result = cached
        if not cached:
            raw = await concept_lookup.ainvoke({"concept_name": concept})
            if raw and raw.get("found"):
                await self.cache_store.set_concept(state["interview_id"], concept, raw)
                result = raw

        if not result:
            return ""

        # Select context based on question mode:
        # clarify → common misconception (directly relevant to wrong belief)
        # follow_up / retrieve → real-world example (grounds hint without naming mechanism)
        mode = state.get("question_mode", "retrieve")

        if mode == "clarify":
            misconceptions = result.get("common_misconceptions", [])
            return f"Relevant misconception in this area: {misconceptions[0]}" if misconceptions else ""

        examples = result.get("examples", [])
        return f"Real-world application: {examples[0]}" if examples else ""

    async def _check_semantic_repetition(
            self,
            feedback_text: str,
            components: FeedbackComponents,
            used_template: str,
            recent_feedbacks: list[str],
            state: InterviewState,
            config: RunnableConfig
    ) -> tuple[str, FeedbackComponents, str]:
        recent_str = "\n".join(f"- {f}" for f in recent_feedbacks)

        check = await self.repetition_chain.ainvoke({
            "new_feedback": feedback_text,
            "recent_feedbacks": recent_str,
        }, config=config)

        if "similar" not in check.content.lower():
            return feedback_text, components, used_template # different feedback - return as it is
        
        # similar feedback - regenerate
        score = state["current_evaluation"]["overall_score"]
        diversity_tone = (
            self._get_tone_guidance(score)
            + " IMPORTANT: Use a completely different angle from recent feedback."
        )

        new_components = await self.feedback_chain.ainvoke({
            "question": state["current_question"]["text"],
            "response": state["candidate_response"],
            "score_band": self.composer._get_score_band(score),
            "tone_guidance": diversity_tone,
            "concept_context": ""
        }, config=config)

        new_feedback_text, new_template = self.composer.compose(
            new_components,
            score,
            state["question_count"],
            state.get("previous_feedback_structures", [])

        )
        return new_feedback_text, new_components, new_template

    async def execute(
            self,
            state: InterviewState,
            config: RunnableConfig
    ) -> dict:
        eval_data = state["current_evaluation"]
        score = eval_data["overall_score"]
        turn = state["question_count"]

        # Detect whether the candidate attempted the actual question topic.
        # covered=[] can happen when the evaluator has no rubric and lists no key points,
        # even for a good on-topic response. Trust the score: score >= 5.0 means
        # the evaluator considered the response on-topic.
        covered = eval_data.get("key_points_covered", [])
        is_off_topic = not covered and score < 5.0

        if is_off_topic:
            # Use a separate LLM prompt that receives ONLY the candidate's response,
            # never the question text. This makes feedback feel human and specific
            # to what the candidate said, while making it impossible to leak the
            # question's answer direction (the LLM can't hint at what it doesn't know).
            try:
                off_topic_text = await self.off_topic_chain.ainvoke(
                    {"candidate_response": state["candidate_response"]},
                    config=config
                )
                off_topic_text = off_topic_text.strip()
            except Exception:
                off_topic_text = ""

            # Fallback if LLM output is too short to pass validation
            if len(off_topic_text.split()) < 15:
                off_topic_text = (
                    "It seems like your response touched on a different area than "
                    "what the question was asking about. Let's refocus on the question."
                )
            return {
                "current_feedback": off_topic_text,
                "previous_feedback_structures": ["off_topic_redirect"],
                "recent_feedbacks": [off_topic_text],
            }

        tone_guidance = self._get_tone_guidance(score)
        concept_context = await self._get_concept_context(state, eval_data, score)

        # LLM generates the components (pydantic)
        components: FeedbackComponents = await self.feedback_chain.ainvoke({
            "question": state["current_question"]["text"],
            "response": state["candidate_response"],
            "score_band": self.composer._get_score_band(score),
            "tone_guidance": tone_guidance,
            "concept_context": concept_context,
        }, config=config)

        # compose into  string
        feedback_text, used_template = self.composer.compose(components, score, turn, state["previous_feedback_structures"])

        # semantic repition check
        recent_feedbacks = state.get("recent_feedbacks", [])
        if len(recent_feedbacks) >= 2:
            feedback_text, components, used_template = await self._check_semantic_repetition(
                feedback_text,
                components,
                used_template,
                recent_feedbacks[-2:],
                state, 
                config)

        # Validation gate
        feedback_output = FeedbackOutput(
            feedback_text=feedback_text,
            strength_acknowledgment=components.strength_acknowledgment,
            gap_hint=components.gap_hint,
            transition_phrase=components.transition_phrase,
            structure_template=used_template
        )

        validation_result: ValidationResult = self.validation_gate.validate(feedback_output.model_dump(), score)

        if not validation_result.is_valid:
            logger.warning(
                "Feedback gate failed: %s", validation_result.feedback
            )
            if self.circuit_breaker.should_retry("feedback"):
                logger.info("Retrying Feedback - circuit breaker budget available.")
                return await self.execute(state, config)
            
            logger.error("Circuit breaker open - using fallback feedback")
            fallback = self.validation_gate.get_fallback()
            fallback_text = fallback["feedback_text"]
            return {
                "current_feedback": fallback_text,
                "previous_feedback_structures": ["fallback"],
                "recent_feedbacks": [fallback_text]
                }

        return {
            "current_feedback": feedback_text,
            "previous_feedback_structures": [used_template],
            "recent_feedbacks": [feedback_text]
        }
