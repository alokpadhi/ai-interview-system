"""
Hybrid relevance grader for CRAG correction loop.

Grading strategy:
  avg_score >= 0.75  →  HIGH   (no LLM, fast path)
  avg_score <= 0.45  →  LOW    (no LLM, fast path)
  0.45 < score < 0.75 → LLM with_structured_output (borderline only)
  LLM failure         → MEDIUM fallback via .with_fallbacks()

Design:
  - Provider-agnostic: BaseChatModel abstraction works across OpenAI/Anthropic/Ollama
  - RunnableLambda wraps pre/post processing into the LCEL chain
  - .with_retry() + .with_fallbacks() for robust LLM config (LangChain >= 1.0)
  - Context penalties applied to score copies — originals never mutated
  - topic_intent used for penalty (not last_topic) — matches retrieval intent
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

from src.rag.cache import RelevanceGrade
from src.rag.models import RetrievalContext, RetrievalResult
from src.utils.llm_factory import get_secondary_llm
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# Thresholds

HIGH_SCORE_THRESHOLD        = 0.75
LOW_SCORE_THRESHOLD         = 0.45
DIFFICULTY_MISMATCH_PENALTY = 0.10
TOPIC_MISMATCH_PENALTY      = 0.15


# Structured output schema
class DocumentGrade(BaseModel):
    """LLM grading decision for borderline retrieval results."""
    grade:    str = Field(description="Relevance grade: HIGH, MEDIUM, or LOW")
    feedback: str = Field(description="One sentence explaining the grade")

    def to_relevance_grade(self) -> RelevanceGrade:
        """Safe coercion — defaults to MEDIUM on unrecognised value."""
        try:
            return RelevanceGrade(self.grade.upper())
        except ValueError:
            logger.warning(
                "Unrecognised grade from LLM: %s — defaulting to MEDIUM", self.grade
            )
            return RelevanceGrade.MEDIUM


# Fallback schema — returned when LLM fails entirely
class _FallbackGrade(BaseModel):
    grade:    str = "MEDIUM"
    feedback: str = "LLM grading unavailable — conservative MEDIUM assigned."


# GradingResult
@dataclass
class GradingResult:
    grade:           RelevanceGrade
    feedback:        str
    avg_score:       float
    used_llm:        bool           = False
    penalised_score: Optional[float] = None


_GRADING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are evaluating whether retrieved interview questions are relevant "
        "to the current retrieval context. Be strict — only grade HIGH if the "
        "questions are clearly appropriate for the difficulty and topic."
    ),
    (
        "human",
        """Retrieval context:
- Difficulty: {difficulty}
- Topic intent: {topic_intent}
- Interview stage: {stage}

Retrieved documents (sample):
{doc_sample}

Average relevance score from vector search: {avg_score:.3f}

Grade the overall retrieval quality as HIGH, MEDIUM, or LOW.
HIGH   = questions are on-topic and at the right difficulty
MEDIUM = mostly relevant but some mismatch
LOW    = off-topic or wrong difficulty throughout"""
    ),
])

# chain builder: with retry to fallback
def _build_grading_chain(llm: BaseChatModel):
    """
    LCEL chain with:
      - RunnableLambda for input formatting (participates in retry/fallback)
      - .with_structured_output() for provider-agnostic Pydantic output
      - .with_retry() for transient failures (up to 2 retries)
      - .with_fallbacks() for complete LLM failure → returns _FallbackGrade

    Provider agnostic: BaseChatModel.with_structured_output() works across
    OpenAI (function calling), Anthropic (tool use), Ollama (JSON mode).
    Reliability depends on underlying model capability.
    """
    structured_llm = llm.with_structured_output(DocumentGrade)

    # fallback chain
    fallback_chain = RunnableLambda(
        lambda _: _FallbackGrade()
    )

    primary_chain = (
        RunnablePassthrough() # pass dict as it is
        | _GRADING_PROMPT
        | structured_llm.with_retry(
            stop_after_attempt=2,
            wait_exponential_jitter=False
        )
    )

    return primary_chain.with_fallbacks([fallback_chain])

class DocumentGrader:
    """
    Hybrid grader: score-based fast paths, LLM only for borderline cases.

    Usage:
        grader = DocumentGrader() # uses get_secondary_llm()
        for testing can use llm=custom_llm
    """
    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        resolved_llm    = llm or get_secondary_llm()
        self._chain     = _build_grading_chain(resolved_llm)

    async def grade(
            self,
            documents: List[RetrievalResult],
            context: RetrievalContext,
            topic_intent: str = ""
    ) -> GradingResult:
        """Grade a list of retrieved documents against the retrieval context.
        Returns GradingResult - never raises
        """

        if not documents:
            return GradingResult(
                grade=RelevanceGrade.LOW,
                feedback="No documents retrieved.",
                avg_score=0.0
            )
        
        # penalities applied to copies - originals kept as it is
        penalized_scores = self._apply_penalties(documents, context, topic_intent)
        avg_penalized = sum(penalized_scores) / len(penalized_scores)
        avg_raw = sum(d.relevance_score for d in documents) / len(documents)

        # Fast path: HIGH
        if avg_penalized >= HIGH_SCORE_THRESHOLD:
            logger.debug("Grader fast-path HIGH | avg_penalized=%.3f", avg_penalized)
            return GradingResult(
                grade=RelevanceGrade.HIGH,
                feedback=f"Strong relevance - avg penalized score {avg_penalized:.3f}",
                avg_score=avg_raw,
                used_llm=False,
                penalised_score=avg_penalized
            )
        
        # fast path: LOW
        if avg_penalized <= LOW_SCORE_THRESHOLD:
            logger.debug("Grader fast-path LOW | avg penalized=%.3f", avg_penalized)
            return GradingResult(
                grade=RelevanceGrade.LOW,
                feedback=f"Poor relevance - avg penalized score {avg_penalized:.3f}",
                avg_score=avg_raw,
                used_llm=False,
                penalised_score=avg_penalized
            )
        
        # Border line: use LLM
        logger.debug("Grader borderline - calling LLM | avg penalized=%.3f", avg_penalized)
        return await self._llm_grade(documents, context, avg_raw, avg_penalized, topic_intent)
    
    def _apply_penalities(
            self,
            documents: List[RetrievalResult],
            context: RetrievalContext,
            topic_intent: str = ""
    ) -> List[float]:
        """Return adjusted scores as new floats - originals not mutated
        Penalty logic uses topic_intent (retrieval intent) not last_topic
        (conversation topic) — these are distinct concepts:
          last_topic    = what was discussed in the last turn
          topic_intent  = what we're trying to retrieve for if any otherwise skip
        """
        adjusted = []
        for doc in documents:
            score = doc.relevance_score

            # difficulty mismatch
            doc_difficulty = doc.difficulty or ""
            if doc_difficulty and doc_difficulty != context.difficulty_level:
                score -= DIFFICULTY_MISMATCH_PENALTY

            # topic mismatch
            doc_topic = doc.topic or ""

            if doc_topic and topic_intent and doc_topic != topic_intent:
                score -= TOPIC_MISMATCH_PENALTY

            adjusted.append(max(0.0, score))

    async def _llm_grade(
            self,
            documents: List[RetrievalResult],
            context: RetrievalContext,
            avg_raw: float,
            avg_penalized: float,
            topic_intent: str = ""
    ) -> GradingResult:
        """
        Invoke LLM grading chain
        """
        doc_sample = "\n".join(
            f"- [{d.metadata.get('difficulty','?')} | {d.metadata.get('topic','?')}] {d.text[:120]}"
            for d in documents[:3]
        )

        result: DocumentGrade | _FallbackGrade = await self._chain.ainvoke(
            {
            "difficulty":   context.difficulty_level,
            "topic_intent": topic_intent,
            "stage":        getattr(context, "stage", "questioning"),
            "doc_sample":   doc_sample,
            "avg_score":    avg_penalized,
            }
        )

        # checking if LLM used DocumentGrade or _FallbackGrade
        is_fallback = isinstance(result, _FallbackGrade)

        if is_fallback:
            logger.warning("Grader LLM failed - fallback MEDIUM assigned")
            return GradingResult(
                grade=RelevanceGrade.MEDIUM,
                feedback=result.feedback,
                avg_score=avg_raw,
                used_llm=False,
                penalised_score=avg_penalized
            )
        
        grade = result.to_relevance_grade()
        logger.info("Grader LLM decision | grade=%s feedback=%s", grade.value, result.feedback)
        return GradingResult(
            grade=grade,
            feedback=result.feedback,
            avg_score=avg_raw,
            used_llm=True,
            penalised_score=avg_penalized,
        )



