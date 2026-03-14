import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig

from src.utils.logging_config import get_logger
from src.agents.contracts import EvaluationOutput
from src.graph.state import InterviewState
from src.tools.code_validator import code_validator, _contains_code
from src.tools.rubric_tool import rubric_lookup
from src.services.validation import ValidationGateRegistry, CircuitBreaker

logger = get_logger(__name__)


EVAL_COT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict, impartial technical evaluator for AI/ML engineering interviews.
Your evaluations must be grounded, consistent, and defensible. You do not reward confidence — only correctness and understanding.

EVALUATION CONTEXT:
Question: {question}
Candidate Response: {response}

RUBRIC CONTEXT:
{rubric_context}

---

EVALUATION PROTOCOL — follow this sequence exactly:

STEP 1 — CRITERION EVALUATION
For each criterion below, write your reasoning FIRST, then assign a score.
Your score must be a direct consequence of your reasoning. Do not assign a score then justify it.

── TECHNICAL ACCURACY (weight: 40%)
Measures: Is the candidate's understanding fundamentally correct?
Check against rubric key_points and common_mistakes if provided.
  0-2 : Fundamental misunderstanding. Core concepts wrong or dangerously incomplete.
  3-5 : Partially correct. Gets the surface idea but missing or wrong on key concepts.
  6-8 : Largely correct. Minor gaps or imprecision but no fundamental errors.
  9-10: Fully correct. No errors. All critical concepts accurately represented.

── COMPLETENESS (weight: 30%)
Measures: Did the candidate cover what was expected?
Assess against the key_points in the rubric context.
  0-2 : Major aspects entirely missing. Covers less than 30% of expected content.
  3-5 : Partial coverage. Hits some points but misses most key concepts.
  6-8 : Good coverage. Most key points present. Minor omissions only.
  9-10: Comprehensive. All key points addressed. Nothing critical omitted.

── DEPTH (weight: 20%)
Measures: Did the candidate go beyond surface definitions? Do they understand WHY, not just WHAT?
  0-2 : Surface level only. Definitions without mechanism or intuition.
  3-5 : Basic understanding. Explains what it is but not how or why it works.
  6-8 : Solid depth. Explains the mechanism and can reason about tradeoffs.
  9-10: Expert depth. Connects concepts, discusses edge cases, demonstrates intuition.

── CLARITY (weight: 10%)
Measures: Is the explanation coherent and well-structured?
  0-2 : Incoherent or contradictory. Cannot follow the explanation.
  3-5 : Followable but vague. Key points buried or poorly expressed.
  6-8 : Clear and structured. Explanation is easy to follow.
  9-10: Exceptionally clear. Precise language, logical flow, no ambiguity.

STEP 2 — KNOWLEDGE GAP ANALYSIS
Using ONLY the key_points from the rubric context:
- key_points_covered: List key points the candidate explicitly and correctly addressed.
- key_points_missed: List key points that were absent, vague, or incorrect.

If no rubric key_points are available, derive from the question's expected answer scope.

STEP 3 — MISCONCEPTION DETECTION
A gap is something the candidate did not mention.
A misconception is something the candidate stated INCORRECTLY — a wrong belief, not just missing knowledge.
List only genuine misconceptions here. An incomplete answer is not a misconception.

STEP 4 — OVERALL SCORE
Compute a weighted assessment: technical_accuracy (40%) + completeness (30%) + depth (20%) + clarity (10%).
This is your holistic judgment — not arithmetic. If the candidate is technically correct but severely incomplete,
your overall score should reflect that the core understanding is present but the answer is insufficient.
Range: 0.0 to 10.0. One decimal place.

---

CRITICAL CONSTRAINTS:
- Never reward fluency over correctness. A clear wrong answer is worse than an unclear right one.
- Never penalize for communication style if the technical content is sound.
- If candidate_response is empty or off-topic, assign 0 across all criteria.
- Your evaluation_reasoning must be minimum 2 sentences and explain your overall_score specifically.
"""),
    ("human", """Evaluate the candidate's response now. Follow the protocol exactly.
Question: {question}
Response: {response}""")
])

REFLECTION_PROMPT = ChatPromptTemplate.from_messages([
(
"system",
"""
You are reviewing an AI evaluation of a candidate's interview answer.

Your job is NOT to re-evaluate the answer from scratch.

Instead, verify that the evaluation is logically consistent and fair.

Check for these issues:

1. Score inconsistency across dimensions.
2. Overall score inconsistent with sub-scores.
3. High score despite missing key concepts.
4. Missed misconceptions in the candidate answer.
5. Overly harsh or overly generous grading.

score_adjustment must be a plain integer with NO leading + sign.
Use 2 not +2, use -2 not -2.

If the evaluation is correct, return {{"adjustment_needed": false, "reason": "brief reason"}}.

If issues exist, suggest corrected values.

Be conservative — only adjust when clearly necessary.
"""
),
(
"human",
"""
Question:
{question}

Candidate Response:
{response}

Expected Key Points:
{rubric}

Evaluation Produced:
{evaluation}

Return JSON with:

{{
  "adjustment_needed": true | false,
  "reason": "short explanation",
  "score_adjustment": -2 to +2,
  "missed_misconceptions": [],
  "additional_key_points_missed": []
}}
"""
)
])

def build_eval_chain(complex_llm: BaseChatModel):
    """Build the evaluation chain with the given prompt and llm.
    """
    eval_chain = (
        EVAL_COT_PROMPT
        | complex_llm.with_structured_output(EvaluationOutput)
    ).with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=True
    )

    lenient_chain = (
        EVAL_COT_PROMPT
        | complex_llm
        | JsonOutputParser()
    )

    return eval_chain.with_fallbacks([lenient_chain])

class EvaluatorAgent:
    """
    Evaluates candidate responses with CoT + Reflection + optional Self-Consistency.
    
    Key features:
    - Injects topic from current_question into evaluation output
    - Self-consistency: configurable N-sample evaluation with median selection
    - Flags high-divergence evaluations for review
    """
    
    def __init__(
        self, 
        complex_llm: BaseChatModel, 
        fast_llm: BaseChatModel,
        consistency_samples: int = 1
    ):
        self.eval_chain = build_eval_chain(complex_llm)
        self.reflect_chain = REFLECTION_PROMPT | fast_llm | JsonOutputParser()
        self.consistency_samples = consistency_samples  # 1 = standard, 2 = self-consistency
        self.validation_gates = ValidationGateRegistry()
        self.complex_llm = complex_llm
        self.fast_llm = fast_llm
        self.circuit_breaker = CircuitBreaker(max_retries=1)

    async def _build_rubric_context(self, question: dict) -> str:
        """
        Build the rubric context string injected into EVAL_COT_PROMPT.

        Priority:
        1. rubric_lookup by question_id  (authoritative JSON file)
        2. question["target_concepts"]   (dynamic rubric — follow_up / clarify)
        3. question["target_misconception"] (clarify questions)
        4. Graceful "no rubric" fallback string
        """
        question_id = question.get("id", "")
        parts: list[str] = []

        if question_id:
            rubric = await rubric_lookup.ainvoke({"question_id": question_id})
            if rubric.get("found"):
                if rubric["key_points"]:
                    formatted = "\n".join(f" - {kp}" for kp in rubric["key_points"])
                    parts.append(f"Key Points:\n{formatted}")

                if rubric["common_mistakes"]:
                    formatted = "\n".join(f" - {cm}" for cm in rubric["common_mistakes"])
                    parts.append(f"Common Mistakes to watch for:\n{formatted}")
                return "\n\n".join(parts)
        
        # question_id not in rubric_lookup - fall to dynamic_rubric
        target_concepts = question.get("target_concepts", [])
        if target_concepts:
            formatted = "\n".join(f" - {c}" for c in target_concepts)
            parts.append(f"Expected concepts:\n{formatted}")

        target_misconception = question.get("target_misconception")
        if target_misconception:
            parts.append(f"Misconception to address: {target_misconception}")

        if not parts:
            parts.append(
                "No rubric available. Evaluate based on the question's expected scope."
            )

        return "\n\n".join(parts)
    
    def _response_contains_code(self, response: str) -> bool:
        """
        Gate check before calling code_validator tool.
        Delegates to the same _contains_code logic used by the tool itself
        — no duplication.
        """
        return _contains_code(response)
    
    def _apply_reflection(self, eval_dict: dict, reflection: dict) -> dict:
        """
        Apply reflection adjustments to the evaluation dict in-place.

        Reflection output shape (from REFLECTION_PROMPT → JsonOutputParser):
            {
                "adjustment_needed": bool,
                "reason": str,
                "score_adjustment": float,   # -2 to +2
                "missed_misconceptions": list[str],
                "additional_key_points_missed": list[str]
            }

        Conservative by design — only acts when adjustment_needed is True.
        Score is clamped to [0.0, 10.0] and rounded to 1 decimal.
        """
        if not isinstance(reflection, dict):
            logger.warning("Reflection output is not a dict - skipping adjustment")
            return eval_dict
        
        if not reflection.get("adjustment_needed", False):
            logger.debug(
                "Reflection: no adjustment needed. Reason: %s",
                reflection.get("reason", ""),
            )
            return eval_dict
        
        score_adjustment = reflection.get("score_adjustment", 0)
        if score_adjustment:
            current = float(eval_dict.get("overall_score", 5.0))
            adjusted = round(max(0.0, min(10.0, current + score_adjustment)), 1)
            logger.debug(
                "Relfection adjusted score: %.1f -> %.1f (reason: %s)",
                current,
                adjusted,
                reflection.get("reason", "")
            )
            eval_dict["overall_score"] = adjusted

        missed_misconceptions = reflection.get("missed_misconceptions", [])
        if missed_misconceptions:
            existing = eval_dict.get("misconceptions", [])
            eval_dict["misconceptions"] = existing + missed_misconceptions

        additional_missed = reflection.get("additional_key_points_missed", [])
        if additional_missed:
            existing = eval_dict.get("key_points_missed", [])
            eval_dict["key_points_missed"] = existing + additional_missed

        return eval_dict
    
    async def _single_evaluate(
            self,
            state: InterviewState,
            config: RunnableConfig
    ) -> dict:
        """
        Single CoT evaluation pass:
            eval_chain → model_dump() → topic injection
            → (optional) code_validator → reflection → _apply_reflection

        model_dump() is called here so that:
        - _apply_reflection always receives a plain dict
        - execute() always writes a plain dict to state (no Pydantic leak)

        Returns: {"current_evaluation": eval_dict}
        """
        question = state["current_question"]
        response = state.get("candidate_response") or ""

        rubric_context = await self._build_rubric_context(question)

        raw = await self.eval_chain.ainvoke(
            {
                "question": question["text"],
                "response": response,
                "rubric_context": rubric_context
            },
            config=config
        )

        # normalize to plain dict
        if isinstance(raw, EvaluationOutput):
            eval_dict = raw.model_dump()
        else:
            eval_dict = dict(raw)

        # topic, question_id injection
        eval_dict["topic"] = question.get("topic", "general")
        eval_dict["question_id"] = question.get("id", "")

        # code validation: if response have code block
        if self._response_contains_code(response):
            code_result = await code_validator.ainvoke({"response": response})
            if code_result.get("code_detected") and not code_result.get("is_valid",True):
                errors = code_result.get("errors", [])
                existing = eval_dict.get("misconceptions", [])
                eval_dict["misconceptions"] = existing + [
                    f"Code syntax error: {e}" for e in errors
                ]
        reflection = await self.reflect_chain.ainvoke(
            {
                "question": question["text"],
                "response": response,
                "rubric": rubric_context,
                "evaluation": eval_dict
            },
            config=config
        )

        eval_dict = self._apply_reflection(eval_dict, reflection)

        return {
            "current_evaluation": eval_dict
        }
    
    # only public method to call by graph node.
    async def execute(
            self, 
            state: InterviewState,
            config: RunnableConfig
    ) -> dict:
        """
        main method. called by the langgraph graph node.
        Asyncio.wait_for() timeout is applied at the graph level.
        """
        self.circuit_breaker.reset() # reset before triggering evaluator for fresh budget
        gate = self.validation_gates.get("evaluator")
        question = state["current_question"]

        try:
            if self.consistency_samples <= 1:
                result = await self._single_evaluate(state, config)
                eval_dict = result["current_evaluation"]
            else:
                results = await asyncio.gather(
                    *[
                        self._single_evaluate(state, config)
                        for _ in range(self.consistency_samples)
                    ],
                    return_exceptions=True
                )
                valid = [r for r in results if isinstance(r, dict)]

                if not valid:
                    logger.error("All self-consistency evaluation failed")
                    eval_dict = gate.get_fallback()
                    eval_dict["topic"] = question.get("topic", "unknown")
                    return {"current_evaluation": eval_dict}
                
                scores = [r["current_evaluation"]["overall_score"] for r in valid]
                sorted_pairs = sorted(zip(scores, valid), key=lambda x: x[0])
                median_idx = len(sorted_pairs) // 2
                eval_dict = sorted_pairs[median_idx][1]["current_evaluation"]

                divergence = max(scores) - min(scores)
                if divergence > 2.0:
                    eval_dict["needs_human_review"] = True
                    eval_dict["consistency_divergence"] = divergence
                    logger.warning(
                        "High self-consistency divergence: %.1f - flagged for review",
                        divergence
                    )

        except Exception as exc:
            logger.error(
                "Evaluation pipeline failed: %s", exc, exc_info=True
            )
            eval_dict = gate.get_fallback()
            eval_dict["topic"] = question.get("topic", "unknown")
            return {"current_evaluation": eval_dict}
        
        validation_result = gate.validate(eval_dict, question)

        if not validation_result.is_valid:
            logger.warning(
                "Evaluator gate failed: %s", validation_result.feedback
            )
            if self.circuit_breaker.should_retry("evaluator"):
                logger.info("Retrying evaluator - circuit breaker budget available.")
                return await self.execute(state, config)
            
            logger.error("Circuit breaker open - using fallback evaluation")
            eval_dict = gate.get_fallback()
            eval_dict["topic"] = question.get("topic", "unknown")

        return {"current_evaluation": eval_dict}


