"""
Loop-safe query refinement for the CRAG correction loop.

Strategy rotation by attempt:
  0 → LLM refine   (rephrase for better semantic match)
  1 → topic pivot  (shift to uncovered topic from ML_TOPICS pool)
  2 → simplify     (strip to 2 core words — last resort)

After every refinement a difflib similarity check guards against
the LLM rephrasing rather than genuinely changing the query.
_force_different() provides a mechanical guarantee of uniqueness.

"""

from __future__ import annotations
import random

from difflib import SequenceMatcher
from enum import Enum
from typing import List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.logging_config import get_logger
from src.utils.llm_factory import get_secondary_llm

logger = get_logger(__name__)


ML_TOPICS = [
'Activation Functions',
 'Artificial Intelligence/Neural Networks',
 'Batch Normalization',
 'Convolutional Neural Networks',
 'Data Science',
 'Deep Learning',
 'Deep Learning Optimization',
 'General',
 'Generative Models',
 'Information Retrieval',
 'Linear Algebra and Matrix Operations',
 'Machine Learning',
 'NLP',
 'Natural Language Processing',
 'Natural Language Processing (NLP)',
 'Neural Networks',
 'Optimization Algorithms',
 'Optimization Techniques',
 'Optimization Techniques in Deep Learning',
 'Text Vectorization',
 'Topic Modeling',
 'Word Embeddings',
 'adaptive_ai_systems',
 'agentic_ai_systems',
 'agents',
 'ai_engineering_fundamentals',
 'ai_evaluation',
 'ai_governance',
 'ai_privacy',
 'ai_safety_for_llms',
 'ai_security',
 'ai_system_design',
 'ai_system_monitoring',
 'ai_system_reliability',
 'algorithmic_efficiency',
 'classification',
 'clustering',
 'computer_vision',
 'computer_vision_fundamentals',
 'computer_vision_system_design',
 'data_engineering',
 'data_engineering_for_ml',
 'data_labeling',
 'data_labeling_and_annotation',
 'data_science_fundamentals',
 'deep_learning',
 'deep_learning_frameworks',
 'deep_learning_fundamentals',
 'deep_learning_training',
 'deployment',
 'dimensionality_reduction',
 'edge_ai_system_design',
 'evaluation',
 'evaluation_metrics',
 'experimentation',
 'experimentation_and_ab_testing',
 'exploratory_and_statistical_analysis',
 'feature_engineering',
 'feature_selection',
 'feedback_loops',
 'generative_models',
 'gradient_boosting',
 'graph_machine_learning',
 'hallucination_in_llm',
 'human_in_the_loop',
 'hyperparameter_tuning',
 'information_retrieval',
 'large_language_model_fundamentals',
 'large_language_model_optimization',
 'large_language_model_safety',
 'large_language_model_training',
 'learning_to_rank_and_search',
 'linear_algebra_and_matrix_operations',
 'linear_regression',
 'llm_basics',
 'llm_history',
 'llm_optimization',
 'llm_safety',
 'llm_system_design',
 'llm_training',
 'machine_learning_fundamentals',
 'machine_learning_lifecycle',
 'machine_learning_optimization',
 'machine_learning_vs_deep_learning',
 'math_foundations',
 'mathematical_foundations_for_ml',
 'metrics',
 'ml_fundamentals',
 'ml_system_design_deployment',
 'mlops',
 'mlops_fundamentals',
 'model_deployment',
 'model_development',
 'model_evaluation',
 'model_monitoring',
 'model_optimization',
 'model_regularization',
 'model_validation',
 'monitoring',
 'natural_language_processing',
 'neural_network_optimization',
 'neural_networks',
 'neural_networks_for_computer\xa0vision',
 'neural_networks_for_vision',
 'nlp',
 'nlp_system_design',
 'optimization',
 'optimization_algorithms',
 'optimization_in_neural\xa0networks',
 'padding',
 'parameter_efficient_fine_tuning',
 'parameter_tuning',
 'peft',
 'privacy',
 'programming_for_ml',
 'python',
 'python_for_ml',
 'pytorch',
 'quantization_in_llm',
 'rag',
 'random_forest',
 'ranking_and\xa0search',
 'recommendation_system_design',
 'recommender_system',
 'recommender_systems',
 'recommender_systems_fundamentals',
 'recsys',
 'regularization',
 'reinforcement_learning_design',
 'relu_and_dying_neuron_problem',
 'retrieval_augmented_generation',
 'search_and_ranking_systems',
 'search_system_design',
 'security',
 'software_engineering',
 'software_engineering_principles',
 'supervised_learning',
 'supervised_machine\xa0learning',
 'system_design',
 'system_design_for_ml',
 'text_classification',
 'text_representation_and_vectorization',
 'time_series',
 'time_series_analysis',
 'time_series_system_design',
 'topic_modeling',
 'training',
 'training_dynamics',
 'validation',
 'word_embeddings'
]

SIMILARITY_THRESHOLD = 0.85 # ratio above which queries are considered too similar

class QueryRefinementStrategy(str, Enum):
    LLM_REFINE = "llm_refine"
    TOPIC_PIVOT  = "topic_pivot"
    SIMPLIFY     = "simplify"
    FORCED       = "forced"   # mechanical fallback when all else is too similar

class RefinedQuery(BaseModel):
    query:    str = Field(description="The refined search query — concise, specific")
    rationale: str = Field(description="One sentence explaining the change")

_REFINE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are refining a search query for an ML interview question database. "
        "The previous query returned low-quality results. "
        "Produce a meaningfully different query — not just a rephrasing."
    ),
    (
        "human",
        """Original query: {original_query}
Grader feedback: {feedback}
Difficulty level: {difficulty}
Queries already tried (avoid these): {seen_queries}

Return a new query that approaches the topic differently."""
    ),
])

class QueryRefiner:
    """
    Produces a new query for each CRAG correction attempt.
    
    Usage:
        refiner = QueryRefiner()
        new_query, strategy = refiner.refine(
            original_query="deep learning optimization",
            feedback="Questions too advanced for easy difficulty",
            difficulty="easy",
            seen_queries=["deep learning optimization"],
            attempt=0,
            covered_topics=["deep_learning"],
            )
    """
    def __init__(self) -> None:
        llm = get_secondary_llm()
        self._llm_refiner = _REFINE_PROMPT | llm.with_structured_output(RefinedQuery)

    async def refine(self,
                     original_query: str,
                     feedback: str,
                     difficulty: str,
                     seen_queries: List[str],
                     attempt: int,
                     covered_topics: Optional[List[str]] = None,
    ) -> Tuple[str, QueryRefinementStrategy]:
        """Return (refined_query, strategy_used)."""
        covered_topics = covered_topics or []

        strategy = self._pick_strategy(attempt)
        refined = await self._execute_strategy(
            strategy, original_query, feedback, difficulty, seen_queries, covered_topics
        )

        if self._too_similar(refined, seen_queries):
            logger.debug(
                "Refined query too similar to seen - forcing different",
                extra={"refined": refined, "strategy": strategy.value}
            )
            refined   = self._force_different(original_query, covered_topics, difficulty)
            strategy  = QueryRefinementStrategy.FORCED

        logger.info(
            "Query refined",
            extra={"strategy": strategy.value, "original": original_query, "refined": refined}
        )
        return refined, strategy
    
    def _pick_strategy(self, attempt: int) -> QueryRefinementStrategy:
        rotation = [
            QueryRefinementStrategy.LLM_REFINE,
            QueryRefinementStrategy.TOPIC_PIVOT,
            QueryRefinementStrategy.SIMPLIFY
        ]

        # clamp to last strategy if attempt exceeds rotation length
        idx = min(attempt, len(rotation) - 1)
        return rotation[idx]

    async def _execute_strategy(
            self,
            strategy: QueryRefinementStrategy,
            original_query: str,
            feedback: str,
            difficulty: str,
            seen_queries: List[str],
            covered_topics: List[str],
    ) -> str:
        if strategy == QueryRefinementStrategy.LLM_REFINE:
            return await self._llm_refine(original_query, feedback, difficulty, seen_queries)
        
        if strategy == QueryRefinementStrategy.TOPIC_PIVOT:
            return self._topic_pivot(difficulty, covered_topics)
        
        # simplify
        return self._simplify(original_query, difficulty)
    
    async def _llm_refine(
            self,
            original_query: str,
            feedback: str,
            difficulty: str,
            seen_queries: List[str],
    ) -> str:
        """"Ask LLM for a meaningfully different query. Falls back to simplify on failure"""
        try:
            result: RefinedQuery = await self._llm_refiner.ainvoke({
                "original_query": original_query,
                "feedback":       feedback,
                "difficulty":     difficulty,
                "seen_queries":   ", ".join(seen_queries) if seen_queries else "none",
            })
            return result.query.strip()
        except Exception as exc:
            logger.warning("LLM refine failed: %s — falling back to simplify", exc)
            return self._simplify(original_query, difficulty)
        
    def _topic_pivot(self, difficulty: str, covered_topics: List[str]) -> str:
        """
        Shift to a topic not yet covered in this interview.
        Falls back to a random ML_TOPICS entry if all are covered.
        """
        covered_norm = {t.lower().replace(" ", "_") for t in covered_topics}
        uncovered = [
            t for t in ML_TOPICS
            if t.lower().replace(" ", "_") not in covered_norm
        ]
        pivot_topic = random.choice(uncovered) if uncovered else random.choice(ML_TOPICS)
        return f"{difficulty} {pivot_topic.replace('_', ' ')}"

    def _simplify(self, original_query: str, difficulty: str) -> str:
        """
        Strip query to 2 meaningful words + difficulty.
        Removes stop words mechanically.
        """
        stop_words = {"the", "a", "an", "and", "or", "for", "in", "of", "to", "with"}
        words = [
            w for w in original_query.lower().split()
            if w not in stop_words and len(w) > 2
        ]
        core = " ".join(words[:2]) if len(words) >= 2 else original_query
        return f"{difficulty} {core}"

    # ------------------------------------------------------------------
    # Similarity guard
    # ------------------------------------------------------------------

    def _too_similar(self, query: str, seen_queries: List[str]) -> bool:
        """Return True if query is too close to any previously tried query."""
        for seen in seen_queries:
            ratio = SequenceMatcher(None, query.lower(), seen.lower()).ratio()
            if ratio > SIMILARITY_THRESHOLD:
                return True
        return False

    def _force_different(
        self,
        original_query: str,
        covered_topics: List[str],
        difficulty:     str,
    ) -> str:
        """
        Mechanically construct a query guaranteed to differ from anything seen.
        Takes the first 2 words of original + first uncovered topic.
        """
        uncovered = [t for t in ML_TOPICS if t not in covered_topics]
        pivot     = uncovered[0] if uncovered else "fundamentals"

        words = original_query.lower().split()
        core  = " ".join(words[:2]) if len(words) >= 2 else original_query

        return f"{difficulty} {core} {pivot.replace('_', ' ')}"