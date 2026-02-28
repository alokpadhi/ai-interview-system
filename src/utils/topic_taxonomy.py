"""
Canonical topic taxonomy for the AI Interview System.

This module is the single source of truth for valid topic names used
across the entire pipeline:

  - ingest_data_to_chromadb.py  → normalizes raw topics at ingest time
  - scripts/normalize_topics.py → re-normalizes existing JSON source files
  - query_refiner.py            → uses CANONICAL_TOPICS as the ML_TOPICS pool

All canonical topics follow snake_case convention.
"""

from __future__ import annotations

import unicodedata
from difflib import get_close_matches
from typing import Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Canonical taxonomy  (~40 topics)
# ---------------------------------------------------------------------------

CANONICAL_TOPICS: frozenset[str] = frozenset({
    # Foundations
    "machine_learning_fundamentals",
    "deep_learning_fundamentals",
    "math_foundations",
    "statistics_and_probability",
    "linear_algebra",
    "programming_for_ml",

    # Core ML methods
    "supervised_learning",
    "classification",
    "linear_regression",
    "gradient_boosting",
    "random_forest",
    "clustering",
    "dimensionality_reduction",
    "feature_engineering",
    "feature_selection",
    "model_regularization",
    "model_evaluation",
    "hyperparameter_tuning",

    # Deep learning
    "neural_networks",
    "convolutional_neural_networks",
    "activation_functions",
    "batch_normalization",
    "optimization_algorithms",
    "training_dynamics",

    # NLP & LLMs
    "natural_language_processing",
    "word_embeddings",
    "text_classification",
    "text_representation",
    "topic_modeling",
    "large_language_models",
    "llm_optimization",
    "llm_safety",
    "hallucination_in_llm",
    "parameter_efficient_fine_tuning",
    "quantization",
    "retrieval_augmented_generation",

    # Computer vision
    "computer_vision",

    # Systems & recommenders
    "recommender_systems",
    "information_retrieval",
    "search_and_ranking",
    "time_series",

    # MLOps & engineering
    "mlops",
    "model_deployment",
    "model_monitoring",
    "data_engineering",
    "data_labeling",
    "experimentation",
    "software_engineering",

    # AI safety & governance
    "ai_safety_and_governance",
    "ai_security",
    "privacy",

    # System design
    "ml_system_design",
    "agentic_ai",

    # Catch-all
    "general",
    "coding",
})


# ---------------------------------------------------------------------------
# Explicit raw-to-canonical mapping
# Generated from the full topic audit across all 6 source JSON files.
# ---------------------------------------------------------------------------

_RAW_TO_CANONICAL: dict[str, str] = {
    # ---- Miscellaneous / catch-all ----------------------------------------
    "general": "general",
    "data science": "machine_learning_fundamentals",
    "data_science": "machine_learning_fundamentals",
    "data_science_fundamentals": "machine_learning_fundamentals",
    "analysis techniques": "statistics_and_probability",
    "analysis_techniques": "statistics_and_probability",
    "exploratory_and_statistical_analysis": "statistics_and_probability",

    # ---- Math ----------------------------------------------------------------
    "linear algebra and matrix operations": "linear_algebra",
    "linear_algebra_and_matrix_operations": "linear_algebra",
    "math_foundations": "math_foundations",
    "mathematical_foundations_for_ml": "math_foundations",
    "convex_optimization": "optimization_algorithms",
    "convex optimization": "optimization_algorithms",

    # ---- ML fundamentals -----------------------------------------------------
    "machine learning": "machine_learning_fundamentals",
    "machine_learning": "machine_learning_fundamentals",
    "machine_learning_fundamentals": "machine_learning_fundamentals",
    "ml_fundamentals": "machine_learning_fundamentals",
    "machine learning algorithm efficiency": "machine_learning_fundamentals",
    "machine_learning_algorithm_efficiency": "machine_learning_fundamentals",
    "machine_learning_vs_deep_learning": "machine_learning_fundamentals",
    "machine learning vs deep learning": "machine_learning_fundamentals",
    "machine_learning_lifecycle": "mlops",
    "machine_learning_optimization": "optimization_algorithms",

    # ---- Supervised learning -------------------------------------------------
    "supervised_learning": "supervised_learning",
    "supervised learning": "supervised_learning",
    "supervised_machine\xa0learning": "supervised_learning",
    "supervised_machine learning": "supervised_learning",

    # ---- Classification & regression -----------------------------------------
    "classification": "classification",
    "linear_regression": "linear_regression",
    "linear regression": "linear_regression",
    "decision_trees": "gradient_boosting",

    # ---- Gradient boosting & forests -----------------------------------------
    "gradient_boosting": "gradient_boosting",
    "random_forest": "random_forest",

    # ---- Clustering & dimensionality reduction --------------------------------
    "clustering": "clustering",
    "dimensionality_reduction": "dimensionality_reduction",

    # ---- Feature work --------------------------------------------------------
    "feature_engineering": "feature_engineering",
    "feature_selection": "feature_selection",

    # ---- Model quality -------------------------------------------------------
    "model_regularization": "model_regularization",
    "regularization": "model_regularization",
    "model_evaluation": "model_evaluation",
    "evaluation": "model_evaluation",
    "evaluation_metrics": "model_evaluation",
    "metrics": "model_evaluation",
    "validation": "model_evaluation",
    "model_validation": "model_evaluation",
    "hyperparameter_tuning": "hyperparameter_tuning",
    "parameter_tuning": "hyperparameter_tuning",
    "batch_size_selection": "hyperparameter_tuning",

    # ---- Deep learning -------------------------------------------------------
    "deep learning": "deep_learning_fundamentals",
    "deep_learning": "deep_learning_fundamentals",
    "deep_learning_fundamentals": "deep_learning_fundamentals",
    "deep learning optimization": "optimization_algorithms",
    "deep_learning_optimization": "optimization_algorithms",
    "deep learning techniques": "deep_learning_fundamentals",
    "deep_learning_techniques": "deep_learning_fundamentals",
    "deep learning frameworks": "deep_learning_fundamentals",
    "deep_learning_frameworks": "deep_learning_fundamentals",
    "deep_learning_training": "training_dynamics",

    # ---- Neural networks -----------------------------------------------------
    "neural networks": "neural_networks",
    "neural_networks": "neural_networks",
    "artificial intelligence/neural networks": "neural_networks",
    "artificial_intelligence/neural_networks": "neural_networks",
    "neural_network_optimization": "optimization_algorithms",
    "neural networks for computer\xa0vision": "convolutional_neural_networks",
    "neural_networks_for_computer\xa0vision": "convolutional_neural_networks",
    "neural_networks_for_vision": "computer_vision",

    # ---- CNN / vision --------------------------------------------------------
    "convolutional neural networks": "convolutional_neural_networks",
    "convolutional_neural_networks": "convolutional_neural_networks",
    "convolutional neural networks (cnn)": "convolutional_neural_networks",
    "convolutional_neural_networks_(cnn)": "convolutional_neural_networks",
    "cnn": "convolutional_neural_networks",
    "computer_vision": "computer_vision",
    "computer_vision_fundamentals": "computer_vision",
    "computer_vision_system_design": "computer_vision",

    # ---- Activation / batch norm ---------------------------------------------
    "activation functions": "activation_functions",
    "activation_functions": "activation_functions",
    "activation functions in neural networks": "activation_functions",
    "activation_functions_in_neural_networks": "activation_functions",
    "relu activation function and dying node problem": "activation_functions",
    "relu_activation_function_and_dying_node_problem": "activation_functions",
    "relu_and_dying_neuron_problem": "activation_functions",
    "batch normalization": "batch_normalization",
    "batch_normalization": "batch_normalization",
    "padding": "convolutional_neural_networks",

    # ---- Optimization --------------------------------------------------------
    "optimization algorithms": "optimization_algorithms",
    "optimization_algorithms": "optimization_algorithms",
    "optimization techniques": "optimization_algorithms",
    "optimization_techniques": "optimization_algorithms",
    "optimization techniques in deep learning": "optimization_algorithms",
    "optimization_techniques_in_deep_learning": "optimization_algorithms",
    "optimization techniques for online learning": "optimization_algorithms",
    "optimization_techniques_for_online_learning": "optimization_algorithms",
    "optimization": "optimization_algorithms",
    "model_optimization": "optimization_algorithms",
    "optimization_in_neural\xa0networks": "optimization_algorithms",
    "optimization in neural networks": "optimization_algorithms",

    # ---- Training dynamics ---------------------------------------------------
    "training": "training_dynamics",
    "training_dynamics": "training_dynamics",
    "large_language_model_training": "training_dynamics",
    "llm_training": "training_dynamics",


    # ---- NLP -----------------------------------------------------------------
    "nlp": "natural_language_processing",
    "natural language processing": "natural_language_processing",
    "natural_language_processing": "natural_language_processing",
    "natural language processing (nlp)": "natural_language_processing",
    "natural_language_processing_(nlp)": "natural_language_processing",
    "natural_language_processing (nlp)": "natural_language_processing",
    "nlp_system_design": "natural_language_processing",

    # ---- Text ----------------------------------------------------------------
    "text vectorization": "text_representation",
    "text_vectorization": "text_representation",
    "text_representation_and_vectorization": "text_representation",
    "text_classification": "text_classification",
    "topic modeling": "topic_modeling",
    "topic_modeling": "topic_modeling",
    "word embeddings": "word_embeddings",
    "word_embeddings": "word_embeddings",

    # ---- LLMs ----------------------------------------------------------------
    "large_language_model_fundamentals": "large_language_models",
    "llm_basics": "large_language_models",
    "llm_history": "large_language_models",
    "hallucination_in_llm": "hallucination_in_llm",
    "large_language_model_optimization": "llm_optimization",
    "llm_optimization": "llm_optimization",
    "large_language_model_safety": "llm_safety",
    "llm_safety": "llm_safety",
    "ai_safety_for_llms": "llm_safety",
    "parameter_efficient_fine_tuning": "parameter_efficient_fine_tuning",
    "peft": "parameter_efficient_fine_tuning",
    "quantization_in_llm": "quantization",

    # ---- RAG -----------------------------------------------------------------
    "rag": "retrieval_augmented_generation",
    "retrieval_augmented_generation": "retrieval_augmented_generation",

    # ---- Information retrieval & search ------------------------------------
    "information retrieval": "information_retrieval",
    "information_retrieval": "information_retrieval",
    "search_and_ranking_systems": "search_and_ranking",
    "search_system_design": "search_and_ranking",
    "learning_to_rank_and_search": "search_and_ranking",
    "ranking_and\xa0search": "search_and_ranking",
    "ranking and search": "search_and_ranking",

    # ---- Recommender systems ------------------------------------------------
    "recommender_system": "recommender_systems",
    "recommender_systems": "recommender_systems",
    "recommender_systems_fundamentals": "recommender_systems",
    "recsys": "recommender_systems",
    "recommendation_system_design": "recommender_systems",

    # ---- Time series --------------------------------------------------------
    "time_series": "time_series",
    "time_series_analysis": "time_series",
    "time_series_system_design": "time_series",

    # ---- MLOps ---------------------------------------------------------------
    "mlops": "mlops",
    "mlops_fundamentals": "mlops",
    "deployment": "model_deployment",
    "model_deployment": "model_deployment",
    "model_development": "model_deployment",
    "model_monitoring": "model_monitoring",
    "monitoring": "model_monitoring",
    "ai_system_monitoring": "model_monitoring",
    "ai_system_reliability": "model_monitoring",
    "ml_system_design_deployment": "ml_system_design",
    "machine_learning_lifecycle": "mlops",

    # ---- Data engineering ---------------------------------------------------
    "data_engineering": "data_engineering",
    "data_engineering_for_ml": "data_engineering",
    "data_labeling": "data_labeling",
    "data_labeling_and_annotation": "data_labeling",

    # ---- Experimentation ----------------------------------------------------
    "experimentation": "experimentation",
    "experimentation_and_ab_testing": "experimentation",

    # ---- Software engineering -----------------------------------------------
    "software_engineering": "software_engineering",
    "software_engineering_principles": "software_engineering",
    "programming_for_ml": "programming_for_ml",
    "python_for_ml": "programming_for_ml",
    "python": "programming_for_ml",
    "pytorch": "programming_for_ml",
    "algorithmic_efficiency": "programming_for_ml",

    # ---- AI safety & governance ---------------------------------------------
    "ai_governance": "ai_safety_and_governance",
    "ai_privacy": "privacy",
    "privacy": "privacy",
    "security": "ai_security",
    "ai_security": "ai_security",

    # ---- System design ------------------------------------------------------
    "system_design": "ml_system_design",
    "system_design_for_ml": "ml_system_design",
    "ai_system_design": "ml_system_design",
    "ai_engineering_fundamentals": "ml_system_design",
    "llm_system_design": "large_language_models",
    "edge_ai_system_design": "ml_system_design",
    "adaptive_ai_systems": "ml_system_design",
    "reinforcement_learning_design": "ml_system_design",
    "graph_machine_learning": "ml_system_design",
    "feedback_loops": "ml_system_design",
    "human_in_the_loop": "ml_system_design",
    "ai_evaluation": "model_evaluation",

    # ---- Agentic AI ----------------------------------------------------------
    "agents": "agentic_ai",
    "agentic_ai_systems": "agentic_ai",

    # ---- Coding --------------------------------------------------------------
    "coding": "coding",

    # ---- Generative models ---------------------------------------------------
    "generative models": "neural_networks",
    "generative_models": "neural_networks",
}


# ---------------------------------------------------------------------------
# TopicNormalizer
# ---------------------------------------------------------------------------

class TopicNormalizer:
    """
    Normalize a raw topic string to an entry in CANONICAL_TOPICS.

    Normalization pipeline:
      1. Unicode NFKC normalize + strip
      2. Replace common separators / non-canonical whitespace with '_'
      3. Lowercase
      4. Exact match in _RAW_TO_CANONICAL  →  return canonical
      5. Exact match already in CANONICAL_TOPICS  →  return as-is
      6. Fuzzy match via difflib.get_close_matches (cutoff=0.70)  →  best match
      7. Final fallback:  "general"  (with a warning log)
    """

    def normalize(self, raw_topic: str) -> str:
        """Return the canonical form of *raw_topic*."""
        if not raw_topic:
            return "general"

        # Step 1 — unicode normalise (replaces \xa0 with a regular space)
        cleaned = unicodedata.normalize("NFKC", raw_topic).strip()

        # Step 2 — collapse hyphens / whitespace sequences to single _
        import re
        cleaned = re.sub(r"[\s\-]+", "_", cleaned)

        # Step 3 — lowercase
        key = cleaned.lower()

        # Step 4 — explicit mapping (includes all known raw forms from the audit)
        if key in _RAW_TO_CANONICAL:
            return _RAW_TO_CANONICAL[key]

        # Also try the original (pre-lowercase) version in case of exact match
        if raw_topic.strip() in _RAW_TO_CANONICAL:
            return _RAW_TO_CANONICAL[raw_topic.strip()]

        # Step 5 — already canonical?
        if key in CANONICAL_TOPICS:
            return key

        # Step 6 — fuzzy fallback
        matches = get_close_matches(key, CANONICAL_TOPICS, n=1, cutoff=0.70)
        if matches:
            canonical = matches[0]
            logger.warning(
                "TopicNormalizer: fuzzy-matched %r → %r", raw_topic, canonical
            )
            return canonical

        # Step 7 — unknown; default to general
        logger.warning(
            "TopicNormalizer: unknown topic %r — falling back to 'general'", raw_topic
        )
        return "general"
