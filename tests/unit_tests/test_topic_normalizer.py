"""
Unit tests for TopicNormalizer and CANONICAL_TOPICS.
"""

import pytest
from src.utils.topic_taxonomy import CANONICAL_TOPICS, TopicNormalizer


@pytest.fixture
def normalizer():
    return TopicNormalizer()


class TestCanonicalTopics:
    """Sanity checks on the CANONICAL_TOPICS constant."""

    def test_is_frozenset(self):
        assert isinstance(CANONICAL_TOPICS, frozenset)

    def test_non_empty(self):
        assert len(CANONICAL_TOPICS) > 0

    def test_all_snake_case(self):
        """Every canonical topic must be lowercase snake_case (no spaces, no \xa0)."""
        for topic in CANONICAL_TOPICS:
            assert topic == topic.lower(), f"{topic!r} is not lowercase"
            assert " " not in topic, f"{topic!r} contains a space"
            assert "\xa0" not in topic, f"{topic!r} contains a non-breaking space"

    def test_general_present(self):
        """'general' is the fallback topic — must exist."""
        assert "general" in CANONICAL_TOPICS

    def test_coding_present(self):
        assert "coding" in CANONICAL_TOPICS

    def test_no_duplicates(self):
        """frozenset guarantees uniqueness, but verify the list form too."""
        topics_list = list(CANONICAL_TOPICS)
        assert len(topics_list) == len(set(topics_list))


class TestTopicNormalizerExplicitMapping:
    """Test cases drawn from the real audit of all 6 source JSON files."""

    def test_already_canonical_passthrough(self, normalizer):
        for topic in ["deep_learning_fundamentals", "natural_language_processing",
                      "recommender_systems", "coding", "general"]:
            assert normalizer.normalize(topic) == topic

    def test_title_case(self, normalizer):
        assert normalizer.normalize("Deep Learning") == "deep_learning_fundamentals"
        assert normalizer.normalize("Machine Learning") == "machine_learning_fundamentals"
        assert normalizer.normalize("Natural Language Processing") == "natural_language_processing"

    def test_abbreviation_nlp(self, normalizer):
        """NLP, nlp → natural_language_processing"""
        assert normalizer.normalize("NLP") == "natural_language_processing"
        assert normalizer.normalize("nlp") == "natural_language_processing"

    def test_acronym_recsys(self, normalizer):
        assert normalizer.normalize("recsys") == "recommender_systems"

    def test_recommender_variants(self, normalizer):
        assert normalizer.normalize("recommender_system") == "recommender_systems"
        assert normalizer.normalize("recommender_systems_fundamentals") == "recommender_systems"
        assert normalizer.normalize("recommendation_system_design") == "recommender_systems"

    def test_nonbreaking_space_vision(self, normalizer):
        """neural_networks_for_computer\xa0vision → convolutional_neural_networks"""
        raw = "neural_networks_for_computer\xa0vision"
        assert normalizer.normalize(raw) == "convolutional_neural_networks"

    def test_nonbreaking_space_optimization(self, normalizer):
        raw = "optimization_in_neural\xa0networks"
        assert normalizer.normalize(raw) == "optimization_algorithms"

    def test_nonbreaking_space_ranking(self, normalizer):
        raw = "ranking_and\xa0search"
        assert normalizer.normalize(raw) == "search_and_ranking"

    def test_nonbreaking_space_supervised(self, normalizer):
        raw = "supervised_machine\xa0learning"
        assert normalizer.normalize(raw) == "supervised_learning"

    def test_peft_alias(self, normalizer):
        assert normalizer.normalize("peft") == "parameter_efficient_fine_tuning"

    def test_rag_alias(self, normalizer):
        assert normalizer.normalize("rag") == "retrieval_augmented_generation"

    def test_mlops_aliases(self, normalizer):
        assert normalizer.normalize("mlops_fundamentals") == "mlops"
        assert normalizer.normalize("machine_learning_lifecycle") == "mlops"

    def test_model_deployment_aliases(self, normalizer):
        assert normalizer.normalize("deployment") == "model_deployment"
        assert normalizer.normalize("model_development") == "model_deployment"

    def test_evaluation_aliases(self, normalizer):
        assert normalizer.normalize("evaluation") == "model_evaluation"
        assert normalizer.normalize("evaluation_metrics") == "model_evaluation"
        assert normalizer.normalize("metrics") == "model_evaluation"
        assert normalizer.normalize("validation") == "model_evaluation"
        assert normalizer.normalize("model_validation") == "model_evaluation"

    def test_optimization_aliases(self, normalizer):
        assert normalizer.normalize("optimization") == "optimization_algorithms"
        assert normalizer.normalize("Optimization Techniques") == "optimization_algorithms"
        assert normalizer.normalize("model_optimization") == "optimization_algorithms"

    def test_regularization_alias(self, normalizer):
        assert normalizer.normalize("regularization") == "model_regularization"

    def test_data_science_alias(self, normalizer):
        assert normalizer.normalize("Data Science") == "machine_learning_fundamentals"

    def test_agents_alias(self, normalizer):
        assert normalizer.normalize("agents") == "agentic_ai"
        assert normalizer.normalize("agentic_ai_systems") == "agentic_ai"

    def test_python_aliases(self, normalizer):
        assert normalizer.normalize("python") == "programming_for_ml"
        assert normalizer.normalize("pytorch") == "programming_for_ml"
        assert normalizer.normalize("python_for_ml") == "programming_for_ml"

    def test_quantization_alias(self, normalizer):
        assert normalizer.normalize("quantization_in_llm") == "quantization"

    def test_llm_aliases(self, normalizer):
        assert normalizer.normalize("llm_basics") == "large_language_models"
        assert normalizer.normalize("llm_history") == "large_language_models"
        assert normalizer.normalize("large_language_model_fundamentals") == "large_language_models"
        assert normalizer.normalize("llm_safety") == "llm_safety"
        assert normalizer.normalize("large_language_model_safety") == "llm_safety"

    def test_time_series_aliases(self, normalizer):
        assert normalizer.normalize("time_series_analysis") == "time_series"
        assert normalizer.normalize("time_series_system_design") == "time_series"


class TestTopicNormalizerEdgeCases:
    """Boundary & robustness tests."""

    def test_empty_string_returns_general(self, normalizer):
        assert normalizer.normalize("") == "general"

    def test_whitespace_stripped(self, normalizer):
        assert normalizer.normalize("  deep_learning  ") == "deep_learning_fundamentals"

    def test_completely_unknown_returns_general(self, normalizer):
        result = normalizer.normalize("xyzzy_not_a_real_topic_12345")
        assert result == "general"

    def test_output_always_in_canonical_topics(self, normalizer):
        """Every output of normalize() must be a canonical topic."""
        raw_samples = [
            "NLP", "nlp", "recsys", "Deep Learning", "ML", "PEFT",
            "neural_networks_for_computer\xa0vision", "data science",
            "llm_basics", "ranking_and\xa0search", "totally_unknown_xyz",
            "optimization", "padding", "cnn",
        ]
        for raw in raw_samples:
            result = normalizer.normalize(raw)
            assert result in CANONICAL_TOPICS, (
                f"normalize({raw!r}) returned {result!r} which is not in CANONICAL_TOPICS"
            )

    def test_idempotent(self, normalizer):
        """Normalizing an already-normalized topic is a no-op."""
        for topic in sorted(CANONICAL_TOPICS):
            assert normalizer.normalize(topic) == topic, (
                f"normalize({topic!r}) is not idempotent: got {normalizer.normalize(topic)!r}"
            )
