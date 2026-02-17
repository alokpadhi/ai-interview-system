"""
Unit tests for VectorRetriever.
"""

import pytest
import json
from pathlib import Path
from src.rag.retriever import VectorRetriever
from src.rag.models import RetrievalResult, RetrievalContext
from scripts.ingest_data_to_chromadb import DataIngestion


class TestVectorRetriever:
    """Test suite for VectorRetriever"""
    
    @pytest.fixture
    def retriever(self, vector_store_temp, sample_questions, sample_concepts, tmp_path):
        """
        Fixture that provides retriever with ingested test data.
        
        This runs once per test - each test gets fresh data.
        """
        # 1. Save sample_questions and sample_concepts to temp JSON files
        questions_file = tmp_path / "test_questions.json"
        concepts_file = tmp_path / "test_concepts.json"
        
        with open(questions_file, 'w') as f:
            json.dump(sample_questions, f)
        
        with open(concepts_file, 'w') as f:
            json.dump(sample_concepts, f)
        
        # 2. Use DataIngestion to ingest them
        ingestion = DataIngestion(vector_store_temp, reset_collections=True)
        ingestion.ingest_questions(str(questions_file))
        ingestion.ingest_concepts(str(concepts_file))
        
        # 3. Return VectorRetriever(vector_store_temp)
        return VectorRetriever(vector_store_temp)
    
    def test_retrieve_questions_basic(self, retriever):
        """Test basic question retrieval without filters"""
        results = retriever.retrieve_questions(
            query="optimization algorithms",
            n_result=10
        )
        
        # Assert: results returned, correct types, valid scores
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, RetrievalResult)
            assert isinstance(result.id, str)
            assert isinstance(result.text, str)
            assert 0.0 <= result.relevance_score <= 1.0
    
    def test_retrieve_questions_with_difficulty(self, retriever):
        """Test retrieval with difficulty filter"""
        results = retriever.retrieve_questions(
            query="machine learning",
            difficulty="medium"
        )
        
        # All results should have difficulty="medium"
        assert len(results) > 0
        for result in results:
            assert result.difficulty == "medium"
    
    def test_retrieve_questions_with_topic(self, retriever):
        """Test retrieval with topic filter"""
        results = retriever.retrieve_questions(
            query="neural networks",
            topic="deep_learning"
        )
        
        # All results should have topic="deep_learning"
        if len(results) > 0:
            for result in results:
                assert result.topic == "deep_learning"
    
    def test_retrieve_questions_multiple_filters(self, retriever):
        """Test retrieval with multiple filters combined"""
        results = retriever.retrieve_questions(
            query="machine learning",
            difficulty="easy",
            topic="evaluation",
            question_type="conceptual"
        )
        
        # All results should match all filters
        if len(results) > 0:
            for result in results:
                assert result.difficulty == "easy"
                assert result.topic == "evaluation"
                assert result.question_type == "conceptual"
    
    def test_retrieve_questions_exclude_ids(self, retriever):
        """Test that exclude_ids filters correctly"""
        # 1. Get results without exclusion
        results_initial = retriever.retrieve_questions(
            query="machine learning",
            n_result=5
        )
        
        assert len(results_initial) > 0
        
        # 2. Note first result's ID
        excluded_id = results_initial[0].id
        
        # 3. Query again excluding that ID
        results_filtered = retriever.retrieve_questions(
            query="machine learning",
            exclude_ids={excluded_id},
            n_result=5
        )
        
        # 4. Assert that ID not in new results
        result_ids = [r.id for r in results_filtered]
        assert excluded_id not in result_ids
    
    def test_retrieve_questions_n_results(self, retriever):
        """Test that n_results parameter is respected"""
        results = retriever.retrieve_questions(
            query="machine learning",
            n_result=2
        )
        
        # Should return at most 2 results
        assert len(results) <= 2
    
    def test_retrieve_questions_sorted_by_relevance(self, retriever):
        """Test that results are sorted by relevance (highest first)"""
        results = retriever.retrieve_questions(query="optimization", n_result=5)
        
        # Check scores are descending
        if len(results) > 1:
            scores = [r.relevance_score for r in results]
            assert scores == sorted(scores, reverse=True)
    
    def test_retrieve_concepts_basic(self, retriever):
        """Test concept retrieval"""
        concepts = retriever.retrieve_concepts(query="overfitting")
        
        # Should find concepts
        assert len(concepts) > 0
        assert isinstance(concepts[0], RetrievalResult)
        
        # Should have concept_name in metadata
        for concept in concepts:
            assert "concept_name" in concept.metadata or concept.concept_name is not None
    
    def test_retrieve_concepts_with_category(self, retriever):
        """Test concept retrieval with category filter"""
        concepts = retriever.retrieve_concepts(
            query="optimization",
            category="optimization"
        )
        
        # All results should have category="optimization"
        if len(concepts) > 0:
            for concept in concepts:
                assert concept.metadata.get("category") == "optimization"
    
    def test_retrieval_result_properties(self, retriever):
        """Test that RetrievalResult convenience properties work"""
        results = retriever.retrieve_questions(query="test", n_result=1)
        
        if results:
            result = results[0]
            # Access properties - should not raise errors
            _ = result.difficulty
            _ = result.topic
            _ = result.question_type
    
    def test_empty_results(self, retriever):
        """Test handling of queries with no matching results"""
        results = retriever.retrieve_questions(
            query="xyzabc123",  # Gibberish query
            difficulty="impossible",  # Non-existent difficulty
            topic="fake_topic"
        )
        
        # Should return empty list, not error
        assert results == []
    
    def test_relevance_score_range(self, retriever):
        """Test that all relevance scores are in valid range [0, 1]"""
        results = retriever.retrieve_questions(query="machine learning", n_result=10)
        
        for result in results:
            assert 0.0 <= result.relevance_score <= 1.0
    
    def test_collection_stats(self, retriever):
        """Test getting collection statistics"""
        stats = retriever.get_collection_stats("interview_questions")
        
        assert "name" in stats
        assert "count" in stats
        assert stats["count"] > 0
    
    def test_get_all_collections(self, retriever):
        """Test getting all collection names"""
        collections = retriever.get_all_collections()
        
        assert isinstance(collections, list)
        assert "interview_questions" in collections
        assert "ml_concepts" in collections


class TestRetrievalResultModel:
    """Test suite for RetrievalResult Pydantic model"""
    
    def test_valid_creation(self):
        """Test creating valid RetrievalResult"""
        result = RetrievalResult(
            id="test_1",
            text="Test question",
            relevance_score=0.8,
            metadata={"difficulty": "medium"}
        )
        
        assert result.id == "test_1"
        assert result.text == "Test question"
        assert result.relevance_score == 0.8
        assert result.metadata["difficulty"] == "medium"
    
    def test_invalid_relevance_score(self):
        """Test that invalid relevance_score raises validation error"""
        with pytest.raises(Exception):  # Pydantic ValidationError
            RetrievalResult(
                id="test",
                text="Test",
                relevance_score=1.5,  # Invalid!
                metadata={}
            )
        
        with pytest.raises(Exception):
            RetrievalResult(
                id="test",
                text="Test",
                relevance_score=-0.1,  # Invalid!
                metadata={}
            )
    
    def test_convenience_properties(self):
        """Test convenience property accessors"""
        result = RetrievalResult(
            id="test",
            text="Test",
            relevance_score=0.5,
            metadata={
                "difficulty": "hard",
                "topic": "ml",
                "question_type": "coding",
                "concept_name": "Gradient Descent"
            }
        )
        
        assert result.difficulty == "hard"
        assert result.topic == "ml"
        assert result.question_type == "coding"
        assert result.concept_name == "Gradient Descent"
    
    def test_str_representation(self):
        """Test string representation is readable"""
        result = RetrievalResult(
            id="test",
            text="What is machine learning?",
            relevance_score=0.856,
            metadata={}
        )
        
        str_repr = str(result)
        # Should contain score and text
        assert "0.856" in str_repr
        assert "machine learning" in str_repr.lower()


class TestRetrievalContext:
    """Test suite for RetrievalContext model"""
    
    def test_should_adapt_difficulty(self):
        """Test difficulty adaptation logic"""
        # Test case 1: High scores → should return True
        context_high = RetrievalContext(
            performance_trajectory=[8.5, 9.0, 8.8]
        )
        assert context_high.should_adapt_difficulty() is True
        
        # Test case 2: Low scores → should return True
        context_low = RetrievalContext(
            performance_trajectory=[4.0, 3.5, 4.5]
        )
        assert context_low.should_adapt_difficulty() is True
        
        # Test case 3: Mixed scores → should return False
        context_mixed = RetrievalContext(
            performance_trajectory=[6.0, 7.0, 6.5]
        )
        assert context_mixed.should_adapt_difficulty() is False
        
        # Test case 4: Not enough data → should return False
        context_insufficient = RetrievalContext(
            performance_trajectory=[8.0, 9.0]
        )
        assert context_insufficient.should_adapt_difficulty() is False
    
    def test_get_performance_trend(self):
        """Test performance trend detection"""
        # Test case 1: Improving scores → "improving"
        context_improving = RetrievalContext(
            performance_trajectory=[5.0, 6.0, 7.0, 8.0]
        )
        assert context_improving.get_performance_trend() == "improving"
        
        # Test case 2: Declining scores → "declining"
        context_declining = RetrievalContext(
            performance_trajectory=[8.0, 7.0, 6.0, 5.0]
        )
        assert context_declining.get_performance_trend() == "declining"
        
        # Test case 3: Stable scores → "stable"
        context_stable = RetrievalContext(
            performance_trajectory=[7.0, 7.1, 6.9, 7.0]
        )
        assert context_stable.get_performance_trend() == "stable"
        
        # Test case 4: Insufficient data → "stable"
        context_insufficient = RetrievalContext(
            performance_trajectory=[7.0]
        )
        assert context_insufficient.get_performance_trend() == "stable"