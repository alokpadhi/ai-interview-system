"""
Integration tests for retrieval quality.
Tests that ingested data is actually retrievable with good relevance.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from scripts.ingest_data_to_chromadb import DataIngestion
from src.data.vector_store import VectorStore


class TestRetrievalQuality:
    """Test suite for retrieval quality after ingestion"""
    
    @pytest.fixture(scope="class")
    def ingested_vector_store(self):
        """
        Fixture that ingests sample data once for all tests in this class.
        Using scope="class" for efficiency.
        """
        # Define sample data directly (avoiding function-scoped fixture dependency)
        sample_questions = [
            {
                "id": "q_test_001",
                "text": "Explain gradient descent and its variants",
                "difficulty": "medium",
                "topic": "optimization",
                "question_type": "conceptual",
                "estimated_time_minutes": 5,
                "tags": ["ml", "optimization"],
                "reference_answer": "Gradient descent is an optimization algorithm..."
            },
            {
                "id": "q_test_002",
                "text": "What is backpropagation in neural networks?",
                "difficulty": "hard",
                "topic": "deep_learning",
                "question_type": "conceptual",
                "estimated_time_minutes": 7,
                "tags": ["deep_learning", "neural_networks"],
                "reference_answer": "Backpropagation is..."
            },
            {
                "id": "q_test_003",
                "text": "Define overfitting and how to prevent it",
                "difficulty": "easy",
                "topic": "evaluation",
                "question_type": "conceptual",
                "estimated_time_minutes": 4,
                "tags": ["ml", "evaluation"],
                "reference_answer": "Overfitting occurs when..."
            }
        ]
        
        sample_concepts = [
            {
                "id": "concept_test_001",
                "concept_name": "Gradient Descent",
                "explanation": "Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function.",
                "category": "optimization",
                "simple_explanation": "Think of walking downhill to find the lowest point.",
                "examples": ["Training neural networks", "Linear regression"],
                "related_concepts": ["learning_rate", "SGD", "momentum"]
            },
            {
                "id": "concept_test_002",
                "concept_name": "Overfitting",
                "explanation": "Overfitting occurs when a model learns the training data too well, including noise.",
                "category": "evaluation",
                "simple_explanation": "Memorizing answers instead of understanding concepts.",
                "examples": ["High training accuracy, low test accuracy"],
                "related_concepts": ["regularization", "cross_validation"]
            }
        ]
        
        # Create temporary directory for this test class
        tempdir = tempfile.mkdtemp()
        vector_store = VectorStore(persist_directory=tempdir)
        
        # Create temporary files for sample data
        temp_path = Path(tempdir)
        questions_file = temp_path / "questions.json"
        concepts_file = temp_path / "concepts.json"
        
        # Save sample data to temp files
        with open(questions_file, "w") as f:
            json.dump(sample_questions, f)
        
        with open(concepts_file, "w") as f:
            json.dump(sample_concepts, f)
        
        # Ingest data
        ingestion = DataIngestion(vector_store, reset_collections=True)
        ingestion.ingest_questions(str(questions_file))
        ingestion.ingest_concepts(str(concepts_file))
        
        # Yield the vector store for tests
        yield vector_store
        
        # Cleanup after all tests in class complete
        shutil.rmtree(tempdir)
    
    def test_semantic_search_relevance(self, ingested_vector_store):
        """Test that semantically similar queries return relevant results"""
        # Query for optimization-related content
        results = ingested_vector_store.query(
            collection_name="interview_questions",
            query_text="optimization algorithms for machine learning",
            n_results=3
        )
        
        # Assert results were returned
        assert len(results["ids"][0]) > 0
        
        # Assert "gradient descent" question is in results (most relevant)
        documents = results["documents"][0]
        assert any("gradient descent" in doc.lower() for doc in documents), \
            "Expected 'gradient descent' in results for optimization query"
        
        # Check that the most relevant result is first (lowest distance)
        distances = results["distances"][0]
        assert distances[0] <= distances[-1], \
            "Results should be ordered by relevance (ascending distance)"
        
        # Check distance/relevance score is reasonable (cosine distance should be < 1.0 for relevant results)
        assert distances[0] < 1.0, \
            f"Top result distance {distances[0]} is too high, indicating poor relevance"
    
    def test_metadata_filtering(self, ingested_vector_store):
        """Test that metadata filters work correctly"""
        # Test 1: Filter by difficulty="medium"
        results = ingested_vector_store.query(
            collection_name="interview_questions",
            query_text="machine learning",
            n_results=5,
            where={"difficulty": "medium"}
        )
        
        # Assert all results have difficulty="medium"
        metadatas = results["metadatas"][0]
        for metadata in metadatas:
            assert metadata["difficulty"] == "medium", \
                f"Expected difficulty='medium', got '{metadata['difficulty']}'"
        
        # Test 2: Filter by difficulty="hard"
        results_hard = ingested_vector_store.query(
            collection_name="interview_questions",
            query_text="machine learning",
            n_results=5,
            where={"difficulty": "hard"}
        )
        
        metadatas_hard = results_hard["metadatas"][0]
        for metadata in metadatas_hard:
            assert metadata["difficulty"] == "hard", \
                f"Expected difficulty='hard', got '{metadata['difficulty']}'"
        
        # Test 3: Filter by topic
        results_topic = ingested_vector_store.query(
            collection_name="interview_questions",
            query_text="machine learning",
            n_results=5,
            where={"topic": "optimization"}
        )
        
        metadatas_topic = results_topic["metadatas"][0]
        for metadata in metadatas_topic:
            assert metadata["topic"] == "optimization", \
                f"Expected topic='optimization', got '{metadata['topic']}'"
        
        # Test 4: Filter by question_type
        results_type = ingested_vector_store.query(
            collection_name="interview_questions",
            query_text="explain",
            n_results=5,
            where={"question_type": "conceptual"}
        )
        
        metadatas_type = results_type["metadatas"][0]
        for metadata in metadatas_type:
            assert metadata["question_type"] == "conceptual", \
                f"Expected question_type='conceptual', got '{metadata['question_type']}'"
    
    def test_embedding_normalization(self, ingested_vector_store):
        """Test that retrieved documents have normalized embeddings"""
        # Query the collection
        results = ingested_vector_store.query(
            collection_name="interview_questions",
            query_text="machine learning concepts",
            n_results=3
        )
        
        # Get distances
        distances = results["distances"][0]
        
        # For cosine distance with normalized vectors, distance should be in [0, 2]
        # In practice, for relevant results, it should be much lower (< 1.0)
        for i, distance in enumerate(distances):
            assert 0 <= distance <= 2, \
                f"Distance {distance} at position {i} is outside valid range [0, 2]"
        
        # Verify distances are reasonable for semantic search
        # The closest result should have a relatively low distance
        assert distances[0] < 1.5, \
            f"Closest result has distance {distances[0]}, which is too high"
        
        # Get the actual embeddings to verify they exist
        collection = ingested_vector_store.client.get_collection("interview_questions")
        all_results = collection.get(include=["embeddings"])
        
        # Verify embeddings exist and are not null
        assert all_results["embeddings"] is not None
        assert len(all_results["embeddings"]) > 0
        
        # Verify each embedding is a valid vector
        for embedding in all_results["embeddings"]:
            assert embedding is not None
            assert len(embedding) == 768  # Expected dimension for BAAI/bge-base-en-v1.5
            assert all(isinstance(x, (int, float)) for x in embedding)
    
    def test_retrieval_count(self, ingested_vector_store):
        """Test that n_results parameter is respected"""
        # Test 1: Query with n_results=1
        results_1 = ingested_vector_store.query(
            collection_name="interview_questions",
            query_text="machine learning",
            n_results=1
        )
        assert len(results_1["ids"][0]) == 1, \
            f"Expected 1 result, got {len(results_1['ids'][0])}"
        
        # Test 2: Query with n_results=2
        results_2 = ingested_vector_store.query(
            collection_name="interview_questions",
            query_text="machine learning",
            n_results=2
        )
        assert len(results_2["ids"][0]) == 2, \
            f"Expected 2 results, got {len(results_2['ids'][0])}"
        
        # Test 3: Query with n_results=3
        results_3 = ingested_vector_store.query(
            collection_name="interview_questions",
            query_text="machine learning",
            n_results=3
        )
        assert len(results_3["ids"][0]) == 3, \
            f"Expected 3 results, got {len(results_3['ids'][0])}"
        
        # Test 4: Query with n_results=10 when only 3 docs exist
        # Should return all available documents without error
        results_10 = ingested_vector_store.query(
            collection_name="interview_questions",
            query_text="machine learning",
            n_results=10
        )
        # Should return 3 results (all available), not error
        assert len(results_10["ids"][0]) == 3, \
            f"Expected 3 results (all available), got {len(results_10['ids'][0])}"
        
        # Verify no errors occurred
        assert results_10["ids"] is not None
        assert results_10["documents"] is not None
    
    def test_concept_lookup(self, ingested_vector_store):
        """Test concept retrieval for feedback agent"""
        # Test 1: Search for "overfitting" concept
        results = ingested_vector_store.query(
            collection_name="ml_concepts",
            query_text="overfitting",
            n_results=2
        )
        
        # Assert results were returned
        assert len(results["ids"][0]) > 0, "No results returned for overfitting query"
        
        # Assert correct concept is returned (should be top result)
        top_document = results["documents"][0][0]
        assert "overfitting" in top_document.lower(), \
            f"Expected 'overfitting' in top result, got: {top_document}"
        
        # Verify metadata is present
        top_metadata = results["metadatas"][0][0]
        assert "category" in top_metadata
        assert top_metadata["category"] == "evaluation"
        
        # Test 2: Search for "gradient descent" concept
        results_gd = ingested_vector_store.query(
            collection_name="ml_concepts",
            query_text="optimization algorithm",
            n_results=2
        )
        
        # Should return gradient descent as most relevant
        top_doc_gd = results_gd["documents"][0][0]
        assert "gradient descent" in top_doc_gd.lower(), \
            f"Expected 'gradient descent' for optimization query, got: {top_doc_gd}"
        
        # Test 3: Verify concept format includes both name and explanation
        for doc in results["documents"][0]:
            assert ":" in doc, \
                "Concept document should be formatted as 'name: explanation'"
            assert len(doc) > 20, \
                "Concept document should have substantial content"
        
        # Test 4: Verify all concepts are retrievable
        stats = ingested_vector_store.get_collection_stats("ml_concepts")
        assert stats["count"] == 2, \
            f"Expected 2 concepts in collection, got {stats['count']}"
    
    def test_cross_collection_independence(self, ingested_vector_store):
        """Test that questions and concepts collections are independent"""
        # Query questions collection
        questions_results = ingested_vector_store.query(
            collection_name="interview_questions",
            query_text="gradient descent",
            n_results=5
        )
        
        # Query concepts collection
        concepts_results = ingested_vector_store.query(
            collection_name="ml_concepts",
            query_text="gradient descent",
            n_results=5
        )
        
        # Verify both collections exist and have data
        assert len(questions_results["ids"][0]) > 0
        assert len(concepts_results["ids"][0]) > 0
        
        # Verify IDs are different (no overlap between collections)
        question_ids = set(questions_results["ids"][0])
        concept_ids = set(concepts_results["ids"][0])
        assert question_ids.isdisjoint(concept_ids), \
            "Question and concept IDs should not overlap"
        
        # Verify document formats are different
        # Questions should have the raw question text
        # Concepts should have "name: explanation" format
        question_doc = questions_results["documents"][0][0]
        concept_doc = concepts_results["documents"][0][0]
        
        # Concepts should have colon separator
        assert ":" in concept_doc, "Concept should have 'name: explanation' format"
        
        # Verify metadata schemas are different
        question_meta = questions_results["metadatas"][0][0]
        concept_meta = concepts_results["metadatas"][0][0]
        
        # Questions have 'difficulty', concepts have 'category'
        assert "difficulty" in question_meta or "topic" in question_meta
        assert "category" in concept_meta or "concept_name" in concept_meta
    
    def test_retrieval_with_no_results(self, ingested_vector_store):
        """Test behavior when metadata filter returns no results"""
        # Query with a filter that shouldn't match anything
        results = ingested_vector_store.query(
            collection_name="interview_questions",
            query_text="machine learning",
            n_results=5,
            where={"difficulty": "nonexistent_difficulty"}
        )
        
        # Should return empty results, not error
        assert len(results["ids"][0]) == 0, \
            "Expected 0 results for non-matching filter"
        assert len(results["documents"][0]) == 0
        assert len(results["metadatas"][0]) == 0
    
    def test_relevance_ranking_order(self, ingested_vector_store):
        """Test that results are properly ranked by relevance"""
        # Query for a specific topic
        results = ingested_vector_store.query(
            collection_name="interview_questions",
            query_text="neural networks and backpropagation",
            n_results=3
        )
        
        distances = results["distances"][0]
        documents = results["documents"][0]
        
        # Verify distances are in ascending order (most relevant first)
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1], \
                f"Distances not in ascending order: {distances[i]} > {distances[i+1]}"
        
        # The top result should be most relevant to the query
        # For "neural networks and backpropagation", expect backpropagation question first
        top_doc = documents[0].lower()
        assert "backpropagation" in top_doc or "neural" in top_doc, \
            f"Expected neural network related content in top result, got: {documents[0]}"