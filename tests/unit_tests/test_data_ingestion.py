"""
Unit tests for data ingestion pipeline.
"""

import pytest
import json
from pathlib import Path
import tempfile
import numpy as np

from scripts.ingest_data_to_chromadb import DataIngestion


class TestDataIngestion:
    """Test suite for DataIngestion class"""
    
    def test_load_json_valid_file(self, tmp_path, vector_store_temp):
        """Test loading valid JSON file"""
        # 1. Create temp JSON file with sample data
        test_data = [
            {"id": "test_1", "text": "Sample question 1"},
            {"id": "test_2", "text": "Sample question 2"}
        ]
        json_file = tmp_path / "test_data.json"
        with open(json_file, "w") as f:
            json.dump(test_data, f)
        
        # 2. Call ingestion.load_json()
        ingestion = DataIngestion(vector_store_temp)
        loaded_data = ingestion.load_json(str(json_file))
        
        # 3. Assert data is loaded correctly
        assert loaded_data == test_data
        assert len(loaded_data) == 2
        assert loaded_data[0]["id"] == "test_1"
    
    def test_load_json_missing_file(self, vector_store_temp):
        """Test loading non-existent file raises error"""
        ingestion = DataIngestion(vector_store_temp)
        with pytest.raises(FileNotFoundError):
            ingestion.load_json("nonexistent.json")
    
    def test_load_json_invalid_json(self, tmp_path, vector_store_temp):
        """Test loading invalid JSON raises error"""
        # 1. Create temp file with invalid JSON
        invalid_json_file = tmp_path / "invalid.json"
        with open(invalid_json_file, "w") as f:
            f.write("{invalid json content")
        
        # 2. Try to load it
        ingestion = DataIngestion(vector_store_temp)
        
        # 3. Assert ValueError is raised
        with pytest.raises(ValueError, match="json file couldn't be loaded"):
            ingestion.load_json(str(invalid_json_file))
    
    def test_ingest_questions(self, vector_store_temp, sample_questions, tmp_path):
        """Test question ingestion"""
        # 1. Save sample_questions to temp JSON file
        json_file = tmp_path / "questions.json"
        with open(json_file, "w") as f:
            json.dump(sample_questions, f)
        
        # 2. Create DataIngestion instance
        ingestion = DataIngestion(vector_store_temp, reset_collections=True)
        
        # 3. Call ingest_questions()
        count = ingestion.ingest_questions(str(json_file))
        
        # 4. Assert correct count returned
        assert count == len(sample_questions)
        assert count == 3
        
        # 5. Verify collection was created
        collections = vector_store_temp.list_collections()
        assert "interview_questions" in collections
        
        # 6. Query to verify data is retrievable
        stats = vector_store_temp.get_collection_stats("interview_questions")
        assert stats["count"] == 3
        
        # Query for a specific question
        results = vector_store_temp.query(
            collection_name="interview_questions",
            query_text="gradient descent",
            n_results=1
        )
        assert len(results["ids"][0]) > 0
        assert "gradient descent" in results["documents"][0][0].lower()
    
    def test_ingest_concepts(self, vector_store_temp, sample_concepts, tmp_path):
        """Test concept ingestion"""
        # 1. Save sample_concepts to temp JSON file
        json_file = tmp_path / "concepts.json"
        with open(json_file, "w") as f:
            json.dump(sample_concepts, f)
        
        # 2. Create DataIngestion instance
        ingestion = DataIngestion(vector_store_temp, reset_collections=True)
        
        # 3. Call ingest_concepts()
        count = ingestion.ingest_concepts(str(json_file))
        
        # 4. Assert correct count returned
        assert count == len(sample_concepts)
        assert count == 2
        
        # 5. Verify collection was created
        collections = vector_store_temp.list_collections()
        assert "ml_concepts" in collections
        
        # 6. Query to verify data is retrievable
        stats = vector_store_temp.get_collection_stats("ml_concepts")
        assert stats["count"] == 2
        
        # Query for a specific concept
        results = vector_store_temp.query(
            collection_name="ml_concepts",
            query_text="optimization algorithm",
            n_results=1
        )
        assert len(results["ids"][0]) > 0
        assert "gradient descent" in results["documents"][0][0].lower()
    
    def test_ingest_with_reset(self, vector_store_temp, sample_questions, tmp_path):
        """Test that reset flag clears existing data"""
        # 1. Ingest data once
        json_file = tmp_path / "questions.json"
        with open(json_file, "w") as f:
            json.dump(sample_questions, f)
        
        ingestion = DataIngestion(vector_store_temp, reset_collections=True)
        count1 = ingestion.ingest_questions(str(json_file))
        assert count1 == 3
        
        # Verify initial count
        stats1 = vector_store_temp.get_collection_stats("interview_questions")
        assert stats1["count"] == 3
        
        # 2. Ingest again with reset=True (should succeed and replace data)
        ingestion_with_reset = DataIngestion(vector_store_temp, reset_collections=True)
        count2 = ingestion_with_reset.ingest_questions(str(json_file))
        assert count2 == 3
        
        # Verify only 3 documents exist (not 6)
        stats = vector_store_temp.get_collection_stats("interview_questions")
        assert stats["count"] == 3
    
    def test_verify_ingestion(self, vector_store_temp, sample_questions, tmp_path):
        """Test ingestion verification"""
        # 1. Ingest known number of questions
        json_file = tmp_path / "questions.json"
        with open(json_file, "w") as f:
            json.dump(sample_questions, f)
        
        ingestion = DataIngestion(vector_store_temp, reset_collections=True)
        count = ingestion.ingest_questions(str(json_file))
        
        # 2. Call verify_ingestion() with correct count
        result_correct = ingestion.verify_ingestion("interview_questions", count)
        
        # 3. Assert returns True
        assert result_correct is True
        
        # 4. Call verify_ingestion() with wrong count
        result_wrong = ingestion.verify_ingestion("interview_questions", count + 10)
        
        # 5. Assert returns False
        assert result_wrong is False
    
    def test_empty_file_ingestion(self, vector_store_temp, tmp_path):
        """Test ingesting empty JSON file"""
        # Create empty JSON file
        json_file = tmp_path / "empty.json"
        with open(json_file, "w") as f:
            json.dump([], f)
        
        ingestion = DataIngestion(vector_store_temp, reset_collections=True)
        count = ingestion.ingest_questions(str(json_file))
        
        assert count == 0
    
    def test_metadata_preservation(self, vector_store_temp, sample_questions, tmp_path):
        """Test that metadata is properly preserved during ingestion"""
        json_file = tmp_path / "questions.json"
        with open(json_file, "w") as f:
            json.dump(sample_questions, f)
        
        ingestion = DataIngestion(vector_store_temp, reset_collections=True)
        ingestion.ingest_questions(str(json_file))
        
        # Query and check metadata
        results = vector_store_temp.query(
            collection_name="interview_questions",
            query_text="gradient descent",
            n_results=1
        )
        
        metadata = results["metadatas"][0][0]
        assert "difficulty" in metadata
        assert "topic" in metadata
        assert "tags" in metadata
        # Tags should be JSON string
        assert isinstance(metadata["tags"], str)
        tags = json.loads(metadata["tags"])
        assert isinstance(tags, list)


class TestEmbeddingVerification:
    """Test suite for verifying embeddings are created properly"""
    
    def test_embeddings_are_generated(self, vector_store_temp, sample_questions, tmp_path):
        """Test that embeddings are actually generated for documents"""
        json_file = tmp_path / "questions.json"
        with open(json_file, "w") as f:
            json.dump(sample_questions, f)
        
        ingestion = DataIngestion(vector_store_temp, reset_collections=True)
        ingestion.ingest_questions(str(json_file))
        
        # Get the collection and retrieve documents with embeddings
        collection = vector_store_temp.client.get_collection("interview_questions")
        
        # Get all documents with embeddings
        results = collection.get(include=["embeddings", "documents"])
        
        # Verify embeddings exist
        assert results["embeddings"] is not None
        assert len(results["embeddings"]) == 3
        
        # Verify each embedding has correct dimensions (768 for BAAI/bge-base-en-v1.5)
        for embedding in results["embeddings"]:
            assert embedding is not None
            assert len(embedding) == 768  # BGE-base-en-v1.5 embedding dimension
    
    def test_embedding_similarity(self, vector_store_temp, tmp_path):
        """Test that similar documents have similar embeddings"""
        # Create documents with similar and dissimilar content
        test_data = [
            {
                "id": "similar_1",
                "text": "Machine learning is a subset of artificial intelligence"
            },
            {
                "id": "similar_2", 
                "text": "Machine learning is part of AI and focuses on learning from data"
            },
            {
                "id": "different_1",
                "text": "The weather today is sunny and warm"
            }
        ]
        
        json_file = tmp_path / "similarity_test.json"
        with open(json_file, "w") as f:
            json.dump(test_data, f)
        
        ingestion = DataIngestion(vector_store_temp, reset_collections=True)
        ingestion.ingest_questions(str(json_file))
        
        # Query with first similar document
        results = vector_store_temp.query(
            collection_name="interview_questions",
            query_text="Machine learning is a subset of artificial intelligence",
            n_results=3
        )
        
        # The most similar document should be the query itself or the other ML document
        # The weather document should be least similar (highest distance)
        distances = results["distances"][0]
        ids = results["ids"][0]
        
        # First result should be most similar (lowest distance)
        assert distances[0] < distances[-1]
        
        # The dissimilar document should not be in top 2
        assert "different_1" not in ids[:2] or distances[ids.index("different_1")] > distances[0]
    
    def test_embedding_consistency(self, vector_store_temp, tmp_path):
        """Test that same text produces consistent embeddings"""
        test_data = [
            {"id": "test_1", "text": "What is gradient descent?"}
        ]
        
        json_file = tmp_path / "consistency_test.json"
        with open(json_file, "w") as f:
            json.dump(test_data, f)
        
        # Ingest once
        ingestion1 = DataIngestion(vector_store_temp, reset_collections=True)
        ingestion1.ingest_questions(str(json_file))
        
        collection = vector_store_temp.client.get_collection("interview_questions")
        results1 = collection.get(ids=["test_1"], include=["embeddings"])
        embedding1 = results1["embeddings"][0]
        
        # Ingest again with reset
        ingestion2 = DataIngestion(vector_store_temp, reset_collections=True)
        ingestion2.ingest_questions(str(json_file))
        
        # Refresh collection reference after reset
        collection = vector_store_temp.client.get_collection("interview_questions")
        results2 = collection.get(ids=["test_1"], include=["embeddings"])
        embedding2 = results2["embeddings"][0]
        
        # Embeddings should be identical
        assert np.allclose(embedding1, embedding2, rtol=1e-5)
    
    def test_query_returns_relevant_results(self, vector_store_temp, sample_questions, tmp_path):
        """Test that semantic search returns relevant results"""
        json_file = tmp_path / "questions.json"
        with open(json_file, "w") as f:
            json.dump(sample_questions, f)
        
        ingestion = DataIngestion(vector_store_temp, reset_collections=True)
        ingestion.ingest_questions(str(json_file))
        
        # Query for optimization-related content
        results = vector_store_temp.query(
            collection_name="interview_questions",
            query_text="optimization algorithms for training models",
            n_results=2
        )
        
        # Should return gradient descent question as most relevant
        top_result = results["documents"][0][0]
        assert "gradient descent" in top_result.lower()
        
        # Query for neural network content
        results = vector_store_temp.query(
            collection_name="interview_questions",
            query_text="neural network training process",
            n_results=2
        )
        
        # Should return backpropagation question
        top_result = results["documents"][0][0]
        assert "backpropagation" in top_result.lower() or "neural" in top_result.lower()
    
    def test_concepts_embedding_format(self, vector_store_temp, sample_concepts, tmp_path):
        """Test that concepts are embedded with proper text format"""
        json_file = tmp_path / "concepts.json"
        with open(json_file, "w") as f:
            json.dump(sample_concepts, f)
        
        ingestion = DataIngestion(vector_store_temp, reset_collections=True)
        ingestion.ingest_concepts(str(json_file))
        
        # Get documents to verify format
        collection = vector_store_temp.client.get_collection("ml_concepts")
        results = collection.get(include=["documents", "embeddings"])
        
        # Verify embeddings exist
        assert len(results["embeddings"]) == 2
        
        # Verify document format includes both name and explanation
        for doc in results["documents"]:
            assert ":" in doc  # Format is "name: explanation"
            assert len(doc) > 10  # Should have substantial content
        
        # Verify embeddings have correct dimensions (768 for BAAI/bge-base-en-v1.5)
        for embedding in results["embeddings"]:
            assert len(embedding) == 768  # BGE-base-en-v1.5 embedding dimension