"""
Pytest configuration and shared fixtures.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from src.data.embeddings import EmbeddingService, get_embedding_service
from src.data.vector_store import VectorStore
from src.utils.config import get_settings


@pytest.fixture(scope="session")
def embedding_service():
    """
    Session-scoped embedding service.
    Loaded once for entire test session (faster tests).
    """
    return get_embedding_service()


@pytest.fixture(scope="function")
def vector_store_temp():
    """
    Function-scoped vector store with temporary directory.
    Creates fresh ChromaDB for each test (isolation).
    Cleanup after test completes.
    """
    tempdir = tempfile.mkdtemp()
    vs = VectorStore(persist_directory=tempdir)
    yield vs
    shutil.rmtree(tempdir)


@pytest.fixture
def sample_questions():
    """Sample questions for testing"""
    return [
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


@pytest.fixture
def sample_concepts():
    """Sample concepts for testing"""
    return [
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