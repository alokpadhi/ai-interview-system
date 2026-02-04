import numpy as np
import pytest

from src.data.embeddings import get_embedding_service


@pytest.fixture(scope="session")
def embedding_service():
    """
    Session-scoped fixture to reuse the singleton embedding service
    across all tests (avoids reloading model).
    """
    return get_embedding_service()


def test_singleton_pattern(embedding_service):
    service1 = get_embedding_service()
    service2 = get_embedding_service()
    assert service1 is service2, "EmbeddingService should be a singleton"


def test_single_embedding_dimension(embedding_service):
    emb = embedding_service.embed("gradient descent optimization")
    assert len(emb) == 768, "BGE-base should output 768 dimensions"


def test_embedding_is_normalized(embedding_service):
    emb = embedding_service.embed("gradient descent optimization")
    norm = np.linalg.norm(emb)
    assert 0.99 < norm < 1.01, f"Embedding norm should be ~1.0, got {norm}"


def test_batch_embedding_shape(embedding_service):
    texts = [
        "What is gradient descent?",
        "Explain backpropagation",
        "Define overfitting"
    ]
    embeddings = embedding_service.embed_batch(texts)

    assert len(embeddings) == len(texts), "Should return one embedding per input text"
    assert all(len(e) == 768 for e in embeddings), "Each embedding should be 768-dim"


def test_similarity_computation(embedding_service):
    sim = embedding_service.compute_similarity(
        "machine learning algorithms",
        "ML models and methods"
    )
    assert sim > 0.5, f"Similar phrases should have high similarity, got {sim}"
