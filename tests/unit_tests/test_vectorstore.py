import pytest
from src.data.vector_store import get_vector_store


TEST_COLLECTION = "test_collection"


@pytest.fixture(scope="function")
def vector_store():
    vs = get_vector_store()

    # Pre-clean: ensure clean state before test
    try:
        vs.delete_collection(TEST_COLLECTION)
    except Exception:
        pass

    yield vs

    # Post-clean: cleanup after test
    try:
        vs.delete_collection(TEST_COLLECTION)
    except Exception:
        pass


def test_create_collection(vector_store):
    collection = vector_store.create_collection(
        name="test_collection",
        metadata={"description": "Test collection"},
        reset=True
    )

    assert collection.name == "test_collection"


def test_add_documents(vector_store):
    vector_store.create_collection(
        name="test_collection",
        metadata={"description": "Test collection"},
        reset=True
    )

    documents = [
        "Gradient descent is an optimization algorithm",
        "Backpropagation computes gradients",
        "Overfitting occurs when model memorizes training data"
    ]
    metadatas = [
        {"topic": "optimization", "difficulty": "medium"},
        {"topic": "deep_learning", "difficulty": "hard"},
        {"topic": "evaluation", "difficulty": "easy"}
    ]
    ids = ["doc1", "doc2", "doc3"]

    vector_store.add_documents(
        collection_name="test_collection",
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    stats = vector_store.get_collection_stats("test_collection")
    assert stats["count"] == 3


def test_semantic_search(vector_store):
    vector_store.create_collection(
        name="test_collection",
        metadata={"description": "Test collection"},
        reset=True
    )

    vector_store.add_documents(
        collection_name="test_collection",
        documents=[
            "Gradient descent is an optimization algorithm",
            "Backpropagation computes gradients",
            "Overfitting occurs when model memorizes training data"
        ],
        metadatas=[
            {"topic": "optimization", "difficulty": "medium"},
            {"topic": "deep_learning", "difficulty": "hard"},
            {"topic": "evaluation", "difficulty": "easy"}
        ],
        ids=["doc1", "doc2", "doc3"]
    )

    results = vector_store.query(
        collection_name="test_collection",
        query_text="optimization algorithms",
        n_results=2
    )

    assert len(results["documents"][0]) == 2
    assert any("optimization" in doc.lower() for doc in results["documents"][0])


def test_filtered_search(vector_store):
    vector_store.create_collection(
        name="test_collection",
        metadata={"description": "Test collection"},
        reset=True
    )

    vector_store.add_documents(
        collection_name="test_collection",
        documents=[
            "Gradient descent is an optimization algorithm",
            "Backpropagation computes gradients",
            "Overfitting occurs when model memorizes training data"
        ],
        metadatas=[
            {"topic": "optimization", "difficulty": "medium"},
            {"topic": "deep_learning", "difficulty": "hard"},
            {"topic": "evaluation", "difficulty": "easy"}
        ],
        ids=["doc1", "doc2", "doc3"]
    )

    results = vector_store.query(
        collection_name="test_collection",
        query_text="training",
        n_results=5,
        where={"difficulty": "easy"}
    )

    docs = results["documents"][0]
    assert len(docs) >= 1
    assert any("training" in doc.lower() or "overfitting" in doc.lower() for doc in docs)


def test_collection_stats(vector_store):
    vector_store.create_collection(
        name="test_collection",
        metadata={"description": "Test collection"},
        reset=True
    )

    stats = vector_store.get_collection_stats("test_collection")

    assert stats["name"] == "test_collection"
    assert "count" in stats
    assert "metadata" in stats


def test_list_collections(vector_store):
    vector_store.create_collection(
        name="test_collection",
        metadata={"description": "Test collection"},
        reset=True
    )

    collections = vector_store.list_collections()
    assert "test_collection" in collections
