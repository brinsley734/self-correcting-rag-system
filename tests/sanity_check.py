import pytest
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Fixtures allow for efficient resource management across tests
@pytest.fixture(scope="module")
def qdrant():
    return QdrantClient(url="http://localhost:6333")

@pytest.fixture(scope="module")
def encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def test_qdrant_connectivity(qdrant):
    """Test 1: Verify Qdrant connection and collection existence."""
    collection_info = qdrant.get_collection("k8s_docs")
    assert collection_info is not None
    assert collection_info.points_count > 0, "Collection 'k8s_docs' is empty"

def test_model_initialization(encoder):
    """Test 2: Ensure the embedding model loads correctly."""
    assert encoder is not None, "SentenceTransformer model failed to initialize"

def test_semantic_retrieval_performance(qdrant, encoder):
    """Test 3: Validate that the dense retrieval layer returns relevant results."""
    semantic_query = "scaling app pods automatically based on system demand"
    query_vector = encoder.encode(semantic_query).tolist()
    
    dense_results = qdrant.search(
        collection_name="k8s_docs",
        query_vector=query_vector,
        limit=3
    )
    
    assert len(dense_results) > 0, "No results returned for semantic query"
    assert dense_results[0].score > 0.4, f"Semantic similarity score too low: {dense_results[0].score}"

# Allows manual execution: python tests/sanity_check.py
if __name__ == "__main__":
    print("Running diagnostics via pytest.main()...")
    pytest.main([__file__])