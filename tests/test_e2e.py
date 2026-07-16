import sys
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Ensure absolute import path maps down to your 'src' layer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.generation import generate_answer

def run_end_to_end_test():
    print("=== STARTING END-TO-END RAG PIPELINE EVALUATION ===")
    
    # 1. Initialize retrieval infrastructure
    print("\n[1/3] Fetching vector matching models...")
    qdrant_client = QdrantClient(url="http://localhost:6333", check_compatibility=False)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 2. Define a query present inside your 1,019 indexed points
    test_query = "How do we scale app pods automatically based on demand?"
    print(f"\n[2/3] Querying Qdrant storage contexts for: '{test_query}'")
    
    query_vector = model.encode(test_query).tolist()
    dense_results = qdrant_client.search(
        collection_name="k8s_docs",
        query_vector=query_vector,
        limit=3
    )
    
    # 3. Pass through generation pipelines and guards
    print("\n[3/3] Feeding contexts to prompt assembly and LLM engine...")
    # Switched to the lighter llama3.2 parameter model to ensure stability on your MacBook Air
    response = generate_answer(query=test_query, dense_results=dense_results, model_name="llama3.2")
    
    print("\n=== SYSTEM INFERENCE RESPONSE ===")
    print(response)
    print("=================================")

if __name__ == "__main__":
    run_end_to_end_test()