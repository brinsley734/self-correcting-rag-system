import sys
# Force this to be a standard synchronous run to bypass uvloop/asyncio conflicts
print("Testing library imports...")
from sentence_transformers import SentenceTransformer
print("Model import successful.")

from qdrant_client import AsyncQdrantClient
print("Qdrant import successful.")

print("All imports passed. System is stable.")