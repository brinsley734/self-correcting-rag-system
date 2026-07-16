from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging

# Import the dynamic application settings
from src.config import settings

logger = logging.getLogger(__name__)

class QdrantIndexer:
    def __init__(self, collection_name: str = "k8s_docs"): 
        self.collection_name = collection_name
        self.client = None # Client is initialized as None to prevent hanging on import
        
        # Load a lightning-fast local 384-dimensional embedding transformer engine
        # Forced device to "cpu" to prevent macOS Apple Silicon driver segmentation faults
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def _ensure_connected(self):
        """Lazy connection: only connect when we actually need to perform operations."""
        if self.client is None:
            logger.info("Establishing lazy connection to Qdrant...")
            self.client = QdrantClient(
                host=settings.QDRANT_HOST, 
                port=settings.QDRANT_PORT,
                timeout=60.0
            )
            self._setup_collection()

    def _setup_collection(self):
        """Creates the internal storage collection space schema layout if missing."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                logger.info(f"Configuring fresh Qdrant collection space: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
        except Exception as e:
            logger.error(f"Failed standing up Qdrant storage node layer: {str(e)}")

    def index_manifest(self, manifest: List[Dict[str, Any]]):
        """Encodes child segments into dense vectors and upserts them to database storage using chunked batches."""
        self._ensure_connected() # Establish connection just before use
        
        points = []
        for parent in manifest:
            parent_id = parent["parent_id"]
            parent_content = parent["parent_content"]
            
            for child in parent["child_chunks"]:
                child_id = child["child_id"]
                content_text = child["content"]
                
                # Calculate vector multi-dimensional representation metrics array
                vector = self.encoder.encode(content_text).tolist()
                
                # Flatten meta metrics to optimize vector search filtering payloads
                payload = {
                    "parent_ref_id": parent_id,
                    "parent_context_body": parent_content,
                    "content": content_text,
                    **child["metadata"]
                }
                
                points.append(PointStruct(
                    id=child_id,
                    vector=vector,
                    payload=payload
                ))
                
        if points:
            batch_size = 100
            total_points = len(points)
            logger.info(f"Beginning batched upsert of {total_points} serialized points into '{self.collection_name}'...")
            
            for i in range(0, total_points, batch_size):
                batch = points[i:i + batch_size]
                try:
                    self.client.upsert(collection_name=self.collection_name, points=batch)
                    logger.debug(f"Successfully upserted points index range [{i} : {min(i + batch_size, total_points)}]")
                except Exception as e:
                    logger.error(f"Failed during batch upsert operation at index window starting at {i}: {str(e)}")
                    raise e
                    
            logger.info(f"Successfully committed {total_points} vectorized nodes to Qdrant storage context.")