from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class QdrantIndexer:
    def __init__(self, collection_name: str = "k8s_docs"):
        # Connect to the qdrant container service running in our Docker network space
        self.client = QdrantClient(host="qdrant", port=6333)
        self.collection_name = collection_name
        
        # Load a lightning-fast local 384-dimensional embedding transformer engine
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
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
        """Encodes child segments into dense vectors and upserts them to database storage."""
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
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Successfully committed {len(points)} vectorized nodes to Qdrant storage context.")