import os
import json
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.processing.chunker import HierarchicalChunker
from src.vector_store.qdrant_index import QdrantIndexer
from src.search.bm25_index import BM25StorageEngine

# Configure logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IngestionPipeline:
    def __init__(self, input_dir: str = "./data/raw/ukjobs"):
        self.input_dir = input_dir
        print("DEBUG: Initializing SentenceTransformer...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("DEBUG: SentenceTransformer loaded successfully.")
        
        print("DEBUG: Initializing Qdrant...")
        # Updated collection name for your job search data
        self.qdrant_indexer = QdrantIndexer(collection_name="uk_jobs_data")
        print("DEBUG: Qdrant initialized.")
        
        self.chunker = HierarchicalChunker(parent_size=2000, child_size=400, overlap=50)
        self.bm25_engine = BM25StorageEngine(index_path="storage/bm25_index.pkl")

    def load_raw_documents(self) -> List[Dict[str, Any]]:
        abs_input_dir = os.path.abspath(self.input_dir)
        print(f"DEBUG: Looking for files in: {abs_input_dir}")
        
        documents = []
        if not os.path.exists(abs_input_dir):
            logger.error(f"Target input directory missing: {abs_input_dir}")
            return documents

        files = os.listdir(abs_input_dir)
        print(f"DEBUG: Found {len(files)} files in directory.")

        for file_name in files:
            if file_name.endswith(".json"):
                file_path = os.path.join(abs_input_dir, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        doc_data = json.load(f)
                        
                        content = doc_data.get("raw_content") or doc_data.get("content") or doc_data.get("text")
                        
                        if content:
                            documents.append({
                                "raw_content": content,
                                "source_url": doc_data.get("source_url") or "",
                                "title": doc_data.get("title") or file_name
                            })
                except Exception as e:
                    logger.error(f"Error reading file {file_name}: {str(e)}")
        
        logger.info(f"Loaded {len(documents)} raw documentation entries from disk storage.")
        return documents

    def run(self):
        """Executes the full pipeline: Load -> Chunk -> Dense Vector Upsert -> BM25 Index."""
        logger.info("Initializing full master ingestion workflow...")
        
        # 1. Load data
        raw_docs = self.load_raw_documents()
        if not raw_docs:
            logger.error("No valid documents found. Terminating pipeline execution.")
            return

        # 2. Process hierarchical manifest layout
        full_manifest: List[Dict[str, Any]] = []
        for doc in raw_docs:
            manifest_segments = self.chunker.chunk_document(doc)
            full_manifest.extend(manifest_segments)

        logger.info(f"Hierarchical processing complete. Total processing units mapped: {len(full_manifest)} parents.")

        # 3. Stream to Qdrant vector database space
        logger.info("Streaming dense vector representations to Qdrant backend server...")
        self.qdrant_indexer.index_manifest(full_manifest)

        # 4. Build and save sparse lookups matrix
        logger.info("Assembling structural keyword indices for BM25 persistence...")
        self.bm25_engine.build_and_save_index(full_manifest)

        logger.info("Ingestion pipeline executed successfully. All vectors and sparse tables committed.")

if __name__ == "__main__":
    pipeline = IngestionPipeline(input_dir="./data/raw/ukjobs")
    pipeline.run()