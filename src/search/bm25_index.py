import os
import pickle
import logging
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class BM25StorageEngine:
    def __init__(self, index_path: str = "storage/bm25_index.pkl"):
        self.index_path = index_path
        self.bm25 = None
        self.doc_manifest = []  # Tracks child chunks mapped to indexes
        
        # Ensure our local storage directory layer exists safely
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self.load_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple helper tokenizer converting strings into filtered word lists."""
        return text.lower().split()

    def build_and_save_index(self, manifest: List[Dict[str, Any]]):
        """Builds a BM25 index over all child fragments and flushes it to disk."""
        corpus = []
        new_doc_entries = []
        
        for parent in manifest:
            parent_id = parent["parent_id"]
            parent_content = parent["parent_content"]
            
            for child in parent["child_chunks"]:
                content_text = child["content"]
                corpus.append(self._tokenize(content_text))
                
                # Keep tracking links layout to pair seamlessly with dense matches later
                new_doc_entries.append({
                    "child_id": child["child_id"],
                    "parent_ref_id": parent_id,
                    "parent_context_body": parent_content,
                    "content": content_text,
                    **child["metadata"]
                })
                
        if corpus:
            self.doc_manifest.extend(new_doc_entries)
            self.bm25 = BM25Okapi(corpus)
            
            # Persist data snapshot binary to disk space volume map
            with open(self.index_path, "wb") as f:
                pickle.dump({"bm25": self.bm25, "doc_manifest": self.doc_manifest}, f)
            logger.info(f"Successfully constructed and saved BM25 matrix tracking {len(corpus)} sparse items.")

    def load_index(self):
        """Loads a pre-built sparse lookup index from disk if it exists."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25 = data["bm25"]
                    self.doc_manifest = data["doc_manifest"]
                logger.info("Successfully loaded active BM25 index artifact straight from storage partition.")
            except Exception as e:
                logger.error(f"Failed reloading saved BM25 inverted indices layout: {str(e)}")