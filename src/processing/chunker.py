from typing import List, Dict, Any
import uuid
from src.processing.extractor import MetadataExtractor

class HierarchicalChunker:
    def __init__(self, parent_size: int = 2000, child_size: int = 400, overlap: int = 50):
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap
        # Initialize extractor
        self.extractor = MetadataExtractor()

    def _sliding_window(self, text: str, size: int, overlap: int) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + size
            chunks.append(text[start:end])
            start += (size - overlap)
        return chunks

    def chunk_document(self, doc_payload: Dict[str, str]) -> List[Dict[str, Any]]:
        raw_text = doc_payload.get("raw_content", "")
        source_url = doc_payload.get("source_url", "")
        doc_title = doc_payload.get("title", "")
        
        parent_texts = self._sliding_window(raw_text, self.parent_size, self.overlap)
        processed_manifest = []

        for p_idx, p_text in enumerate(parent_texts):
            parent_id = str(uuid.uuid4())
            child_texts = self._sliding_window(p_text, self.child_size, self.overlap)
            
            children_meta = []
            for c_idx, c_text in enumerate(child_texts):
                # Run extraction on child chunk content
                extracted_tags = self.extractor.extract_metadata(c_text)
                
                children_meta.append({
                    "child_id": str(uuid.uuid4()),
                    "parent_ref_id": parent_id,
                    "content": c_text,
                    "metadata": {
                        "source_url": source_url,
                        "title": doc_title,
                        "hierarchy_position": f"P{p_idx}-C{c_idx}",
                        # Inject clean extracted entities
                        "entities": extracted_tags["kubernetes_components"],
                        "intents": extracted_tags["suggested_actions"]
                    }
                })
                
            processed_manifest.append({
                "parent_id": parent_id,
                "parent_content": p_text,
                "child_chunks": children_meta,
                "metadata": {
                    "source_url": source_url,
                    "title": doc_title,
                    "parent_index": p_idx
                }
            })
            
        return processed_manifest