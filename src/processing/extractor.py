from pydantic import BaseModel
from typing import List, Dict, Any
import re
# Import your MetadataExtractor here
# from src.processing.extractor import MetadataExtractor 

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_title: str
    content: str
    metadata: Dict[str, Any]

# Paste or reference your class here
class MetadataExtractor:
    def extract_metadata(self, text: str) -> Dict[str, List[str]]:
        found_components = []
        found_actions = []
        
        k8s_terms = ["pod", "service", "deployment", "replica", "ingress", "kubelet", "cluster", "node"]
        for term in k8s_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text.lower()):
                found_components.append(term)
                
        action_terms = ["create", "apply", "delete", "expose", "scale", "configure"]
        for action in action_terms:
            if re.search(r'\b' + re.escape(action) + r'\b', text.lower()):
                found_actions.append(action)

        return {
            "kubernetes_components": list(set(found_components)),
            "suggested_actions": list(set(found_actions))
        }

def split_text_recursive(text: str, max_chars: int = 1000) -> List[str]:
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_chars:
            chunks.append(" ".join(current_chunk))
            overlap_words = current_chunk[-15:] if len(current_chunk) > 15 else []
            current_chunk = overlap_words + [word]
            current_length = sum(len(w) + 1 for w in current_chunk)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def generate_chunks(doc_id: str, title: str, full_text: str, file_metadata: Dict[str, Any]) -> List[DocumentChunk]:
    """Generates a list of structured chunks, dynamically enriching each with extracted entity tags."""
    text_segments = split_text_recursive(full_text)
    extractor = MetadataExtractor()
    chunks = []
    
    for i, segment in enumerate(text_segments):
        # 1. Base metadata passed down from the file level
        chunk_metadata = file_metadata.copy()
        chunk_metadata["chunk_index"] = i
        
        # 2. Run your specific text-based regex analyzer
        semantic_tags = extractor.extract_metadata(segment)
        
        # 3. Merge them cleanly into the chunk's core metadata dict
        chunk_metadata.update(semantic_tags)
        
        chunks.append(DocumentChunk(
            chunk_id=f"{doc_id}_chunk_{i}",
            doc_title=title,
            content=segment,
            metadata=chunk_metadata
        ))
        
    return chunks