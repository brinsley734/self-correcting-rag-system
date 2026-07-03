from typing import Dict, List
import re

class MetadataExtractor:
    def extract_metadata(self, text: str) -> Dict[str, List[str]]:
        """
        Analyzes a segment of text to pull out core technical keywords, 
        potential target components, and command types.
        """
        # Simple rule-based heuristic tags for demonstration
        found_components = []
        found_actions = []
        
        # Look for typical Kubernetes terms
        k8s_terms = ["pod", "service", "deployment", "replica", "ingress", "kubelet", "cluster", "node"]
        for term in k8s_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text.lower()):
                found_components.append(term)
                
        # Look for operational commands or focus keywords
        action_terms = ["create", "apply", "delete", "expose", "scale", "configure"]
        for action in action_terms:
            if re.search(r'\b' + re.escape(action) + r'\b', text.lower()):
                found_actions.append(action)

        return {
            "kubernetes_components": list(set(found_components)),
            "suggested_actions": list(set(found_actions))
        }