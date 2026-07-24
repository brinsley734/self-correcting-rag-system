import asyncio
from typing import Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder

logger = logging.getLogger(__name__)

class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._lock = asyncio.Lock()
        
        # Default model identifiers
        self.current_embedding_model_name: str = "all-MiniLM-L6-v2"
        self.current_llm_model: str = "mistral"
        self.current_reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
        
        # Loaded model instances
        self.current_embedding_model: Optional[SentenceTransformer] = None
        self.current_reranker_model: Optional[CrossEncoder] = None
        
        self._initialized = False

    def initialise(self) -> None:
        """Synchronous initialization hook for application startup to load local models into memory."""
        if self._initialized:
            return
        logger.info("Loading initial local embedding and reranking models...")
        self.current_embedding_model = SentenceTransformer(self.current_embedding_model_name)
        self.current_reranker_model = CrossEncoder(self.current_reranker_model_name)
        self._initialized = True
        logger.info("ModelRegistry initialized successfully with active local model instances.")

    async def get_active_models(self) -> Dict[str, str]:
        async with self._lock:
            return {
                "embedding_model": self.current_embedding_model_name,
                "llm_model": self.current_llm_model,
                "reranker_model": self.current_reranker_model_name
            }

    async def set_active_models(
        self, 
        embedding_model: Optional[str] = None, 
        llm_model: Optional[str] = None,
        reranker_model: Optional[str] = None
    ):
        async with self._lock:
            if embedding_model and embedding_model != self.current_embedding_model_name:
                logger.info(f"Swapping local embedding model to: {embedding_model}")
                self.current_embedding_model = SentenceTransformer(embedding_model)
                self.current_embedding_model_name = embedding_model
                
            if llm_model:
                self.current_llm_model = llm_model
                
            if reranker_model and reranker_model != self.current_reranker_model_name:
                logger.info(f"Swapping local reranker model to: {reranker_model}")
                self.current_reranker_model = CrossEncoder(reranker_model)
                self.current_reranker_model_name = reranker_model

def get_registry() -> ModelRegistry:
    return ModelRegistry()