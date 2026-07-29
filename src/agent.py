#agent.py
import os
import logging
import time
import uuid
import json
import asyncio
from typing import List, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from src.generation import generate_answer_stream
print("Loaded generate_answer_stream from:", generate_answer_stream.__module__)
print("Signature:", generate_answer_stream.__code__.co_varnames)
from src.config import settings


logger = logging.getLogger(__name__)

class AutonomousRAGAgent:
    """Autonomous RAG Agent handling contextual retrieval, regional routing, and generation loops."""
    
    def __init__(self, qdrant_client: AsyncQdrantClient, collection_name: str):
        self.client = qdrant_client
        self.collection_name = collection_name

    async def query_and_learn_stream(
        self,
        question: str,
        top_chunks: List[models.ScoredPoint],
        encoder,
        model_name: str = "llama3.2"
    ):
        """Asynchronously streams generated answers while incorporating regional routing and context."""
        q_lower = question.lower()
        
        # --- Issue 3: Regional Keyword Routing (Scotland & Wales) ---
        scotland_markers = ["scotland", "edinburgh", "glasgow", "aberdeen", "dundee", "scottish"]
        welsh_markers = ["wales", "cardiff", "swansea", "newport", "welsh"]
        
        is_scotland = any(marker in q_lower for marker in scotland_markers)
        is_wales = any(marker in q_lower for marker in welsh_markers)
        
        regional_context_modifier = ""
        if is_scotland:
            regional_context_modifier = " [Context Focus: Scottish regional guidelines, employment standards, and opportunities in Scotland]"
            logger.info("Regional Keyword Routing: Detected Scotland context.")
        elif is_wales:
            regional_context_modifier = " [Context Focus: Welsh regional guidelines, employment standards, and opportunities in Wales]"
            logger.info("Regional Keyword Routing: Detected Wales context.")

        # Extract payload contents from top Qdrant chunks
        context_texts = []
        for hit in top_chunks:
            if hit.payload:
                text_content = hit.payload.get("content") or hit.payload.get("text") or ""
                if text_content:
                    context_texts.append(text_content)
                    
        combined_context = "\n\n".join(context_texts)
        if regional_context_modifier:
            combined_context += f"\n{regional_context_modifier}"

        # Stream generation chunks using the generation module
        async for chunk in generate_answer_stream(
            query=question,
            dense_results=top_chunks,
            model_name=model_name
        ):
            yield chunk