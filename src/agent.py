import os
import logging
import time
import uuid
import json as _json
from typing import List
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from src.generation import build_final_prompt, client, FALLBACK_MODELS
from src.config import settings

logger = logging.getLogger(__name__)


class AutonomousRAGAgent:
    """
    Autonomous RAG Agent handling contextual retrieval,
    regional routing, and streaming generation.

    Key design decision (dissertation note):
        The agent builds combined_context from Qdrant results
        and enriches it with a regional modifier BEFORE passing
        it to the LLM. This ensures Scotland/Wales/Belfast context
        is preserved in the prompt rather than being discarded
        by a second call to assemble_context() inside generation.py.
    """

    def __init__(self, qdrant_client: AsyncQdrantClient, collection_name: str):
        self.client = qdrant_client
        self.collection_name = collection_name

    async def query_and_learn_stream(
        self,
        question: str,
        top_chunks: List[models.ScoredPoint],
        encoder,
        model_name: str = "mistral"
    ):
        """
        Async generator that streams the LLM answer for a question with model failover.

        Steps:
          1. Detect UK regional context (Scotland, Wales, etc.)
          2. Extract text from Qdrant top_chunks
          3. Append regional modifier to context if detected
          4. Build prompt and stream from Ollama with model failover chain
          5. Yield each token as SSE JSON: data: {"content": "..."}
        """
        q_lower = question.lower()

        # ── Regional Keyword Routing ──────────────────────────────────
        # Detect UK regions to append focused context hints to the prompt.
        # This improves answer relevance for regional salary/job questions.

        scotland_markers = [
            "scotland", "edinburgh", "glasgow",
            "aberdeen", "dundee", "scottish"
        ]
        welsh_markers = [
            "wales", "cardiff", "swansea", "newport", "welsh"
        ]
        northern_ireland_markers = [
            "northern ireland", "belfast", "derry", "londonderry"
        ]
        english_region_markers = [
            "manchester", "birmingham", "leeds", "liverpool",
            "bristol", "sheffield", "newcastle", "nottingham",
            "leicester", "coventry", "reading", "southampton",
            "oxford", "cambridge", "exeter", "plymouth", "hull",
            "sunderland", "brighton", "norwich"
        ]

        regional_context_modifier = ""

        if any(m in q_lower for m in scotland_markers):
            regional_context_modifier = (
                "\n[Context Focus: Scottish regional job market. "
                "Edinburgh and Glasgow are the primary tech hubs in Scotland. "
                "Salaries are typically 10-15% below London but cost of living "
                "is significantly lower. Apply Scottish employment law context.]"
            )
            logger.info("Regional Keyword Routing: Detected Scotland context.")

        elif any(m in q_lower for m in welsh_markers):
            regional_context_modifier = (
                "\n[Context Focus: Welsh regional job market. "
                "Cardiff is the primary tech hub in Wales. "
                "Salaries typically range £30k-£78k for software engineers. "
                "Welsh Government and financial services are major employers. "
                "Cost of living is 30-35% lower than London.]"
            )
            logger.info("Regional Keyword Routing: Detected Wales context.")

        elif any(m in q_lower for m in northern_ireland_markers):
            regional_context_modifier = (
                "\n[Context Focus: Northern Ireland job market. "
                "Belfast is the primary tech hub. "
                "Salaries typically range £28k-£65k for software engineers. "
                "Lower competition than London with significantly lower cost of living. "
                "Same UK Skilled Worker visa rules apply as rest of UK.]"
            )
            logger.info("Regional Keyword Routing: Detected Northern Ireland context.")

        elif any(m in q_lower for m in english_region_markers):
            regional_context_modifier = (
                "\n[Context Focus: English regional job market outside London. "
                "Salaries are typically 20-30% below London rates but with "
                "lower cost of living. Strong hybrid and remote opportunities available.]"
            )
            logger.info("Regional Keyword Routing: Detected English region context.")

        # ── Extract context from Qdrant chunks ────────────────────────
        context_texts = []
        for hit in top_chunks:
            if hit.payload:
                text_content = (
                    hit.payload.get("content") or
                    hit.payload.get("text") or ""
                ).strip()
                if text_content:
                    context_texts.append(text_content)

        combined_context = "\n\n".join(context_texts)

        # Append regional modifier AFTER corpus context
        # so the LLM sees both the retrieved data AND the regional hint
        if regional_context_modifier:
            combined_context += regional_context_modifier

        # ── Fallback if no context ────────────────────────────────────
        if not combined_context:
            yield 'data: {"content": "I could not find relevant information for this query in the knowledge base."}\n\n'
            return

        # ── Build prompt and stream from LLM with Failover ───────────
        final_prompt = build_final_prompt(question, combined_context)

        # Build model attempt order using the explicit priority chain
        models_to_try = [m for m in FALLBACK_MODELS]
        if model_name and model_name not in models_to_try:
            models_to_try.insert(0, model_name)

        last_error = None
        for attempt_model in models_to_try:
            response = None
            try:
                logger.info(f"[LLM] Attempting model: {attempt_model}")
                response = client.chat.completions.create(
                    model=attempt_model,
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=0.0,
                    max_tokens=400,
                    stream=True,
                    timeout=30,
                )

                # Signal which model is responding
                yield f'data: {{"content": "", "model": "{attempt_model}"}}\n\n'

                chunk_count = 0
                for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        chunk_count += 1
                        yield f'data: {{"content": {_json.dumps(delta)}}}\n\n'

                if chunk_count > 0:
                    logger.info(f"[LLM] Success with model: {attempt_model}")
                    return  # Success — stop trying other models

            except GeneratorExit:
                return
            except Exception as e:
                last_error = str(e)
                logger.error(f"[LLM] Model {attempt_model} failed: {e}")
                logger.info(f"[LLM] Trying next fallback...")
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass
                continue  # Try next model

        # All models failed
        error_msg = f"All models unavailable. Last error: {last_error}"
        logger.error(f"[LLM] {error_msg}")
        yield f'data: {{"content": "Service temporarily unavailable. Please try again in a moment."}}\n\n'