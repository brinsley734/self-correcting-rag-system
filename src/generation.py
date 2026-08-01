import json
import os
import logging
import requests
from openai import OpenAI
from duckduckgo_search import DDGS

# --- BEGIN MONKEY-PATCH FOR OPENAI/PYDANTIC BUG ---
import openai._compat as _compat
_original_model_dump = _compat.model_dump
def _patched_model_dump(model, *, exclude_unset=False, by_alias=None, **kw):
    return _original_model_dump(
        model,
        exclude_unset=exclude_unset,
        by_alias=bool(by_alias) if by_alias is not None else False,
        **kw
    )
_compat.model_dump = _patched_model_dump
# --- END MONKEY-PATCH ---

logging.getLogger("openai").setLevel(logging.INFO)

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama")
)

FALLBACK_MODELS = [
    "mistral",
    "llama3.2",
    "phi3",
    "llama3",
    "gemma",
]


def is_model_available(model_name: str) -> bool:
    """
    Check if a model is available in Ollama.
    Calls GET http://localhost:11434/api/tags
    and checks if model_name is in the list.
    Returns True if available, False if not.
    """
    try:
        resp = requests.get(
            "http://localhost:11434/api/tags",
            timeout=3
        )
        if resp.status_code == 200:
            models = [m["name"].split(":")[0]
                      for m in resp.json().get("models", [])]
            return model_name in models
    except Exception:
        pass
    return False


def assemble_context(dense_results, threshold=0.3, query=None):
    """
    Filters and joins text payloads from Qdrant hits into a single context string.
    Implements a semantic score threshold and keyword relevance check to prevent hallucination.
    """
    if not dense_results:
        return None  # Trigger fallback search

    first_result = dense_results[0]
    score = getattr(first_result, "score", None)
    if score is not None and score < threshold:
        return None  # Trigger fallback search

    extracted_chunks = []
    for hit in dense_results:
        payload = getattr(hit, "payload", hit)
        if isinstance(payload, dict):
            text_content = (payload.get("content") or payload.get("text") or "").strip()
        else:
            text_content = str(payload).strip()

        if text_content:
            extracted_chunks.append(f"- {text_content}")

    if not extracted_chunks:
        return None

    # Relevance check: If query provided, check if retrieved chunks match query intent
    if query:
        query_words = set(query.lower().split())
        stop_words = {"what", "how", "is", "are", "the", "a", "an", "in", "for", 
                      "to", "of", "and", "or", "do", "i", "can", "you", "me", "with"}
        meaningful_words = query_words - stop_words

        combined_text = " ".join(extracted_chunks).lower()
        matches = sum(1 for word in meaningful_words if word in combined_text)
        
        # If fewer than 2 meaningful query words appear in retrieved context → fallback
        if len(meaningful_words) > 2 and matches < 2:
            print(f"[RAG] Low relevance ({matches} keyword matches) — triggering web search")
            return None

    return "\n\n".join(extracted_chunks)


def search_web_for_context(query, max_results=3):
    """
    Performs a live web search to gather context when local vector retrieval fails,
    using multiple DDGS backend options for resilience.
    """
    backends = ["api", "html", "lite"]
    for backend in backends:
        try:
            with DDGS() as ddgs:
                results = [r.get("body", "") for r in ddgs.text(query, max_results=max_results, backend=backend) if r.get("body")]
                if results:
                    print(f"[Web Fallback] Successfully fetched web context using backend '{backend}' for: {query}")
                    return "\n\n".join([f"- {res}" for res in results])
        except Exception as e:
            print(f"[Web Fallback Error] Backend '{backend}' failed: {e}")
            continue
    return None


def build_final_prompt(query, context_string):
    """
    Wraps the query and structured context into explicit system instructions.
    """
    prompt = f"""You are a precise career, technical, and visa advisor assistant.
Use ONLY the provided context blocks below to answer the user's query accurately. Do not invent details not present in the context.

---
PROVIDED CONTEXT:
{context_string}
---

User Query: {query}
Answer:"""
    return prompt.strip()


def build_web_search_prompt(query, context_string):
    """
    Prompt for web search results — less restrictive than corpus prompt.
    Tells the LLM to synthesise from web sources rather than refuse.
    Used by international_stream_generator in main.py.
    """
    prompt = f"""You are a helpful career, salary, and job market advisor.
Use the web search results below to answer the user's question.
Include specific salary figures, currencies, cities, and any relevant
details from the search results. If the results mention ranges, include them.

---
WEB SEARCH RESULTS:
{context_string}
---

User Query: {query}
Answer:"""
    return prompt.strip()


def generate_answer(query, dense_results, model_name="mistral"):
    """
    Non-streaming answer generation with fallback logic.
    """
    context = assemble_context(dense_results, threshold=0.3, query=query)

    if context is None:
        print("Local context missing or below threshold. Searching web dynamically...")
        context = search_web_for_context(query)

    if not context:
        return "I could not find relevant information locally or via web search to answer your query."

    final_prompt = build_final_prompt(query, context)

    # Build model attempt order
    models_to_try = [model_name] + [
        m for m in FALLBACK_MODELS if m != model_name
    ]
    seen = set()
    models_to_try = [
        m for m in models_to_try
        if not (m in seen or seen.add(m))
    ]

    last_error = None
    for attempt_model in models_to_try:
        try:
            print(f"[LLM] Attempting model: {attempt_model}")
            response = client.chat.completions.create(
                model=attempt_model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.0,
                max_tokens=400,
                timeout=30,
            )
            print(f"[LLM] Success with model: {attempt_model}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = str(e)
            print(f"[LLM] Model {attempt_model} failed: {e}")
            print(f"[LLM] Trying next fallback...")
            continue

    return f"Service temporarily unavailable. Last error: {last_error}"


async def generate_answer_stream(query, dense_results, model_name="mistral"):
    """
    Async generator that streams LLM response chunks in SSE-compatible JSON format with model failover.
    """
    context = assemble_context(dense_results, threshold=0.3, query=query)

    if context is None:
        print("Local context missing or below threshold. Searching web dynamically...")
        context = search_web_for_context(query)

    if not context:
        yield 'data: {"content": "I could not find relevant information locally or via web search to answer your query."}\n\n'
        return

    final_prompt = build_final_prompt(query, context)

    # Build model attempt order
    models_to_try = [model_name] + [
        m for m in FALLBACK_MODELS if m != model_name
    ]
    seen = set()
    models_to_try = [
        m for m in models_to_try
        if not (m in seen or seen.add(m))
    ]

    last_error = None
    for attempt_model in models_to_try:
        response = None
        try:
            print(f"[LLM] Attempting model: {attempt_model}")
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
                    yield f'data: {{"content": {json.dumps(delta)}}}\n\n'

            if chunk_count > 0:
                print(f"[LLM] Success with model: {attempt_model}")
                return  # Success — stop trying other models

        except GeneratorExit:
            return
        except Exception as e:
            last_error = str(e)
            print(f"[LLM] Model {attempt_model} failed: {e}")
            print(f"[LLM] Trying next fallback...")
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
            continue  # Try next model

    # All models failed
    error_msg = f"All models unavailable. Last error: {last_error}"
    print(f"[LLM] {error_msg}")
    yield f'data: {{"content": "Service temporarily unavailable. Please try again in a moment."}}\n\n'