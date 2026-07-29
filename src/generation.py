import json
import os
import logging
from openai import OpenAI
from ddgs import DDGS

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


def assemble_context(dense_results, threshold=0.4):
    """
    Filters and joins text payloads from Qdrant hits into a single context string.
    Implements a strict semantic score threshold to prevent hallucination.
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

    return "\n\n".join(extracted_chunks) if extracted_chunks else None


def search_web_for_context(query, max_results=3):
    """
    Performs a live web search to gather context when local vector retrieval fails.
    """
    try:
        with DDGS() as ddgs:
            results = [r["body"] for r in ddgs.text(query, max_results=max_results)]
            if results:
                print(f"[Web Fallback] Successfully fetched web context for: {query}")
                return "\n\n".join([f"- {res}" for res in results])
    except Exception as e:
        print(f"[Web Fallback Error] {e}")
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


def generate_answer(query, dense_results, model_name="llama3.2"):
    """
    Non-streaming answer generation. Used for cache population and batch evaluation.
    """
    context = assemble_context(dense_results, threshold=0.4)

    if context is None:
        print("Local context missing or below threshold. Searching web dynamically...")
        context = search_web_for_context(query)

    if not context:
        return "I could not find relevant information locally or via web search to answer your query."

    final_prompt = build_final_prompt(query, context)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.0,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error executing generation inference block: {e}"


async def generate_answer_stream(query, dense_results, model_name="llama3.2"):
    """
    Async generator that streams LLM response chunks in SSE-compatible JSON format.

    Each yielded chunk is formatted as:
        data: {"content": "text fragment here"}\n\n

    The frontend reads each chunk, strips "data: ", parses JSON,
    and appends parsed.content to the displayed message.

    Flow:
        1. Assemble context from Qdrant results
        2. If context score below threshold → web search fallback
        3. Build prompt and stream from Ollama via OpenAI-compatible API
        4. Yield each token delta as SSE JSON chunk
        5. Clean up stream on completion or client disconnect
    """
    context = assemble_context(dense_results, threshold=0.4)

    if context is None:
        print("Local context missing or below threshold. Searching web dynamically...")
        context = search_web_for_context(query)

    if not context:
        yield 'data: {"content": "I could not find relevant information locally or via web search to answer your query."}\n\n'
        return

    final_prompt = build_final_prompt(query, context)
    response = None

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.0,
            max_tokens=400,
            stream=True
        )

        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                # Yield SSE-formatted JSON so frontend can parse parsed.content
                yield f'data: {{"content": {json.dumps(delta)}}}\n\n'

    except GeneratorExit:
        # Client disconnected — exit cleanly without error
        pass
    except Exception as e:
        yield f'data: {{"content": "Error executing generation inference block: {str(e)}"}}\n\n'
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass