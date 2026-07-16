import os
from openai import OpenAI

# Initialize the inference client wrapper.
# Default base_url points to a local Ollama service, but can be overridden via env vars.
client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama") 
)

def assemble_context(dense_results, threshold=0.4):
    """
    Filters and joins text payloads from Qdrant hits into a single context string.
    Implements a strict semantic score threshold to prevent hallucination.
    """
    if not dense_results or dense_results[0].score < threshold:
        return None  # Trigger hallucination guardrail

    extracted_chunks = []
    for hit in dense_results:
        # Changed from "text" to "content" to match your index schema payload key
        text_content = hit.payload.get("content", "").strip()
        if text_content:
            extracted_chunks.append(f"- {text_content}")
            
    return "\n\n".join(extracted_chunks)


def build_final_prompt(query, context_string):
    """
    Wraps the query and structured context into explicit system instructions.
    """
    prompt = f"""You are a precise technical support assistant specializing in Kubernetes documentation.
Use ONLY the provided context blocks below to answer the user's query. 

Strict Rules:
1. If the answer cannot be found completely within the context, say exactly: "I cannot find the answer within the provided context."
2. Do not use outside knowledge or extrapolate past the facts listed.

---
PROVIDED CONTEXT:
{context_string}
---

User Query: {query}
Answer:"""
    return prompt.strip()


def generate_answer(query, dense_results, model_name="llama3.2"):
    """
    Coordinates context assembly, evaluates threshold checks, and executes 
    the completed prompt layout against the configured LLM generation engine.
    """
    # 1. Assemble the retrieved contexts from vector storage search
    context = assemble_context(dense_results, threshold=0.4)
    
    # 2. Short-circuit early if hallucination guardrail fallback was triggered
    if context is None:
        return "I cannot find the answer within the provided context."
        
    # 3. Build the structured engineering prompt layout
    final_prompt = build_final_prompt(query, context)
        
    try:
        # 4. Fire the inference request payload to the model provider
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.0,  # Zeroed out to optimize for deterministic fact extraction
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error executing generation inference block: {e}"