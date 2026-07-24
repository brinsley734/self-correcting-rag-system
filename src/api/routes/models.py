from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from src.core.model_registry import get_registry

router = APIRouter(tags=["models"])
router = APIRouter(tags=["models"])

class ModelUpdateModel(BaseModel):
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None
    reranker_model: Optional[str] = None

@router.get("/status")
async def get_model_status():
    registry = get_registry()
    models = await registry.get_active_models()
    return {
        "status": "success",
        "active_models": models
    }

@router.post("/swap")
async def swap_models(payload: ModelUpdateModel):
    registry = get_registry()
    await registry.set_active_models(
        embedding_model=payload.embedding_model,
        llm_model=payload.llm_model,
        reranker_model=payload.reranker_model
    )
    updated_models = await registry.get_active_models()
    return {
        "status": "success",
        "message": "Models swapped successfully. Note: Re-ingestion may be required if the embedding model changed.",
        "active_models": updated_models
    }