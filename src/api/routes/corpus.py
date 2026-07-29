import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from qdrant_client import AsyncQdrantClient
from src.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

async_qdrant = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
COLLECTION_NAME = "uk_jobs_data"

@router.get("/learned-entries")
async def get_learned_entries() -> Dict[Any, Any]:
    """
    Retrieves all corpus entries dynamically learned via the DuckDuckGo web search fallback.
    """
    try:
        response = await async_qdrant.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            with_vectors=False,
            limit=100
        )
        
        records = response[0] if isinstance(response, tuple) else getattr(response, "points", [])

        entries = []
        for record in records:
            payload = record.payload or {}
            source = payload.get("source", "")
            # Catch entries from web fallback or containing learned timestamps
            if source == "web_learned" or "learned_at" in payload:
                entries.append({
                    "id": record.id,
                    "content": payload.get("content") or payload.get("text"),
                    "url": payload.get("url"),
                    "learned_at": payload.get("learned_at")
                })

        return {
            "status": "success",
            "total_learned_entries": len(entries),
            "entries": entries
        }

    except Exception as e:
        logger.error(f"Failed to fetch learned entries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))