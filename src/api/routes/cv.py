import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
from io import BytesIO
from docx import Document
from src.core.model_registry import get_registry
from qdrant_client import AsyncQdrantClient
from src.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

async_qdrant = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
COLLECTION_NAME = "uk_jobs_data"

def extract_text_from_file(filename: str, file_bytes: bytes) -> str:
    """Extracts text content from uploaded document formats (.docx)."""
    if filename.endswith(".docx"):
        doc = Document(BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    elif filename.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(BytesIO(file_bytes))
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise HTTPException(status_code=400, detail="Failed to parse PDF file format.")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .docx or .pdf file.")

@router.post("/analyse")
async def analyse_cv(file: UploadFile = File(...)) -> Dict[Any, Any]:
    """
    Parses an uploaded CV file, embeds its text, searches the job corpus,
    and returns matched jobs, skills found, missing keywords, and visa compatibility.
    """
    registry = get_registry()
    encoder = registry.current_embedding_model
    
    if encoder is None:
        raise HTTPException(status_code=503, detail="Embedding model is still loading.")

    file_bytes = await file.read()
    cv_text = extract_text_from_file(file.filename, file_bytes)
    
    if not cv_text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file contains no readable text.")

    try:
        query_vector = encoder.encode(cv_text[:2000]).tolist()
        
        response = await async_qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3
        )
        search_results = response.points

        matched_jobs = []
        corpus_texts = []
        for hit in search_results:
            if hit.payload:
                content = hit.payload.get("content") or hit.payload.get("text") or ""
                corpus_texts.append(content)
                matched_jobs.append({
                    "score": round(hit.score, 2),
                    "snippet": content[:300] + "..."
                })

        common_skills = ["java", "python", "fastapi", "sql", "oracle", "docker", "aws", "react", "node.js", "javascript"]
        cv_lower = cv_text.lower()
        
        found_keywords = [skill for skill in common_skills if skill in cv_lower]
        missing_keywords = [skill for skill in common_skills if skill not in cv_lower][:5]

        corpus_blob = " ".join(corpus_texts).lower()
        visa_match = "Likely Eligible" if any(term in corpus_blob for term in ["sponsorship", "visa", "skilled worker"]) else "Review Required"

        return {
            "status": "success",
            "filename": file.filename,
            "matched_jobs": matched_jobs,
            "keywords_found": found_keywords,
            "missing_keywords": missing_keywords,
            "visa_sponsorship_match": visa_match
        }

    except Exception as e:
        logger.error(f"CV analysis pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))