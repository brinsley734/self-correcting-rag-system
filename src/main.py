import os
import logging
import time
import uuid
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from src.config import settings
from src.ingestion.pipeline import IngestionPipeline
from src.core.model_registry import get_registry
from src.api.routes import models as model_routes

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Cache Initialization ---
async def setup_cache_collection(client: AsyncQdrantClient):
    cache_name = "query_cache"
    if not await client.collection_exists(cache_name):
        await client.create_collection(
            collection_name=cache_name,
            vectors_config=models.VectorParams(
                size=384, 
                distance=models.Distance.COSINE
            )
        )
        logger.info(f"Initialized cache collection: {cache_name}")

async def get_from_cache(client: AsyncQdrantClient, query_vector: List[float], threshold: float = 0.95):
    """Searches the cache for a similar question."""
    hits = await client.search(
        collection_name="query_cache",
        query_vector=query_vector,
        limit=1,
        with_payload=True
    )
    if hits and hits[0].score >= threshold:
        return hits[0].payload
    return None

async def save_to_cache(client: AsyncQdrantClient, query_vector: List[float], answer: str, question: str):
    """Saves a new query-answer pair to the cache."""
    await client.upsert(
        collection_name="query_cache",
        points=[{
            "id": str(uuid.uuid4()),
            "vector": query_vector,
            "payload": {"answer": answer, "question": question}
        }]
    )

# --- State Container ---
async_qdrant = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
COLLECTION_NAME = "uk_jobs_data"

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Model Registry singleton...")
    registry = get_registry()
    registry.initialise()
    
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
    await setup_cache_collection(async_qdrant)
    
    yield
    
    await app.state.http_client.aclose()
    logger.info("Shutdown complete.")

# --- App Initialization ---
app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan,
    description="Asynchronous backend engine with reranking, performance monitoring, and semantic caching.",
    version=settings.VERSION
)

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount Static Frontend & UI Dashboard ---
os.makedirs("static/benchmarks", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", tags=["UI Dashboard"])
async def read_index():
    return FileResponse("static/index.html")

# --- Mount Routers ---
app.include_router(model_routes.router, prefix="/api/v1/models", tags=["Model Management"])

# --- Data Structures ---
class QueryRequest(BaseModel):
    question: str
    limit: Optional[int] = 3

class QueryResponse(BaseModel):
    question: str
    answer: str
    context_used: List[str]

# --- Background Execution ---
def execute_pipeline():
    try:
        logger.info("Background thread spinning up master ingestion pipeline process...")
        pipeline = IngestionPipeline()
        pipeline.run()
    except Exception as e:
        logger.error(f"Critical breakdown in background ingestion workflow: {str(e)}")

# --- Endpoints ---
@app.get("/health", tags=["System"])
async def health_check():
    try:
        await async_qdrant.get_collections()
        return {"status": "healthy", "database_connection": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Infrastructure down: {str(e)}")

@app.post("/api/v1/ingest", tags=["Ingestion"])
async def trigger_ingestion(background_tasks: BackgroundTasks):
    background_tasks.add_task(execute_pipeline)
    return JSONResponse(status_code=202, content={"status": "initiated"})

@app.post("/query", response_model=QueryResponse, tags=["Retrieval & Generation"])
async def process_rag_query(payload: QueryRequest):
    registry = get_registry()
    encoder = registry.current_embedding_model
    reranker = registry.current_reranker_model
    
    if encoder is None or reranker is None:
        raise HTTPException(status_code=503, detail="Models are still loading.")
    
    start_total = time.time()
    query_vector = encoder.encode(payload.question).tolist()
    
    cached_hit = await get_from_cache(async_qdrant, query_vector)
    if cached_hit:
        logger.info("Cache HIT: Returning stored answer.")
        return QueryResponse(
            question=payload.question, 
            answer=cached_hit["answer"], 
            context_used=["[CACHED]"]
        )
    
    try:
        start_retrieval = time.time()
        search_results = await async_qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=10 
        )
        
        if search_results:
            pairs = [(payload.question, hit.payload.get("content")) for hit in search_results]
            scores = reranker.predict(pairs)
            ranked_results = sorted(zip(search_results, scores), key=lambda x: x[1], reverse=True)
            top_chunks = [hit[0].payload.get("content") for hit in ranked_results[:payload.limit]]
        else:
            top_chunks = []
            
        retrieval_time = time.time() - start_retrieval
        context_str = "\n---\n".join(set(top_chunks)) if top_chunks else "No relevant context found."
        
        ollama_payload = {
            "model": "llama3.2",
            "messages": [
                {"role": "system", "content": "You are a helpful expert agent."},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {payload.question}"}
            ],
            "stream": False
        }
        
        http_client = app.state.http_client
        gen_res = await http_client.post(f"{settings.OLLAMA_BASE_URL}/api/chat", json=ollama_payload)
        gen_res.raise_for_status()
        
        llm_answer = gen_res.json()["message"]["content"]
        
        await save_to_cache(async_qdrant, query_vector, llm_answer, payload.question)
        
        total_time = time.time() - start_total
        logger.info(f"Retrieval/Rerank: {retrieval_time:.2f}s | Total: {total_time:.2f}s | Cache MISS")
        
        return QueryResponse(question=payload.question, answer=llm_answer, context_used=top_chunks)

    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))