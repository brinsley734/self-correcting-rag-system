from fastapi import FastAPI, HTTPException, Query
from src.config import settings
from src.ingestion.scraper import KubeScraper
from src.processing.chunker import HierarchicalChunker
from src.vector_store.qdrant_index import QdrantIndexer
from src.search.bm25_index import BM25StorageEngine

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

scraper = KubeScraper()
chunker = HierarchicalChunker()
indexer = QdrantIndexer()
bm25_engine = BM25StorageEngine()

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "project": settings.PROJECT_NAME}

@app.get("/api/v1/ingest", tags=["Ingestion"])
def ingest_pipeline_endpoint(url: str = Query(..., description="The official target Kubernetes documentation URL")):
    scraped_data = scraper.scrape_page(url)
    if not scraped_data:
        raise HTTPException(status_code=400, detail="Failed parsing documentation text target layer.")
        
    chunked_manifest = chunker.chunk_document(scraped_data)
    indexer.index_manifest(chunked_manifest)
    bm25_engine.build_and_save_index(chunked_manifest)
    
    return {
        "status": "success",
        "pipeline_summary": {
            "title": scraped_data["title"],
            "total_structural_parents_processed": len(chunked_manifest),
            "indexes_status": "Qdrant and BM25 synced successfully"
        }
    }

@app.get("/api/v1/search", tags=["Retrieval"])
def hybrid_search(query: str = Query(..., description="The query to search the Kubernetes documentation")):
    """
    Executes an operational smoke test by matching terms against both dense (vector) 
    and sparse (BM25) indexes to gather top candidates.
    """
    results = {}

    # 1. Sparse Match via BM25
    if bm25_engine.bm25:
        tokenized_query = query.lower().split()
        # Retrieve scores across all entries
        scores = bm25_engine.bm25.get_scores(tokenized_query)
        # Pair entries with their indices, sort, and grab top 3 items
        top_bm25_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
        
        results["sparse_bm25_matches"] = [
            {
                "child_id": bm25_engine.doc_manifest[idx]["child_id"],
                "content": bm25_engine.doc_manifest[idx]["content"],
                "score": float(scores[idx]),
                "parent_context": bm25_engine.doc_manifest[idx]["parent_context_body"]
            }
            for idx in top_bm25_indices if scores[idx] > 0
        ]
    else:
        results["sparse_bm25_matches"] = []

    # 2. Dense Match via Qdrant Client
    try:
        # Encode our query string into a vector array
        query_vector = indexer.encoder.encode(query).tolist()
        dense_matches = indexer.client.search(
            collection_name=indexer.collection_name,
            query_vector=query_vector,
            limit=3
        )
        results["dense_vector_matches"] = [
            {
                "child_id": point.id,
                "content": point.payload.get("content"),
                "score": float(point.score),
                "parent_context": point.payload.get("parent_context_body")
            }
            for point in dense_matches
        ]
    except Exception as e:
        results["dense_vector_matches"] = [f"Qdrant match exception: {str(e)}"]

    return {
        "query": query,
        "hybrid_results": results
    }