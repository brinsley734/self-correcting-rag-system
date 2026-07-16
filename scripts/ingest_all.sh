#!/bin/bash
set -e

echo "=== Starting Full Ingestion Pipeline ==="

# Ensure target directories exist
mkdir -p data/processed

# Run the ingestion orchestration pipeline via python module path
# Using python3 to match your verified environment
python3 -m src.ingestion.pipeline \
    --input-dir "data/raw/ukjobs" \
    --bm25-output "data/processed/bm25_index.pkl"

echo "=== Ingestion Pipeline Completed Successfully ==="