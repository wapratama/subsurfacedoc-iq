#!/bin/bash
set -e

echo "=== SubsurfaceIQ Startup ==="

# Check 1: Vector index exists
if [ ! -f "search/vector_index.db" ]; then
    echo "❌ Vector index not found at search/vector_index.db"
    echo ""
    echo "Build it first on your local machine:"
    echo "  uv run python ingestion/pipeline.py"
    echo "  uv run python search/indexer.py"
    echo ""
    echo "Then re-run: docker compose up -d"
    exit 1
fi

# Check 2: DuckDB corpus exists
if [ ! -f "volve_rag.duckdb" ]; then
    echo "❌ Corpus not found at volve_rag.duckdb"
    echo "  uv run python ingestion/pipeline.py"
    exit 1
fi

echo "✅ Indexes found — starting app"

# Init Postgres tables
uv run python monitoring/db.py

# Start feedback sync in background
uv run python monitoring/sync_feedback.py &

# Start Streamlit
exec uv run streamlit run app/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0