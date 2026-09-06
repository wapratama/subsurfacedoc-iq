"""
tests/test_smoke.py

Smoke tests for all SubsurfaceIQ phases.
Run: uv run pytest tests/ -v

These tests verify the full pipeline is wired correctly
without making LLM API calls (mocked where possible).
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Phase 1: Corpus ───────────────────────────────────────────────────────────

def test_corpus_sample_exists():
    path = Path("data/samples/corpus_sample.csv")
    assert path.exists(), "Corpus sample not committed"

    import pandas as pd
    df = pd.read_csv(path)
    assert len(df) > 0
    required_cols = {"doc_id", "well_id", "well_series", "date", "content"}
    assert required_cols.issubset(df.columns)


def test_ground_truth_exists():
    path = Path("data/ground_truth.csv")
    assert path.exists(), "Run evaluation/generate_gt.py first"

    import pandas as pd
    df = pd.read_csv(path)
    assert len(df) >= 200, f"Too few Q&A pairs: {len(df)}"
    assert "question" in df.columns
    assert "doc_id"   in df.columns


# ── Phase 2: Search ───────────────────────────────────────────────────────────

def test_indexes_exist():
    assert Path("search/text_index.pkl").exists(),  "Run search/indexer.py"
    assert Path("search/vector_index.db").exists(), "Run search/indexer.py"


def test_text_search_returns_results():
    from search import text_search
    results = text_search("mud weight Hugin Formation", num_results=3)
    assert len(results) == 3
    for r in results:
        assert "doc_id"   in r
        assert "well_id"  in r
        assert "content"  in r


def test_vector_search_returns_results():
    from search import vector_search
    results = vector_search("drilling formation evaluation", num_results=3)
    assert len(results) == 3


def test_hybrid_search_returns_results():
    from search import hybrid_search
    results = hybrid_search("daily drilling summary", num_results=5)
    assert len(results) == 5


def test_well_filter_works():
    from search import text_search
    results = text_search(
        "drilling operations",
        num_results=5,
        filter_dict={"well_series": "production"}
    )
    for r in results:
        assert r["well_series"] == "production", \
            f"Filter failed: got {r['well_series']}"


# ── Phase 3: RAG ──────────────────────────────────────────────────────────────

def test_rag_result_structure():
    """Test RAG result structure without making API call."""
    from rag import RAGResult
    result = RAGResult(
        answer="Test answer",
        input_tokens=1000,
        output_tokens=100,
        cost=0.00016,
        sources=[{"doc_id": "test", "well_id": "15/9-F-15 A",
                   "date": "2010-03-15", "content": "test content"}],
        search_mode="hybrid_reranked",
        query="test query",
    )
    assert result.total_tokens == 1100
    assert result.cost < 0.01


def test_compute_cost():
    from rag.rag_base import compute_cost
    cost = compute_cost(7000, 200, "gpt-5.4-mini")
    assert 0.0001 < cost < 0.01, f"Unexpected cost: {cost}"


# ── Phase 4: Evaluation ───────────────────────────────────────────────────────

def test_metrics_hit_rate():
    from evaluation.metrics import hit_rate
    matrix = [[1,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0]]
    assert hit_rate(matrix) == pytest.approx(2/3)


def test_metrics_mrr():
    from evaluation.metrics import mrr
    matrix = [[1,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0]]
    expected = (1.0 + 0.5 + 0.0) / 3
    assert mrr(matrix) == pytest.approx(expected)


def test_evaluation_results_exist():
    path = Path("evaluation/results.json")
    assert path.exists(), "Run evaluation/run_eval.py first"

    import json
    results = json.loads(path.read_text())
    assert "retrieval" in results
    assert "answer"    in results
    assert len(results["retrieval"]) >= 4


# ── Phase 5: UI ───────────────────────────────────────────────────────────────

def test_feedback_db_schema():
    import sqlite3
    db_path = Path("data/feedback.db")
    if not db_path.exists():
        pytest.skip("Run the app first to create feedback.db")

    conn = sqlite3.connect(str(db_path))
    cols = [r[1] for r in conn.execute(
        "PRAGMA table_info(feedback)"
    ).fetchall()]
    conn.close()

    required = {"timestamp","query","answer","score","input_tokens","cost"}
    assert required.issubset(set(cols))


# ── Phase 7: Reproducibility ──────────────────────────────────────────────────

def test_lockfile_exists():
    assert Path("uv.lock").exists(), "Run uv lock"


def test_env_example_has_required_keys():
    env_example = Path(".env.example").read_text()
    required_keys = [
        "OPENAI_API_KEY",
        "POSTGRES_HOST",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]
    for key in required_keys:
        assert key in env_example, f"Missing from .env.example: {key}"


def test_dockerfile_exists():
    assert Path("Dockerfile").exists()


def test_docker_compose_exists():
    import yaml
    content = Path("docker-compose.yml").read_text()
    config  = yaml.safe_load(content)
    services = config.get("services", {})
    assert "app"      in services
    assert "postgres" in services
    assert "grafana"  in services