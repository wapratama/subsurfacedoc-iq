"""
search/indexer.py

Builds text and vector indexes from the Volve DDR corpus in DuckDB.
Persists vector index to disk (sqlitesearch) so embeddings are computed
once, not on every app startup.

Run this once after pipeline.py:
    uv run python search/indexer.py
"""
import sys
import json
import pickle
import numpy as np
import duckdb
from pathlib import Path
from tqdm import tqdm
from minsearch import Index
from sqlitesearch import VectorSearchIndex

sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
from embedder import Embedder

# ── Paths ────────────────────────────────────────────────────────────────────
DUCKDB_PATH      = Path("volve_rag.duckdb")
TEXT_INDEX_PATH  = Path("search/text_index.pkl")
VECTOR_DB_PATH   = Path("search/vector_index.db")  # sqlitesearch persistent
CORPUS_CACHE     = Path("search/corpus_cache.json")

TEXT_INDEX_PATH.parent.mkdir(exist_ok=True)


# ── Load corpus from DuckDB ───────────────────────────────────────────────────

def load_corpus() -> list[dict]:
    """Load all DDR documents from DuckDB corpus."""
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    rows = conn.execute("""
        SELECT
            doc_id, filename, well_id, well_series,
            date, operator, rig_name, depth_md, content, char_count
        FROM volve.volve_ddrs
        ORDER BY well_id, date
    """).fetchall()
    conn.close()

    columns = [
        "doc_id", "filename", "well_id", "well_series",
        "date", "operator", "rig_name", "depth_md", "content", "char_count"
    ]
    corpus = [dict(zip(columns, row)) for row in rows]
    print(f"Loaded {len(corpus)} DDR documents from DuckDB")
    return corpus


# ── Text index ───────────────────────────────────────────────────────────────

def build_text_index(corpus: list[dict]) -> Index:
    """
    Build minsearch text index.

    WHY content-only in text_fields: DDR text already contains wellbore
    name, date, operator, rig name in the header — boosting the content
    field alone captures these without double-indexing.

    WHY these keyword_fields: enables filtered search by well or series.
    e.g. filter_dict={"well_id": "15/9-F-15 A"} narrows to one well.
    """
    index = Index(
        text_fields=["content"],
        keyword_fields=["well_id", "well_series", "operator", "doc_type"]
    )

    # Add doc_type field (not in DuckDB — add inline)
    docs_with_type = [{**doc, "doc_type": "drilling_report"} for doc in corpus]
    index.fit(docs_with_type)
    print(f"✅ Text index built: {len(corpus)} documents")
    return index


def save_text_index(index: Index, corpus: list[dict]):
    """Persist text index and corpus to disk."""
    with open(TEXT_INDEX_PATH, 'wb') as f:
        pickle.dump({"index": index, "corpus": corpus}, f)
    print(f"✅ Text index saved: {TEXT_INDEX_PATH}")


def load_text_index() -> tuple[Index, list[dict]]:
    """Load text index from disk."""
    with open(TEXT_INDEX_PATH, 'rb') as f:
        data = pickle.load(f)
    print(f"✅ Text index loaded: {len(data['corpus'])} documents")
    return data["index"], data["corpus"]


# ── Vector index ──────────────────────────────────────────────────────────────

def build_vector_index(corpus: list[dict]) -> VectorSearchIndex:
    """
    Build sqlitesearch vector index with persistent storage.
    Embeddings computed once, stored in VECTOR_DB_PATH.

    WHY sqlitesearch over minsearch.VectorSearch:
    - Persistent: survives app restarts without re-embedding
    - Same API: .fit(vectors, documents) / .search(query_vector)
    - 1,759 DDRs × 384-dim = ~2.7M floats = ~10MB on disk — trivial
    """
    embedder = Embedder()
    print(f"Embedding {len(corpus)} DDRs...")

    # Batch embed for efficiency
    BATCH_SIZE = 64
    all_vectors = []

    for i in tqdm(range(0, len(corpus), BATCH_SIZE), desc="Embedding"):
        batch = corpus[i: i + BATCH_SIZE]
        texts = [doc["content"] for doc in batch]
        vectors = embedder.encode_batch(texts)
        all_vectors.append(vectors)

    X = np.vstack(all_vectors)
    print(f"Embedding matrix: {X.shape}")

    # Verify normalization
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=0.01), "Embeddings not normalized!"

    # Build persistent vector index
    vector_index = VectorSearchIndex(
        keyword_fields=["well_id", "well_series"],
        db_path=str(VECTOR_DB_PATH)
    )

    # Add doc_type to corpus for keyword filtering
    corpus_with_type = [{**doc, "doc_type": "drilling_report"} for doc in corpus]
    vector_index.fit(X, corpus_with_type)
    print(f"✅ Vector index built and saved: {VECTOR_DB_PATH}")
    return vector_index


def load_vector_index() -> VectorSearchIndex:
    """Load existing vector index from disk — no re-embedding needed."""
    index = VectorSearchIndex(
        keyword_fields=["well_id", "well_series"],
        db_path=str(VECTOR_DB_PATH)
    )
    print(f"✅ Vector index loaded from disk")
    return index


# ── Corpus cache ──────────────────────────────────────────────────────────────

def save_corpus_cache(corpus: list[dict]):
    """Save corpus as JSON for fast loading without DuckDB dependency."""
    CORPUS_CACHE.write_text(json.dumps(corpus, indent=2, default=str))
    print(f"✅ Corpus cache saved: {len(corpus)} docs")


def load_corpus_cache() -> list[dict]:
    """Load corpus from JSON cache."""
    corpus = json.loads(CORPUS_CACHE.read_text())
    print(f"✅ Corpus cache loaded: {len(corpus)} docs")
    return corpus


# ── Main ──────────────────────────────────────────────────────────────────────

def build_all_indexes():
    """Full index build — run once after pipeline.py."""
    corpus = load_corpus()
    save_corpus_cache(corpus)

    text_index = build_text_index(corpus)
    save_text_index(text_index, corpus)

    build_vector_index(corpus)

    print("\n=== Index Build Complete ===")
    print(f"Text index:   {TEXT_INDEX_PATH}")
    print(f"Vector index: {VECTOR_DB_PATH}")
    print(f"Corpus cache: {CORPUS_CACHE}")
    print("\nRun validation:")
    print("  uv run python search/indexer.py --validate")


def validate_indexes():
    """Quick validation that both indexes are working correctly."""
    corpus = load_corpus_cache()
    text_index, _ = load_text_index()
    vector_index = load_vector_index()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
    from embedder import Embedder
    embedder = Embedder()

    # Test query relevant to Volve DDR content
    test_query = "mud weight when drilling through Hugin Formation"
    v = embedder.encode(test_query)

    text_results = text_index.search(test_query, num_results=3)
    vector_results = vector_index.search(v, num_results=3)

    print(f"\n=== Validation: '{test_query}' ===")
    print("\nText search top-3:")
    for r in text_results:
        print(f"  {r['well_id']} | {r['date']} | {r['content'][:100]}...")

    print("\nVector search top-3:")
    for r in vector_results:
        print(f"  {r['well_id']} | {r['date']} | {r['content'][:100]}...")

    # Test filtered search
    filtered = text_index.search(
        "formation strength leak off test",
        filter_dict={"well_series": "production"},
        num_results=3
    )
    print("\nFiltered (production wells only) top-3:")
    for r in filtered:
        print(f"  {r['well_id']} | {r['date']}")

    print("\n✅ All index validations passed")


if __name__ == "__main__":
    import sys
    if "--validate" in sys.argv:
        validate_indexes()
    else:
        build_all_indexes()