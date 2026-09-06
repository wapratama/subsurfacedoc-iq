"""
search/retrieval.py

All search functions for SubsurfaceIQ.
Implements: text, vector, hybrid (RRF), HyDE, and reranked search.

Design: each function is a pure callable that takes (query, **kwargs)
and returns list[dict]. This makes swapping search methods trivial
in both the RAG layer and the evaluation harness.
"""
import sys
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
from embedder import Embedder

# Lazy-loaded singletons — initialized once on first call
_embedder      = None
_text_index    = None
_vector_index  = None
_reranker      = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_indexes():
    global _text_index, _vector_index
    if _text_index is None:
        from search.indexer import load_text_index, load_vector_index
        _text_index, _ = load_text_index()
        _vector_index  = load_vector_index()
    return _text_index, _vector_index


def _get_reranker():
    """
    Lazy-load cross-encoder reranker.
    WHY lazy: sentence-transformers is heavy — don't load unless needed.
    WHY ms-marco-MiniLM-L-6-v2: fast, strong, widely validated for
    passage re-ranking. Runs on CPU in < 100ms for 10 candidates.
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


# ── Core search functions ─────────────────────────────────────────────────────

def text_search(
    query: str,
    num_results: int = 5,
    filter_dict: dict | None = None,
) -> list[dict]:
    """
    Keyword search using minsearch TF-IDF.
    Supports optional filter by well_id, well_series, or operator.
    """
    text_index, _ = _get_indexes()
    return text_index.search(
        query,
        num_results=num_results,
        filter_dict=filter_dict or {}
    )


def vector_search(
    query: str,
    num_results: int = 5,
    filter_dict: dict | None = None,
) -> list[dict]:
    """
    Semantic search using ONNX all-MiniLM-L6-v2 embeddings.
    Vectors are L2-normalized — dot product = cosine similarity.
    """
    _, vector_index = _get_indexes()
    embedder = _get_embedder()
    v = embedder.encode(query)
    return vector_index.search(v, num_results=num_results)


def rrf(
    result_lists: list[list[dict]],
    k: int = 1,
    num_results: int = 5
) -> list[dict]:
    """
    Reciprocal Rank Fusion.
    k=1: best MRR from Module 4 evaluation on similar corpus.
    Key: (doc_id) uniquely identifies a DDR (one doc = one DDR, no chunks).
    """
    scores = {}
    docs   = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc["doc_id"]
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key]   = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def hybrid_search(
    query: str,
    num_results: int = 5,
    k: int = 1,
    filter_dict: dict | None = None,
) -> list[dict]:
    """
    Hybrid search: text + vector → RRF fusion.
    Wider retrieval window (10) before fusion to num_results.
    k=1 from Module 4 finding — best MRR on this corpus type.
    """
    text_r   = text_search(query, num_results=10, filter_dict=filter_dict)
    vector_r = vector_search(query, num_results=10)
    return rrf([text_r, vector_r], k=k, num_results=num_results)


def hyde_search(
    query: str,
    num_results: int = 5,
    model: str = "gpt-5.4-mini",
) -> list[dict]:
    """
    HyDE — Hypothetical Document Embeddings.
    Generates a fake DDR-style answer, embeds it, searches with that vector.

    WHY for Volve: users ask informal questions ("what mud weight for Hugin?")
    while DDRs use formal drilling engineering language. HyDE bridges the gap
    by generating text in the DDR register before embedding.
    """
    client = OpenAI()
    hyde_prompt = f"""
You are a drilling engineer writing a section of a Volve field Daily Drilling Report.
Write a short, technical paragraph (3-5 sentences) that would answer this question,
using realistic drilling engineering vocabulary as found in North Sea DDRs.

Question: {query}

Write only the paragraph, no preamble.
""".strip()

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": hyde_prompt}]
    )
    hypothetical_answer = response.output_text.strip()

    _, vector_index = _get_indexes()
    embedder = _get_embedder()
    v = embedder.encode(hypothetical_answer)
    results = vector_index.search(v, num_results=num_results)

    # Attach the hypothetical answer for debugging/monitoring
    for r in results:
        r["_hyde_answer"] = hypothetical_answer
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Cross-encoder reranking of candidate documents.
    WHY: bi-encoder embeddings may conflate "formation pressure" (geology)
    with "formation damage" (drilling) — cross-encoder evaluates the
    (query, document) pair jointly, catching these distinctions.
    """
    reranker = _get_reranker()
    pairs  = [(query, doc["content"][:512]) for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def hybrid_reranked_search(
    query: str,
    num_results: int = 5,
    filter_dict: dict | None = None,
) -> list[dict]:
    """
    Full pipeline: hybrid retrieval (wide) → cross-encoder reranking (narrow).
    This is the highest-quality search mode — used as default in app.
    Retrieves 15 candidates first, reranks to top num_results.
    """
    candidates = hybrid_search(
        query, num_results=15, filter_dict=filter_dict
    )
    return rerank(query, candidates, top_k=num_results)


# ── Search mode registry ──────────────────────────────────────────────────────
# Maps UI display name → callable
# Used by both app/streamlit_app.py and evaluation/run_eval.py

SEARCH_MODES = {
    "Hybrid + Rerank (Best)":  hybrid_reranked_search,
    "Hybrid":                  hybrid_search,
    "Semantic (Vector)":       vector_search,
    "Keyword (Text)":          text_search,
    "HyDE":                    hyde_search,
}