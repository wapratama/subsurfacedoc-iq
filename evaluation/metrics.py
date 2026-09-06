"""
evaluation/metrics.py

Retrieval and answer quality evaluation metrics.
Reuses Module 4 patterns — adapted for Volve DDR corpus.

Key difference from homework 4:
  Ground truth uses 'doc_id' not 'filename' as the match key.
  One DDR = one document (no chunking) so doc_id is the unique identifier.
  A hit = returned result's doc_id matches the question's doc_id.
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
from embedder import Embedder

_embedder = None

def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def compute_relevance(
    search_fn,
    question: str,
    correct_doc_id: str,
    num_results: int = 5,
) -> list[int]:
    """
    Run search and return binary relevance list.
    1 = correct DDR is at this position, 0 = not.
    e.g. [0, 0, 1, 0, 0] = correct doc found at rank 3.
    """
    try:
        results = search_fn(question, num_results=num_results)
    except TypeError:
        # Some search functions don't accept num_results kwarg
        results = search_fn(question)

    return [1 if r["doc_id"] == correct_doc_id else 0 for r in results]


def hit_rate(relevance_matrix: list[list[int]]) -> float:
    """Fraction of queries where correct doc appears in top-k results."""
    return sum(any(row) for row in relevance_matrix) / len(relevance_matrix)


def mrr(relevance_matrix: list[list[int]]) -> float:
    """
    Mean Reciprocal Rank.
    Rewards finding the correct doc at a high rank position.
    Rank 1 = 1.0, Rank 2 = 0.5, Rank 3 = 0.33, not found = 0.0
    """
    scores = []
    for row in relevance_matrix:
        score = 0.0
        for rank, hit in enumerate(row, start=1):
            if hit:
                score = 1.0 / rank
                break
        scores.append(score)
    return float(np.mean(scores))


def evaluate(
    search_fn,
    ground_truth: list[dict],
    num_results: int = 5,
) -> dict[str, float]:
    """
    Evaluate a search function over the full ground truth dataset.
    Returns {"hit_rate": float, "mrr": float}.
    """
    relevance_matrix = [
        compute_relevance(
            search_fn,
            item["question"],
            item["doc_id"],
            num_results,
        )
        for item in ground_truth
    ]
    return {
        "hit_rate": round(hit_rate(relevance_matrix), 4),
        "mrr":      round(mrr(relevance_matrix),      4),
    }


# ── Answer quality metrics ────────────────────────────────────────────────────

def cosine_similarity_eval(
    generated: str,
    reference: str,
) -> float:
    """
    Method 1: Semantic similarity between generated and reference answer.
    Both embedded with same ONNX model → dot product = cosine similarity.
    Range: -1 to 1, where 1 = identical meaning, 0 = unrelated.
    """
    embedder = _get_embedder()
    v1 = embedder.encode(generated)
    v2 = embedder.encode(reference)
    return float(np.dot(v1, v2))


def llm_judge_eval(
    question: str,
    generated: str,
    context: str,
    model: str = "gpt-5.4-mini",
) -> dict:
    """
    Method 2: LLM-as-judge for answer relevance and accuracy.
    Evaluates WITHOUT a reference answer — uses retrieved context instead.
    Score 1–5, with explanation for Grafana dashboard display.

    WHY context-based not reference-based: DDRs don't have pre-written
    reference answers — the judge uses the same retrieved DDR content
    to assess whether the generated answer is grounded and accurate.
    """
    client = OpenAI()
    judge_prompt = f"""
You are evaluating an AI assistant that answers questions about
Volve oil field drilling reports.

Rate the GENERATED ANSWER from 1 to 5 using this scale:
  5 — Accurate, cites specific well/date, fully answers the question
  4 — Mostly accurate with minor gaps or missing citation
  3 — Partially answers, some relevant info but incomplete
  2 — Vague or only loosely related to the question
  1 — Inaccurate, hallucinated, or completely off-topic

QUESTION: {question}

RETRIEVED CONTEXT (what the assistant had access to):
{context[:2000]}

GENERATED ANSWER:
{generated}

Respond ONLY with valid JSON:
{{"score": <1-5>, "explanation": "<one sentence why>"}}
""".strip()

    try:
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": judge_prompt}]
        )
        import json
        raw = response.output_text.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return {
            "score":       int(result.get("score", 0)),
            "explanation": str(result.get("explanation", "")),
        }
    except Exception as e:
        return {"score": 0, "explanation": f"Judge error: {e}"}