"""
rag/__init__.py — Public RAG API for SubsurfaceIQ.
"""
from rag.rag_base import VolveRAG, RAGResult, compute_cost

__all__ = ["VolveRAG", "RAGResult", "compute_cost"]