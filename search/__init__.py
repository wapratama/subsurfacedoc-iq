"""
search/__init__.py

Public API for SubsurfaceIQ search layer.
Import from here, not from submodules directly.
"""
from search.retrieval import (
    text_search,
    vector_search,
    hybrid_search,
    hyde_search,
    hybrid_reranked_search,
    SEARCH_MODES,
)

__all__ = [
    "text_search",
    "vector_search",
    "hybrid_search",
    "hyde_search",
    "hybrid_reranked_search",
    "SEARCH_MODES",
]