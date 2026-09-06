"""
rag/rag_base.py

VolveRAG: domain-aware RAG for Volve Daily Drilling Reports.

Inherits from RAGBase (homework rag_helper.py) and overrides:
  - search()        → uses hybrid_reranked_search from Phase 2
  - build_context() → includes DDR metadata (well, date, operator)
  - rag()           → returns RAGResult with answer + usage + sources
  - instructions    → petroleum engineering domain prompt

The context format is deliberately structured so the LLM can cite
specific wells and dates — critical for a technical assistant where
"well F-15 A on 2010-03-15" is more useful than a generic answer.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Resolve rag_helper.py from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_helper import RAGBase

from search import hybrid_reranked_search, hybrid_search, text_search, vector_search


# ── System prompt ─────────────────────────────────────────────────────────────

VOLVE_INSTRUCTIONS = """
You are a petroleum engineering assistant specializing in the Volve oil field,
operated by Statoil (now Equinor) in the Norwegian North Sea (Block 15/9).

Your knowledge base is 1,759 Daily Drilling Reports (DDRs) covering all
26 wellbores drilled between 1980 and 2018.

RULES:
1. Always cite the specific well (e.g. 15/9-F-15 A) and date when referencing a DDR.
2. If multiple DDRs support the answer, synthesize them and note the date range.
3. If the answer is not in the provided context, respond:
   "This information is not available in the retrieved DDRs. Try refining your query
   or filtering by a specific well."
4. Use petroleum engineering terminology accurately:
   - Depths in meters MD (measured depth) and TVD (true vertical depth)
   - Mud weight in kg/m³ or ppg — state which unit the report uses
   - Formation names: Hugin, Skagerrak, Sleipner, Ty, Lista, Sele, Balder
5. Never fabricate drilling parameters, depths, or formation data.
""".strip()


PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT — Retrieved Daily Drilling Reports:
{context}
""".strip()


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    """
    Carries the complete output of one RAG call.
    Structured so Phase 5 (UI) and Phase 6 (monitoring) can consume
    all fields without re-parsing the answer string.
    """
    answer:       str
    input_tokens:  int
    output_tokens: int
    cost:         float
    sources:      list[dict] = field(default_factory=list)
    search_mode:  str = "hybrid_reranked"
    query:        str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ── Cost calculator ───────────────────────────────────────────────────────────

def compute_cost(input_tokens: int, output_tokens: int,
                 model: str = "gpt-5.4-mini") -> float:
    """
    Compute approximate USD cost for one LLM call.
    gpt-5.4-mini pricing: $0.15/1M input, $0.60/1M output tokens.
    Update prices here if model changes.
    """
    prices = {
        "gpt-5.4-mini": (0.15, 0.60),
        "gpt-4.1":      (2.00, 8.00),
    }
    input_price, output_price = prices.get(model, (0.15, 0.60))
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


# ── VolveRAG ──────────────────────────────────────────────────────────────────

class VolveRAG(RAGBase):
    """
    Volve-specific RAG subclass.

    Usage:
        from rag import VolveRAG
        rag = VolveRAG.create()
        result = rag.rag("What mud weight was used drilling the Hugin Formation?")
        print(result.answer)
        print(f"Sources: {[s['well_id'] + ' ' + s['date'] for s in result.sources]}")
    """

    def __init__(self, search_mode: str = "hybrid_reranked", **kwargs):
        super().__init__(**kwargs)
        self.search_mode = search_mode

        # Map mode name → search function
        self._search_fn_map = {
            "hybrid_reranked": hybrid_reranked_search,
            "hybrid":          hybrid_search,
            "vector":          vector_search,
            "text":            text_search,
        }

    @classmethod
    def create(
        cls,
        search_mode: str = "hybrid_reranked",
        model: str = "gpt-5.4-mini",
    ) -> "VolveRAG":
        """
        Factory method — assembles VolveRAG with all dependencies.
        Use this instead of __init__ directly.
        """
        from openai import OpenAI

        # Build a dummy index (not used — we override search())
        # RAGBase requires an index argument
        class _DummyIndex:
            def search(self, *a, **kw): return []

        return cls(
            index=_DummyIndex(),
            llm_client=OpenAI(),
            instructions=VOLVE_INSTRUCTIONS,
            prompt_template=PROMPT_TEMPLATE,
            model=model,
            search_mode=search_mode,
        )

    def search(
        self,
        query: str,
        num_results: int = 5,
        filter_dict: dict | None = None,
    ) -> list[dict]:
        """
        Delegate to the configured search function from Phase 2.
        filter_dict enables per-well or per-series filtering from the UI.
        """
        fn = self._search_fn_map.get(self.search_mode, hybrid_reranked_search)
        if filter_dict:
            return fn(query, num_results=num_results, filter_dict=filter_dict)
        return fn(query, num_results=num_results)

    def build_context(self, search_results: list[dict]) -> str:
        """
        Build LLM context from DDR search results.

        Format: each DDR gets a labeled block with metadata header +
        content. Labels help the LLM cite specific wells and dates
        in its answer rather than giving generic responses.
        """
        blocks = []
        for i, doc in enumerate(search_results, start=1):
            well    = doc.get("well_id",   "Unknown well")
            date    = doc.get("date",      "Unknown date")
            op      = doc.get("operator",  "")
            rig     = doc.get("rig_name",  "")
            depth   = doc.get("depth_md",  "")

            header_parts = [f"[DDR {i}] Well: {well} | Date: {date}"]
            if op:    header_parts.append(f"Operator: {op}")
            if rig:   header_parts.append(f"Rig: {rig}")
            if depth: header_parts.append(f"Depth MD: {depth}m")

            blocks.append(
                " | ".join(header_parts) + "\n" + doc["content"]
            )

        return "\n\n---\n\n".join(blocks)

    def llm(self, prompt: str):
        """Call LLM and return full response object for usage tracking."""
        return self.llm_client.responses.create(
            model=self.model,
            input=[
                {"role": "developer", "content": self.instructions},
                {"role": "user",      "content": prompt},
            ]
        )

    def rag(
        self,
        query: str,
        num_results: int = 5,
        filter_dict: dict | None = None,
    ) -> RAGResult:
        """
        Full RAG pipeline with OTel instrumentation — returns RAGResult with all monitoring fields.
        """
        # Import here — avoids circular import at module level
        try:
            from monitoring.otel_setup import tracer
            otel_available = True
        except Exception:
            otel_available = False

        def _run() -> RAGResult:
            search_results = self.search(query, num_results, filter_dict)
            prompt         = self.build_prompt(query, search_results)
            response       = self.llm(prompt)

            usage         = response.usage
            input_tokens  = usage.input_tokens
            output_tokens = usage.output_tokens
            cost          = compute_cost(input_tokens, output_tokens, self.model)

            return RAGResult(
                answer        = response.output_text,
                input_tokens  = input_tokens,
                output_tokens = output_tokens,
                cost          = cost,
                sources       = search_results,
                search_mode   = self.search_mode,
                query         = query,
            )

        if not otel_available:
            return _run()

        # Wrap in OTel span — captures timing automatically
        with tracer.start_as_current_span("rag") as span:
            # Set attributes before the LLM call
            span.set_attribute("query",       query)
            span.set_attribute("search_mode", self.search_mode)
            span.set_attribute("well_filter", str(filter_dict or "none"))

            result = _run()

            # Set attributes after — now we have token counts
            span.set_attribute("input_tokens",  result.input_tokens)
            span.set_attribute("output_tokens", result.output_tokens)
            span.set_attribute("cost",          result.cost)
            if result.sources:
                span.set_attribute("well_id", result.sources[0].get("well_id", ""))

            return result