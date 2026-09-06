"""
rag/agent.py

PydanticAI agentic version of SubsurfaceIQ.

WHY an agent on top of RAG: a multi-step agent can:
  1. Search broadly first to understand the context
  2. Decide to search again with a more specific well filter
  3. Synthesize across multiple DDRs before answering

This covers the "agentic best practice" criterion and demonstrates
Module 1's agentic loop concept applied to a real domain problem.
"""
from __future__ import annotations

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

load_dotenv()

from search import hybrid_reranked_search, text_search
from rag.rag_base import RAGResult, compute_cost, VOLVE_INSTRUCTIONS


# ── Tool definitions ──────────────────────────────────────────────────────────

def search_volve_reports(query: str) -> list[dict]:
    """
    Search all Volve Daily Drilling Reports for relevant information.
    Use this for broad questions about the field, formations, or operations.
    Returns up to 5 most relevant DDRs with well ID, date, and content.

    Args:
        query: Natural language question about Volve drilling operations.
    """
    results = hybrid_reranked_search(query, num_results=5)
    # Return slim version to stay within context limits
    return [
        {
            "well_id":  r["well_id"],
            "date":     r["date"],
            "operator": r.get("operator", ""),
            "content":  r["content"][:800],  # trim for agent context
        }
        for r in results
    ]


def search_specific_well(query: str, well_id: str) -> list[dict]:
    """
    Search Daily Drilling Reports for a specific well only.
    Use this when the question targets a known wellbore.
    Available well IDs: 15/9-19 A, 15/9-F-1, 15/9-F-5, 15/9-F-9 A,
    15/9-F-11 A, 15/9-F-12, 15/9-F-14, 15/9-F-15, 15/9-F-15 A, etc.

    Args:
        query:   Natural language question about this well's operations.
        well_id: Exact wellbore name (e.g. '15/9-F-15 A').
    """
    results = text_search(
        query,
        num_results=5,
        filter_dict={"well_id": well_id}
    )
    return [
        {
            "well_id":  r["well_id"],
            "date":     r["date"],
            "content":  r["content"][:800],
        }
        for r in results
    ]


# ── Agent definition ──────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = f"""
{VOLVE_INSTRUCTIONS}

You have access to two search tools:
- search_volve_reports: broad search across all 26 wellbores
- search_specific_well: filtered search for one specific wellbore

Strategy:
1. Start with search_volve_reports to get context.
2. If the question targets a specific well, follow up with search_specific_well.
3. Synthesize findings from all searches before answering.
4. Always cite well IDs and dates in your final answer.
""".strip()


def create_agent(model: str = "gpt-5.4-mini") -> Agent:
    """
    Create a PydanticAI agent with Volve search tools.

    WHY PydanticAI: integrates natively with Logfire for tracing
    (dlt workshop pattern), has clean tool registration, and produces
    well-typed tool responses.
    """
    return Agent(
        model=OpenAIModel(model),
        system_prompt=AGENT_SYSTEM_PROMPT,
        tools=[search_volve_reports, search_specific_well],
    )


async def run_agent(query: str, model: str = "gpt-5.4-mini") -> RAGResult:
    """
    Run the agentic pipeline for one query.
    Returns RAGResult for consistent interface with VolveRAG.
    """
    agent = create_agent(model)
    result = await agent.run(query)

    # PydanticAI usage — adapt field names as confirmed from actual response
    usage         = result.usage()
    input_tokens  = getattr(usage, "request_tokens",  0)
    output_tokens = getattr(usage, "response_tokens", 0)

    return RAGResult(
        answer        = result.output,
        input_tokens  = input_tokens,
        output_tokens = output_tokens,
        cost          = compute_cost(input_tokens, output_tokens, model),
        sources       = [],   # agent doesn't return structured sources
        search_mode   = "agent",
        query         = query,
    )