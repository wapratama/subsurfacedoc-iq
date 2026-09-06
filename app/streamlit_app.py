"""
app/streamlit_app.py

SubsurfaceIQ — Volve Field Drilling Intelligence Assistant
Streamlit UI with search mode selection, well filtering,
source citation, and feedback collection.

Run: uv run streamlit run app/streamlit_app.py
"""
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag import VolveRAG, RAGResult
from search import SEARCH_MODES

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SubsurfaceIQ",
    page_icon="🛢️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS — petroleum engineering aesthetic ─────────────────────────────────────

st.markdown("""
<style>
  /* Base */
  .stApp { background-color: #0B1929; color: #F8FAFC; }
  .block-container { max-width: 860px; padding-top: 2rem; }

  /* Header */
  .app-header {
    border-bottom: 1px solid #1E3A5F;
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
  }
  .app-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #F8FAFC;
    letter-spacing: -0.01em;
  }
  .app-subtitle {
    font-size: 0.85rem;
    color: #94A3B8;
    margin-top: 0.2rem;
  }

  /* Answer block */
  .answer-block {
    background: #1E3A5F;
    border-left: 3px solid #F59E0B;
    border-radius: 0 4px 4px 0;
    padding: 1.25rem 1.5rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #F8FAFC;
    margin: 1rem 0;
  }

  /* Source cards */
  .source-card {
    background: #112336;
    border: 1px solid #1E3A5F;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
  }
  .source-meta {
    color: #F59E0B;
    font-weight: 600;
    margin-bottom: 0.3rem;
    font-size: 0.8rem;
  }
  .source-snippet {
    color: #94A3B8;
    line-height: 1.5;
  }

  /* Token badge */
  .token-badge {
    display: inline-block;
    background: #112336;
    border: 1px solid #1E3A5F;
    border-radius: 3px;
    padding: 0.15rem 0.5rem;
    font-size: 0.75rem;
    color: #94A3B8;
    margin-right: 0.5rem;
  }

  /* Feedback buttons */
  div[data-testid="column"] button {
    border-radius: 4px;
    font-size: 0.85rem;
    padding: 0.3rem 1rem;
  }

  /* Input */
  .stTextArea textarea {
    background-color: #112336 !important;
    border: 1px solid #1E3A5F !important;
    color: #F8FAFC !important;
    font-size: 0.95rem;
    border-radius: 4px;
  }
  .stSelectbox select {
    background-color: #112336 !important;
    color: #F8FAFC !important;
  }
  .stButton > button[kind="primary"] {
    background-color: #F59E0B;
    color: #0B1929;
    font-weight: 600;
    border: none;
  }
  .stButton > button[kind="primary"]:hover {
    background-color: #D97706;
  }

  /* Hide default streamlit branding */
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Feedback logging ──────────────────────────────────────────────────────────

FEEDBACK_DB = Path("data/feedback.db")
FEEDBACK_DB.parent.mkdir(exist_ok=True)


def init_feedback_db():
    conn = sqlite3.connect(str(FEEDBACK_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            query       TEXT NOT NULL,
            answer      TEXT NOT NULL,
            score       INTEGER NOT NULL,   -- 1=helpful, 0=not helpful
            well_id     TEXT,
            search_mode TEXT,
            input_tokens  INTEGER,
            output_tokens INTEGER,
            cost        REAL
        )
    """)
    conn.commit()
    conn.close()


def log_feedback(result: RAGResult, score: int, well_filter: str | None):
    conn = sqlite3.connect(str(FEEDBACK_DB))
    conn.execute("""
        INSERT INTO feedback
        (timestamp, query, answer, score, well_id, search_mode,
         input_tokens, output_tokens, cost)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        result.query,
        result.answer,
        score,
        well_filter or "all",
        result.search_mode,
        result.input_tokens,
        result.output_tokens,
        result.cost,
    ))
    conn.commit()
    conn.close()


# ── Well catalog ──────────────────────────────────────────────────────────────

WELL_OPTIONS = {
    "All wells": None,
    # Exploration
    "15/9-19 A  (Exploration)":   {"well_id": "15/9-19 A"},
    "15/9-19 B  (Exploration)":   {"well_id": "15/9-19 B"},
    "15/9-19 S  (Exploration)":   {"well_id": "15/9-19 S"},
    "15/9-19 ST2 (Exploration)":  {"well_id": "15/9-19 ST2"},
    "15/9-19 BT2 (Exploration)":  {"well_id": "15/9-19 BT2"},
    # Production — F-series
    "15/9-F-1  (Production)":     {"well_id": "15/9-F-1"},
    "15/9-F-1 A (Production)":    {"well_id": "15/9-F-1 A"},
    "15/9-F-4  (Production)":     {"well_id": "15/9-F-4"},
    "15/9-F-5  (Production)":     {"well_id": "15/9-F-5"},
    "15/9-F-7  (Production)":     {"well_id": "15/9-F-7"},
    "15/9-F-9  (Production)":     {"well_id": "15/9-F-9"},
    "15/9-F-9 A (Production)":    {"well_id": "15/9-F-9 A"},
    "15/9-F-11 (Production)":     {"well_id": "15/9-F-11"},
    "15/9-F-11 A (Production)":   {"well_id": "15/9-F-11 A"},
    "15/9-F-12 (Production)":     {"well_id": "15/9-F-12"},
    "15/9-F-14 (Production)":     {"well_id": "15/9-F-14"},
    "15/9-F-15 (Production)":     {"well_id": "15/9-F-15"},
    "15/9-F-15 A (Production)":   {"well_id": "15/9-F-15 A"},
    "15/9-F-15 B (Production)":   {"well_id": "15/9-F-15 B"},
    "15/9-F-15 C (Production)":   {"well_id": "15/9-F-15 C"},
    "15/9-F-15 D (Production)":   {"well_id": "15/9-F-15 D"},
    # Series filter
    "── Series filters ──": None,
    "All exploration wells":  {"well_series": "exploration"},
    "All production wells":   {"well_series": "production"},
}


# ── RAG singleton ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading search indexes...")
def get_rag(search_mode: str) -> VolveRAG:
    return VolveRAG.create(search_mode=search_mode)


# ── Session state ─────────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "result":         None,
        "feedback_given": False,
        "query_count":    0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    init_feedback_db()
    init_session()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🛢️ SubsurfaceIQ</div>
        <div class="app-subtitle">
            Volve Field · 26 wells · 1,759 Daily Drilling Reports · 1980–2018
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    col_search, col_well = st.columns([1, 1])

    with col_search:
        search_mode_label = st.selectbox(
            "Search mode",
            options=list(SEARCH_MODES.keys()),
            index=0,  # "Hybrid + Rerank (Best)" is default
            help="Hybrid+Rerank gives best results. Keyword is fastest.",
        )

    with col_well:
        well_label = st.selectbox(
            "Well filter",
            options=list(WELL_OPTIONS.keys()),
            index=0,  # "All wells" is default
            help="Filter results to a specific well or series.",
        )

    filter_dict = WELL_OPTIONS.get(well_label)
    # Skip separator options
    if well_label.startswith("──"):
        filter_dict = None

    # Map display label → internal mode name for VolveRAG
    MODE_MAP = {
        "Hybrid + Rerank (Best)": "hybrid_reranked",
        "Hybrid":                 "hybrid",
        "Semantic (Vector)":      "vector",
        "Keyword (Text)":         "text",
        "HyDE":                   "hyde",
    }
    search_mode = MODE_MAP.get(search_mode_label, "hybrid_reranked")

    # ── Query input ───────────────────────────────────────────────────────────
    query = st.text_area(
        "Ask a question about Volve drilling operations",
        placeholder=(
            "e.g. What mud weight was used when drilling the Hugin Formation?\n"
            "     What were the main drilling problems on well F-15 A?\n"
            "     When was the first production well spudded?"
        ),
        height=100,
        label_visibility="collapsed",
    )

    search_clicked = st.button("Search DDRs", type="primary", use_container_width=True)

    # ── Run RAG ───────────────────────────────────────────────────────────────
    if search_clicked and query.strip():
        st.session_state.feedback_given = False
        st.session_state.result         = None

        with st.spinner("Searching 1,759 drilling reports..."):
            rag = get_rag(search_mode)
            result = rag.rag(
                query.strip(),
                num_results=5,
                filter_dict=filter_dict,
            )
            st.session_state.result       = result
            st.session_state.query_count += 1

    # ── Display result ────────────────────────────────────────────────────────
    result: RAGResult | None = st.session_state.result

    if result:
        # Answer
        st.markdown(
            f'<div class="answer-block">{result.answer}</div>',
            unsafe_allow_html=True,
        )

        # Token / cost metadata strip
        st.markdown(
            f'<span class="token-badge">↑ {result.input_tokens} tokens in</span>'
            f'<span class="token-badge">↓ {result.output_tokens} tokens out</span>'
            f'<span class="token-badge">${result.cost:.5f}</span>'
            f'<span class="token-badge">{result.search_mode}</span>',
            unsafe_allow_html=True,
        )

        # Feedback
        if not st.session_state.feedback_given:
            st.markdown(
                "<div style='margin-top:1rem; font-size:0.82rem; "
                "color:#94A3B8;'>Was this answer helpful?</div>",
                unsafe_allow_html=True,
            )
            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])
            with fb_col1:
                if st.button("👍 Yes"):
                    log_feedback(result, score=1, well_filter=well_label)
                    st.session_state.feedback_given = True
                    st.rerun()
            with fb_col2:
                if st.button("👎 No"):
                    log_feedback(result, score=0, well_filter=well_label)
                    st.session_state.feedback_given = True
                    st.rerun()
        else:
            st.markdown(
                "<div style='color:#94A3B8; font-size:0.82rem; "
                "margin-top:0.75rem;'>✓ Feedback recorded</div>",
                unsafe_allow_html=True,
            )

        # Source DDRs
        if result.sources:
            st.markdown(
                "<div style='margin-top:1.5rem; font-size:0.82rem; "
                "color:#94A3B8; margin-bottom:0.5rem;'>"
                f"Retrieved from {len(result.sources)} DDRs</div>",
                unsafe_allow_html=True,
            )
            for i, src in enumerate(result.sources, 1):
                well    = src.get("well_id",  "Unknown")
                date    = src.get("date",     "Unknown")
                op      = src.get("operator", "")
                snippet = src.get("content",  "")[:280].replace("\n", " ")

                meta = f"[{i}] {well}  ·  {date}"
                if op:
                    meta += f"  ·  {op}"

                st.markdown(f"""
                <div class="source-card">
                    <div class="source-meta">{meta}</div>
                    <div class="source-snippet">{snippet}…</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    elif not search_clicked:
        st.markdown("""
        <div style="color:#1E3A5F; font-size:0.82rem; margin-top:3rem;
                    text-align:center; padding-bottom:2rem;">
            Volve open dataset · Equinor ASA · CC BY-NC-SA 4.0
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()