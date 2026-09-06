# SubsurfaceDoc-IQ
An Agentic RAG Assistant for Upstream Oil & Gas Document Intelligence (Case Study on Volve Field)

### Volve Field Drilling Intelligence Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![uv](https://img.shields.io/badge/deps-uv-orange)](https://github.com/astral-sh/uv)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED)](https://docker.com)
[![LLM Zoomcamp](https://img.shields.io/badge/DataTalks.Club-LLM%20Zoomcamp%202026-green)](https://github.com/DataTalksClub/llm-zoomcamp)

---

## Problem Statement

The Equinor Volve oil field (Norwegian North Sea) generated over
1,759 Daily Drilling Reports (DDRs) across 26 wellbores drilled between 1980
and 2018. These reports document critical operational data: formation tops,
mud weights, drilling incidents, pressure tests, geological observations, and
completion decisions. This data spread across thousands of pages of unstructured text.

A petroleum engineer asking *"what mud weight was used when drilling through
the Hugin Formation on well 15/9-F-15 A?"* has no practical way to find that
answer without manually reading dozens of reports. Multiply that across 26
wells and 38 years of operations, and the knowledge buried in these documents
becomes effectively inaccessible.

**SubsurfaceDocIQ** indexes all 1,759 DDRs and answers natural-language drilling
questions in seconds, citing the exact well and date of every source, making
the Volve knowledge base fully searchable for the first time.

> This system is designed to be field-agnostic. The same architecture can be
> applied to any operator's DDR corpus worldwide, addressing a universal pain
> point in upstream oil and gas data management.

---

## Architecture

```
Equinor Volve Open Dataset
    (1,759 HTML Daily Drilling Reports, 26 wellbores, 1980–2018)
              │
              ▼
┌─────────────────────────────┐
│  INGESTION (Phase 1)        │
│  BeautifulSoup HTML extract │
│  dlt pipeline → DuckDB      │
│  ~1,650 content-rich DDRs   │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  SEARCH (Phase 2)           │
│  minsearch text index       │
│  sqlitesearch vector index  │
│  ONNX all-MiniLM-L6-v2      │
│  Hybrid RRF + reranking     │
│  HyDE query expansion       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  RAG + AGENT (Phase 3)      │
│  VolveRAG (domain prompt)   │
│  gpt-5.4-mini (OpenAI)      │
│  PydanticAI agentic version │
└─────────────┬───────────────┘
              │
         ┌────┴────┐
         ▼         ▼
┌──────────────┐  ┌────────────────────────┐
│  STREAMLIT   │  │  MONITORING            │
│  (Phase 5)   │  │  OTel → Postgres       │
│  Well filter │  │  Grafana (5 panels)    │
│  👍👎 feedback│  │  Feedback sync bridge  │
└──────────────┘  └────────────────────────┘
```

---

## Tech Stack

| Component             | Technology                                    |
|-----------------------|-----------------------------------------------|
| Ingestion             | dlt + DuckDB                                  |
| HTML extraction       | BeautifulSoup4                                |
| Text search           | minsearch                                     |
| Vector search         | sqlitesearch + ONNX all-MiniLM-L6-v2 (384-dim)|
| Reranking             | cross-encoder/ms-marco-MiniLM-L-6-v2          |
| Query expansion       | HyDE (Hypothetical Document Embeddings)       |
| LLM                   | OpenAI gpt-5.4-mini (Responses API)           |
| Agent framework       | PydanticAI                                    |
| UI                    | Streamlit                                     |
| Observability         | OpenTelemetry + Postgres + Grafana            |
| Containerization      | Docker Compose                                |
| Dependency management | uv (locked via uv.lock)                       |

---

## Dataset

- **Source:** Equinor Volve Field Open Dataset
- **License:** Equinor Open Data Licence (free for research and education)
- **Access:** https://www.equinor.com/energy/volve-data-sharing

> Free registration required. ~40,000 files total; this project uses only text-rich PDF reports (~50–80 documents).

### Corpus used in this project

| Property | Value |
|---|---|
| Document type | Daily Drilling Reports (HTML) |
| Total files | 1,759 |
| After quality filter | ~1,650 content-rich reports |
| Wellbores | 26 (5 exploration + 21 production) |
| Well series | 15/9-19 (exploration), 15/9-F (production) |
| Date range | 1980–2018 |
| Total size | ~29.5 MB |

### The 26 wellbores

| Series | Wellbores |
|---|---|
| Exploration (15/9-19) | 19 A, 19 B, 19 BT2, 19 S, 19 ST2 |
| Production (15/9-F) | F-1, F-1 A, F-1 B, F-1 C, F-4, F-5, F-7, F-9, F-9 A, F-10, F-11, F-11 T2, F-11 A, F-11 B, F-12, F-14, F-15, F-15 A, F-15 B, F-15 C, F-15 D |

### How to get your DATABRICKS_TOKEN?

In your Databricks workspace:
- Click your avatar (top right) → Settings
Developer → Access tokens → Generate new token
- Name it subsurfacedoc-iq-ingest, expiry 90 days
- Copy immediately — shown only once
- Paste into .env as DATABRICKS_TOKEN=dapi...

Your DATABRICKS_HOST is the URL in your browser bar:
https://your-workspace.cloud.databricks.com.

---


## Quick Start

### Prerequisites

- Python 3.11+ with `uv`: `pip install uv`
- Docker Desktop with Docker Compose
- OpenAI API key
- Equinor Volve dataset access

### Step 1 — Clone and configure

```bash
git clone https://github.com/wapratama/subsurfacedoc-iq
cd subsurfacedoc-iq
cp .env.example .env
```

Open `.env` and fill in your keys:

```bash
OPENAI_API_KEY=sk-...
POSTGRES_HOST=localhost
POSTGRES_DB=subsurface_monitoring
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin
GRAFANA_ADMIN_PASSWORD=admin
```

### Step 2 — Get the Volve data

Register at https://www.equinor.com/energy/volve-data-sharing (free,
instant access).

Download the HTML Daily Drilling Reports from:
```
Well_technical_data/Daily Drilling Report - HTML Version/
```

Place all `.html` files into:
```
data/raw/ddr_html/
```

Expected: ~1,759 files, ~29.5 MB total. The `data/Volve.xlsx` inventory
documents exactly which files were used.

### Step 3 — Install dependencies

```bash
uv sync   # installs exact locked versions from uv.lock
```

### Step 4 — Download ONNX embedder model

```bash
uv run python models/download.py
```

This downloads `all-MiniLM-L6-v2` ONNX weights into `models/` (~25 MB).

### Step 5 — Build corpus and indexes

```bash
# Extract HTML → DuckDB corpus (~2 minutes)
uv run python ingestion/pipeline.py

# Build text + vector search indexes (~60-90 seconds on CPU)
uv run python search/indexer.py
```

Expected output:
```
✅ Text index built: 1650 documents
Embedding 1650 DDRs...  [████████████] 100%
✅ Vector index built and saved: search/vector_index.db
```

### Step 6 — Start all services

```bash
docker compose up -d --build
```

Wait ~15 seconds for services to initialize, then provision Grafana:

```bash
uv run python monitoring/grafana/init.py
```

Expected output:
```
✅ Grafana is healthy
✅ Datasource provisioned
✅ Dashboard provisioned
✅ Grafana ready at http://localhost:3000
```

### Step 7 — Open the app

| Service | URL | Credentials |
|---|---|---|
| SubsurfaceIQ | http://localhost:8501 | — |
| Grafana Dashboard | http://localhost:3000 | admin / admin |

---

## Project Structure

```
subsurfacedoc-iq/
│
├── data/
│   ├── raw/ddr_html/          # Volve DDR HTML files (not committed)
│   ├── samples/
│   │   └── corpus_sample.csv  # 15-row sample for verification
│   ├── ground_truth.csv       # 360 Q&A pairs for evaluation
│   └── feedback.db            # SQLite feedback log (Streamlit → Postgres bridge)
│
├── ingestion/
│   ├── extract.py             # HTML → structured text (26-well mapping, NULL cleaning)
│   └── pipeline.py            # dlt resource → DuckDB volve.volve_ddrs table
│
├── models/
│   ├── embedder.py            # ONNX all-MiniLM-L6-v2 wrapper (mean-pool + L2-norm)
│   └── download.py            # Downloads ONNX model weights from HuggingFace
│
├── search/
│   ├── indexer.py             # Builds + persists text index (pkl) + vector index (db)
│   ├── retrieval.py           # text, vector, hybrid, HyDE, hybrid_reranked + SEARCH_MODES
│   └── __init__.py
│
├── rag/
│   ├── rag_base.py            # VolveRAG: domain prompt, context builder, RAGResult, OTel
│   ├── agent.py               # PydanticAI agent with search_volve_reports + search_specific_well
│   └── __init__.py
│
├── evaluation/
│   ├── generate_gt.py         # Stratified ground truth generation (5 Q/DDR × 72 DDRs)
│   ├── metrics.py             # hit_rate, mrr, cosine_similarity_eval, llm_judge_eval
│   ├── run_eval.py            # Full evaluation: 5 retrieval methods + answer quality
│   └── results.json           # Committed evaluation results
│
├── monitoring/
│   ├── db.py                  # Postgres schema (TIMESTAMPTZ), connection factory
│   ├── otel_setup.py          # PostgresSpanExporter, TracerProvider setup
│   ├── sync_feedback.py       # SQLite → Postgres feedback bridge (polls every 10s)
│   └── grafana/
│       ├── init.py            # Auto-provisions datasource + dashboard via Grafana HTTP API
│       ├── provisioning/      # Grafana YAML config files
│       └── dashboards/
│           └── subsurfaceiq.json  # 5-panel dashboard definition
│
├── app/
│   └── streamlit_app.py       # UI: query, well filter, answer, sources, 👍👎 feedback
│
├── scripts/
│   └── entrypoint.sh          # Startup guard: checks indexes exist, starts services
│
├── tests/
│   └── test_smoke.py          # Smoke tests for all phases (pytest)
│
├── docker-compose.yml         # app + postgres + grafana (3 services)
├── Dockerfile                 # python:3.11-slim + uv sync --frozen
├── pyproject.toml
├── uv.lock                    # Exact locked dependencies (reproducibility)
├── .env.example               # All required env var keys (no values)
└── README.md
```

---

## Running Evaluation

```bash
# Generate ground truth Q&A pairs (once — ~$0.10, ~5 minutes)
uv run python evaluation/generate_gt.py

# Run full retrieval + answer quality evaluation (~10 minutes)
uv run python evaluation/run_eval.py
```

Results are printed as a ranked table and saved to `evaluation/results.json`.
The ground truth CSV is committed so evaluation is fully reproducible.

---

## Running Tests

```bash
uv run pytest tests/ -v
```

All smoke tests must pass before submission. Tests cover corpus integrity,
index loading, all 5 search modes, well filter correctness, RAG result
structure, evaluation metric math, feedback DB schema, and reproducibility
files.

---

## Monitoring Dashboard

The Grafana dashboard at http://localhost:3000 includes five panels:

| Panel | Type | SQL Source |
|---|---|---|
| Recent Queries | Table | `spans WHERE span_name='rag' ORDER BY timestamp DESC LIMIT 10` |
| Input Tokens Over Time | Time series | `spans.input_tokens` by timestamp |
| Response Latency (ms) | Time series | `spans.duration_ms` by timestamp |
| User Feedback | Pie chart | `feedback GROUP BY score` |
| Cost Per Query (USD) | Time series | `spans.cost` by timestamp |

All data flows: Streamlit → SQLite feedback.db → sync bridge → Postgres →
Grafana. OTel spans flow: VolveRAG.rag() → PostgresSpanExporter → Postgres
→ Grafana.

---

## Best Practices Implemented

Three bonus-point techniques from Module 6 are implemented in
`search/retrieval.py`:

**1. Hybrid Search with RRF**
Combines text (TF-IDF) and vector (semantic) search results using
Reciprocal Rank Fusion. `k=1` selected from empirical evaluation on
this corpus type (Module 4 finding). Documents appearing in both result
lists score approximately 2× higher than documents appearing in only one.

**2. HyDE — Hypothetical Document Embeddings**
For informal queries ("why didn't they produce from the lower zone?"),
generates a hypothetical DDR-style answer using gpt-5.4-mini, then embeds
that answer for retrieval. The hypothetical text uses the same formal
petroleum engineering vocabulary as the actual DDRs, bridging the
vocabulary gap between informal questions and technical documents.

**3. Cross-Encoder Reranking**
After hybrid retrieval (15 candidates), `cross-encoder/ms-marco-MiniLM-L-6-v2`
evaluates each (query, chunk) pair jointly to rerank to the top 5. Catches
false positives where bi-encoder embeddings conflate semantically adjacent
but topically different terms (e.g. "formation pressure" vs "formation damage").

---

## Design Decisions

**Why HTML DDRs only (not PDFs, DOC, XLSX)?**
The 1,759 HTML DDRs are the cleanest and most structurally consistent text
source in the Volve dataset. BeautifulSoup extraction is deterministic —
no PDF scan detection, no OCR, no binary format parsers. This keeps the
ingestion pipeline simple and reproducible, which directly serves the
evaluation and monitoring criteria. Multi-format expansion is planned (see
Future Improvements).

**Why no chunking?**
Each DDR covers exactly one 24-hour period of operations — it is already
a natural semantic unit averaging 2,000–3,500 characters. Applying a
sliding window would artificially split "Formation drilled: Hugin" from
the mud weight recorded in the same report's next paragraph. One DDR =
one document = one index entry.

**Why `k=1` in RRF?**
Empirical finding from Module 4 evaluation on a similar corpus type: `k=1`
aggressively rewards top-rank agreement between text and vector search,
producing higher MRR than the standard `k=60`. Confirmed by running
`hybrid_search` at k=1, 50, 100, 200 on the ground truth dataset.

**Why sqlitesearch over minsearch.VectorSearch?**
sqlitesearch persists the vector index to disk, so the 60–90 second
embedding computation runs once (at `search/indexer.py`) and never again
on app startup. minsearch.VectorSearch stores vectors in RAM and requires
re-embedding on every restart — unusable in a containerized deployment.

---

## Future Improvements

The following are planned and noted in the codebase but deferred to
keep the initial submission focused and reproducible:

### Additional Document Types
- **Well completion reports** (PDF) — final well summaries, DST results,
  perforation decisions per wellbore
- **Petrophysical interpretation reports** (PDF) — formation evaluation,
  porosity/saturation analysis per zone
- **Biostratigraphy reports** (PDF) — geological age determination
- **Geochemistry reports** (PDF) — fluid analysis, source rock correlation
- **Drilling programmes** (DOC) — pre-drill well design and objectives
- **Production data** (XLSX) — monthly well-level oil/gas/water volumes,
  GOR, water cut trends

### Technical Enhancements
- **Multi-format extraction:** PyMuPDF for PDF, python-docx for Word,
  pandas for XLSX — with `doc_type`-aware chunking (section-aware for
  reports, smaller windows for production data tables)
- **Norwegian DDR handling:** ~15% of early exploration DDRs (15/9-19
  series, 1992–1997) are in Norwegian. A multilingual embedding model
  (e.g. `paraphrase-multilingual-MiniLM-L12-v2`) would improve retrieval
  on these documents
- **Well-level cross-document reasoning:** correlate DDR formation
  observations with completion findings from the same wellbore across
  time — enables questions like "how did drilling problems on F-15 affect
  the completion design?"
- **OSDU-compliant metadata schema** for enterprise integration with
  Schlumberger Delfi / Halliburton iEnergy ecosystems

### Deployment
- **Cloud deployment** (GCP Cloud Run or Render) for public access
- **Scheduled corpus refresh** via Kestra (Module 3 patterns) — ingest
  new DDRs automatically when added to the source dataset
- **Production vector database** (Qdrant or FAISS) for >100K document
  scale beyond minsearch's in-memory limits

### Application Layer
- **Comparative well analysis:** "compare mud programme across F-15 A
  and F-15 B" — multi-well context synthesis
- **Timeline reconstruction:** given a well, reconstruct its full
  drilling history from spud to completion in chronological order
- **Anomaly detection:** flag DDRs with unusually high NPT (non-productive
  time) or mud losses for operational review

---

## Acknowledgements

- **Dataset:** Equinor ASA for releasing the Volve field dataset under
  an open licence, a landmark contribution to petroleum data science ([Original Volve license terms](https://cdn.equinor.com/files/h61q9gi9/global/de6532f6134b9a953f6c41bac47a0c055a3712d3.pdf)).
- **Course:** DataTalks.Club LLM Zoomcamp 2026 by [Alexey Grigorev](https://www.linkedin.com/in/agrigorev/) for
  the structured curriculum that made this project possible
- **Libraries:** The open-source ecosystem — dlt, minsearch, sqlitesearch,
  sentence-transformers, BeautifulSoup4, Streamlit, OpenTelemetry,
  PydanticAI, and Grafana

---

*For your information, some markdown file (README & others), and some code in my repo are created with the help of AI assistant (ChatGPT, Claude, and/or Gemini) using some references from LLM Zoomcamp repo and some adjustment from me. For your main references, please visit the [website](https://datatalks.club/docs/courses/llm-zoomcamp/) and original [Github repo](https://github.com/DataTalksClub/llm-zoomcamp/).*

**_Thanks for your visit & Keep Learning!!!_**