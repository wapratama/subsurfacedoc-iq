"""
monitoring/db.py

Postgres connection and table initialization for SubsurfaceIQ monitoring.
All monitoring tables created here — single source of truth for schema.

WHY Postgres over SQLite for monitoring:
- Grafana connects natively to Postgres via built-in datasource
- Concurrent writes from OTel exporter + feedback sync without locking
- Timezone-aware timestamps (TIMESTAMPTZ) — critical for Grafana display
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        dbname=os.environ.get("POSTGRES_DB",   "subsurface_monitoring"),
        user=os.environ.get("POSTGRES_USER",   "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", "admin"),
    )


def init_tables():
    """
    Create all monitoring tables if they don't exist.
    Called once at app startup and by docker-compose healthcheck.

    CRITICAL: Use TIMESTAMPTZ (not TIMESTAMP) everywhere.
    Without timezone info, Grafana's default UTC display shows wrong
    times for Jakarta-based users and audit trails break.
    """
    conn = get_connection()
    cur  = conn.cursor()

    # Spans table — OTel trace data from each RAG call
    cur.execute("""
        CREATE TABLE IF NOT EXISTS spans (
            id            SERIAL PRIMARY KEY,
            timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            span_name     TEXT        NOT NULL,
            query         TEXT,
            well_id       TEXT,
            search_mode   TEXT,
            input_tokens  INTEGER,
            output_tokens INTEGER,
            cost          REAL,
            duration_ms   REAL,
            start_time    BIGINT,
            end_time      BIGINT
        )
    """)

    # Feedback table — user 👍/👎 from Streamlit
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id            SERIAL PRIMARY KEY,
            timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            query         TEXT        NOT NULL,
            answer        TEXT        NOT NULL,
            score         INTEGER     NOT NULL,
            well_id       TEXT,
            search_mode   TEXT,
            input_tokens  INTEGER,
            output_tokens INTEGER,
            cost          REAL
        )
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Postgres tables initialized")


if __name__ == "__main__":
    init_tables()