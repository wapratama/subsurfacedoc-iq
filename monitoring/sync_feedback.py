"""
monitoring/sync_feedback.py

Bridges SQLite feedback (Phase 5) → Postgres (Phase 6 Grafana).
Runs as a background process alongside the Streamlit app.
Polls every 10 seconds — lightweight, no dependencies on OTel.
"""
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from monitoring.db import get_connection

SQLITE_PATH = Path("data/feedback.db")


def ensure_synced_column():
    """Add sync tracking column if this is first run."""
    conn = sqlite3.connect(str(SQLITE_PATH))
    try:
        conn.execute("ALTER TABLE feedback ADD COLUMN synced INTEGER DEFAULT 0")
        conn.commit()
    except Exception:
        pass  # Column already exists
    conn.close()


def sync_once() -> int:
    """Sync all unsynced rows. Returns count synced."""
    if not SQLITE_PATH.exists():
        return 0

    sqlite_conn = sqlite3.connect(str(SQLITE_PATH))
    sqlite_conn.row_factory = sqlite3.Row
    rows = sqlite_conn.execute(
        "SELECT * FROM feedback WHERE synced = 0 ORDER BY id"
    ).fetchall()

    if not rows:
        sqlite_conn.close()
        return 0

    pg_conn = get_connection()
    pg_cur  = pg_conn.cursor()
    synced  = 0

    for row in rows:
        try:
            # Parse timestamp — SQLite stores as ISO string
            ts = row["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)

            pg_cur.execute("""
                INSERT INTO feedback
                (timestamp, query, answer, score, well_id,
                 search_mode, input_tokens, output_tokens, cost)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                ts,
                row["query"],
                row["answer"],
                row["score"],
                row.get("well_id"),
                row.get("search_mode"),
                row.get("input_tokens"),
                row.get("output_tokens"),
                row.get("cost"),
            ))

            sqlite_conn.execute(
                "UPDATE feedback SET synced=1 WHERE id=?", (row["id"],)
            )
            synced += 1

        except Exception as e:
            print(f"  Row {row['id']} failed: {e}")

    pg_conn.commit()
    sqlite_conn.commit()
    pg_cur.close()
    pg_conn.close()
    sqlite_conn.close()
    return synced


def run():
    print("Feedback sync started — polling every 10s")
    ensure_synced_column()
    while True:
        try:
            n = sync_once()
            if n > 0:
                print(f"[{datetime.now(timezone.utc).isoformat()}] Synced {n} rows")
        except Exception as e:
            print(f"Sync error: {e}")
        time.sleep(10)


if __name__ == "__main__":
    run()