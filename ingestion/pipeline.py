"""
ingestion/pipeline.py

dlt pipeline: Volve Data → DuckDB corpus.
Run this once to build the knowledge base, re-run to refresh.
"""
import dlt
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Import from same package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion.extract import extract_pdf, ExtractedDoc

MANIFEST_FILE = Path("data/processed/manifest.json")
SKIP_FILE = Path("data/processed/scanned_files.txt")
MIN_QUALITY_SCORE = 0.4  # skip docs with fewer than 40% text pages


def load_skip_list() -> set[str]:
    if SKIP_FILE.exists():
        return set(SKIP_FILE.read_text().splitlines())
    return set()


@dlt.resource(name="volve_ddrs", write_disposition="replace",
    columns={
        "doc_id":      {"data_type": "text",   "nullable": False},
        "filename":    {"data_type": "text",   "nullable": False},
        "doc_type":    {"data_type": "text",   "nullable": False},
        "well_id":     {"data_type": "text",   "nullable": False},
        "well_series": {"data_type": "text",   "nullable": True},
        "date":        {"data_type": "text",   "nullable": True},
        "operator":    {"data_type": "text",   "nullable": True},
        "rig_name":    {"data_type": "text",   "nullable": True},
        "depth_md":    {"data_type": "text",   "nullable": True},
        "content":     {"data_type": "text",   "nullable": False},
        "char_count":  {"data_type": "bigint", "nullable": True},
    }
)

def volve_ddrs():
    html_files = sorted(Path("data/raw/ddr_html").glob("*.html"))
    print(f"Found {len(html_files)} HTML files")

    yielded = skipped_placeholder = skipped_error = 0

    for html_path in html_files:
        doc = extract_ddr(html_path)
        if doc is None:
            skipped_placeholder += 1
            continue
        yielded += 1
        yield doc

    print(f"\nDone: {yielded} yielded")
    print(f"      {skipped_placeholder} skipped (placeholder/empty)")
    print(f"      {skipped_error} skipped (read errors)")

def run():
    pipeline = dlt.pipeline(
        pipeline_name="volve_corpus",
        destination="duckdb",
        dataset_name="volve",
    )

    load_info = pipeline.run(volve_pages())
    print(load_info)

    # Immediate verification
    import duckdb
    conn = duckdb.connect("volve_corpus.duckdb")

    print("\n--- Corpus Summary ---")
    print(conn.execute("""
        SELECT
            doc_type,
            COUNT(*) as pages,
            AVG(char_count)::INT as avg_chars,
            COUNT(DISTINCT filename) as docs
        FROM volve.volve_pages
        GROUP BY doc_type
        ORDER BY pages DESC
    """).df().to_string(index=False))

    total = conn.execute("SELECT COUNT(*) FROM volve.volve_pages").fetchone()[0]
    print(f"\nTotal pages in corpus: {total}")


if __name__ == "__main__":
    run()