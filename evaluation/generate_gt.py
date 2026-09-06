"""
evaluation/generate_gt.py

Generates ground truth Q&A pairs from a sample of Volve DDRs.
Uses LLM structured output — same pattern as Module 4 homework.

Strategy:
- Sample ~72 DDRs (5% of 1,759) — enough for meaningful evaluation
- Stratified by well_series and well_id for coverage
- 5 questions per DDR = 360 total ground truth pairs
- Questions target CONTENT, not structure — avoids homogeneity inflation

Run once, commit ground_truth.csv to repo.
"""
import json
import time
import random
import pandas as pd
import duckdb
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DUCKDB_PATH    = Path("volve_rag.duckdb")
OUTPUT_PATH    = Path("data/ground_truth.csv")
SAMPLE_SIZE    = 72    # 5 per DDR × 72 DDRs = 360 Q&A pairs
QUESTIONS_PER  = 5
MODEL          = "gpt-5.4-mini"
RANDOM_SEED    = 42

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Ground truth prompt ───────────────────────────────────────────────────────

GT_INSTRUCTIONS = """
You are a petroleum engineer studying Volve field drilling operations.
Given one Daily Drilling Report (DDR) from the Volve North Sea field,
write 5 questions a drilling engineer might ask that are directly
answered by this specific report.

Rules:
- Each question must be answerable ONLY from THIS report's content
- Use different words from the report — no direct copying of phrases
- Include a mix: factual (what depth?), operational (why was X done?),
  technical (what mud weight?), and contextual (what formation?)
- Questions should sound like real engineer queries, not exam questions
- Do NOT ask about the report's filename, format, or metadata
""".strip()


class Questions(BaseModel):
    questions: list[str]


# ── Stratified sampler ────────────────────────────────────────────────────────

def load_stratified_sample(n: int = SAMPLE_SIZE) -> list[dict]:
    """
    Stratified sample across well_series and well_id.
    Ensures both exploration and production wells are represented,
    and no single well dominates the ground truth dataset.
    WHY stratified: random sampling on 1,759 DDRs would over-sample
    high-frequency wells (F-15 family) and under-sample exploration wells.
    """
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    df = conn.execute("""
        SELECT doc_id, filename, well_id, well_series, date, content
        FROM volve.volve_ddrs
        WHERE char_count > 500   -- skip thin placeholder DDRs
        ORDER BY well_id, date
    """).df()
    conn.close()

    print(f"Total usable DDRs: {len(df)}")
    print(f"By series: {df['well_series'].value_counts().to_dict()}")

    # Sample proportionally from each well_id
    random.seed(RANDOM_SEED)
    sampled = (
        df.groupby("well_id", group_keys=False)
          .apply(lambda g: g.sample(
              min(len(g), max(1, n // df["well_id"].nunique())),
              random_state=RANDOM_SEED
          ))
          .sample(frac=1, random_state=RANDOM_SEED)  # shuffle
          .head(n)
          .reset_index(drop=True)
    )
    print(f"\nSampled {len(sampled)} DDRs from {sampled['well_id'].nunique()} wells")
    print(sampled.groupby('well_series')['well_id'].count().to_string())
    return sampled.to_dict('records')


# ── LLM structured output ─────────────────────────────────────────────────────

def generate_questions(doc: dict, client: OpenAI) -> tuple[list[str], int]:
    """
    Generate 5 questions for one DDR using structured output.
    Returns (questions, input_tokens) — tokens for Q1-style cost tracking.
    """
    user_prompt = f"""
Well: {doc['well_id']}
Date: {doc['date']}

REPORT CONTENT:
{doc['content'][:3000]}
""".strip()

    response = client.responses.parse(
        model=MODEL,
        input=[
            {"role": "developer", "content": GT_INSTRUCTIONS},
            {"role": "user",      "content": user_prompt},
        ],
        text_format=Questions,
    )

    questions = response.output_parsed.questions
    input_tokens = response.usage.input_tokens
    return questions, input_tokens


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    client = OpenAI()
    sample = load_stratified_sample()

    records      = []
    all_tokens   = []
    failed       = 0

    for doc in tqdm(sample, desc="Generating questions"):
        try:
            questions, input_tokens = generate_questions(doc, client)
            all_tokens.append(input_tokens)

            for q in questions:
                records.append({
                    "question":    q,
                    "filename":    doc["filename"],
                    "doc_id":      doc["doc_id"],
                    "well_id":     doc["well_id"],
                    "well_series": doc["well_series"],
                    "date":        doc["date"],
                })

            # Rate limit guard — 1 req/sec is safe for gpt-5.4-mini
            time.sleep(1.0)

        except Exception as e:
            print(f"\nFailed: {doc['doc_id']}: {e}")
            failed += 1
            continue

    # Save ground truth
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)

    # Report — matches Q1 pattern from homework 4
    avg_tokens = sum(all_tokens) / len(all_tokens) if all_tokens else 0
    print(f"\n=== Ground Truth Generation Complete ===")
    print(f"Total Q&A pairs: {len(records)}")
    print(f"Failed DDRs:     {failed}")
    print(f"Avg input tokens per call: {avg_tokens:.0f}")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"\nSample questions:")
    for r in records[:3]:
        print(f"  [{r['well_id']} | {r['date']}] {r['question']}")


if __name__ == "__main__":
    run()