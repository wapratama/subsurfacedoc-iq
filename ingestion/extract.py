"""
ingestion/extract.py

Extracts and cleans text from Volve Daily Drilling Report HTML files.
Grounded in actual DDR structure observed from Volve dataset 2025.

Key structural facts (verified from actual files):
- Template: bulma CSS framework, consistent section headers via CSS classes
- Sections: Wellbore info tables + variable content sections (Operations,
  Drilling Fluid, Lithology, Survey Station, Core Info, Welltest Info)
- NULL sentinel: -999.99 (WITSML standard for missing values)
- Language: English (uppercase style in older reports pre-2000)
- Date range: 1980-2018 (but pre-1992 files are placeholder summaries)
- 26 wellbores across 2 well series: 15/9-19 (exploration) and 15/9-F (production)
"""

import re
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime


# ── Well name mapping (exact 26 wellbores, ordered longest-first for prefix match) ──
WELL_NAME_MAP = {
    "15_9_19_BT2":  "15/9-19 BT2",
    "15_9_19_ST2":  "15/9-19 ST2",
    "15_9_19_A":    "15/9-19 A",
    "15_9_19_B":    "15/9-19 B",
    "15_9_19_S":    "15/9-19 S",
    "15_9_F_1_A":   "15/9-F-1 A",
    "15_9_F_1_B":   "15/9-F-1 B",
    "15_9_F_1_C":   "15/9-F-1 C",
    "15_9_F_1":     "15/9-F-1",
    "15_9_F_4":     "15/9-F-4",
    "15_9_F_5":     "15/9-F-5",
    "15_9_F_7":     "15/9-F-7",
    "15_9_F_9_A":   "15/9-F-9 A",
    "15_9_F_9":     "15/9-F-9",
    "15_9_F_10":    "15/9-F-10",
    "15_9_F_11_T2": "15/9-F-11 T2",
    "15_9_F_11_A":  "15/9-F-11 A",
    "15_9_F_11_B":  "15/9-F-11 B",
    "15_9_F_11":    "15/9-F-11",
    "15_9_F_12":    "15/9-F-12",
    "15_9_F_14":    "15/9-F-14",
    "15_9_F_15_A":  "15/9-F-15 A",
    "15_9_F_15_B":  "15/9-F-15 B",
    "15_9_F_15_C":  "15/9-F-15 C",
    "15_9_F_15_D":  "15/9-F-15 D",
    "15_9_F_15":    "15/9-F-15",
}

# Minimum useful content — below this a DDR is a placeholder
MIN_CONTENT_CHARS = 300


def extract_well_id(filename: str) -> str:
    """
    Extract well ID using longest-prefix match against known 26 wellbores.
    Must be longest-first to avoid '15_9_F_15' matching before '15_9_F_15_A'.
    """
    stem = Path(filename).stem  # remove .html
    for prefix, well_name in WELL_NAME_MAP.items():
        if stem.startswith(prefix):
            return well_name
    return "unknown"


def extract_date(filename: str) -> str:
    """
    Extract ISO date from filename.
    Pattern: {well_prefix}_{YYYY}_{MM}_{DD}.html
    """
    match = re.search(r'(\d{4})_(\d{2})_(\d{2})(?:\.html)?$', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return "unknown"


def parse_wellbore_info(soup: BeautifulSoup) -> dict:
    """
    Extract structured metadata from wellbore info tables.
    Returns dict of key→value from ColHead/ColValue pairs.
    """
    metadata = {}
    for table in soup.find_all('table', class_='ddr-report-wellbore-info-table'):
        for row in table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 2:
                key   = cells[0].get_text(strip=True).rstrip(':')
                value = cells[1].get_text(strip=True)
                # Skip empty values and NULL sentinels
                if value and value != '-999.99':
                    metadata[key] = value
    return metadata


def extract_section_text(soup: BeautifulSoup, section_name: str) -> str:
    """
    Extract text content of a named DDR section.
    Sections are identified by TextHeader class followed by TextData content.
    Also captures table content (Operations remarks, Lithology, etc.)
    """
    content_parts = []
    found = False

    for tag in soup.find_all(['div', 'p', 'table']):
        # Detect section header
        if tag.get('class') and 'TextHeader' in tag.get('class', []):
            if section_name.lower() in tag.get_text(strip=True).lower():
                found = True
                continue
            elif found:
                break  # hit next section header

        if found:
            text = tag.get_text(separator=' ', strip=True)
            # Clean NULL sentinels
            text = text.replace('-999.99', '')
            text = re.sub(r'\s+', ' ', text).strip()
            if text and len(text) > 3:
                content_parts.append(text)

    return '\n'.join(content_parts)


def clean_extracted_text(text: str) -> str:
    """
    Remove DDR-specific noise from extracted text.
    - NULL sentinel values (-999.99)
    - Orphan header lines (field name with no value)
    - Excessive whitespace
    - Unit-only lines like "()" or "(m)" with no values
    """
    # Replace NULL sentinel
    text = text.replace('-999.99', '')

    lines = text.split('\n')
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip lines that are just empty field names (end with ":" or "()")
        if re.match(r'^[\w\s]+\(\):\s*$', line):
            continue
        if re.match(r'^[\w\s]+\s*\(\)\s*$', line):
            continue
        # Skip lines shorter than 2 chars
        if len(line) < 2:
            continue
        cleaned.append(line)

    # Collapse multiple blank lines
    text = '\n'.join(cleaned)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def build_rag_text(metadata: dict, soup: BeautifulSoup) -> str:
    """
    Build the final embeddable text representation of a DDR.

    Structure: metadata header → key sections in priority order.
    Designed to be both human-readable and retrieval-optimized.
    Sections ordered by RAG value: activities first (richest narrative),
    then operations details, then technical measurements.
    """
    parts = []

    # ── Metadata header ────────────────────────────────────────────────────
    wellbore  = metadata.get('Wellbore', metadata.get('wellbore', ''))
    period    = metadata.get('period', '')
    operator  = metadata.get('Operator', '')
    rig       = metadata.get('Rig Name', '')
    depth_md  = metadata.get('Depth mMd', metadata.get('Depth mMD', ''))
    dist      = metadata.get('Dist Drilled (m)', '')

    header_lines = [f"DAILY DRILLING REPORT — Volve Field"]
    if wellbore:  header_lines.append(f"Wellbore: {wellbore}")
    if period:    header_lines.append(f"Period: {period}")
    if operator:  header_lines.append(f"Operator: {operator}")
    if rig:       header_lines.append(f"Rig: {rig}")
    if depth_md:  header_lines.append(f"Current Depth MD: {depth_md} m")
    if dist:      header_lines.append(f"Distance Drilled: {dist} m")

    parts.append('\n'.join(header_lines))

    # ── Priority sections — order determines embedding emphasis ─────────────
    SECTIONS = [
        "Summary of activities",       # richest narrative — goes first
        "Summary of planned activities",
        "Operations",                  # hour-by-hour remarks
        "Lithology Information",       # formation descriptions
        "Core Information",            # coring results
        "Welltest Information",        # DST/flow test results
        "Gas Reading Information",     # hydrocarbon shows
        "Pore Pressure",
        "Survey Station",
        "Drilling Fluid",
    ]

    for section in SECTIONS:
        section_text = extract_section_text(soup, section)
        if section_text and len(section_text.strip()) > 10:
            parts.append(f"\n{section.upper()}:\n{section_text}")

    full_text = '\n\n'.join(parts)
    return clean_extracted_text(full_text)


def extract_ddr(html_path: Path) -> dict | None:
    """
    Extract one Daily Drilling Report HTML into a structured dict
    ready for dlt ingestion → DuckDB → minsearch indexing.

    Returns None for placeholder/empty reports.
    """
    try:
        content = html_path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        try:
            content = html_path.read_bytes().decode('latin-1', errors='replace')
        except Exception as e:
            print(f"  Read error {html_path.name}: {e}")
            return None

    soup = BeautifulSoup(content, 'html.parser')

    # Extract metadata first
    metadata = parse_wellbore_info(soup)

    # Add wellbore and period from SubReportHeader
    for tag in soup.find_all(class_='DDR-SubReportHeader'):
        text = tag.get_text(strip=True)
        if text.startswith('Wellbore:'):
            metadata['Wellbore'] = text.replace('Wellbore:', '').strip()
        elif text.startswith('Period:'):
            metadata['period'] = text.replace('Period:', '').strip()

    # Build RAG-optimized text
    rag_text = build_rag_text(metadata, soup)

    # Reject placeholder DDRs
    if len(rag_text) < MIN_CONTENT_CHARS:
        return None

    # Reject reports where activities section is literally "None"
    activities = extract_section_text(soup, "Summary of activities")
    if activities.strip().lower() in ('none', '', 'n/a'):
        return None

    well_id = extract_well_id(html_path.name)
    date    = extract_date(html_path.name)

    # Well series: exploration (15/9-19) vs production (15/9-F)
    well_series = "exploration" if "19" in well_id.split("/")[-1] else "production"

    return {
        "doc_id":      html_path.stem,
        "filename":    html_path.name,
        "doc_type":    "drilling_report",
        "well_id":     well_id,
        "well_series": well_series,
        "date":        date,
        "operator":    metadata.get('Operator', ''),
        "rig_name":    metadata.get('Rig Name', ''),
        "depth_md":    metadata.get('Depth mMd', metadata.get('Depth mMD', '')),
        "content":     rag_text,
        "char_count":  len(rag_text),
    }


# ── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: uv run python ingestion/extract.py <path_to_ddr.html>")
        sys.exit(1)

    result = extract_ddr(Path(sys.argv[1]))
    if result:
        print(f"well_id:     {result['well_id']}")
        print(f"well_series: {result['well_series']}")
        print(f"date:        {result['date']}")
        print(f"operator:    {result['operator']}")
        print(f"rig_name:    {result['rig_name']}")
        print(f"depth_md:    {result['depth_md']}")
        print(f"char_count:  {result['char_count']}")
        print(f"\n--- Content Preview (first 800 chars) ---")
        print(result['content'][:800])
    else:
        print("Result: None (placeholder or insufficient content)")