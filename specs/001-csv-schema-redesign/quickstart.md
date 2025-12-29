# Quick Start Guide: CSV Schema v2

**Feature**: CSV Schema Redesign for Citation and Search Result Tracking
**Target Audience**: Developers implementing the schema changes
**Estimated Reading Time**: 10 minutes

## Overview

This guide provides a quick walkthrough of implementing the CSV schema v2 redesign in the marimo_api_pipeline.py notebook. The changes enable citation-level analysis by expanding each API response into multiple CSV rows (one per citation).

---

## Implementation Checklist

- [ ] Update CSV_COLUMNS definition (Line ~150)
- [ ] Add helper functions (domain extraction, row expansion)
- [ ] Modify OpenAIAdapter.parse_response() for citation extraction
- [ ] Modify AnthropicAdapter.parse_response() for citation extraction
- [ ] Modify GeminiAdapter.parse_response() for citation extraction
- [ ] Update _append_row_local() for multiple row writes
- [ ] Add CFG["csv_schema_version"] parameter
- [ ] Test with real API responses
- [ ] Update CLAUDE.md documentation

---

## Step 1: Update CSV_COLUMNS (Line ~150)

**Location**: Cell defining CSV_COLUMNS (around line 150)

**Action**: Replace the existing 19-column list with the new 20-column schema

```python
CSV_COLUMNS = [
    # Narrative information (input)
    "narrative_id",           # old: note_id
    "narrative_type",         # NEW: classification from input CSV
    "narrative_prompt",       # old: note_text

    # Model information
    "model_name",             # old: provider
    "model_version",          # old: model

    # Answer information (output)
    "answer_id",              # old: run_id
    "answer_prompt",          # old: user_prompt
    "answer_text",            # old: response_text
    "answer_raw_json",        # old: raw_json
    "answer_timestamp",       # old: requested_at
    "answer_citation_list",   # NEW: JSON array of URLs

    # Source information (citation tracking - NEW)
    "source_id",              # NEW: citation identifier
    "source_url",             # NEW: citation URL
    "source_domain",          # NEW: extracted domain

    # Result information (Perplexity search results - NEW)
    "result_id",              # NEW: search result identifier
    "result_url",             # NEW: search result URL
    "result_domain",          # NEW: extracted domain
    "result_title",           # NEW: search result title
    "result_snippet",         # NEW: search result snippet
    "result_rank",            # NEW: search result rank (1-N)
]
```

**Why**: This defines the new schema with 20 columns instead of 19

---

## Step 2: Add Helper Functions

**Location**: New cell after CSV_COLUMNS definition or existing utility cell

**Action**: Add domain extraction and row expansion functions

```python
import re
from urllib.parse import urlparse

def extract_domain(url: str) -> str:
    """Extract domain from URL, removing only 'www.' prefix."""
    if not url:
        return ""

    try:
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"

        parsed = urlparse(url)
        domain = parsed.netloc

        if not domain:
            return ""

        # Remove port if present
        domain = domain.split(':')[0]

        # Remove only www. prefix (keep other subdomains)
        if domain.startswith('www.'):
            domain = domain[4:]

        return domain.lower()

    except Exception:
        return ""


def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text using regex."""
    url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    return re.findall(url_pattern, text or "")


def expand_citations_to_rows(base_row: dict, citations: list, results: list = None) -> list:
    """
    Expand one answer into multiple rows (one per citation).

    Args:
        base_row: Dict with narrative_*, model_*, answer_* fields
        citations: List of citation URLs or dicts with {url, title, ...}
        results: List of search result dicts (Perplexity only) or None

    Returns:
        List of row dicts ready for CSV writing
    """
    # Normalize citations to dict format
    normalized_citations = []
    for c in citations:
        if isinstance(c, str):
            normalized_citations.append({"url": c})
        else:
            normalized_citations.append(c)

    # Handle empty citations
    if not normalized_citations:
        normalized_citations = [{}]  # One row with empty source_* fields

    # Handle Perplexity results
    if results is None:
        results = [{}]  # Non-Perplexity case
    elif not results:
        results = [{}]  # Perplexity with no results

    # Expand rows (Cartesian product for Perplexity)
    expanded = []
    for i, citation in enumerate(normalized_citations):
        for j, result in enumerate(results):
            row = base_row.copy()

            # Citation fields
            if citation:
                row["source_id"] = f"{base_row['answer_id']}_source_{i}"
                row["source_url"] = citation.get("url", "")
                row["source_domain"] = extract_domain(citation.get("url", ""))
            else:
                row["source_id"] = ""
                row["source_url"] = ""
                row["source_domain"] = ""

            # Result fields (Perplexity only)
            if result and results != [{}]:
                row["result_id"] = f"{base_row['answer_id']}_result_{j}"
                row["result_url"] = result.get("url", "")
                row["result_domain"] = extract_domain(result.get("url", ""))
                row["result_title"] = result.get("title", "")
                row["result_snippet"] = result.get("snippet", "")
                row["result_rank"] = j + 1  # 1-based rank
            else:
                row["result_id"] = ""
                row["result_url"] = ""
                row["result_domain"] = ""
                row["result_title"] = ""
                row["result_snippet"] = ""
                row["result_rank"] = ""

            expanded.append(row)

    return expanded
```

**Why**: These helper functions extract domains and expand rows for citation tracking

---

## Step 3: Update Adapter parse_response Methods

### OpenAIAdapter (Line ~689-742)

**Action**: Modify `parse_response` to extract citations and return them

```python
def parse_response(self, resp: Any, note_id: str, note_text: str) -> dict:
    """Parse OpenAI response and extract citations."""
    choice = resp.choices[0] if resp.choices else None
    msg = choice.message if choice else None
    finish_reason = choice.finish_reason if choice else None

    response_text = msg.content if msg else ""

    # Extract citations using URL regex
    citations = extract_urls_from_text(response_text)

    # Token usage
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    # Perplexity search_results (if available)
    search_results = getattr(resp, "search_results", None)
    results = []
    if search_results and self.cfg.name == "perplexity":
        results = [
            {
                "url": r.get("url"),
                "title": r.get("title"),
                "snippet": r.get("snippet"),
            }
            for r in search_results
        ]

    return {
        "response_text": response_text,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "raw_json": json.dumps(resp.model_dump() if hasattr(resp, "model_dump") else {}),
        "citations": citations,  # NEW: list of URLs
        "results": results,       # NEW: Perplexity results
    }
```

**Why**: Extracts citations from text and search_results for row expansion

### AnthropicAdapter (Line ~744-810)

**Action**: Add citation extraction from tool_use blocks

```python
def parse_response(self, resp: Any, note_id: str, note_text: str) -> dict:
    """Parse Anthropic response and extract citations from tool usage."""
    content = getattr(resp, "content", None) or []

    # Extract text content
    text_parts = [block.text for block in content if getattr(block, "type", None) == "text"]
    response_text = " ".join(text_parts)

    # Extract citations from tool_use blocks
    citations = []
    for block in content:
        if getattr(block, "type", None) == "tool_use":
            tool_name = getattr(block, "name", None)
            if tool_name == "web_search":
                tool_input = getattr(block, "input", {})
                results = tool_input.get("results", [])
                for result in results:
                    if result.get("url"):
                        citations.append(result.get("url"))

    # Token usage
    usage = getattr(resp, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) if usage else None
    output_tokens = getattr(usage, "output_tokens", None) if usage else None

    return {
        "response_text": response_text,
        "finish_reason": getattr(resp, "stop_reason", None),
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": (input_tokens or 0) + (output_tokens or 0) if (input_tokens and output_tokens) else None,
        "raw_json": resp.model_dump_json() if hasattr(resp, "model_dump_json") else "{}",
        "citations": citations,  # NEW: list of URLs from tool_use
        "results": [],           # No search results for Claude
    }
```

**Why**: Extracts citations from Claude's web_search tool usage

### GeminiAdapter (Line ~812-883)

**Action**: Add citation extraction from grounding_metadata

```python
def parse_response(self, resp: Any, note_id: str, note_text: str) -> dict:
    """Parse Gemini response and extract citations from grounding metadata."""
    candidates = getattr(resp, "candidates", None) or []

    if not candidates:
        return {
            "response_text": "",
            "finish_reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "raw_json": "{}",
            "citations": [],
            "results": [],
        }

    candidate = candidates[0]
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []

    response_text = " ".join([p.text for p in parts if hasattr(p, "text")])

    # Extract citations from grounding_metadata
    citations = []
    grounding_metadata = getattr(candidate, "grounding_metadata", None)
    if grounding_metadata:
        grounding_chunks = getattr(grounding_metadata, "grounding_chunks", [])
        for chunk in grounding_chunks:
            web_chunk = getattr(chunk, "web", None)
            if web_chunk:
                uri = getattr(web_chunk, "uri", None)
                if uri:
                    citations.append(uri)

    # Usage metadata
    usage = getattr(resp, "usage_metadata", None)
    prompt_token_count = getattr(usage, "prompt_token_count", None) if usage else None
    candidates_token_count = getattr(usage, "candidates_token_count", None) if usage else None
    total_token_count = getattr(usage, "total_token_count", None) if usage else None

    return {
        "response_text": response_text,
        "finish_reason": getattr(candidate, "finish_reason", None),
        "prompt_tokens": prompt_token_count,
        "completion_tokens": candidates_token_count,
        "total_tokens": total_token_count,
        "raw_json": json.dumps({"candidates": [c.__dict__ for c in candidates]}),
        "citations": citations,  # NEW: list of URLs from grounding_metadata
        "results": [],           # No search results for Gemini
    }
```

**Why**: Extracts citations from Gemini's grounding metadata structure

---

## Step 4: Update Row Builder Functions (Line ~461-497)

### Modify _base_row_local

**Action**: Update column names to v2 schema

```python
def _base_row_local(cfg, provider_config, note_id, note_text, prompt_id):
    """Create base row with narrative, model, and answer metadata."""
    narrative_type = ""  # NEW: from input CSV if available

    return {
        # Narrative fields (renamed)
        "narrative_id": note_id,          # old: note_id
        "narrative_type": narrative_type,  # NEW
        "narrative_prompt": note_text,    # old: note_text

        # Model fields (renamed)
        "model_name": provider_config.name,  # old: provider
        "model_version": provider_config.model,  # old: model

        # Answer fields (renamed + new)
        "answer_id": prompt_id,           # old: run_id
        "answer_prompt": "",              # old: user_prompt (filled later)
        "answer_text": "",                # old: response_text (filled later)
        "answer_raw_json": "",            # old: raw_json (filled later)
        "answer_timestamp": datetime.now(timezone.utc).isoformat(),  # old: requested_at
        "answer_citation_list": "[]",     # NEW (filled later)

        # Source fields (NEW - filled during row expansion)
        "source_id": "",
        "source_url": "",
        "source_domain": "",

        # Result fields (NEW - filled during row expansion)
        "result_id": "",
        "result_url": "",
        "result_domain": "",
        "result_title": "",
        "result_snippet": "",
        "result_rank": "",
    }
```

**Why**: Creates base row with v2 column names and new empty fields

### Modify _success_row_local

**Action**: Update to populate answer_* fields and citation_list

```python
def _success_row_local(base_row, user_prompt, parsed):
    """Populate success row with answer data."""
    row = base_row.copy()
    row["answer_prompt"] = user_prompt
    row["answer_text"] = parsed.get("response_text", "")
    row["answer_raw_json"] = parsed.get("raw_json", "{}")
    row["answer_citation_list"] = json.dumps(parsed.get("citations", []))  # NEW
    return row
```

**Why**: Fills in answer_* fields and serializes citation list to JSON

---

## Step 5: Update _append_row_local (Line ~898-933)

**Action**: Modify to write multiple expanded rows instead of single row

```python
def _append_row_local(cfg, provider_name, base_row, citations, results=None):
    """
    Write expanded rows for a single API response.

    Args:
        cfg: Configuration dict
        provider_name: Provider name for output filename
        base_row: Base row dict with narrative_*, model_*, answer_* fields
        citations: List of citation URLs or dicts
        results: List of search result dicts (Perplexity only) or None
    """
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / f"{provider_name}.csv"

    # Expand rows (citations × results for Perplexity)
    expanded_rows = expand_citations_to_rows(base_row, citations, results)

    # Write rows incrementally
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in expanded_rows:
            writer.writerow(row)
```

**Why**: Implements row expansion and writes multiple rows per API response

### Update Call Sites

**Action**: Update calls to `_append_row_local` in the main query function

```python
# In query_one_provider_async function
parsed = adapter.parse_response(resp, note_id, note_text)
success_row = _success_row_local(base_row, user_prompt, parsed)

# Extract citations and results from parsed response
citations = parsed.get("citations", [])
results = parsed.get("results", [])  # Perplexity only

# Write expanded rows
_append_row_local(cfg, provider_config.name, success_row, citations, results)
```

**Why**: Passes citations and results to _append_row_local for expansion

---

## Step 6: Add CFG Parameter (Line ~65)

**Action**: Add schema version parameter to CFG dictionary

```python
CFG = {
    # ... existing config ...

    # Schema versioning (NEW)
    "csv_schema_version": "v2",  # Options: "v1" (old), "v2" (new)

    # ... rest of config ...
}
```

**Why**: Enables future backward compatibility and version tracking

---

## Step 7: Handle narrative_type Column

**Action**: Modify load_input_csv function to read narrative_type if present

```python
def load_input_csv(cfg):
    """Load input CSV and validate required columns."""
    input_csv = Path(cfg["input_csv"])
    note_text_col = cfg["note_text_col"]
    note_id_col = cfg["note_id_col"]

    if not input_csv.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {str(input_csv)}")

    df_in = pd.read_csv(input_csv)
    if int(cfg["max_rows"]) > 0:
        df_in = df_in.head(int(cfg["max_rows"]))

    # Validate required columns
    if note_text_col not in df_in.columns:
        raise ValueError(f"'{note_text_col}' column not found in input CSV")

    # Optional narrative_type column (NEW)
    has_narrative_type = "narrative_type" in df_in.columns
    if not has_narrative_type:
        df_in["narrative_type"] = ""  # Add empty column if missing

    return df_in, has_narrative_type
```

**Action**: Update _base_row_local to accept narrative_type parameter

```python
def _base_row_local(cfg, provider_config, note_id, note_text, narrative_type, prompt_id):
    """Create base row with narrative, model, and answer metadata."""
    return {
        "narrative_id": note_id,
        "narrative_type": narrative_type,  # Now from parameter
        "narrative_prompt": note_text,
        # ... rest of row ...
    }
```

**Action**: Update call sites to pass narrative_type from input CSV row

**Why**: Supports optional narrative_type column without breaking existing CSVs

---

## Step 8: Testing

### Manual Validation Checklist

1. **Schema structure**:
   - [ ] CSV has 20 columns in correct order
   - [ ] Column names match CSV_COLUMNS definition

2. **Row expansion**:
   - [ ] 0 citations → 1 row (empty source_* fields)
   - [ ] 2 citations → 2 rows with same answer_id
   - [ ] Perplexity: 2 citations × 3 results = 6 rows

3. **Domain extraction**:
   - [ ] `https://www.example.com` → `example.com` (www removed)
   - [ ] `https://blog.example.com` → `blog.example.com` (subdomain kept)
   - [ ] Invalid URL → empty string

4. **Citation extraction**:
   - [ ] OpenAI: URLs extracted from text
   - [ ] Claude: URLs from tool_use blocks
   - [ ] Gemini: URLs from grounding_metadata
   - [ ] Perplexity: Both citations and search_results

5. **Data integrity**:
   - [ ] All rows with same answer_id have identical answer_* fields
   - [ ] answer_citation_list matches number of source_* rows
   - [ ] result_* fields empty for non-Perplexity

### Test with Sample Data

```python
# In a notebook cell, run a small test batch
CFG["max_rows"] = 3  # Test with 3 narratives
CFG["run"] = True
CFG["confirm"] = "RUN"

# Run pipeline
# ... (execute pipeline cells) ...

# Verify output
import pandas as pd
df = pd.read_csv("llm_runs/openai.csv")
print(f"Total rows: {len(df)}")
print(f"Unique answer_ids: {df['answer_id'].nunique()}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
print(df[['answer_id', 'source_id', 'source_url', 'source_domain']].head(10))
```

**Expected Results**:
- 3 narratives × 1 provider × ~2 avg citations = ~6 rows
- CSV has 20 columns
- source_id follows pattern `{answer_id}_source_0`, `_source_1`, etc.

---

## Step 9: Update Documentation

**Action**: Update CLAUDE.md Known Limitations section

```markdown
## CSV Schema Version

**Current Schema**: v2 (2025-12-29)

**Breaking Changes from v1**:
- 9 columns renamed (note_id → narrative_id, provider → model_name, etc.)
- 9 columns removed (base_url, token counts, latency, search_results_json, error)
- 10 columns added (narrative_type, source_*, result_*)
- Row expansion: 1 answer → N rows (one per citation)

**Migration**: Set `CFG["csv_schema_version"] = "v2"` and run pipeline. Analysis scripts must be updated for new column names.
```

**Why**: Documents breaking changes for future users

---

## Common Issues and Solutions

### Issue 1: ModuleNotFoundError for urllib.parse

**Solution**: `urllib.parse` is in Python standard library (no install needed). Check Python version ≥3.11.

### Issue 2: Row count mismatch (fewer rows than expected)

**Solution**: Check citation extraction logic. Add debug print statements:
```python
print(f"Extracted {len(citations)} citations for answer_id {base_row['answer_id']}")
```

### Issue 3: Domain extraction returns empty for valid URLs

**Solution**: Check for missing https:// prefix. The extract_domain function adds it automatically, but verify input format.

### Issue 4: Perplexity rows not expanding correctly

**Solution**: Verify both citations AND results are passed to expand_citations_to_rows:
```python
citations = parsed.get("citations", [])
results = parsed.get("results", [])  # Must be list, not None
_append_row_local(cfg, provider_name, success_row, citations, results)
```

### Issue 5: CSV columns out of order

**Solution**: Ensure CSV_COLUMNS list matches DictWriter fieldnames exactly. Order matters for CSV consistency.

---

## Next Steps

After implementation:

1. **Test with subset of data** (`CFG["max_rows"] = 10`)
2. **Validate row counts** match expected citation expansion
3. **Run full pipeline** on production data
4. **Archive old v1 CSVs** for reference
5. **Update analysis scripts** to use new column names
6. **Create migration script** (optional, if historical data conversion needed)

---

## Additional Resources

- **spec.md**: Full feature specification with requirements and success criteria
- **data-model.md**: Detailed entity definitions and relationships
- **research.md**: Implementation patterns and best practices
- **contracts/csv-schema-v2.md**: Complete CSV schema contract with validation rules

For questions or issues, refer to the specification documents in `/specs/001-csv-schema-redesign/`.
