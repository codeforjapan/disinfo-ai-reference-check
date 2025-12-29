# Data Model: CSV Schema v2

**Feature**: CSV Schema Redesign for Citation and Search Result Tracking
**Version**: 2.0.0
**Status**: Draft

## Overview

This document defines the data model for CSV schema v2, which restructures output from a flat 19-column format to a normalized 20-column format with row expansion for citations and search results.

## Key Design Decisions

1. **Row Expansion**: One API response creates N rows (one per citation). This enables standard CSV operations (filter, group, join) without JSON parsing.
2. **Denormalization**: Answer data (answer_text, answer_timestamp) is duplicated across expanded rows for query efficiency.
3. **Optional Fields**: narrative_type and result_* columns are optional/provider-specific rather than always populated.
4. **Domain Extraction**: Domains auto-extracted from URLs for easier aggregation (e.g., grouping by news source).

## Entities

### Narrative (Input Entity)

Represents a disinformation narrative to be fact-checked.

**Attributes**:
- `narrative_id` (string, unique): Unique identifier for the narrative. Maps from input CSV column (configurable via CFG["note_id_col"], default "narrative_id").
- `narrative_type` (string, optional): Classification of narrative type (e.g., "misinformation", "satire", "true"). Read from input CSV if column exists; empty string if missing.
- `narrative_prompt` (string, required): The text content of the narrative. Maps from input CSV column (configurable via CFG["note_text_col"], default "prompt").

**Source**: Input CSV file (CFG["input_csv"], default "narratives.csv")

**Validation Rules**:
- narrative_prompt MUST NOT be empty or null (enforced in load_input_csv)
- narrative_id SHOULD be unique (not enforced; duplicates create separate answer_ids)
- narrative_type is optional; empty string if column missing

**Example**:
```python
{
    "narrative_id": "N001",
    "narrative_type": "misinformation",
    "narrative_prompt": "このワクチンは危険である"
}
```

---

### Model (Configuration Entity)

Represents an AI provider/model combination.

**Attributes**:
- `model_name` (string, required): Provider name. Maps from ProviderConfig.name (e.g., "openai", "claude", "gemini", "grok", "perplexity").
- `model_version` (string, required): Specific model identifier. Maps from ProviderConfig.model (e.g., "gpt-4", "claude-3-5-sonnet-20241022", "gemini-2.0-flash-exp").

**Source**: PROVIDERS list configuration (marimo_api_pipeline.py, lines 196-331)

**Example**:
```python
{
    "model_name": "openai",
    "model_version": "gpt-4-turbo-2024-04-09"
}
```

---

### Answer (Output Entity)

Represents a fact-checking response from a model for a specific narrative.

**Attributes**:
- `answer_id` (string, unique): Unique identifier for this response. Generated as UUID or run-specific ID. Maps from old "run_id".
- `answer_prompt` (string, required): The exact user prompt sent to the model. Constructed from CFG["user_prompt_template"] with {note_text} substitution. Maps from old "user_prompt".
- `answer_text` (string, required): The model's response text. Extracted from API response (varies by provider). Maps from old "response_text".
- `answer_timestamp` (string, required): ISO 8601 formatted datetime when request was made. Maps from old "requested_at".
- `answer_citation_list` (string, required): JSON array of citation URLs. Extracted from provider response; format: `["url1", "url2"]`. Empty array `[]` if no citations. NEW COLUMN.
- `answer_raw_json` (string, required): Complete JSON serialization of API response for debugging/auditing. Maps from old "raw_json".

**Relationships**:
- One Answer belongs to one Narrative (answer → narrative_id)
- One Answer belongs to one Model (answer → model_name + model_version)
- One Answer has many Citations (1:N, expanded into separate rows)
- One Answer has many SearchResults (1:N, Perplexity only, expanded into separate rows)

**Derivation**: Created from API response via ProviderAdapter.parse_response()

**Example**:
```python
{
    "answer_id": "550e8400-e29b-41d4-a716-446655440000",
    "answer_prompt": "このワクチンは危険である\n\n「はい」または「いいえ」で回答してください...",
    "answer_text": "いいえ; このワクチンは安全です。厚生労働省の公式データによると...",
    "answer_timestamp": "2025-12-29T17:30:00Z",
    "answer_citation_list": "[\"https://www.mhlw.go.jp/vaccine\", \"https://www.who.int/vaccine-safety\"]",
    "answer_raw_json": "{\"id\": \"chatcmpl-...\", \"choices\": [...], ...}"
}
```

---

### Citation/Source (Output Entity - Row Expansion)

Represents a source cited by the model in its answer. Each citation creates a separate CSV row.

**Attributes**:
- `source_id` (string, required): Citation identifier within answer scope. Format: "{answer_id}_source_{index}" (e.g., "550e8400_source_1"). NEW COLUMN.
- `source_url` (string, required): Full URL of the cited source. Extracted from answer_citation_list. NEW COLUMN.
- `source_domain` (string, required): Domain extracted from source_url (e.g., "www.mhlw.go.jp" or "mhlw.go.jp"). Auto-extracted using urllib.parse. Empty string if URL unparseable. NEW COLUMN.

**Relationships**:
- Many Citations belong to one Answer (N:1)
- Each Citation is represented as one CSV row with Answer data duplicated

**Derivation**: Extracted from API response by provider-specific logic:
- OpenAI: Regex/URL parsing of message.content
- Claude: Parse tool usage history in raw response
- Gemini: Parse grounding_metadata
- xAI/Grok: Parse citation fields
- Perplexity: Parse citations array

**Row Expansion Logic**:
```python
# Pseudo-code
if len(citations) == 0:
    create 1 row with empty source_* fields
else:
    for each citation in citations:
        create 1 row with source_id, source_url, source_domain populated
        duplicate answer_* fields across all rows
```

**Example**:
```python
# Answer has 2 citations → creates 2 rows
Row 1: {
    "source_id": "550e8400_source_0",
    "source_url": "https://www.mhlw.go.jp/vaccine",
    "source_domain": "www.mhlw.go.jp",
    # ... answer_* fields duplicated ...
}
Row 2: {
    "source_id": "550e8400_source_1",
    "source_url": "https://www.who.int/vaccine-safety",
    "source_domain": "www.who.int",
    # ... answer_* fields duplicated ...
}
```

---

### SearchResult (Output Entity - Perplexity Only)

Represents a search engine result retrieved by Perplexity. Only applicable to Perplexity provider.

**Attributes**:
- `result_id` (string, provider-specific): Search result identifier within answer scope. Format: "{answer_id}_result_{index}". NEW COLUMN.
- `result_url` (string, provider-specific): URL of the search result. NEW COLUMN.
- `result_domain` (string, provider-specific): Domain extracted from result_url. NEW COLUMN.
- `result_title` (string, provider-specific): Title of the search result page. NEW COLUMN.
- `result_snippet` (string, provider-specific): Text excerpt from the search result. NEW COLUMN.
- `result_rank` (integer, provider-specific): Position in search results (1-based index). NEW COLUMN.

**Relationships**:
- Many SearchResults belong to one Answer (N:1, Perplexity only)
- SearchResults and Citations have a Cartesian product relationship for row expansion

**Derivation**: Extracted from Perplexity API response field `search_results`

**Row Expansion Logic (Perplexity only)**:
```python
# Pseudo-code
citations = extract_citations(response)  # e.g., 2 citations
results = extract_search_results(response)  # e.g., 3 results

if len(citations) == 0:
    citations = [None]  # ensure at least 1 row
if len(results) == 0:
    results = [None]  # non-Perplexity case

# Cartesian product: 2 citations × 3 results = 6 rows
for citation in citations:
    for result in results:
        create 1 row with citation data + result data + answer data
```

**Example (Perplexity with 2 citations × 2 results = 4 rows)**:
```python
Row 1: {source_id: "...0", source_url: "url1", result_id: "...0", result_url: "result1", result_rank: 1, ...}
Row 2: {source_id: "...0", source_url: "url1", result_id: "...1", result_url: "result2", result_rank: 2, ...}
Row 3: {source_id: "...1", source_url: "url2", result_id: "...0", result_url: "result1", result_rank: 1, ...}
Row 4: {source_id: "...1", source_url: "url2", result_id: "...1", result_url: "result2", result_rank: 2, ...}
```

**Non-Perplexity Providers**: All result_* fields are empty strings.

---

## CSV Schema Definition

### New Schema (v2)

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

**Total Columns**: 20 (was 19 in v1)

### Removed Columns (from v1)

- `base_url`: Redundant (provider-specific, not needed for analysis)
- `system_prompt`: Fixed value, stored in CFG, not needed per-row
- `finish_reason`: Debugging metadata, available in answer_raw_json
- `prompt_tokens`: Token usage metadata, available in answer_raw_json
- `completion_tokens`: Token usage metadata, available in answer_raw_json
- `total_tokens`: Token usage metadata, available in answer_raw_json
- `latency_ms`: Performance metadata, can be recalculated if needed
- `search_results_json`: Replaced by structured result_* columns
- `error`: Error rows not written to output CSV (logged separately)

**Rationale**: Remove redundant metadata that clutters analysis and can be derived from answer_raw_json. Focus on research-relevant columns.

---

## Schema Migration

### Version Detection

```python
# In CFG dictionary
CFG = {
    "csv_schema_version": "v2",  # Options: "v1" (old schema) or "v2" (new schema)
    # ... other config ...
}
```

### Migration Path

**From v1 to v2**:
1. Update CFG["csv_schema_version"] = "v2"
2. Run pipeline to generate new CSV files
3. Old v1 CSVs remain unchanged in llm_runs/ (manual archiving recommended)
4. Analysis scripts must be updated to use new column names

**Historical Data**: Out of scope. A separate migration script is needed to convert v1 CSVs to v2 format.

---

## Implementation Notes

### Domain Extraction Function

```python
from urllib.parse import urlparse

def extract_domain(url: str) -> str:
    """Extract domain from URL. Returns empty string if unparseable."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        # Return netloc (includes subdomain: www.example.com)
        # Alternative: Remove 'www.' if desired (parsed.netloc.replace('www.', ''))
        return parsed.netloc or ""
    except Exception:
        return ""
```

**Decision**: Keep subdomains (www.example.com) rather than removing them, as some sources use different subdomains meaningfully (en.wikipedia.org vs ja.wikipedia.org).

### Row Expansion Function

```python
def expand_rows(base_row: dict, citations: list, results: list) -> list:
    """
    Expand one answer into multiple rows (citations × results).

    Args:
        base_row: Dict with narrative_*, model_*, answer_* fields
        citations: List of dicts with {url: str} or empty list
        results: List of dicts with Perplexity result fields or empty list

    Returns:
        List of row dicts ready for CSV writing
    """
    if not citations:
        citations = [{}]  # Empty citation (all source_* fields empty)
    if not results:
        results = [{}]  # Empty result (all result_* fields empty)

    expanded = []
    for i, citation in enumerate(citations):
        for j, result in enumerate(results):
            row = base_row.copy()

            # Add citation fields
            if citation:
                row["source_id"] = f"{base_row['answer_id']}_source_{i}"
                row["source_url"] = citation.get("url", "")
                row["source_domain"] = extract_domain(citation.get("url", ""))
            else:
                row["source_id"] = ""
                row["source_url"] = ""
                row["source_domain"] = ""

            # Add result fields (Perplexity only)
            if result:
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

### CSV Writing Update

Modify `_append_row_local()` to handle multiple rows:

```python
def _append_row_local(cfg, provider_name, rows: list):
    """
    Write multiple expanded rows to provider CSV.

    Args:
        cfg: Configuration dict
        provider_name: Provider name for output filename
        rows: List of row dicts (from expand_rows)
    """
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / f"{provider_name}.csv"

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
```

---

## Validation Rules

### Input Validation
- narrative_prompt MUST NOT be empty (existing validation)
- narrative_type column is optional in input CSV
- If narrative_type column exists but row value is empty, use empty string

### Output Validation
- answer_id MUST be unique per API call
- answer_citation_list MUST be valid JSON array (even if empty: `[]`)
- source_id format MUST match "{answer_id}_source_{index}"
- result_id format MUST match "{answer_id}_result_{index}" (Perplexity only)
- result_rank MUST be 1-based integer or empty string
- All result_* fields MUST be empty for non-Perplexity providers

### Row Count Validation
- Zero citations → 1 row (empty source_* fields)
- N citations × 0 results → N rows
- N citations × M results (Perplexity) → N×M rows

---

## Examples

### Example 1: OpenAI with 2 Citations

**Input**:
```python
narrative = {
    "narrative_id": "N001",
    "narrative_type": "misinformation",
    "narrative_prompt": "ワクチンは危険"
}
```

**API Response**: OpenAI returns answer with 2 URLs in text

**Output CSV Rows** (2 rows):
```csv
narrative_id,narrative_type,narrative_prompt,model_name,model_version,answer_id,answer_prompt,answer_text,answer_raw_json,answer_timestamp,answer_citation_list,source_id,source_url,source_domain,result_id,result_url,result_domain,result_title,result_snippet,result_rank
N001,misinformation,ワクチンは危険,openai,gpt-4,550e8400,{prompt},いいえ; ワクチンは安全...,{json},2025-12-29T17:30:00Z,"[""https://mhlw.go.jp"",""https://who.int""]",550e8400_source_0,https://mhlw.go.jp,mhlw.go.jp,,,,,
N001,misinformation,ワクチンは危険,openai,gpt-4,550e8400,{prompt},いいえ; ワクチンは安全...,{json},2025-12-29T17:30:00Z,"[""https://mhlw.go.jp"",""https://who.int""]",550e8400_source_1,https://who.int,who.int,,,,,
```

### Example 2: Perplexity with 2 Citations × 3 Search Results

**Output CSV Rows** (6 rows):
```
2 citations × 3 results = 6 rows total
Each row has citation data + result data + answer data (all duplicated)
```

---

## Change Summary

| Aspect | Old (v1) | New (v2) | Impact |
|--------|----------|----------|--------|
| Row structure | 1 answer = 1 row | 1 answer = N rows (per citation) | Enables citation-level analysis |
| Columns | 19 columns | 20 columns | +1 net (removed 9, added 10) |
| Citation data | Bundled in search_results_json | Structured in source_* columns | Standard CSV operations |
| Perplexity results | Bundled in search_results_json | Structured in result_* columns | Detailed search analysis |
| Metadata | Includes tokens, latency, base_url | Removed (available in raw_json) | Cleaner schema |
| Schema version | Implicit (no config) | Explicit CFG["csv_schema_version"] | Backward compatibility |
