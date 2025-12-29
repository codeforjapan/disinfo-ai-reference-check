# Research Findings: CSV Schema Redesign

**Feature**: CSV Schema Redesign for Citation and Search Result Tracking
**Date**: 2025-12-29
**Status**: Complete

## Overview

This document consolidates research findings for implementing citation extraction, URL domain parsing, CSV row expansion, and schema versioning for the CSV schema redesign feature.

---

## 1. Citation Extraction Patterns

### Decision Summary

Each LLM provider returns citations in different formats requiring provider-specific extraction logic within the `ProviderAdapter.parse_response()` method.

### OpenAI (Regex-based URL Extraction)

**Decision**: Extract URLs from message.content using regex pattern matching

**Implementation Pattern**:
```python
import re
from typing import List

def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from OpenAI message content using regex."""
    # RFC 3986 compliant URL pattern
    url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    return re.findall(url_pattern, text or "")
```

**Rationale**: OpenAI responses embed URLs in natural language text without structured citation fields. Regex is the only reliable extraction method. The pattern handles:
- HTTP and HTTPS protocols
- Subdomains (www, api, blog, etc.)
- Paths and query parameters
- URL-safe special characters

**Alternatives Considered**:
- Manual string splitting: Rejected due to inability to handle URLs embedded mid-sentence
- NLP-based extraction: Rejected as over-engineered for simple URL pattern matching

---

### Anthropic/Claude (Tool Usage Parsing)

**Decision**: Extract citations from `tool_use` blocks in response.content

**Implementation Pattern**:
```python
def extract_claude_citations(resp: Any) -> List[Dict[str, str]]:
    """Extract web search results from Claude API response content blocks."""
    content = getattr(resp, "content", None) or []
    citations = []

    for block in content:
        if getattr(block, "type", None) == "tool_use":
            tool_name = getattr(block, "name", None)
            if tool_name == "web_search":
                tool_input = getattr(block, "input", {})
                results = tool_input.get("results", [])
                for result in results:
                    citations.append({
                        "url": result.get("url"),
                        "title": result.get("title"),
                        "snippet": result.get("snippet"),
                    })

    return citations
```

**Rationale**: Claude's web_search_20250305 tool returns structured data in response.content blocks. This structured approach is more reliable than parsing generated text and provides rich metadata (title, snippet) beyond just URLs.

**Alternatives Considered**:
- Parse text response like OpenAI: Rejected because it discards structured tool usage data
- Use raw_json parsing: Considered but tool_use blocks are already exposed in response object attributes

---

### Google Gemini (Grounding Metadata Extraction)

**Decision**: Extract citations from `grounding_metadata.grounding_chunks`

**Implementation Pattern**:
```python
def extract_gemini_citations(resp: Any) -> List[Dict[str, str]]:
    """Extract grounding metadata from Gemini API response."""
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return []

    grounding_metadata = getattr(candidates[0], "grounding_metadata", None)
    if not grounding_metadata:
        return []

    grounding_chunks = getattr(grounding_metadata, "grounding_chunks", [])
    citations = []

    for chunk in grounding_chunks:
        web_chunk = getattr(chunk, "web", None)
        if web_chunk:
            citations.append({
                "url": getattr(web_chunk, "uri", None),
                "title": getattr(web_chunk, "title", None),
            })

    return citations
```

**Rationale**: Gemini provides structured citation data via grounding_metadata when Google Search tool is enabled. This is the canonical source for Gemini citations and includes both URLs and page titles.

**Alternatives Considered**:
- Text parsing: Rejected as grounding_metadata is the authoritative source
- Search query extraction: Considered but grounding_chunks provides actual cited sources, not queries

---

### xAI/Grok (Citations Array)

**Decision**: Extract from `citations` array when `return_citations: true` in search_parameters

**Implementation Pattern**:
```python
def extract_xai_citations(resp: Any) -> List[Dict[str, str]]:
    """Extract citations from xAI/Grok API response."""
    citations = getattr(resp, "citations", None)
    if not citations:
        return []

    return [
        {
            "url": citation.get("url"),
            "title": citation.get("title"),
            "snippet": citation.get("snippet"),
        }
        for citation in citations
    ]
```

**Rationale**: xAI returns structured citations array when enabled. Similar to Perplexity's approach but requires explicit `return_citations: true` in search_parameters configuration.

**Alternatives Considered**:
- Parse from text response: Rejected as citations array is more reliable and structured
- Treat as OpenAI-compatible without citations: Rejected as we want to capture available citation data

---

### Perplexity (Dual Structure: Citations + Search Results)

**Decision**: Extract both `citations` array AND `search_results` array for complete data

**Implementation Pattern**:
```python
def extract_perplexity_data(resp: Any) -> Dict[str, Any]:
    """Extract both citations and search_results from Perplexity response."""
    citations = getattr(resp, "citations", None) or []
    search_results = getattr(resp, "search_results", None) or []

    return {
        "citations": [{"url": url} for url in citations],
        "search_results": [
            {
                "url": result.get("url"),
                "title": result.get("title"),
                "snippet": result.get("snippet"),
                "date": result.get("date"),
                "last_updated": result.get("last_updated"),
                "source": result.get("source"),
            }
            for result in search_results
        ]
    }
```

**Rationale**: Perplexity's unique dual structure requires capturing both:
- `citations`: URLs actually referenced in the response text
- `search_results`: Full search engine results with metadata (title, snippet, date)

These are different: citations may be a subset of search_results, or include URLs not in search_results. The Cartesian product (citations × search_results) provides complete linkage for research analysis.

**Alternatives Considered**:
- Use only citations: Rejected as search_results provide valuable context about search quality
- Use only search_results: Rejected as citations indicate what the model actually used vs what was available
- Merge into single list: Rejected as it loses the semantic difference between cited and searched

---

## 2. URL Domain Extraction

### Decision Summary

Use `urllib.parse.urlparse()` with **removal of ONLY `www.` prefix**, preserving all other subdomains for semantic significance.

### Implementation Pattern

```python
from urllib.parse import urlparse
from typing import Optional

def extract_domain(url: str) -> str:
    """
    Extract domain from URL, removing only 'www.' prefix.

    Returns empty string if URL is unparseable.

    Examples:
        "https://www.example.com/page" → "example.com"
        "https://blog.example.com/page" → "blog.example.com"
        "https://api.github.com/repos" → "api.github.com"
        "https://ja.wikipedia.org/wiki/Page" → "ja.wikipedia.org"
    """
    if not url:
        return ""

    try:
        # Handle missing scheme
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"

        parsed = urlparse(url)
        domain = parsed.netloc

        if not domain:
            return ""

        # Remove port if present
        domain = domain.split(':')[0]

        # Remove ONLY www. prefix (keep other subdomains)
        if domain.startswith('www.'):
            domain = domain[4:]

        # Handle IDN (Internationalized Domain Names)
        try:
            domain = domain.encode('ascii').decode('idna')
        except (UnicodeError, UnicodeDecodeError):
            pass  # Already ASCII or invalid IDN

        return domain.lower()

    except Exception:
        return ""
```

### Rationale

**Why remove `www.` but keep other subdomains?**

1. **www is semantically meaningless**: It's a legacy convention with no semantic value (www.example.com and example.com typically serve identical content)

2. **Subdomains carry information**: Different subdomains represent different services or content:
   - `blog.example.com` vs `api.example.com` are functionally different
   - `en.wikipedia.org` vs `ja.wikipedia.org` serve different language content
   - Research integrity requires distinguishing these

3. **Consistency with DNS**: Subdomains are part of the authoritative domain structure in DNS

4. **Analysis flexibility**: Researchers can aggregate by root domain (example.com) in post-processing if needed, but cannot reconstruct subdomain information if discarded

### Edge Case Handling

**Non-standard protocols** (ftp://, file://, mailto:):
- **Decision**: Return empty string
- **Rationale**: Research focuses on web-accessible citations; non-HTTP(S) protocols are rare

**Malformed URLs**:
- **Decision**: Catch all exceptions, return empty string
- **Rationale**: Invalid URLs shouldn't crash pipeline; empty domain signals data quality issue

**Internationalized Domain Names (IDN)**:
- **Decision**: Attempt IDNA decoding; if fails, keep original
- **Rationale**: Support international domains while maintaining robustness

**Missing scheme** (e.g., "example.com" instead of "https://example.com"):
- **Decision**: Prepend "https://" before parsing
- **Rationale**: Many citation strings omit scheme; assuming HTTPS is reasonable default

### Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| Remove all subdomains (keep only root domain) | Loses semantic information (blog vs api, en vs ja); cannot be reconstructed |
| Keep all subdomains including www | Duplicates identical sources (www.example.com vs example.com); clutters analysis |
| Use TLD extraction library (tldextract) | Over-engineered; adds dependency; stdlib urllib.parse is sufficient |
| Normalize to root domain with eTLD+1 | Complex implementation; researchers can aggregate in post-processing if needed |

---

## 3. CSV Row Expansion Patterns

### Decision Summary

Use **generator pattern** for memory-efficient row expansion with immediate CSV writing (no buffering).

### Core Pattern: Generator-Based Expansion

```python
from typing import Iterator, Dict, Any, List

def expand_citations_to_rows(
    base_row: Dict[str, Any],
    citations: List[Dict[str, str]]
) -> Iterator[Dict[str, Any]]:
    """
    Expand one API response into multiple CSV rows (one per citation).

    Memory-efficient: yields rows one at a time instead of building list.

    Args:
        base_row: Base data (narrative_*, model_*, answer_*) shared across all rows
        citations: List of citation dicts with source_url, source_domain, etc.

    Yields:
        One dict per citation (or one dict with empty source_* fields if no citations)
    """
    if not citations:
        # Handle empty citation list: create 1 row with source_* fields = ""
        row = base_row.copy()
        row["source_id"] = ""
        row["source_url"] = ""
        row["source_domain"] = ""
        yield row
    else:
        # Yield one row per citation
        for i, citation in enumerate(citations):
            row = base_row.copy()
            row["source_id"] = f"{base_row['answer_id']}_source_{i}"
            row["source_url"] = citation.get("url", "")
            row["source_domain"] = citation.get("domain", "")
            yield row
```

**Rationale**:
- **Memory efficiency**: Generator yields rows one at a time; no large lists in memory
- **Incremental persistence**: Rows can be written to CSV immediately as generated
- **Simplicity**: Single pass through citations, no complex state management
- **Constitution compliance**: Aligns with "Incremental Data Persistence" principle (no memory batching)

### Perplexity Pattern: Cartesian Product

```python
def expand_perplexity_rows(
    base_row: Dict[str, Any],
    citations: List[str],
    search_results: List[Dict[str, Any]]
) -> Iterator[Dict[str, Any]]:
    """
    Expand Perplexity response with Cartesian product of citations × search_results.

    Examples:
        2 citations × 3 search_results = 6 rows
        2 citations × 0 search_results = 2 rows (result_* fields empty)
        0 citations × 3 search_results = 1 row (source_* and result_* fields empty)
    """
    if not citations:
        citations = [None]  # Ensure at least 1 row
    if not search_results:
        search_results = [None]  # Non-Perplexity case

    result_rank = 0
    for i, citation_url in enumerate(citations):
        for j, result in enumerate(search_results):
            row = base_row.copy()

            # Citation fields
            if citation_url:
                row["source_id"] = f"{base_row['answer_id']}_source_{i}"
                row["source_url"] = citation_url
                row["source_domain"] = extract_domain(citation_url)
            else:
                row["source_id"] = ""
                row["source_url"] = ""
                row["source_domain"] = ""

            # Result fields (Perplexity only)
            if result:
                result_rank = j + 1  # 1-based ranking
                row["result_id"] = f"{base_row['answer_id']}_result_{j}"
                row["result_url"] = result.get("url", "")
                row["result_domain"] = extract_domain(result.get("url", ""))
                row["result_title"] = result.get("title", "")
                row["result_snippet"] = result.get("snippet", "")
                row["result_rank"] = result_rank
            else:
                row["result_id"] = ""
                row["result_url"] = ""
                row["result_domain"] = ""
                row["result_title"] = ""
                row["result_snippet"] = ""
                row["result_rank"] = ""

            yield row
```

**Rationale**:
- **Complete data linkage**: Each (citation, search_result) pair gets one row for analysis
- **Handles empty cases gracefully**: 0 citations or 0 results still produces 1 row
- **Preserves all data**: No information loss from original API response
- **Research value**: Enables analysis like "which cited sources appeared in top 3 search results?"

### Incremental CSV Writing

```python
def append_expanded_rows(
    csv_path: Path,
    row_generator: Iterator[Dict[str, Any]],
    fieldnames: List[str]
) -> int:
    """
    Append expanded rows to CSV file incrementally.

    Returns:
        Number of rows written
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    rows_written = 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in row_generator:
            writer.writerow(row)
            rows_written += 1

    return rows_written
```

**Rationale**:
- **Incremental writes**: No buffering, each row written immediately
- **Constitution compliance**: Aligns with "Incremental Data Persistence" principle
- **Robustness**: Partial results saved even if later API calls fail
- **Real-time monitoring**: Researchers can inspect output while pipeline runs

### Integration with Adapter Pattern

```python
# In _append_row_local function (marimo_api_pipeline.py ~line 898)
def _append_row_local(cfg, provider_name, base_row, citations, results):
    """
    Write expanded rows for a single API response.

    Modified to accept citations and results instead of single row dict.
    """
    output_dir = Path(cfg["output_dir"])
    csv_path = output_dir / f"{provider_name}.csv"

    # Generate expanded rows
    if provider_name == "perplexity":
        row_generator = expand_perplexity_rows(base_row, citations, results)
    else:
        row_generator = expand_citations_to_rows(base_row, citations)

    # Write incrementally
    return append_expanded_rows(csv_path, row_generator, CSV_COLUMNS)
```

### Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| Build list of all expanded rows, then write batch | Violates "Incremental Data Persistence" principle; risks data loss on crash |
| Write each expanded row in separate CSV write operation | Excessive file I/O overhead; slower performance |
| Use pandas DataFrame for row expansion | Over-engineered; adds memory overhead; csv.DictWriter is sufficient |
| Lazy expansion only when CSV is read | Breaks incremental persistence; forces researchers to implement expansion logic |

---

## 4. Backward Compatibility Strategies

### Decision Summary

Implement **configuration-driven schema selection** with explicit version parameter (`CFG["csv_schema_version"]`) and optional auto-detection of existing file schemas.

### Schema Versioning Structure

```python
# In marimo_api_pipeline.py CFG definition (~line 65)
CFG = {
    "csv_schema_version": "v2",  # Options: "v1" (old schema), "v2" (new schema)
    # ... other config ...
}

# Old schema (v1) - 19 columns
CSV_COLUMNS_V1 = [
    "run_id", "prompt_id", "requested_at",
    "provider", "model", "base_url",
    "note_id", "note_text",
    "system_prompt", "user_prompt",
    "response_text", "finish_reason",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "latency_ms", "search_results_json", "error", "raw_json",
]

# New schema (v2) - 20 columns
CSV_COLUMNS_V2 = [
    # Narrative information
    "narrative_id", "narrative_type", "narrative_prompt",
    # Model information
    "model_name", "model_version",
    # Answer information
    "answer_id", "answer_prompt", "answer_text",
    "answer_raw_json", "answer_timestamp", "answer_citation_list",
    # Source information (citations)
    "source_id", "source_url", "source_domain",
    # Result information (Perplexity search results)
    "result_id", "result_url", "result_domain",
    "result_title", "result_snippet", "result_rank",
]

# Active schema selection
CSV_COLUMNS = CSV_COLUMNS_V2 if CFG["csv_schema_version"] == "v2" else CSV_COLUMNS_V1
```

**Rationale**:
- **Explicit versioning**: Clear configuration parameter documents breaking change
- **Minimal code duplication**: Two column list definitions, single selection point
- **Easy switching**: Change one config value to toggle schema version
- **Documentation value**: Version numbers communicate compatibility expectations

### Migration Path

**Option 1: Clean break (recommended)**
- Set `CFG["csv_schema_version"] = "v2"`
- Run pipeline to generate new v2 CSV files
- Archive old v1 CSVs manually
- Update analysis scripts for new column names

**Option 2: Gradual migration**
- Keep `CFG["csv_schema_version"] = "v1"` initially
- Develop and test new analysis scripts against v2 schema on subset of data
- Switch to v2 when ready
- Migrate historical data if needed (separate script, out of scope)

**Option 3: Parallel schemas**
- Generate both v1 and v2 outputs simultaneously (different output directories)
- Transition analysis scripts gradually
- **Not recommended**: Doubles I/O and storage; violates simplicity

### Implementation Pattern

```python
def _get_active_schema_columns(cfg: dict) -> List[str]:
    """Get active CSV schema columns based on configuration."""
    version = cfg.get("csv_schema_version", "v2")  # Default to v2

    if version == "v1":
        return CSV_COLUMNS_V1
    elif version == "v2":
        return CSV_COLUMNS_V2
    else:
        raise ValueError(f"Unsupported csv_schema_version: {version}")

# Use in CSV writing
def _append_row_local(cfg, provider_name, ...):
    columns = _get_active_schema_columns(cfg)
    # ... write with columns ...
```

### Handling Column Name Changes

**Mapping approach** (if v1 support maintained):
```python
V1_TO_V2_COLUMN_MAP = {
    "run_id": "answer_id",
    "requested_at": "answer_timestamp",
    "note_id": "narrative_id",
    "note_text": "narrative_prompt",
    "provider": "model_name",
    "model": "model_version",
    "user_prompt": "answer_prompt",
    "response_text": "answer_text",
    "raw_json": "answer_raw_json",
}

def map_row_v1_to_v2_columns(v1_row: dict) -> dict:
    """Map v1 row data to v2 column names."""
    v2_row = {}
    for v1_col, v2_col in V1_TO_V2_COLUMN_MAP.items():
        if v1_col in v1_row:
            v2_row[v2_col] = v1_row[v1_col]
    return v2_row
```

**Decision**: **Do NOT implement automatic column mapping** in production code

**Rationale**:
- **Simplicity**: Adding mapping logic complicates row builder functions
- **Clean break**: v2 is a breaking change; explicit version selection is clearer
- **Maintenance burden**: Mapping logic must be maintained indefinitely
- **User control**: Researchers explicitly choose schema version; no hidden conversions

### Documentation and Migration Guidance

**In CLAUDE.md Known Limitations section**:
```markdown
## Schema Version Migration

The CSV output schema changed in schema v2 (December 2025) from 19 to 20 columns.

**Breaking changes**:
- 9 columns renamed (note_id → narrative_id, etc.)
- 9 columns removed (base_url, token counts, latency, etc.)
- 10 columns added (narrative_type, source_*, result_*)
- Row expansion: 1 answer → N rows (one per citation)

**Migration**:
1. Set `CFG["csv_schema_version"] = "v2"` in configuration cell
2. Run pipeline to generate new v2 CSV files
3. Update analysis scripts to use new column names
4. Archive old v1 CSVs for reference

**Historical data**: A separate migration script is needed to convert v1 CSVs to v2 format (out of scope for this feature).
```

### Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| Auto-detect schema from existing CSV header | Adds complexity; explicit version control is clearer |
| Dual schema support with runtime switching | Over-engineered; doubles code paths and testing surface |
| Versioned output directories (v1/, v2/) | Clutters file structure; version in config is sufficient |
| Embed schema version in CSV file (metadata row) | Non-standard; breaks CSV parsing in external tools |
| Schema migration script in main pipeline | Out of scope; separate migration tool is cleaner |

---

## Implementation Priority

Based on research findings, implement in this order:

1. **Domain extraction function** (standalone, easily testable)
2. **Row expansion generator** (core logic, independent of providers)
3. **OpenAI citation extraction** (simplest: regex-based)
4. **Perplexity citation + results extraction** (most complex: Cartesian product)
5. **Claude, Gemini, xAI extraction** (structured data, similar patterns)
6. **Schema versioning** (configuration-driven, low risk)

---

## Testing Recommendations

### Unit Testing Approach

While no test framework is currently in place, manual validation should cover:

1. **Domain extraction edge cases**:
   - Standard URLs: `https://example.com/path` → `example.com`
   - www removal: `https://www.example.com` → `example.com`
   - Subdomain preservation: `https://blog.example.com` → `blog.example.com`
   - Malformed URLs: `not-a-url` → `""`
   - Missing scheme: `example.com/path` → `example.com`

2. **Row expansion counts**:
   - 0 citations → 1 row (empty source_*)
   - 3 citations → 3 rows
   - 2 citations × 3 results (Perplexity) → 6 rows
   - 0 citations × 3 results (Perplexity) → 1 row (all fields empty)

3. **Citation extraction accuracy**:
   - OpenAI: Parse 2-3 URLs from sample response text
   - Perplexity: Verify both citations and search_results extracted
   - Claude: Verify tool_use blocks parsed correctly

4. **Schema version switching**:
   - Set `CFG["csv_schema_version"] = "v1"` → verify 19 columns
   - Set `CFG["csv_schema_version"] = "v2"` → verify 20 columns

### Validation Against Real Data

Test against existing CSV outputs in `llm_runs/` directory:
- Parse v1 CSVs to extract search_results_json
- Compare extracted citations with new extraction logic
- Verify row counts match expected expansion (1 answer → N citations)

---

## References

- Python urllib.parse documentation: https://docs.python.org/3/library/urllib.parse.html
- RFC 3986 (URI Generic Syntax): https://www.rfc-editor.org/rfc/rfc3986
- OpenAI API documentation: https://platform.openai.com/docs/api-reference
- Anthropic Claude API documentation: https://docs.anthropic.com/claude/reference
- Google Gemini API documentation: https://ai.google.dev/docs
- Perplexity API documentation: https://docs.perplexity.ai/reference/post_chat_completions
