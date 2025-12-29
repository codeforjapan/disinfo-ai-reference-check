# CSV Schema v2 Contract

**Version**: 2.0.0
**Date**: 2025-12-29
**Status**: Draft

## Overview

This document defines the CSV output schema contract for the disinformation research pipeline. This schema is used by all provider adapters (OpenAI, Claude, Gemini, xAI/Grok, Perplexity) to write results to provider-specific CSV files.

## Schema Definition

### Column List (20 columns)

```python
CSV_COLUMNS = [
    # Narrative information (from input CSV)
    "narrative_id",           # string, unique identifier for narrative
    "narrative_type",         # string, optional classification (e.g., "misinformation")
    "narrative_prompt",       # string, required narrative text content

    # Model information (from provider configuration)
    "model_name",             # string, provider name (e.g., "openai", "claude")
    "model_version",          # string, specific model (e.g., "gpt-4", "claude-3-5-sonnet")

    # Answer information (from API response)
    "answer_id",              # string, unique identifier for this response (UUID)
    "answer_prompt",          # string, exact user prompt sent to model
    "answer_text",            # string, model's response text
    "answer_raw_json",        # string, complete JSON serialization of API response
    "answer_timestamp",       # string, ISO 8601 datetime when request was made
    "answer_citation_list",   # string, JSON array of citation URLs (e.g., "[\"url1\", \"url2\"]")

    # Source information (citation tracking - one row per citation)
    "source_id",              # string, citation identifier ("{answer_id}_source_{index}")
    "source_url",             # string, full URL of cited source
    "source_domain",          # string, domain extracted from source_url (e.g., "example.com")

    # Result information (Perplexity search results only)
    "result_id",              # string, search result identifier ("{answer_id}_result_{index}")
    "result_url",             # string, search result URL
    "result_domain",          # string, domain extracted from result_url
    "result_title",           # string, search result page title
    "result_snippet",         # string, text excerpt from search result
    "result_rank",            # integer, position in search results (1-based)
]
```

---

## Field Specifications

### Narrative Fields

#### `narrative_id`
- **Type**: string
- **Required**: Yes
- **Source**: Input CSV column (configurable via CFG["note_id_col"])
- **Constraints**: Should be unique per narrative (not enforced; duplicates create separate answer_ids)
- **Example**: `"N001"`, `"narrative_12345"`

#### `narrative_type`
- **Type**: string
- **Required**: No (optional)
- **Source**: Input CSV column "narrative_type" (if exists)
- **Default**: Empty string `""` if column missing or value empty
- **Constraints**: Free-form text classification
- **Example**: `"misinformation"`, `"satire"`, `"true"`, `""`

#### `narrative_prompt`
- **Type**: string
- **Required**: Yes
- **Source**: Input CSV column (configurable via CFG["note_text_col"])
- **Constraints**: MUST NOT be empty or null (validated before API calls)
- **Example**: `"このワクチンは危険である"`

---

### Model Fields

#### `model_name`
- **Type**: string
- **Required**: Yes
- **Source**: ProviderConfig.name
- **Constraints**: One of: `"openai"`, `"claude"`, `"gemini"`, `"grok"`, `"perplexity"`
- **Example**: `"openai"`, `"claude"`

#### `model_version`
- **Type**: string
- **Required**: Yes
- **Source**: ProviderConfig.model
- **Constraints**: Provider-specific model identifier
- **Example**: `"gpt-4-turbo-2024-04-09"`, `"claude-3-5-sonnet-20241022"`, `"gemini-2.0-flash-exp"`

---

### Answer Fields

#### `answer_id`
- **Type**: string (UUID)
- **Required**: Yes
- **Source**: Generated per API response (uuid.uuid4() or run-specific ID)
- **Constraints**: MUST be unique per API response
- **Example**: `"550e8400-e29b-41d4-a716-446655440000"`

#### `answer_prompt`
- **Type**: string
- **Required**: Yes
- **Source**: Constructed from CFG["user_prompt_template"] with {note_text} substitution
- **Constraints**: MUST match the exact prompt sent to API
- **Example**: `"このワクチンは危険である\n\n「はい」または「いいえ」で回答してください..."`

#### `answer_text`
- **Type**: string
- **Required**: Yes
- **Source**: Extracted from API response (provider-specific path)
- **Constraints**: Full model response text
- **Example**: `"いいえ; このワクチンは安全です。厚生労働省の公式データによると..."`

#### `answer_raw_json`
- **Type**: string (JSON serialized)
- **Required**: Yes
- **Source**: json.dumps() of complete API response object
- **Constraints**: MUST be valid JSON; used for debugging and auditing
- **Example**: `"{\"id\": \"chatcmpl-123\", \"choices\": [...], ...}"`

#### `answer_timestamp`
- **Type**: string (ISO 8601 datetime)
- **Required**: Yes
- **Source**: datetime.now(timezone.utc).isoformat()
- **Constraints**: MUST be ISO 8601 format with timezone
- **Example**: `"2025-12-29T17:30:00Z"`, `"2025-12-29T17:30:00.123456+00:00"`

#### `answer_citation_list`
- **Type**: string (JSON array)
- **Required**: Yes
- **Source**: Extracted from API response, serialized as JSON
- **Constraints**: MUST be valid JSON array; empty if no citations: `"[]"`
- **Example**: `"[\"https://www.mhlw.go.jp/vaccine\", \"https://www.who.int/safety\"]"`

---

### Source Fields (Citation Tracking)

#### `source_id`
- **Type**: string
- **Required**: Yes (can be empty string if no citations)
- **Source**: Generated during row expansion
- **Format**: `"{answer_id}_source_{index}"` (0-indexed)
- **Empty**: `""` if no citations for this answer
- **Example**: `"550e8400_source_0"`, `"550e8400_source_1"`, `""`

#### `source_url`
- **Type**: string
- **Required**: Yes (can be empty string if no citations)
- **Source**: Extracted from API response (provider-specific)
- **Constraints**: Full URL including scheme (https://)
- **Empty**: `""` if no citations for this answer
- **Example**: `"https://www.mhlw.go.jp/vaccine"`, `""`

#### `source_domain`
- **Type**: string
- **Required**: Yes (can be empty string if no citations)
- **Source**: Auto-extracted from source_url using urllib.parse
- **Format**: Domain without www. prefix, with other subdomains preserved
- **Empty**: `""` if source_url is empty or unparseable
- **Example**: `"mhlw.go.jp"`, `"blog.example.com"`, `"ja.wikipedia.org"`, `""`

---

### Result Fields (Perplexity Search Results Only)

#### `result_id`
- **Type**: string
- **Required**: Yes (can be empty string for non-Perplexity)
- **Source**: Generated during row expansion (Perplexity only)
- **Format**: `"{answer_id}_result_{index}"` (0-indexed)
- **Empty**: `""` for all non-Perplexity providers
- **Example**: `"550e8400_result_0"`, `"550e8400_result_1"`, `""`

#### `result_url`
- **Type**: string
- **Required**: Yes (can be empty string for non-Perplexity)
- **Source**: Perplexity API response field `search_results[].url`
- **Empty**: `""` for non-Perplexity providers or if no search results
- **Example**: `"https://www.example.com/article"`, `""`

#### `result_domain`
- **Type**: string
- **Required**: Yes (can be empty string for non-Perplexity)
- **Source**: Auto-extracted from result_url
- **Empty**: `""` for non-Perplexity or if result_url empty/unparseable
- **Example**: `"example.com"`, `""`

#### `result_title`
- **Type**: string
- **Required**: Yes (can be empty string for non-Perplexity)
- **Source**: Perplexity API response field `search_results[].title`
- **Empty**: `""` for non-Perplexity providers
- **Example**: `"公式サイト - ワクチン安全性"`, `""`

#### `result_snippet`
- **Type**: string
- **Required**: Yes (can be empty string for non-Perplexity)
- **Source**: Perplexity API response field `search_results[].snippet`
- **Constraints**: Can be long (10KB+); no truncation
- **Empty**: `""` for non-Perplexity providers
- **Example**: `"ワクチンの安全性については、厚生労働省の調査により..."`, `""`

#### `result_rank`
- **Type**: integer (as string in CSV) or empty string
- **Required**: Yes (can be empty string for non-Perplexity)
- **Source**: Generated during row expansion (1-based index)
- **Constraints**: 1-based rank (1 = first result, 2 = second, etc.)
- **Empty**: `""` for non-Perplexity providers
- **Example**: `"1"`, `"2"`, `"3"`, `""`

---

## Row Expansion Rules

### Standard Providers (OpenAI, Claude, Gemini, xAI/Grok)

**Rule**: One row per citation

**Examples**:

| Scenario | Citations | Output Rows | source_* Fields | result_* Fields |
|----------|-----------|-------------|-----------------|-----------------|
| No citations | 0 | 1 | All empty `""` | All empty `""` |
| 1 citation | 1 | 1 | Populated | All empty `""` |
| 3 citations | 3 | 3 | Populated (one per row) | All empty `""` |

**Implementation**:
```python
if len(citations) == 0:
    create 1 row with source_* = "" and result_* = ""
else:
    for each citation:
        create 1 row with source_* populated and result_* = ""
        duplicate answer_* and narrative_* fields across all rows
```

### Perplexity Provider

**Rule**: Cartesian product of citations × search_results

**Examples**:

| Scenario | Citations | Search Results | Output Rows | source_* Fields | result_* Fields |
|----------|-----------|----------------|-------------|-----------------|-----------------|
| No citations, no results | 0 | 0 | 1 | All empty `""` | All empty `""` |
| 2 citations, no results | 2 | 0 | 2 | Populated | All empty `""` |
| No citations, 3 results | 0 | 3 | 1 | All empty `""` | All empty `""` (no citations to link) |
| 2 citations, 3 results | 2 | 3 | 6 | Populated | Populated (Cartesian product) |

**Implementation**:
```python
if len(citations) == 0:
    citations = [None]  # Ensure at least 1 iteration
if len(search_results) == 0:
    search_results = [None]  # Ensure at least 1 iteration

for citation in citations:
    for result in search_results:
        create 1 row with citation and result data
        duplicate answer_* and narrative_* fields
```

---

## Data Integrity Constraints

### Cross-Row Consistency

**Invariant**: All expanded rows from the same API response MUST have identical values for:
- `narrative_id`
- `narrative_type`
- `narrative_prompt`
- `model_name`
- `model_version`
- `answer_id` (CRITICAL: same answer_id across all expanded rows)
- `answer_prompt`
- `answer_text`
- `answer_raw_json`
- `answer_timestamp`
- `answer_citation_list`

**Rationale**: Enables joining rows back to original answer by grouping on `answer_id`

### Citation List Synchronization

**Invariant**: The number of expanded rows with non-empty `source_*` fields MUST match the array length in `answer_citation_list`

**Example**:
- `answer_citation_list = "[\"url1\", \"url2\", \"url3\"]"` → 3 rows with source_* populated

**Exception**: Perplexity Cartesian product creates M×N rows where M = len(citations)

### Empty Field Representation

**Rule**: Optional/empty fields use empty string `""`, NOT NULL

**Rationale**:
- Consistent CSV parsing (no NULL handling needed)
- Clear distinction between "no data" (empty string) vs "missing field" (would be NULL)
- Simplifies analysis scripts (no null checks required)

---

## Provider-Specific Contracts

### OpenAI

**Citation Extraction**: Regex URL parsing from `message.content`

**Empty Fields**:
- `result_*`: Always empty `""`
- `source_*`: Empty if no URLs found in text

**Example Row**:
```csv
narrative_id,narrative_type,narrative_prompt,model_name,model_version,answer_id,answer_prompt,answer_text,answer_raw_json,answer_timestamp,answer_citation_list,source_id,source_url,source_domain,result_id,result_url,result_domain,result_title,result_snippet,result_rank
N001,misinformation,ワクチンは危険,openai,gpt-4,550e8400,{prompt},いいえ...,{json},2025-12-29T17:30:00Z,"[""https://mhlw.go.jp""]",550e8400_source_0,https://mhlw.go.jp,mhlw.go.jp,,,,,
```

### Anthropic/Claude

**Citation Extraction**: Parse `tool_use` blocks with `name="web_search"` from `response.content`

**Empty Fields**:
- `result_*`: Always empty `""`
- `source_*`: Empty if no tool_use blocks

**Example Row**: Similar to OpenAI, with potentially richer citation metadata (title, snippet from tool_use)

### Google Gemini

**Citation Extraction**: Parse `grounding_metadata.grounding_chunks[].web`

**Empty Fields**:
- `result_*`: Always empty `""`
- `source_*`: Empty if no grounding_metadata

**Example Row**: Similar to OpenAI/Claude

### xAI/Grok

**Citation Extraction**: Parse `citations` array from response (when `return_citations: true`)

**Empty Fields**:
- `result_*`: Always empty `""`
- `source_*`: Empty if citations array empty or missing

**Example Row**: Similar to OpenAI/Claude

### Perplexity

**Citation Extraction**: Parse both `citations` array AND `search_results` array

**Empty Fields**:
- `result_*`: Populated with search result metadata
- `source_*`: Populated with citation URLs

**Example Row (Cartesian product)**:
```csv
narrative_id,narrative_type,narrative_prompt,model_name,model_version,answer_id,answer_prompt,answer_text,answer_raw_json,answer_timestamp,answer_citation_list,source_id,source_url,source_domain,result_id,result_url,result_domain,result_title,result_snippet,result_rank
N001,misinformation,ワクチンは危険,perplexity,sonar,550e8400,{prompt},いいえ...,{json},2025-12-29T17:30:00Z,"[""https://mhlw.go.jp"",""https://who.int""]",550e8400_source_0,https://mhlw.go.jp,mhlw.go.jp,550e8400_result_0,https://news.com/article,news.com,ワクチン安全性,厚労省によると...,1
N001,misinformation,ワクチンは危険,perplexity,sonar,550e8400,{prompt},いいえ...,{json},2025-12-29T17:30:00Z,"[""https://mhlw.go.jp"",""https://who.int""]",550e8400_source_0,https://mhlw.go.jp,mhlw.go.jp,550e8400_result_1,https://blog.com/post,blog.com,ワクチン情報,安全性について...,2
N001,misinformation,ワクチンは危険,perplexity,sonar,550e8400,{prompt},いいえ...,{json},2025-12-29T17:30:00Z,"[""https://mhlw.go.jp"",""https://who.int""]",550e8400_source_1,https://who.int,who.int,550e8400_result_0,https://news.com/article,news.com,ワクチン安全性,厚労省によると...,1
N001,misinformation,ワクチンは危険,perplexity,sonar,550e8400,{prompt},いいえ...,{json},2025-12-29T17:30:00Z,"[""https://mhlw.go.jp"",""https://who.int""]",550e8400_source_1,https://who.int,who.int,550e8400_result_1,https://blog.com/post,blog.com,ワクチン情報,安全性について...,2
```

(4 rows: 2 citations × 2 results)

---

## Validation Rules

### Pre-Write Validation

Before writing any row to CSV, validate:

1. **All required fields present**: Every field in CSV_COLUMNS MUST be present in row dict
2. **answer_id not empty**: MUST be non-empty UUID string
3. **answer_citation_list valid JSON**: MUST be parseable as JSON array (can be empty `[]`)
4. **source_id format**: If non-empty, MUST match pattern `{answer_id}_source_{digit+}`
5. **result_id format**: If non-empty, MUST match pattern `{answer_id}_result_{digit+}`
6. **result_rank type**: If non-empty, MUST be integer or convertible to integer

### Post-Write Validation (Manual Testing)

After pipeline run, verify:

1. **Row count consistency**: Count rows per answer_id matches expected citation count
2. **Field consistency**: All rows with same answer_id have identical answer_* fields
3. **Empty field patterns**: Non-Perplexity providers have all result_* empty
4. **Perplexity Cartesian product**: Verify row count = citations × results

---

## Migration from Schema v1

### Column Mapping

| v1 Column | v2 Column | Notes |
|-----------|-----------|-------|
| `run_id` | `answer_id` | Name change only |
| `prompt_id` | REMOVED | Not needed in v2 |
| `requested_at` | `answer_timestamp` | Name change only |
| `provider` | `model_name` | Name change only |
| `model` | `model_version` | Name change only |
| `base_url` | REMOVED | Redundant metadata |
| `note_id` | `narrative_id` | Name change only |
| `note_text` | `narrative_prompt` | Name change only |
| `system_prompt` | REMOVED | Fixed value, not needed per-row |
| `user_prompt` | `answer_prompt` | Name change only |
| `response_text` | `answer_text` | Name change only |
| `finish_reason` | REMOVED | Available in answer_raw_json |
| `prompt_tokens` | REMOVED | Available in answer_raw_json |
| `completion_tokens` | REMOVED | Available in answer_raw_json |
| `total_tokens` | REMOVED | Available in answer_raw_json |
| `latency_ms` | REMOVED | Can be recalculated if needed |
| `search_results_json` | REPLACED | Now source_* and result_* columns |
| `error` | REMOVED | Error rows not written to CSV |
| `raw_json` | `answer_raw_json` | Name change only |
| N/A | `narrative_type` | NEW COLUMN |
| N/A | `answer_citation_list` | NEW COLUMN |
| N/A | `source_id` | NEW COLUMN |
| N/A | `source_url` | NEW COLUMN |
| N/A | `source_domain` | NEW COLUMN |
| N/A | `result_id` | NEW COLUMN |
| N/A | `result_url` | NEW COLUMN |
| N/A | `result_domain` | NEW COLUMN |
| N/A | `result_title` | NEW COLUMN |
| N/A | `result_snippet` | NEW COLUMN |
| N/A | `result_rank` | NEW COLUMN |

**Total**: 19 v1 columns → 20 v2 columns (removed 9, renamed 10, added 10)

---

## Change Log

### v2.0.0 (2025-12-29)

**Breaking Changes**:
- Restructured from 19 to 20 columns
- Renamed 10 columns to research-focused terminology
- Removed 9 redundant metadata columns
- Added 10 new columns for citation and search result tracking
- Implemented row expansion (1 answer → N rows per citation)

**New Features**:
- Citation-level data structure (source_* columns)
- Perplexity search result details (result_* columns)
- Narrative type classification (narrative_type column)
- Citation list preservation (answer_citation_list column)

**Migration**: Requires `CFG["csv_schema_version"] = "v2"` and potential updates to analysis scripts
