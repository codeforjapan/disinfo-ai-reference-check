# Feature Specification: CSV Schema Redesign for Citation and Search Result Tracking

**Feature Branch**: `001-csv-schema-redesign`
**Created**: 2025-12-29
**Status**: Draft
**Input**: User description: "CSVスキーマの再設計: 出力CSVの構造を、より詳細で構造化されたスキーマに変更する。特にcitation情報とsearch結果を明示的に分離し、Perplexity特有の情報をサポートする。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Citation Analysis Across Providers (Priority: P1)

As a disinformation researcher, I need to analyze which sources different AI models cite when fact-checking narratives, so that I can evaluate the quality and diversity of information sources each model relies on.

**Why this priority**: This is the core value proposition of the schema redesign. Understanding citation patterns is essential for evaluating AI model reliability in fact-checking scenarios. This can deliver immediate research value even without other features.

**Independent Test**: Can be fully tested by running the pipeline with any AI provider (Claude, OpenAI, Gemini), exporting the CSV, and verifying that each cited source appears as a separate row with source_url and source_domain populated.

**Acceptance Scenarios**:

1. **Given** a narrative has been fact-checked by Claude with 3 citations, **When** I export the results to CSV, **Then** I see 3 rows for that narrative with each citation's URL and domain populated in separate rows
2. **Given** a narrative has been fact-checked by OpenAI with no citations, **When** I export the results, **Then** I see 1 row with empty source fields
3. **Given** I have results from multiple providers for the same narrative, **When** I filter by narrative_id, **Then** I can compare which sources each provider cited
4. **Given** I want to analyze citation domains, **When** I group by source_domain, **Then** I can see which news sources, fact-checkers, or websites are most frequently cited

---

### User Story 2 - Perplexity Search Result Deep Dive (Priority: P2)

As a disinformation researcher, I need to examine the detailed search results that Perplexity used (including titles, snippets, and rankings), so that I can understand not just what sources were cited, but what information was available during the search process.

**Why this priority**: This adds critical depth for Perplexity analysis but depends on P1's citation tracking foundation. It enables analysis of search quality and relevance, which is valuable for understanding how search-augmented LLMs work.

**Independent Test**: Can be tested by running the pipeline with Perplexity provider, exporting CSV, and verifying that result_title, result_snippet, and result_rank are populated for each search result returned by Perplexity.

**Acceptance Scenarios**:

1. **Given** Perplexity returns 5 search results for a narrative, **When** I export the CSV, **Then** I see rows with result_id, result_url, result_title, result_snippet, and result_rank (1-5) populated
2. **Given** a search result appears at rank 2, **When** I examine the result_rank field, **Then** it shows "2" indicating its position in search results
3. **Given** I want to analyze search quality, **When** I examine result_snippet fields, **Then** I can see what text excerpts the search engine provided
4. **Given** non-Perplexity providers have been used, **When** I examine their result_* fields, **Then** these fields are empty (not applicable)

---

### User Story 3 - Cross-Model Citation Comparison (Priority: P1)

As a disinformation researcher, I need to compare how different AI models (Claude, OpenAI, Gemini, Perplexity, xAI) cite sources for the same narrative, so that I can identify consensus sources and unique citations that might indicate different search strategies or biases.

**Why this priority**: This delivers the comparative analysis value that justifies the schema redesign. It enables researchers to understand model differences, which is critical for disinformation research methodology.

**Independent Test**: Can be tested by running the pipeline with 2+ providers for the same narrative, exporting CSV, and performing a join on narrative_id to compare cited sources across providers.

**Acceptance Scenarios**:

1. **Given** the same narrative has been processed by 3 different providers, **When** I filter CSV by that narrative_id, **Then** I can see all citations from each provider grouped together
2. **Given** two providers cite the same domain, **When** I analyze source_domain, **Then** I can identify consensus sources
3. **Given** one provider cites a unique source not cited by others, **When** I analyze the data, **Then** I can identify provider-specific citation patterns
4. **Given** I want to measure citation diversity, **When** I count distinct source_urls per provider, **Then** I can quantify how many unique sources each model consulted

---

### User Story 4 - Narrative Type Analysis (Priority: P2)

As a disinformation researcher, I need to categorize narratives by type (misinformation, satire, true, etc.) and analyze how AI models respond differently to each type, so that I can evaluate whether models adapt their fact-checking approach based on narrative characteristics.

**Why this priority**: This enables classification-based analysis which is valuable for research but depends on having the citation and response data (P1, P3) already in place.

**Independent Test**: Can be tested by preparing an input CSV with narrative_type column, running the pipeline, and verifying that the narrative_type is preserved and queryable in the output CSV.

**Acceptance Scenarios**:

1. **Given** input CSV contains narrative_type = "misinformation", **When** pipeline processes the narrative, **Then** output CSV preserves this value in narrative_type column
2. **Given** I want to analyze misinformation responses specifically, **When** I filter by narrative_type = "misinformation", **Then** I see all misinformation narratives with their citations
3. **Given** I have 3 narrative types in my dataset, **When** I group by narrative_type, **Then** I can compare citation patterns across narrative types
4. **Given** a narrative has no type specified in input, **When** processed, **Then** narrative_type is empty (null) in output

---

### User Story 5 - Answer-Level Analysis (Priority: P3)

As a disinformation researcher, I need to analyze the full answer text alongside citations, so that I can evaluate how models incorporate cited sources into their fact-checking responses.

**Why this priority**: This is important for qualitative analysis but less critical than citation tracking (P1) and comparison (P1). It can be added after core citation infrastructure is working.

**Independent Test**: Can be tested by running the pipeline, exporting CSV, and verifying that answer_text contains the full model response and can be analyzed alongside source_url citations.

**Acceptance Scenarios**:

1. **Given** a model provides a 200-word fact-checking response, **When** I export to CSV, **Then** the full answer_text is preserved without truncation
2. **Given** I want to analyze how citations relate to answer content, **When** I read answer_text and source_url together, **Then** I can see which sources supported which claims
3. **Given** multiple citations exist for one answer, **When** I view the CSV rows, **Then** answer_text is duplicated across rows (each row = one citation) maintaining the relationship

---

### Edge Cases

- What happens when a provider returns citations but no answer text? (System creates rows with source_* fields populated but answer_text empty)
- What happens when a provider returns an answer but no citations? (System creates 1 row with answer_text populated but source_* fields empty)
- What happens when Perplexity returns citations that differ from search results? (System records both: citations in source_* fields and search results in result_* fields independently)
- What happens when a citation URL cannot be parsed for domain extraction? (System stores the full URL in source_url and sets source_domain to empty)
- What happens when input CSV lacks narrative_type column? (System treats it as optional and continues processing with empty narrative_type values for all rows)
- What happens when processing the same narrative_id multiple times in the same run? (Each answer gets a unique answer_id, creating separate rows even if narrative_id is duplicated)
- What happens with extremely large result_snippet fields (e.g., 10KB)? (System stores full snippet without truncation, as CSV format supports large text fields)
- What happens when a model returns duplicate citations (same URL cited twice)? (Each citation creates a separate row, preserving the duplicate for accurate citation frequency analysis)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST restructure output CSV from 19 columns to 20 columns with new schema
- **FR-002**: System MUST rename existing columns: note_id → narrative_id, note_text → narrative_prompt, provider → model_name, model → model_version, run_id → answer_id, response_text → answer_text, requested_at → answer_timestamp, raw_json → answer_raw_json, user_prompt → answer_prompt
- **FR-003**: System MUST remove columns: base_url, system_prompt, finish_reason, prompt_tokens, completion_tokens, total_tokens, latency_ms, search_results_json, error
- **FR-004**: System MUST add new column: narrative_type (string, read from input CSV)
- **FR-005**: System MUST add new column: answer_citation_list (JSON array of citation URLs)
- **FR-006**: System MUST add new columns for citation tracking: source_id (string), source_url (string), source_domain (string, auto-extracted from URL)
- **FR-007**: System MUST add new columns for Perplexity search results: result_id (string), result_url (string), result_domain (string), result_title (string), result_snippet (string), result_rank (integer)
- **FR-008**: System MUST expand rows based on citations: 1 answer with N citations creates N rows, each with identical answer_* fields but different source_* fields
- **FR-009**: System MUST expand Perplexity rows based on citations × results: 1 answer with N citations and M search results creates N×M rows
- **FR-010**: System MUST handle zero citations by creating 1 row with empty source_* fields
- **FR-011**: System MUST extract citations from OpenAI responses by parsing URLs in message.content
- **FR-012**: System MUST extract citations from Claude responses by parsing tool usage history in raw_json
- **FR-013**: System MUST extract citations from Gemini responses by parsing grounding_metadata
- **FR-014**: System MUST extract citations from xAI responses by parsing citation fields
- **FR-015**: System MUST extract citations from Perplexity responses by parsing citations field
- **FR-016**: System MUST extract search results from Perplexity responses including: url, title, snippet, rank (position in results list)
- **FR-017**: System MUST auto-extract domain from URLs using standard URL parsing (e.g., "https://example.com/path" → "example.com")
- **FR-018**: System MUST read narrative_type from input CSV (narratives.csv) when column is present
- **FR-019**: System MUST maintain backward compatibility by supporting a schema version configuration parameter
- **FR-020**: System MUST assign unique source_id to each citation row (auto-incrementing or UUID within answer context)
- **FR-021**: System MUST assign unique result_id to each search result (auto-incrementing within answer context)
- **FR-022**: System MUST store complete raw response in answer_raw_json for traceability
- **FR-023**: System MUST preserve answer_timestamp as ISO 8601 formatted datetime
- **FR-024**: System MUST leave result_* columns empty (null) for non-Perplexity providers

### Key Entities

- **Narrative**: The disinformation narrative being fact-checked
  - Attributes: narrative_id (unique identifier), narrative_type (classification category), narrative_prompt (text content)
  - Represents the input being evaluated

- **Model**: The AI provider/model combination used for fact-checking
  - Attributes: model_name (provider name like "openai", "claude"), model_version (specific model like "gpt-4", "claude-3")
  - Represents who performed the fact-checking

- **Answer**: A fact-checking response from a model for a specific narrative
  - Attributes: answer_id (unique per response), answer_text (response content), answer_timestamp (when generated), answer_prompt (exact prompt sent), answer_citation_list (JSON array of cited URLs), answer_raw_json (complete API response)
  - Represents the fact-checking output
  - Has one-to-many relationship with Citations

- **Citation/Source**: A source cited by the model in its answer
  - Attributes: source_id (unique within answer), source_url (full URL), source_domain (extracted domain name)
  - Represents information sources the model relied upon
  - Each citation creates a separate row in the CSV

- **Search Result**: A search engine result retrieved by Perplexity (only applicable to Perplexity provider)
  - Attributes: result_id (unique within answer), result_url (search result URL), result_domain (extracted domain), result_title (result title), result_snippet (result text excerpt), result_rank (position in search results 1-N)
  - Represents what the search engine found (may differ from what was ultimately cited)
  - Each result creates additional row expansion in combination with citations

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Researchers can identify all sources cited by a model for a given narrative in under 30 seconds by filtering on narrative_id
- **SC-002**: Researchers can compare citation patterns across all 5 providers (OpenAI, Claude, Gemini, xAI, Perplexity) for any narrative in under 2 minutes
- **SC-003**: System successfully expands a single answer with 3 citations and 5 search results (Perplexity) into 15 CSV rows with correct data in all fields
- **SC-004**: System successfully expands a single answer with 2 citations and 0 search results (non-Perplexity) into 2 CSV rows with source_* populated and result_* empty
- **SC-005**: Domain extraction correctly identifies domains for 95% of standard URLs (http/https with valid domain names)
- **SC-006**: Citation extraction correctly identifies all citations from provider-specific response formats with 90% accuracy
- **SC-007**: Pipeline processes 100 narratives across 5 providers producing a complete CSV with all 20 columns populated appropriately in under 10 minutes
- **SC-008**: Researchers can calculate citation diversity metrics (unique domains per provider) by grouping on source_domain in under 1 minute
- **SC-009**: Zero data loss: 100% of original answer data (text, timestamp, raw JSON) is preserved in expanded row format
- **SC-010**: Schema clarity: New column names are self-documenting such that a researcher unfamiliar with the system can understand the data structure within 5 minutes of viewing the CSV

## Assumptions

1. **Input CSV Format**: We assume the input CSV (narratives.csv) will be updated to include a narrative_type column. If this column is missing, the pipeline should treat it as optional and leave the field empty rather than failing.

2. **Citation Format Variability**: We assume citations may appear in various formats across providers:
   - OpenAI: URLs embedded in text responses
   - Claude: Structured tool usage in API response
   - Gemini: Grounding metadata structure
   - Perplexity: Dedicated citations array
   - xAI: Citation fields in response

   The system will implement provider-specific parsing logic to handle these differences.

3. **URL Format**: We assume most citations will be standard HTTP/HTTPS URLs. The domain extraction will handle these standard cases. Non-standard URLs (file://, ftp://, etc.) will store the full URL in source_url and leave source_domain empty.

4. **Row Expansion Performance**: We assume that row expansion (especially for Perplexity with citations × results) will not create prohibitively large CSV files. For a typical run with 100 narratives × 5 providers × average 2 citations × average 3 results (Perplexity only), we estimate ~2,000 total rows, which is manageable in CSV format.

5. **Backward Compatibility Need**: We assume existing pipelines and analysis scripts may depend on the old schema. The system will include a configuration option to support both schemas, with a clear migration path.

6. **Citation as Unit of Analysis**: We assume that the primary unit of analysis for researchers is the citation (source), not the answer. This justifies the decision to expand rows by citations rather than keeping citations as a JSON array in a single row.

7. **Perplexity Result Value**: We assume that Perplexity's search results (what the search engine found) are sufficiently different from citations (what the model used) to warrant separate tracking in result_* columns.

8. **Error Handling Strategy**: We assume that errors (API failures, parsing errors) will be logged separately rather than creating error rows in the main output CSV. This keeps the CSV focused on successful results and simplifies analysis.

## Dependencies

1. **Input CSV Schema Change**: The narratives.csv input file must be updated to include narrative_type column (though system treats this as optional if missing).

2. **Provider API Response Formats**: Citation extraction depends on stable API response formats from each provider. Changes to provider APIs may require updates to parsing logic.

3. **URL Parsing Library**: Domain extraction requires a robust URL parsing library (standard library urllib.parse or equivalent) to handle various URL formats.

## Out of Scope

1. **Historical Data Migration**: Converting existing CSVs with the old schema (19 columns) to the new schema (20 columns) is out of scope for this feature. A separate migration script or tool should be created if needed.

2. **Citation Content Extraction**: Extracting or storing the actual content/text from cited URLs (e.g., fetching and archiving web pages) is out of scope. We only store URLs and metadata.

3. **Citation Quality Scoring**: Evaluating or scoring the quality, credibility, or relevance of citations is out of scope. This feature focuses on data structure, not analysis algorithms.

4. **Advanced Search Result Analysis**: Features like duplicate detection across results, relevance scoring of snippets, or semantic analysis of result content are out of scope.

5. **Real-time Citation Validation**: Checking if citation URLs are still accessible, detecting broken links, or validating that URLs lead to expected content is out of scope.

6. **Multi-language Support**: Handling citations or search results in multiple languages, including language detection or translation, is out of scope for this schema redesign.

7. **Performance Optimization**: Advanced optimization techniques like parallel processing, caching, or database indexing are out of scope. This feature focuses on correctness and data structure.

## Notes

- **Schema Version Configuration**: The implementation should include a `CFG["csv_schema_version"]` parameter that allows switching between "v1" (old 19-column schema) and "v2" (new 20-column schema) for backward compatibility during transition period.

- **Citation ID Generation**: The source_id and result_id fields should use a predictable format (e.g., "{answer_id}_source_{index}" and "{answer_id}_result_{index}") to make debugging and data validation easier.

- **JSON Storage for Lists**: The answer_citation_list column should store citations as a valid JSON array (e.g., `["url1", "url2"]`) to enable both programmatic parsing and human readability.

- **Empty vs Null**: The specification uses "empty" to mean empty string ("") for consistency. All optional text fields should use empty strings rather than NULL values to simplify CSV parsing.
