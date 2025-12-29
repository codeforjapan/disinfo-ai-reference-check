# Implementation Plan: CSV Schema Redesign for Citation and Search Result Tracking

**Branch**: `001-csv-schema-redesign` | **Date**: 2025-12-29 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-csv-schema-redesign/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Redesign the output CSV schema from 19 to 20 columns with explicit separation of citation and search result data. Key changes: (1) rename columns to research-focused terminology (narrative_id, answer_text, model_name), (2) remove redundant metadata columns (base_url, token counts, latency), (3) add citation tracking (source_url, source_domain, answer_citation_list), (4) add Perplexity search result columns (result_title, result_snippet, result_rank), (5) implement row expansion logic where each citation creates a separate CSV row for easier analysis.

## Technical Context

**Language/Version**: Python 3.11+ (existing marimo_api_pipeline.py uses Python 3.11+ features)
**Primary Dependencies**: marimo>=0.18.4, pandas>=2.3.3, openai>=2.14.0, anthropic>=0.75.0, google-genai>=1.56.0, python-dotenv>=1.0.0, urllib.parse (stdlib)
**Storage**: CSV files (provider-specific output files in llm_runs/ directory)
**Testing**: Manual validation (no test framework currently in place - testing against real CSV outputs)
**Target Platform**: Linux/macOS/Windows (cross-platform Python, notebook-based execution)
**Project Type**: Single notebook project (all code in marimo_api_pipeline.py)
**Performance Goals**: Process 100 narratives × 5 providers in <10 minutes; row expansion should add <5% overhead
**Constraints**: Must maintain backward compatibility via CFG["csv_schema_version"]; no breaking changes to existing pipeline execution flow; memory-efficient row expansion (no large in-memory buffers)
**Scale/Scope**: Single 1000-line Python notebook; 5 provider adapters to modify; ~20 CSV columns; estimated 2,000-5,000 rows per typical research run

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Verify compliance with constitution principles (see `.specify/memory/constitution.md`):

- [x] **Notebook-First Development**: Feature modifies CSV schema only; all changes within notebook cells; works in all execution modes
- [x] **Provider Adapter Pattern**: Schema changes implemented in adapter parse_response methods; adapters remain isolated; no pipeline logic changes required
- [x] **Async-First with Structured Concurrency**: No changes to async patterns; row expansion happens synchronously within each adapter's response handling
- [x] **Incremental Data Persistence**: Row expansion preserves incremental write pattern; multiple expanded rows written immediately after each API response
- [x] **Configuration-Driven Safety Gates**: New CFG["csv_schema_version"] parameter added for v1/v2 switching; existing safety gates unchanged
- [x] **Error Transparency**: Error handling unchanged; errors still captured with context; schema change is transparent to error flow
- [x] **Prompt Reproducibility**: No prompt changes; existing system_prompt and user_prompt_template preserved in output
- [x] **Input Validation**: New optional validation for narrative_type column (treat as optional if missing); existing validation preserved
- [x] **Output Schema Stability**: VIOLATION JUSTIFIED - CSV_COLUMNS changes from 19 to 20 columns with renames/additions/removals (see Complexity Tracking)
- [x] **Cell Organization**: All changes within existing cell structure; CSV_COLUMNS definition cell updated; helper functions added in appropriate cells
- [x] **Dependency Management**: No new dependencies; uses stdlib urllib.parse for domain extraction; requirements.txt unchanged

**Violations**:
1. **Output Schema Stability** - Justified below in Complexity Tracking section. This is an intentional breaking change with version control and migration documentation.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
marimo_api_pipeline.py           # Single notebook file containing all code
├── [Line 150] CSV_COLUMNS       # Schema definition to be updated
├── [Line 196-331] PROVIDERS     # Provider configurations (unchanged)
├── [Line 355-881] Adapters      # Provider adapters (parse_response methods to be updated)
│   ├── OpenAIAdapter            # Citation extraction from message.content
│   ├── AnthropicAdapter         # Citation extraction from tool usage
│   └── GeminiAdapter            # Citation extraction from grounding_metadata
├── [Line 461-497] Row builders  # Helper functions to create base/success/error rows (to be updated for row expansion)
└── [Line 898-933] CSV writer    # _append_row_local function (to be updated for multiple row writes)

llm_runs/                        # Output directory
├── openai.csv                   # Provider-specific output files (schema v2)
├── claude.csv
├── gemini.csv
├── grok.csv
└── perplexity.csv

narratives.csv                   # Input file (optionally add narrative_type column)
requirements.txt                 # No changes needed (uses stdlib for URL parsing)
```

**Structure Decision**: Single notebook project. All schema changes implemented within existing marimo_api_pipeline.py cells. No new files created. Changes localized to:
1. CSV_COLUMNS definition (line ~150)
2. Provider adapter parse_response methods (lines ~582-882)
3. Row builder helpers (lines ~461-497)
4. CSV writer _append_row_local (lines ~898-933)
5. New helper function for domain extraction and row expansion

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Output Schema Stability (breaking change to CSV_COLUMNS) | Research requirements have evolved to need citation-level analysis. Current schema bundles all citations in search_results_json making per-citation analysis require complex JSON parsing in every analysis script. Row-per-citation structure enables standard CSV operations (filter, group, join). | Keeping schema unchanged and adding new columns would create dual representations (search_results_json + source_* columns) with synchronization issues. JSON-in-CSV approach requires non-standard tooling and fails the "self-documenting" success criterion. Adding columns without removing redundant ones (base_url, token counts, latency) violates simplicity and creates confusion about which fields to use. |

## Phase 0: Research (Complete)

**Status**: ✅ Complete
**Output**: `research.md`

### Key Findings

1. **Citation Extraction Patterns**: Researched provider-specific extraction methods for OpenAI (regex), Claude (tool_use), Gemini (grounding_metadata), xAI (citations array), and Perplexity (dual structure)

2. **URL Domain Extraction**: Decided to use `urllib.parse.urlparse()` with removal of ONLY `www.` prefix while preserving other subdomains for semantic significance

3. **CSV Row Expansion**: Adopted generator pattern for memory-efficient row expansion with immediate CSV writing (no buffering)

4. **Backward Compatibility**: Implemented configuration-driven schema selection with `CFG["csv_schema_version"]` parameter

**Deliverable**: research.md documents all decisions with rationale and alternatives considered

---

## Phase 1: Design & Contracts (Complete)

**Status**: ✅ Complete
**Outputs**: `data-model.md`, `contracts/csv-schema-v2.md`, `quickstart.md`

### Data Model

Defined 5 core entities:
- **Narrative** (input): narrative_id, narrative_type, narrative_prompt
- **Model** (config): model_name, model_version
- **Answer** (output): answer_id, answer_text, answer_timestamp, answer_citation_list, answer_raw_json
- **Citation/Source** (expanded): source_id, source_url, source_domain
- **SearchResult** (Perplexity): result_id, result_url, result_domain, result_title, result_snippet, result_rank

**Row expansion rules**:
- Standard providers: 1 answer → N rows (one per citation)
- Perplexity: 1 answer → N×M rows (citations × search_results)

### CSV Schema Contract

Complete 20-column schema definition with:
- Field specifications (type, source, constraints, examples)
- Provider-specific contracts (OpenAI, Claude, Gemini, xAI, Perplexity)
- Validation rules (pre-write and post-write)
- Row expansion rules with examples
- Migration mapping from v1 (19 columns) to v2 (20 columns)

### Quickstart Guide

Step-by-step implementation guide covering:
- CSV_COLUMNS update (Line ~150)
- Helper functions (domain extraction, row expansion)
- Adapter modifications for citation extraction
- Row builder updates for v2 column names
- CSV writer changes for multiple row writes
- Testing checklist and validation procedures

---

## Phase 2: Task Generation

**Status**: ⏭️ Not Started (use `/speckit.tasks` command)
**Expected Output**: `tasks.md`

Task breakdown will be generated based on:
- research.md findings
- data-model.md entity definitions
- quickstart.md implementation steps
- Constitution compliance requirements

Estimated task categories:
1. Schema definition updates
2. Helper function implementation
3. Provider adapter modifications (5 adapters)
4. Row builder updates
5. CSV writer modifications
6. Testing and validation

---

## Implementation Readiness

### Prerequisites Met

- [x] All technical unknowns resolved (research.md)
- [x] Data model defined (data-model.md)
- [x] CSV schema contract specified (contracts/csv-schema-v2.md)
- [x] Implementation guide created (quickstart.md)
- [x] Constitution principles validated
- [x] Agent context updated (CLAUDE.md)

### Ready for Implementation

The feature is **ready for task breakdown** (`/speckit.tasks`) and subsequent implementation.

**Confidence Level**: High
- Clear technical approach with researched patterns
- Well-defined schema contract with validation rules
- Step-by-step quickstart guide for developers
- Single notebook project simplifies implementation scope
- No new dependencies required (uses stdlib)

**Estimated Complexity**: Medium
- Localized changes within existing file structure
- 5 provider adapters to modify (similar patterns)
- Testing requires manual validation (no test framework)
- Breaking change requires careful documentation

---

## Post-Implementation Checklist

After implementation completes, verify:

1. **Schema Validation**:
   - [ ] CSV has exactly 20 columns in correct order
   - [ ] All column names match CSV_COLUMNS definition
   - [ ] No missing or extra columns

2. **Row Expansion**:
   - [ ] 0 citations → 1 row (empty source_* fields)
   - [ ] N citations → N rows with same answer_id
   - [ ] Perplexity: N citations × M results = N×M rows
   - [ ] All expanded rows have identical answer_* fields

3. **Citation Extraction**:
   - [ ] OpenAI: URLs extracted from text
   - [ ] Claude: Tool_use blocks parsed
   - [ ] Gemini: Grounding_metadata parsed
   - [ ] xAI: Citations array parsed
   - [ ] Perplexity: Both citations and search_results parsed

4. **Domain Extraction**:
   - [ ] www. prefix removed
   - [ ] Other subdomains preserved
   - [ ] Invalid URLs return empty string

5. **Data Integrity**:
   - [ ] answer_id unique per API response
   - [ ] answer_citation_list valid JSON array
   - [ ] source_id format: {answer_id}_source_{index}
   - [ ] result_* empty for non-Perplexity providers

6. **Documentation**:
   - [ ] CLAUDE.md updated with schema v2 notes
   - [ ] Migration guide provided
   - [ ] Breaking changes documented

---

## Next Command

Run `/speckit.tasks` to generate actionable task breakdown for implementation.
