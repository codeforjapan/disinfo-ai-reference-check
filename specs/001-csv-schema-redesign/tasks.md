# Tasks: CSV Schema Redesign for Citation and Search Result Tracking

**Input**: Design documents from `/specs/001-csv-schema-redesign/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/csv-schema-v2.md, quickstart.md

**Tests**: Manual validation only (no test framework in place). Each user story includes validation tasks at the end.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing. Each story delivers a complete, testable increment.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different code sections, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4, US5)
- Include exact file paths or line numbers in descriptions

## Path Conventions

- **Single notebook project**: All code in `marimo_api_pipeline.py`
- Line numbers are approximate and may shift during implementation
- Refer to `quickstart.md` for detailed implementation steps

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add new helper functions and update configuration

- [X] T001 Add CFG["csv_schema_version"] = "v2" parameter in marimo_api_pipeline.py CFG dictionary (line ~65)
- [X] T002 [P] Implement extract_domain(url) helper function in new utility cell after CSV_COLUMNS definition (line ~175)
- [X] T003 [P] Implement extract_urls_from_text(text) helper function for OpenAI citation extraction in same utility cell
- [X] T004 [P] Implement expand_citations_to_rows(base_row, citations, results) helper function for row expansion in same utility cell

**Checkpoint**: ✅ Helper functions ready - can now proceed with schema and adapter updates

---

## Phase 2: Foundational (CSV Schema & Row Builders)

**Purpose**: Core schema changes that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Update CSV_COLUMNS definition from 19 to 20 columns in marimo_api_pipeline.py (line ~150) per contracts/csv-schema-v2.md
- [X] T006 Update _base_row_local function to use new column names (narrative_id, narrative_type, narrative_prompt, model_name, model_version, answer_*, source_*, result_*) in marimo_api_pipeline.py (line ~461)
- [X] T007 Update _success_row_local function to populate answer_* fields and add answer_citation_list as JSON array in marimo_api_pipeline.py (line ~485)
- [X] T008 Update load_input_csv function to optionally read narrative_type column (treat as optional) in marimo_api_pipeline.py (line ~176-201)
- [X] T009 Modify _append_row_local function to accept citations and results parameters instead of single row dict in marimo_api_pipeline.py (line ~898)
- [X] T010 Update _append_row_local to call expand_citations_to_rows and write multiple expanded rows using csv.DictWriter in marimo_api_pipeline.py (line ~898-933)

**Checkpoint**: ✅ Foundation ready - citation extraction and row expansion infrastructure complete

---

## Phase 3: User Story 1 - Citation Analysis Across Providers (Priority: P1) 🎯 MVP

**Goal**: Enable researchers to analyze which sources different AI models cite when fact-checking narratives

**Independent Test**: Run pipeline with any provider (OpenAI, Claude, Gemini), export CSV, verify each cited source appears as separate row with source_url and source_domain populated

### Implementation for User Story 1

- [X] T011 [P] [US1] Update OpenAIAdapter.parse_response to extract citations using extract_urls_from_text in marimo_api_pipeline.py (line ~689-742)
- [X] T012 [P] [US1] Update AnthropicAdapter.parse_response to extract citations from tool_use blocks (web_search tool) in marimo_api_pipeline.py (line ~744-810)
- [X] T013 [P] [US1] Update GeminiAdapter.parse_response to extract citations from grounding_metadata.grounding_chunks in marimo_api_pipeline.py (line ~812-883)
- [X] T014 [US1] Update query_one_provider_async to extract citations from parsed response and pass to _append_row_local in marimo_api_pipeline.py (line ~355-458)
- [X] T015 [US1] Modify all adapter parse_response methods to return {"citations": [...], "results": [], ...} dict structure

### Validation for User Story 1

- [ ] T016 [US1] Manual validation: Run pipeline with CFG["max_rows"] = 3, verify 0 citations → 1 row with empty source_* fields
- [ ] T017 [US1] Manual validation: Run pipeline with OpenAI, verify URLs extracted from text and domain parsed correctly (www. removed, subdomains kept)
- [ ] T018 [US1] Manual validation: Run pipeline with Claude, verify tool_use citations extracted correctly
- [ ] T019 [US1] Manual validation: Run pipeline with Gemini, verify grounding_metadata citations extracted correctly
- [ ] T020 [US1] Manual validation: Verify CSV has 20 columns in correct order per contracts/csv-schema-v2.md
- [ ] T021 [US1] Manual validation: Verify 3 citations → 3 rows with same answer_id and identical answer_* fields

**Checkpoint**: User Story 1 complete - citation tracking works for OpenAI, Claude, Gemini with row expansion

---

## Phase 4: User Story 3 - Cross-Model Citation Comparison (Priority: P1)

**Goal**: Enable comparison of citation patterns across multiple AI models for the same narrative

**Independent Test**: Run pipeline with 2+ providers for same narrative, perform CSV join on narrative_id, verify all citations from each provider visible

**Note**: User Story 3 is implemented before US2 because it's also P1 priority and builds directly on US1 foundation

### Implementation for User Story 3

- [ ] T022 [P] [US3] Update OpenAIAdapter for xAI/Grok provider to extract citations from citations array (when return_citations=true) in marimo_api_pipeline.py (line ~689-742)
- [ ] T023 [US3] Verify all 5 providers (OpenAI, Claude, Gemini, xAI, Perplexity) use consistent citation extraction returning same dict structure

### Validation for User Story 3

- [ ] T024 [US3] Manual validation: Run pipeline with same narrative_id across 3 different providers
- [ ] T025 [US3] Manual validation: Filter CSV by narrative_id, verify all citations from each provider are present with model_name populated
- [ ] T026 [US3] Manual validation: Group by source_domain, verify consensus sources (cited by 2+ providers) identifiable
- [ ] T027 [US3] Manual validation: Count distinct source_urls per model_name, verify citation diversity metrics calculable

**Checkpoint**: User Story 3 complete - cross-model comparison enabled via citation tracking

---

## Phase 5: User Story 2 - Perplexity Search Result Deep Dive (Priority: P2)

**Goal**: Enable detailed analysis of Perplexity search results including titles, snippets, and rankings

**Independent Test**: Run pipeline with Perplexity provider, verify result_title, result_snippet, result_rank populated for each search result

### Implementation for User Story 2

- [ ] T028 [US2] Update OpenAIAdapter.parse_response to extract Perplexity search_results array (when provider is perplexity) in marimo_api_pipeline.py (line ~726-731)
- [ ] T029 [US2] Modify expand_citations_to_rows to handle Perplexity Cartesian product (citations × results) in utility cell
- [ ] T030 [US2] Update _append_row_local to pass results parameter for Perplexity provider in marimo_api_pipeline.py (line ~898-933)
- [ ] T031 [US2] Verify result_* columns (result_id, result_url, result_domain, result_title, result_snippet, result_rank) populated correctly for Perplexity

### Validation for User Story 2

- [ ] T032 [US2] Manual validation: Run pipeline with Perplexity, verify search_results extracted from API response
- [ ] T033 [US2] Manual validation: Verify 2 citations × 3 results = 6 rows with Cartesian product structure
- [ ] T034 [US2] Manual validation: Verify result_rank is 1-based (1, 2, 3, ...) not 0-based
- [ ] T035 [US2] Manual validation: Verify result_* fields are empty for non-Perplexity providers (OpenAI, Claude, Gemini, xAI)
- [ ] T036 [US2] Manual validation: Verify result_snippet can handle long text (10KB+) without truncation

**Checkpoint**: User Story 2 complete - Perplexity search result tracking with Cartesian product expansion

---

## Phase 6: User Story 4 - Narrative Type Analysis (Priority: P2)

**Goal**: Enable classification-based analysis by narrative type (misinformation, satire, true, etc.)

**Independent Test**: Prepare input CSV with narrative_type column, run pipeline, verify narrative_type preserved in output CSV

### Implementation for User Story 4

- [ ] T037 [US4] Update load_input_csv to check for narrative_type column and add it to df_in if missing (with empty string default) in marimo_api_pipeline.py (line ~176-201)
- [ ] T038 [US4] Update _base_row_local to accept narrative_type parameter from input CSV row in marimo_api_pipeline.py (line ~461)
- [ ] T039 [US4] Update call sites in query_one_provider_async to pass narrative_type from df_in row to _base_row_local in marimo_api_pipeline.py (line ~355-458)

### Validation for User Story 4

- [ ] T040 [US4] Manual validation: Create test input CSV with narrative_type column containing "misinformation", "satire", "true" values
- [ ] T041 [US4] Manual validation: Run pipeline, verify narrative_type preserved in output CSV for all rows
- [ ] T042 [US4] Manual validation: Filter output CSV by narrative_type = "misinformation", verify filtering works
- [ ] T043 [US4] Manual validation: Run pipeline with input CSV lacking narrative_type column, verify pipeline continues with empty string values (no failure)
- [ ] T044 [US4] Manual validation: Group by narrative_type, verify citation pattern comparison across types possible

**Checkpoint**: User Story 4 complete - narrative type classification integrated

---

## Phase 7: User Story 5 - Answer-Level Analysis (Priority: P3)

**Goal**: Enable analysis of full answer text alongside citations for qualitative evaluation

**Independent Test**: Run pipeline, verify answer_text contains full model response without truncation

**Note**: This story is largely complete from US1 foundation - just needs validation

### Validation for User Story 5

- [ ] T045 [US5] Manual validation: Run pipeline with long model response (200+ words), verify answer_text preserved without truncation
- [ ] T046 [US5] Manual validation: Verify answer_text duplicated correctly across expanded rows (same value for all rows with same answer_id)
- [ ] T047 [US5] Manual validation: Read answer_text alongside source_url in CSV, verify relationship visible for qualitative analysis
- [ ] T048 [US5] Manual validation: Verify answer_raw_json contains complete API response for debugging/auditing

**Checkpoint**: User Story 5 complete - answer-level analysis validated

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, validation, and final cleanup

- [ ] T049 [P] Update CLAUDE.md with schema v2 notes, breaking changes, and migration guidance
- [ ] T050 [P] Create example narratives.csv with narrative_type column for testing
- [ ] T051 Validate all edge cases from spec.md: no citations, no answer text, Perplexity citation/result mismatch, unparseable URLs, missing narrative_type
- [ ] T052 Run full pipeline validation with CFG["max_rows"] = 10 across all 5 providers
- [ ] T053 Verify CSV row counts match expected expansion (log: "answer_id X created N rows from M citations")
- [ ] T054 Verify data integrity: all rows with same answer_id have identical narrative_*, model_*, and answer_* fields
- [ ] T055 Performance validation: Process 100 narratives × 5 providers in <10 minutes, verify row expansion adds <5% overhead
- [ ] T056 [P] Archive old v1 CSV files in llm_runs/ (rename to .v1.csv for reference)
- [ ] T057 Final validation: Run quickstart.md testing checklist completely

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - US1 (P1): Can start after Foundational - No dependencies on other stories ✅ **MVP TARGET**
  - US3 (P1): Can start after Foundational - No dependencies (just extends US1 to more providers)
  - US2 (P2): Can start after Foundational - Independent but benefits from US1 validation
  - US4 (P2): Can start after Foundational - Independent
  - US5 (P3): Can start after Foundational - Mostly validation of US1 work
- **Polish (Phase 8)**: Depends on all desired user stories being complete

### User Story Independence

All user stories are **independently testable**:

- **US1**: Test with any provider (OpenAI, Claude, Gemini) → citations extracted, rows expanded
- **US3**: Test with multiple providers for same narrative → comparison works
- **US2**: Test with Perplexity alone → search results tracked with Cartesian product
- **US4**: Test with narrative_type column → classification preserved
- **US5**: Test answer text preservation → qualitative analysis possible

### Within Each User Story

- Helper functions before adapter modifications
- All adapters can be modified in parallel (marked [P])
- Validation tasks run after implementation tasks complete
- Manual validation critical (no automated test framework)

### Parallel Opportunities

- **Phase 1 Setup**: All 4 tasks can run in parallel (T001-T004 are independent)
- **Phase 2 Foundational**: T005-T008 can run in parallel (different code sections)
- **US1 Implementation**: T011, T012, T013 can run in parallel (different adapters)
- **US3 Implementation**: Minimal work, extends US1
- **US2 Implementation**: T028-T030 can run sequentially or with T028 first
- **US4 Implementation**: T037-T039 can run sequentially (related code sections)
- **Phase 8 Polish**: T049, T050, T056 can run in parallel (different artifacts)

---

## Parallel Example: User Story 1

```bash
# Launch all adapter modifications for User Story 1 together:
Task: "Update OpenAIAdapter.parse_response to extract citations in marimo_api_pipeline.py (line ~689-742)"
Task: "Update AnthropicAdapter.parse_response to extract citations in marimo_api_pipeline.py (line ~744-810)"
Task: "Update GeminiAdapter.parse_response to extract citations in marimo_api_pipeline.py (line ~812-883)"

# These 3 tasks are independent - each modifies different adapter class
# Can be completed in parallel by 3 developers or sequentially
```

---

## Implementation Strategy

### MVP First (User Story 1 Only) ✅ RECOMMENDED

1. Complete Phase 1: Setup (T001-T004)
2. Complete Phase 2: Foundational (T005-T010) - CRITICAL foundation
3. Complete Phase 3: User Story 1 (T011-T021)
4. **STOP and VALIDATE**: Run validation tasks (T016-T021)
5. Deploy/demo if ready - researchers can now analyze citations from OpenAI/Claude/Gemini

**MVP delivers**:
- Citation tracking for 3 major providers
- Row-per-citation structure for easy analysis
- Domain extraction for source aggregation
- Immediately usable for research

### Incremental Delivery

1. **Foundation** (Phase 1-2): Setup + Foundational → Schema v2 ready
2. **MVP** (Phase 3): User Story 1 → Citation tracking works → Deploy/Demo
3. **Enhancement 1** (Phase 4): User Story 3 → Cross-model comparison → Deploy/Demo
4. **Enhancement 2** (Phase 5): User Story 2 → Perplexity deep dive → Deploy/Demo
5. **Enhancement 3** (Phase 6): User Story 4 → Narrative type analysis → Deploy/Demo
6. **Enhancement 4** (Phase 7): User Story 5 → Answer-level analysis validation → Deploy/Demo
7. **Polish** (Phase 8): Documentation and final validation

Each story adds value without breaking previous stories.

### Parallel Team Strategy

With multiple developers (NOT REQUIRED - single developer can complete sequentially):

1. **Together**: Phase 1 (Setup) + Phase 2 (Foundational)
2. **After Foundational completes**:
   - Developer A: US1 (OpenAI adapter)
   - Developer B: US1 (Claude adapter)
   - Developer C: US1 (Gemini adapter)
3. Merge US1 work, validate together
4. Repeat for US2-US5 if desired

---

## Task Count Summary

- **Phase 1 (Setup)**: 4 tasks
- **Phase 2 (Foundational)**: 6 tasks
- **Phase 3 (US1 - P1)**: 11 tasks (6 implementation + 5 validation)
- **Phase 4 (US3 - P1)**: 6 tasks (2 implementation + 4 validation)
- **Phase 5 (US2 - P2)**: 9 tasks (4 implementation + 5 validation)
- **Phase 6 (US4 - P2)**: 8 tasks (3 implementation + 5 validation)
- **Phase 7 (US5 - P3)**: 4 tasks (0 implementation, 4 validation)
- **Phase 8 (Polish)**: 9 tasks

**Total**: 57 tasks

**Parallel opportunities**: 15 tasks marked [P] (26% of tasks)

**MVP scope** (Phase 1-3): 21 tasks (37% of total) - Delivers core citation tracking value

**Independent test criteria per story**:
- US1: CSV exports show one row per citation with source_url/source_domain
- US3: Multiple providers for same narrative all visible in CSV
- US2: Perplexity result_* fields populated with Cartesian product
- US4: narrative_type preserved from input to output CSV
- US5: answer_text preserved without truncation across expanded rows

---

## Notes

- [P] tasks = different code sections/adapters, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story delivers independently completable and testable increment
- Manual validation critical - no automated test framework exists
- Line numbers approximate - use file search to locate exact functions
- Refer to quickstart.md for detailed implementation steps per task
- Stop at any checkpoint to validate story independently before proceeding
- Breaking change: Archive old CSV files before running pipeline with v2 schema
