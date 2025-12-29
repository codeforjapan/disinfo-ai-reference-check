<!--
Sync Impact Report:
Version: 1.0.0 (Initial constitution)
Ratification Date: 2025-12-29
Changes:
  - Initial constitution created for Marimo Python notebook development
  - Established 6 core principles for research pipeline development
  - Added Research Data Integrity section
  - Added Marimo Development Practices section

Templates Requiring Updates:
  ✅ plan-template.md - Constitution Check section will reference these principles
  ✅ spec-template.md - Functional requirements should align with data integrity and performance standards
  ✅ tasks-template.md - Task organization reflects async patterns and testing discipline
  ✅ checklist-template.md - Checklist items can reference principle compliance
  ✅ agent-file-template.md - Development guidelines can reference active principles

Follow-up TODOs: None
-->

# Disinfo AI Reference Check Constitution

## Core Principles

### I. Notebook-First Development

All features MUST be developed as executable Marimo notebook cells that support both interactive exploration and production execution. Code MUST work identically in three modes: `marimo edit` (interactive), `marimo run` (script), and `python` (standard execution). Configuration MUST be exposed as editable cell variables (e.g., `CFG` dictionary) for interactive experimentation. No functionality may require external scripts or configuration files that break the notebook paradigm.

**Rationale**: Marimo notebooks serve dual purposes—research exploration and production pipelines. Breaking the notebook paradigm forces users to choose between exploration and reliability, undermining the core value proposition.

### II. Provider Adapter Pattern (NON-NEGOTIABLE)

All LLM provider integrations MUST implement the `ProviderAdapter` interface with standardized async methods: `query_async()`, error handling, and response normalization. Each provider adapter MUST handle API-specific differences (authentication, request/response formats, web search integration) internally without leaking implementation details. Provider failures MUST be isolated—one provider's error CANNOT affect other providers' execution. New providers MUST be addable by creating a new adapter class and updating the `PROVIDERS` list, with no changes to pipeline logic.

**Rationale**: The pipeline queries 5+ heterogeneous LLM APIs (OpenAI-compatible, Anthropic, Google Gemini) with fundamentally different interfaces. Without strict adapter isolation, adding providers or handling API changes creates cascading modifications across the codebase.

### III. Async-First with Structured Concurrency

All I/O operations (API calls, file writes) MUST use `asyncio` with explicit concurrency control via semaphores. Synchronous SDK clients (e.g., Google Gemini) MUST be wrapped with `asyncio.to_thread()` to prevent blocking. Concurrency limits MUST be configurable via `CFG["max_concurrency"]` and enforced by a single shared semaphore across all providers. Batch processing MUST use `asyncio.gather()` with proper error isolation—individual failures MUST NOT cancel the entire batch.

**Rationale**: The pipeline makes hundreds of API calls across multiple providers. Without structured concurrency, rate limits are violated, timeout errors proliferate, and debugging becomes impossible due to race conditions.

### IV. Incremental Data Persistence

Results MUST be written to provider-specific CSV files immediately after each API response, not batched in memory. Each row MUST be appended with `csv.DictWriter` in a single write operation to prevent corruption. Output schema (defined in `CSV_COLUMNS`) MUST capture: execution metadata (run_id, latency), provider info (model, endpoint), full request/response (prompts, response_text, raw_json), token usage, and errors. No in-memory result aggregation is permitted—crashes MUST NOT lose completed work.

**Rationale**: Research pipelines run for hours with hundreds of expensive API calls. Losing results to crashes, memory errors, or Jupyter kernel restarts is unacceptable. Incremental writes enable pause/resume and real-time monitoring.

### V. Configuration-Driven Safety Gates

Execution MUST require explicit dual confirmation: `CFG["run"] = True` AND `CFG["confirm"] = "RUN"`. This prevents accidental expensive API calls during interactive exploration. All execution parameters (batch size, concurrency, temperature, max_tokens) MUST be configurable via the `CFG` dictionary with sensible defaults. API keys MUST be loaded from environment variables (`.env` file) with graceful degradation—missing keys result in error rows, not pipeline failure.

**Rationale**: Interactive notebooks make accidental execution trivial (shift+enter), which can burn through API quotas. Research requires parameter experimentation; hardcoded values create friction and increase error risk.

### VI. Error Transparency and Debuggability

All errors MUST be captured with context (provider, note_id, latency_ms) and written to output CSV with `error` field populated. Error messages MUST include actionable information (API error codes, timeout values, validation failures). Successful responses MUST have `error = null` for unambiguous filtering. Stack traces MUST be logged but NOT included in CSV output. Each API call MUST measure and record latency regardless of success/failure state.

**Rationale**: Research pipelines process diverse inputs that trigger edge cases. Silent failures corrupt datasets. Verbose errors clutter output. Structured error capture enables systematic debugging and quality analysis without manual log parsing.

## Research Data Integrity

### Prompt Reproducibility

All prompts MUST be constructed from templates stored in `CFG` dictionary with explicit variable substitution. Both `system_prompt` and `user_prompt_template` MUST be recorded in output CSV for every API call. Template variables (e.g., `{note_text}`) MUST be safely escaped to prevent format string injection. Prompt construction MUST be deterministic—same input CSV + same CFG = identical prompts across runs.

**Rationale**: Research validity requires reproducible prompts. Hardcoded prompts prevent experimentation. Unescaped templates create security risks and silent corruption.

### Input Validation

Empty or null `note_text` MUST generate error rows for all providers with descriptive error messages—no API calls may be made. CSV parsing errors MUST halt execution with clear error messages indicating row numbers and column names. Missing required columns MUST be detected before processing begins.

**Rationale**: Research data integrity depends on valid inputs. Silent skipping corrupts result interpretation. Late failures waste API quota on invalid batches.

### Output Schema Stability

The `CSV_COLUMNS` list defines the canonical output schema and MUST NOT be modified without migration documentation. Adding columns requires appending to the list and updating all adapter implementations. Removing columns requires deprecation warnings and explicit version bump. Column order MUST remain stable to preserve compatibility with downstream analysis scripts.

**Rationale**: Research pipelines accumulate months of historical data. Schema changes break analysis notebooks, visualizations, and comparison studies.

## Marimo Development Practices

### Cell Organization

Configuration MUST live in dedicated cells (e.g., CFG definition). Pure functions and classes MUST be defined in separate cells for reusability. Execution cells MUST be clearly marked (e.g., "Pipeline Execution" header comments). Display cells MUST be isolated at the end—`display()` calls MUST NOT block script execution when running non-interactively.

**Rationale**: Marimo's reactive execution model makes cell organization critical. Poor organization causes unintended re-execution, hidden dependencies, and debugging confusion.

### Dependency Management

All imports MUST happen in the first cell with explicit error handling for optional dependencies (anthropic, google-genai). Missing optional dependencies MUST result in provider-specific errors, not import failures. The `requirements.txt` file MUST list all dependencies with minimum version constraints. Version conflicts MUST be documented in CLAUDE.md Known Limitations section.

**Rationale**: Research environments have diverse setups. Forcing installation of unused provider SDKs creates friction. Import failures block notebook loading, preventing even reading documentation.

### Interactive vs. Script Execution

Code MUST work identically in interactive (`marimo edit`) and script (`marimo run`, `python`) modes. Features that break in script mode (e.g., `display()` hanging) MUST be documented in CLAUDE.md Known Limitations. Safety gates MUST work in both modes—accidental execution risks apply equally.

**Rationale**: Marimo's dual execution modes are a core feature. Mode-specific bugs force users into single-mode workflows, losing the exploration-to-production benefit.

## Governance

This constitution supersedes all other development practices. All code changes MUST comply with these principles. Principle violations MUST be explicitly justified in pull request descriptions with:
1. Which principle is violated
2. Why compliance is impossible or unreasonable
3. What mitigation prevents the violation from spreading

Amendments require:
1. Documentation of impact on existing code
2. Version bump following semantic versioning (MAJOR for breaking principle changes, MINOR for new principles, PATCH for clarifications)
3. Update of all dependent templates (plan, spec, tasks)
4. Migration guide for affected features

Complexity that violates principles MUST be avoided unless explicitly justified. Justifications MUST be documented in the `Complexity Tracking` section of implementation plans.

Development guidance for runtime use is provided in `CLAUDE.md` and must reference these principles where applicable.

**Version**: 1.0.0 | **Ratified**: 2025-12-29 | **Last Amended**: 2025-12-29
