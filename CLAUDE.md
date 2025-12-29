# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Constitution**: Development principles are defined in `.specify/memory/constitution.md`. All code changes must comply with these principles.

## Project Overview

This is a disinformation research pipeline that queries multiple LLM providers (OpenAI, Anthropic/Claude, Google Gemini, xAI/Grok, Perplexity) with narrative prompts from a CSV file and collects their responses with web search capabilities. The tool is designed to evaluate how different AI models respond to fact-checking queries about potentially false narratives.

Built as a Marimo notebook (`marimo_api_pipeline.py`), which provides both interactive notebook interface and executable Python script functionality.

## Running the Pipeline

### Prerequisites

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Required packages:
- `marimo` - Notebook framework
- `pandas` - Data processing
- `python-dotenv` - Environment variable management
- `openai` - OpenAI API client (also used for xAI and Perplexity)
- `anthropic` - Claude API client (optional, only if using Claude)
- `google-genai` - Gemini API client (optional, only if using Gemini)

2. Copy `.env.example` to `.env` and configure API keys:
```bash
cp .env.example .env
```

3. Edit `.env` with your API keys (at least one provider required):
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
XAI_API_KEY=your_key_here
PERPLEXITY_API_KEY=your_key_here
```

### Execution Methods

**Interactive notebook mode:**
```bash
marimo edit marimo_api_pipeline.py
```

**Script mode:**
```bash
marimo run marimo_api_pipeline.py
```

**Standard Python:**
```bash
python marimo_api_pipeline.py
```

### Configuration

The pipeline is controlled via the `CFG` dictionary (defined around line 54). Key settings:

- `input_csv`: Input CSV file path (default: `narratives.csv`)
- `note_text_col`: Column containing the prompt text (default: `prompt`)
- `note_id_col`: Column containing narrative IDs (default: `narrative_id`)
- `max_rows`: Limit number of rows to process (0 = all)
- `output_dir`: Where to save results (default: `llm_runs/`)
- `max_concurrency`: Concurrent API requests across all providers (default: 6)
- `batch_size`: Number of records per processing batch (default: 10)
- `temperature`: Model temperature setting (default: 0.2)
- `max_tokens`: Max response length (default: 512)

**Safety mechanism:** Set both `CFG["run"] = True` and `CFG["confirm"] = "RUN"` to execute.

### Web Search Configuration

Each provider supports web search with provider-specific implementations:
- OpenAI: Uses native `web_search_options` parameter
- Claude: Uses `web_search_20250305` tool with configurable domains/locations
- Gemini: Uses Google Search tool
- xAI/Grok: Uses `search_parameters` with citation control
- Perplexity: Native search with `search_mode` and recency filters

Configure via `CFG["web_search"]` dictionary.

## Architecture

### Core Components

1. **Provider System** (lines 196-331)
   - `ProviderConfig`: Immutable configuration dataclass for each LLM provider
   - `PROVIDERS`: List of all configured providers with API endpoints and models
   - Provider-specific adapters handle API differences between OpenAI-compatible, Anthropic, and Gemini APIs

2. **Adapter Pattern** (lines 582-882)
   - `ProviderAdapter`: Base class defining interface for all providers
   - `OpenAIAdapter`: Handles OpenAI, xAI, and Perplexity (OpenAI-compatible APIs)
   - `AnthropicAdapter`: Handles Claude API with web search tool integration
   - `GeminiAdapter`: Handles Google Gemini with synchronous API wrapped in asyncio

3. **Pipeline Execution** (lines 355-1051)
   - Async/await pattern with semaphore-based concurrency control
   - Processes input CSV in batches
   - For each narrative, queries all available providers in parallel
   - Appends results to provider-specific CSV files incrementally

### Data Flow

1. Load narratives from `narratives.csv` (columns: `narrative_id`, `prompt`, `narrative_type`, `origin_note`)
2. For each narrative, create a `prompt_id` and generate prompts using templates
3. Initialize API clients for all providers with valid API keys
4. Process records in batches (configurable batch size)
5. Within each batch, query all providers concurrently (shared semaphore for rate limiting)
6. Append each response immediately to `llm_runs/{provider}.csv`

### Output Schema

Results saved to `llm_runs/{provider}.csv` with columns defined in `CSV_COLUMNS` (line 139):
- Execution metadata: `run_id`, `prompt_id`, `requested_at`, `latency_ms`
- Provider info: `provider`, `model`, `base_url`
- Input: `note_id`, `note_text`, `system_prompt`, `user_prompt`
- Response: `response_text`, `finish_reason`, `search_results_json`, `raw_json`
- Token usage: `prompt_tokens`, `completion_tokens`, `total_tokens`
- Error handling: `error` (null on success)

## Key Implementation Details

### Client Initialization
- Clients are lazily initialized only for providers with valid API keys
- Errors during client creation are captured in `provider_errors` dict
- Missing API keys result in error rows rather than pipeline failure

### Error Handling
- Per-provider error isolation: one provider's failure doesn't affect others
- Empty note text generates error rows for all providers
- API errors captured with latency measurement and written to output CSV

### Concurrency Model
- Single shared semaphore controls max concurrent requests across all providers
- Batch processing with `asyncio.gather()` for parallelism
- Synchronous Gemini client wrapped with `asyncio.to_thread()`

### Prompt Construction
- System prompt defines assistant behavior and output format expectations
- User prompt template supports variable substitution (currently only `{note_text}`)
- Safe formatting escapes curly braces in note text to prevent template injection

## Dependencies

Primary libraries:
- `marimo`: Notebook framework
- `openai`: OpenAI API client (also used for xAI and Perplexity)
- `anthropic`: Claude API client
- `google-genai`: Gemini API client
- `pandas`: CSV processing and data display
- Standard library: `asyncio`, `csv`, `dataclasses`, `pathlib`, `uuid`

## Known Limitations

1. **OpenAI Web Search**: The `web_search_options` parameter for OpenAI is currently disabled (line 310) as it is still an experimental feature not yet officially supported in the public API.

2. **Provider Isolation**: If a provider's SDK is not installed, that provider will be skipped with an error recorded in the output CSV. At least one provider SDK must be installed for the pipeline to function.

3. **Marimo Display**: When running in script mode (`python marimo_api_pipeline.py`), the final cell that displays CSV results may cause the script to hang. This is a known issue with the `display()` function in non-interactive environments.

## Active Technologies
- Python 3.11+ (existing marimo_api_pipeline.py uses Python 3.11+ features) + marimo>=0.18.4, pandas>=2.3.3, openai>=2.14.0, anthropic>=0.75.0, google-genai>=1.56.0, python-dotenv>=1.0.0, urllib.parse (stdlib) (001-csv-schema-redesign)
- CSV files (provider-specific output files in llm_runs/ directory) (001-csv-schema-redesign)

## Recent Changes
- 001-csv-schema-redesign: Added Python 3.11+ (existing marimo_api_pipeline.py uses Python 3.11+ features) + marimo>=0.18.4, pandas>=2.3.3, openai>=2.14.0, anthropic>=0.75.0, google-genai>=1.56.0, python-dotenv>=1.0.0, urllib.parse (stdlib)
