# Agent Execution Transcript PoC

Proof-of-concept demonstrating **per-turn timing, reasoning capture, and automated reasoning categorization** from the OpenAI Agents SDK using `run_streamed()`.

## What This Proves

OpenAI doesn't expose per-step timings via the API. But by timestamping streaming events as they arrive, we can reconstruct the full execution timeline:

- **Per code interpreter call timing** — how long each code block took (total + execution)
- **Reasoning summaries** — the model's chain of thought between steps
- **Phase breakdown** — reasoning vs. code execution vs. message output
- **Reasoning categorization** — LLM-based classification of reasoning phases into addressable vs. irreducible categories
- **Token usage** — input/output/total per response

## Example Output

From a real run analyzing two data files:

```
PHASE BREAKDOWN:
  Code interpreter 1: 8.9s   (duration: 10,814ms)
  Code interpreter 2: 19.7s  (duration: 20,127ms)
  ...17 calls total

  Total wall clock:   288.1s
  Code interpreter:   114,563ms across 17 calls (43%)
  Other (reasoning/streaming/overhead): 173,537ms (57%)

REASONING BREAKDOWN BY CATEGORY:
  Analytical reasoning (irreducible)     6    52.3s  37.7%  Not reducible
  Schema / structure discovery           5    25.1s  18.1%  ◀ ADDRESSABLE
  Analytical planning                    4    21.8s  15.7%  Pre-built templates
  Error recovery / debugging             3    18.2s  13.1%  ◀ ADDRESSABLE
  Output composition                     3    17.4s  12.5%  Not reducible
  Data assessment                        2     3.9s   2.8%  ◀ ADDRESSABLE

  Addressable:         47.2s (34%) ← eliminated by structured data + schema
  Irreducible:         91.5s (66%) ← actual analytical work

FULL TIME BUDGET:
  Code interpreter:               114.6s  (40%)
  Reasoning (addressable):         47.2s  (16%)
  Reasoning (irreducible):         91.5s  (32%)
  Message output + overhead:       35.0s  (12%)
  TOTAL:                          288.1s

  Estimated addressable total:    81.6s (28%)
  (addressable reasoning + ~30% of CI for file loading)
```

## How It Works

The script:

1. Uploads data files to OpenAI (simulating `data_query_tool` returning file IDs)
2. Creates an `Agent` with `CodeInterpreterTool` and reasoning summaries enabled
3. Runs `Runner.run_streamed()` and iterates over `stream_events()`
4. Timestamps each event: `code_interpreter_call.in_progress`, `.interpreting`, `.completed`, `reasoning_summary_part.added`, `.done`
5. After the run completes, classifies reasoning phases using GPT-4o-mini
6. Prints the full timeline with per-call durations and reasoning categorization

### Streaming Events Captured

| Event | What It Tells You |
|---|---|
| `response.code_interpreter_call.in_progress` | Code block started |
| `response.code_interpreter_call.interpreting` | Execution began inside container |
| `response.code_interpreter_call.completed` | Code block finished (diff = execution time) |
| `response.reasoning_summary_part.added` | Reasoning phase started |
| `response.reasoning_summary_part.done` | Reasoning phase ended (includes summary text) |

### Reasoning Categorization

After the run, reasoning summaries are classified into six categories using GPT-4o-mini:

| Category | Description | Addressable By |
|---|---|---|
| `schema_discovery` | Figuring out file structure, columns, types | Structured data + schema descriptions in prompt |
| `error_recovery` | Debugging DataFrame errors, fixing mistakes | Cleaner data + schema hints (model won't hit dead ends) |
| `data_assessment` | Evaluating data quality and sufficiency | Include confidence metadata with data |
| `analytical_planning` | Planning what analysis to do | Pre-built analysis templates / better prompts |
| `analytical_reasoning` | Actual analytical thinking | Not reducible — this is the analysis |
| `output_composition` | Formatting results, writing summaries | Not reducible — this is the deliverable |

The first three categories (**schema_discovery**, **error_recovery**, **data_assessment**) are directly addressable by providing structured data with rich schema annotations instead of raw Excel files.

## Requirements

```bash
pip install openai-agents openpyxl
```

## Usage

```bash
export OPENAI_API_KEY=sk-...

# With your own data files — edit the file paths in main()
python transcript_poc.py
```

By default the script expects two `.xlsx` files in a `data/` directory. Edit the `file_paths` list and the prompt in `main()` to match your data.

## Configuration

The agent uses:
- **Model:** `gpt-5` (reasoning summaries require gpt-5+ or o-series)
- **Reasoning:** `effort="medium"`, `summary="detailed"`
- **Container:** `{"type": "auto"}` (OpenAI manages lifecycle)

These match the analysis-agent production configuration.

## Integration

This approach requires **no new API features**. It works with the existing `stream_events()` iterator that the analysis-agent already consumes. Two implementation options:

1. **Add timestamps in the existing stream processor** — ~50 lines alongside the current Redis publishing
2. **Enable TracingProcessor** — remove `set_tracing_disabled()`, add a custom processor via `add_trace_processor()`

Both are additive — no changes to user-facing streaming behavior.
