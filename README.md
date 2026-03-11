# Agent Execution Transcript PoC

Proof-of-concept demonstrating **per-turn timing and reasoning capture** from the OpenAI Agents SDK using `run_streamed()`.

## What This Proves

OpenAI doesn't expose per-step timings via the API. But by timestamping streaming events as they arrive, we can reconstruct the full execution timeline:

- **Per code interpreter call timing** — how long each code block took (total + execution)
- **Reasoning summaries** — the model's chain of thought between steps
- **Phase breakdown** — reasoning vs. code execution vs. message output
- **Token usage** — input/output/total per response

## Example Output

From a real run analyzing two real data fimes:

```
PHASE BREAKDOWN:
  Reasoning:          0.0s → 238.8s  (interleaved, ~70s total)
  Code interpreter 1: 8.9s   (duration: 10,814ms)  — Load file 1
  Code interpreter 2: 19.7s  (duration: 20,127ms)  — Load  file 2
  Code interpreter 3: 49.1s  (duration: 2,834ms)   — Inspect data types
  ...
  Code interpreter 15: 190.0s (duration: 24,902ms) — Generate final chart
  ...17 calls total

  Total wall clock:   266.5s
  Code interpreter:   114,563ms across 17 calls (43%)
  Reasoning/overhead: 151,923ms (57%)
```

Reasoning traces show the model's thinking:
- *"Analyzing ROI calculations — I'm considering columns like Cost and..."*
- *"Resolving DataFrame errors — I noticed an error with the DataFrame..."*
- *"Calculating ROI by Channel — I've got the ROI by channel for brand 1..."*

## How It Works

The script:

1. Uploads data files to OpenAI (simulating `data_query_tool` returning file IDs)
2. Creates an `Agent` with `CodeInterpreterTool` and reasoning summaries enabled
3. Runs `Runner.run_streamed()` and iterates over `stream_events()`
4. Timestamps each event: `code_interpreter_call.in_progress`, `.interpreting`, `.completed`, `reasoning_summary_part.added`, `.done`
5. Prints the full timeline with per-call durations

Key streaming events captured:

| Event | What It Tells You |
|---|---|
| `response.code_interpreter_call.in_progress` | Code block started |
| `response.code_interpreter_call.interpreting` | Execution began inside container |
| `response.code_interpreter_call.completed` | Code block finished (diff = execution time) |
| `response.reasoning_summary_part.added` | Reasoning phase started |
| `response.reasoning_summary_part.done` | Reasoning phase ended (includes summary text) |

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

By default the script expects two `.xlsx` files. Edit the `file_paths` list and the prompt in `main()` to match your data.

## Configuration

The agent uses:
- **Model:** `gpt-5` (reasoning summaries require gpt-5+ or o-series)
- **Reasoning:** `effort="medium"`, `summary="auto"`
- **Container:** `{"type": "auto"}` (OpenAI manages lifecycle)

These match the analysis-agent production configuration.

## Integration

This approach requires **no new API features**. It works with the existing `stream_events()` iterator that the analysis-agent already consumes. Two implementation options:

1. **Add timestamps in the existing stream processor** — ~50 lines alongside the current Redis publishing
2. **Enable TracingProcessor** — remove `set_tracing_disabled()`, add a custom processor via `add_trace_processor()`

Both are additive — no changes to user-facing streaming behavior.
