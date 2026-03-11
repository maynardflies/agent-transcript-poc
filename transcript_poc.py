"""
Proof-of-concept: Agent Execution Transcript with Per-Turn Timing

Uses run_streamed() to capture per-event timestamps — proving we can
break down where every second goes in a real analysis request.

Upload your own data files (e.g. Excel, CSV) and edit the prompt
in main() to match your use case.

Requirements:
    pip install openai-agents openpyxl

Environment:
    OPENAI_API_KEY must be set

Run:
    python transcript_poc.py
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agents import Agent, Runner, ModelSettings
from agents.tool import CodeInterpreterTool
from openai import AsyncOpenAI
from openai.types.shared.reasoning import Reasoning


# ---------------------------------------------------------------------------
# 1. Event-based transcript — timestamps each streaming event
# ---------------------------------------------------------------------------

@dataclass
class TranscriptEvent:
    event_type: str
    timestamp: float
    wall_clock: str
    duration_ms: float | None = None
    detail: str = ""


@dataclass
class TranscriptCollector:
    events: list[TranscriptEvent] = field(default_factory=list)
    _pending: dict[str, float] = field(default_factory=dict)

    def record(self, event_type: str, detail: str = ""):
        self.events.append(TranscriptEvent(
            event_type=event_type,
            timestamp=time.monotonic(),
            wall_clock=datetime.now(timezone.utc).isoformat(),
            detail=detail,
        ))

    def record_start(self, key: str, event_type: str, detail: str = ""):
        self._pending[key] = time.monotonic()
        self.record(event_type, detail)

    def record_end(self, key: str, event_type: str, detail: str = ""):
        start = self._pending.pop(key, None)
        duration = round((time.monotonic() - start) * 1000, 1) if start else None
        self.events.append(TranscriptEvent(
            event_type=event_type,
            timestamp=time.monotonic(),
            wall_clock=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration,
            detail=f"{detail} [{duration:.0f}ms]" if duration else detail,
        ))

    def print_transcript(self):
        print("\n" + "=" * 80)
        print("EXECUTION TRANSCRIPT — Per-Turn Timing")
        print("=" * 80)

        if not self.events:
            print("  (no events captured)")
            return

        t0 = self.events[0].timestamp

        # Collect phases for the summary
        phases: list[tuple[str, float, float]] = []  # (name, start_offset, end_offset)
        current_phase = None
        phase_start = 0.0

        print(f"\n{'#':<4} {'T+':>8} {'Duration':>10} {'Event':<25} Detail")
        print(f"{'─'*4} {'─'*8} {'─'*10} {'─'*25} {'─'*50}")

        for i, e in enumerate(self.events, 1):
            offset = e.timestamp - t0
            offset_str = f"{offset:.1f}s"
            dur = f"{e.duration_ms:.0f}ms" if e.duration_ms else ""
            detail = e.detail[:70] if e.detail else ""
            print(f"{i:<4} {offset_str:>8} {dur:>10} {e.event_type:<25} {detail}")

        # Phase summary
        print(f"\n{'─'*80}")
        print("PHASE BREAKDOWN:")

        # Group events into phases
        reasoning_events = [e for e in self.events if e.event_type.startswith("reasoning")]
        ci_events = [e for e in self.events if e.event_type.startswith("code_interpreter")]
        msg_events = [e for e in self.events if e.event_type.startswith("message")]

        # Reasoning phase timing
        if reasoning_events:
            r_start = reasoning_events[0].timestamp - t0
            r_end = reasoning_events[-1].timestamp - t0
            print(f"  Reasoning:          {r_start:.1f}s → {r_end:.1f}s  ({r_end - r_start:.1f}s)")

        # Per code interpreter call
        ci_calls = [e for e in ci_events if "started" in e.detail]
        ci_completes = [e for e in ci_events if e.duration_ms is not None]
        for idx, (start_e, end_e) in enumerate(zip(ci_calls, ci_completes), 1):
            s = start_e.timestamp - t0
            dur = end_e.duration_ms
            print(f"  Code interpreter {idx}: {s:.1f}s  (duration: {dur:.0f}ms)")

        # Total
        total = (self.events[-1].timestamp - t0)
        print(f"\n  Total wall clock:   {total:.1f}s")

        # Token usage summary
        timed = [e for e in self.events if e.duration_ms is not None]
        ci_total = sum(e.duration_ms for e in ci_completes)
        if ci_completes:
            print(f"  Code interpreter:   {ci_total:.0f}ms across {len(ci_completes)} call(s)")
            print(f"  Other (reasoning/streaming/overhead): {total*1000 - ci_total:.0f}ms")
        print()


# ---------------------------------------------------------------------------
# 2. Upload files to OpenAI for code interpreter access
# ---------------------------------------------------------------------------

async def upload_files(client: AsyncOpenAI, file_paths: list[str]) -> list[str]:
    """Upload files and return their OpenAI file IDs."""
    file_ids = []
    for path in file_paths:
        print(f"  Uploading {path.split('/')[-1]}...")
        with open(path, "rb") as f:
            response = await client.files.create(file=f, purpose="assistants")
        file_ids.append(response.id)
        print(f"    → {response.id}")
    return file_ids


# ---------------------------------------------------------------------------
# 3. Run the realistic analysis
# ---------------------------------------------------------------------------

async def main():
    client = AsyncOpenAI()
    collector = TranscriptCollector()

    # Upload data files — same as what data_query_tool returns as file_ids
    # Replace these paths with your own data files
    print("Uploading data files to OpenAI...")
    file_ids = await upload_files(client, [
        "data/brand_a.xlsx",
        "data/brand_b.xlsx",
    ])
    print(f"Uploaded {len(file_ids)} files\n")

    # Use the actual system prompt from the analysis-agent
    system_prompt = (
        "You are a helpful analyst assistant with Python code interpreter capabilities. "
        "Always begin by understanding the user's goal. "
        "After receiving data, evaluate what you have and tell the user whether the "
        "available data is sufficient to proceed. "
        "Before attempting deeper calculations, present a concise analysis plan "
        "(two to three steps) describing how you will answer the question based on "
        "the available sources. "
        "Execute the plan using the code_interpreter tool, leveraging the provided "
        "file IDs, and stream your code, intermediate outputs, and final findings."
    )

    # Create agent matching the production config
    agent = Agent(
        name="Analysis Agent",
        instructions=system_prompt,
        tools=[
            CodeInterpreterTool(
                tool_config={
                    "type": "code_interpreter",
                    "container": {
                        "type": "auto",
                        "file_ids": file_ids,
                    },
                }
            )
        ],
        model="gpt-5",
        model_settings=ModelSettings(
            reasoning=Reasoning(
                effort="medium",
                summary="auto",
            )
        ),
    )

    # Cross-file analysis prompt — forces multi-turn reasoning
    # Replace with a prompt relevant to your data
    prompt = (
        "I've uploaded two data files containing media ROI data for two "
        "different brands.\n\n"
        "I need you to:\n"
        "1. First, explore both files — understand the structure, time periods "
        "covered, and what channels/platforms are present in each.\n"
        "2. Compare total media spend and ROI across both brands for the most "
        "recent full year, broken down by channel. Which brand is getting "
        "better ROI from its media spend?\n"
        "3. Identify which specific platform within each brand is the most "
        "and least efficient (highest/lowest ROI).\n"
        "4. Create a visualization comparing the two brands' ROI by channel.\n\n"
        "Show all intermediate results and explain your reasoning at each step."
    )

    print(f"Running analysis agent...")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    print(f"Capturing per-event timestamps...\n")

    # Track code interpreter call count for pairing start/end
    ci_count = 0

    result = Runner.run_streamed(agent, prompt)

    async for event in result.stream_events():
        etype = getattr(event, "type", "unknown")

        if etype == "raw_response_event":
            data = event.data
            data_type = getattr(data, "type", "")

            # Code interpreter lifecycle
            if data_type == "response.code_interpreter_call.in_progress":
                ci_count += 1
                collector.record_start(
                    f"ci_{ci_count}", "code_interpreter",
                    f"[call {ci_count}] started"
                )

            elif data_type == "response.code_interpreter_call.interpreting":
                collector.record("code_interpreter_exec", f"[call {ci_count}] executing")

            elif data_type == "response.code_interpreter_call.completed":
                first_line = ""
                ci_item = getattr(data, "item", None)
                if ci_item:
                    code = getattr(ci_item, "code", "")
                    first_line = code.split("\n")[0][:60] if code else ""
                collector.record_end(
                    f"ci_{ci_count}", "code_interpreter",
                    f"[call {ci_count}] completed: {first_line}"
                )

            # Reasoning — only boundaries
            elif data_type == "response.reasoning_summary_part.added":
                collector.record("reasoning", "started")
            elif data_type == "response.reasoning_summary_part.done":
                part = getattr(data, "part", None)
                text = getattr(part, "text", "")[:60] if part else ""
                collector.record("reasoning_done", f"{text}")

        elif etype == "run_item_stream_event":
            name = event.name

            if name == "reasoning_item_created":
                raw = getattr(event.item, "raw_item", None)
                summaries = getattr(raw, "summary", []) if raw else []
                if summaries:
                    text = summaries[0].text[:80] if summaries else ""
                    collector.record("reasoning_item", text)

            elif name == "message_output_created":
                raw = getattr(event.item, "raw_item", None)
                text = ""
                if raw:
                    for c in getattr(raw, "content", []):
                        if hasattr(c, "text"):
                            text = c.text[:60]
                            break
                collector.record("message_output", text)

    # Results
    final = getattr(result, "final_output", "(no output)")
    print(f"\n{'─'*80}")
    print(f"FINAL OUTPUT (truncated):")
    print(f"{'─'*80}")
    print(str(final)[:500])
    print("..." if len(str(final)) > 500 else "")

    collector.print_transcript()

    # Token usage
    if hasattr(result, "raw_responses"):
        print("TOKEN USAGE:")
        total_in = total_out = 0
        for i, resp in enumerate(result.raw_responses):
            usage = getattr(resp, "usage", None)
            if usage:
                inp = getattr(usage, "input_tokens", 0)
                out = getattr(usage, "output_tokens", 0)
                total_in += inp
                total_out += out
                print(f"  Response {i+1}: input={inp}, output={out}, "
                      f"total={getattr(usage, 'total_tokens', '?')}")
        print(f"  TOTAL: input={total_in}, output={total_out}, combined={total_in+total_out}")

    # Cleanup uploaded files
    print("\nCleaning up uploaded files...")
    for fid in file_ids:
        await client.files.delete(fid)
        print(f"  Deleted {fid}")


if __name__ == "__main__":
    asyncio.run(main())
