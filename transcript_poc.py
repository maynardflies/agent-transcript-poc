"""
Proof-of-concept: Agent Execution Transcript with Per-Turn Timing

Uses run_streamed() to capture per-event timestamps and auto-categorizes
reasoning phases (via LLM) to show where time is spent and what's
addressable through structured data, better prompts, or schema hints.

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
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agents import Agent, Runner, ModelSettings
from agents.tool import CodeInterpreterTool
from openai import AsyncOpenAI
from openai.types.shared.reasoning import Reasoning


# ---------------------------------------------------------------------------
# 1. Reasoning categorizer (LLM-based, post-hoc)
# ---------------------------------------------------------------------------

REASONING_CATEGORIES = {
    "schema_discovery": {
        "label": "Schema / structure discovery",
        "addressable_by": "Structured data + schema descriptions in prompt",
    },
    "error_recovery": {
        "label": "Error recovery / debugging",
        "addressable_by": "Cleaner data + schema hints (model won't hit dead ends)",
    },
    "analytical_planning": {
        "label": "Analytical planning",
        "addressable_by": "Pre-built analysis templates / better prompts",
    },
    "analytical_reasoning": {
        "label": "Analytical reasoning (irreducible)",
        "addressable_by": "Not reducible — this is the actual analysis",
    },
    "output_composition": {
        "label": "Output composition",
        "addressable_by": "Not reducible — this is the deliverable",
    },
    "data_assessment": {
        "label": "Data assessment",
        "addressable_by": "Include confidence metadata with data",
    },
}


async def classify_reasoning_phases(
    client: AsyncOpenAI, phases: list[tuple[str, float]]
) -> dict[int, str]:
    """Use GPT to classify reasoning summaries into categories. Returns {index: category}."""
    categories = "\n".join(f"- {k}: {v['label']}" for k, v in REASONING_CATEGORIES.items())
    items = "\n".join(
        f"{i}. [{dur:.0f}ms] {text[:200]}"
        for i, (text, dur) in enumerate(phases)
    )

    prompt = f"""Classify each reasoning phase into exactly one category.

Categories:
{categories}

Reasoning phases:
{items}

Respond with ONLY a JSON object mapping phase index (int) to category name (string).
Example: {{"0": "schema_discovery", "1": "analytical_planning"}}"""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return {int(k): v for k, v in result.items()}


# ---------------------------------------------------------------------------
# 2. Event-based transcript — timestamps each streaming event
# ---------------------------------------------------------------------------

@dataclass
class TranscriptEvent:
    event_type: str
    timestamp: float
    wall_clock: str
    duration_ms: float | None = None
    detail: str = ""
    category: str | None = None


@dataclass
class TranscriptCollector:
    events: list[TranscriptEvent] = field(default_factory=list)
    _pending: dict[str, float] = field(default_factory=dict)
    _reasoning_phases: list[tuple[str, float]] = field(default_factory=list)
    reasoning_for_classification: list[tuple[str, float, int]] = field(default_factory=list)

    def record(self, event_type: str, detail: str = "", category: str | None = None):
        self.events.append(TranscriptEvent(
            event_type=event_type,
            timestamp=time.monotonic(),
            wall_clock=datetime.now(timezone.utc).isoformat(),
            detail=detail,
            category=category,
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

    def record_reasoning_start(self):
        self._reasoning_phases.append(("", time.monotonic()))
        self.record("reasoning", "started")

    def record_reasoning_done(self, summary_text: str):
        duration = None
        if self._reasoning_phases:
            _, start = self._reasoning_phases.pop()
            duration = round((time.monotonic() - start) * 1000, 1)

        event_idx = len(self.events)
        self.events.append(TranscriptEvent(
            event_type="reasoning_done",
            timestamp=time.monotonic(),
            wall_clock=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration,
            detail=summary_text[:70],
            category="pending",
        ))
        if duration:
            self.reasoning_for_classification.append((summary_text, duration, event_idx))

    async def classify_reasoning(self, client: AsyncOpenAI):
        """Post-hoc LLM classification of reasoning phases."""
        if not self.reasoning_for_classification:
            return
        phases = [(text, dur) for text, dur, _ in self.reasoning_for_classification]
        print("Classifying reasoning phases with LLM...")
        classifications = await classify_reasoning_phases(client, phases)
        for i, (_, _, event_idx) in enumerate(self.reasoning_for_classification):
            category = classifications.get(i, "uncategorized")
            if event_idx < len(self.events):
                self.events[event_idx].category = category

    def print_transcript(self):
        print("\n" + "=" * 80)
        print("EXECUTION TRANSCRIPT — Per-Turn Timing")
        print("=" * 80)

        if not self.events:
            print("  (no events captured)")
            return

        t0 = self.events[0].timestamp

        print(f"\n{'#':<4} {'T+':>8} {'Duration':>10} {'Event':<25} {'Category':<22} Detail")
        print(f"{'─'*4} {'─'*8} {'─'*10} {'─'*25} {'─'*22} {'─'*40}")

        for i, e in enumerate(self.events, 1):
            offset = e.timestamp - t0
            offset_str = f"{offset:.1f}s"
            dur = f"{e.duration_ms:.0f}ms" if e.duration_ms else ""
            detail = e.detail[:50] if e.detail else ""
            cat = e.category or ""
            print(f"{i:<4} {offset_str:>8} {dur:>10} {e.event_type:<25} {cat:<22} {detail}")

        # Phase summary
        print(f"\n{'─'*80}")
        print("PHASE BREAKDOWN:")

        ci_events = [e for e in self.events if e.event_type.startswith("code_interpreter")]
        ci_calls = [e for e in ci_events if "started" in e.detail]
        ci_completes = [e for e in ci_events if e.duration_ms is not None]
        for idx, (start_e, end_e) in enumerate(zip(ci_calls, ci_completes), 1):
            s = start_e.timestamp - t0
            dur = end_e.duration_ms
            print(f"  Code interpreter {idx}: {s:.1f}s  (duration: {dur:.0f}ms)")

        total = (self.events[-1].timestamp - t0)
        ci_total = sum(e.duration_ms for e in ci_completes)

        print(f"\n  Total wall clock:   {total:.1f}s")
        if ci_completes:
            print(f"  Code interpreter:   {ci_total:.0f}ms across {len(ci_completes)} call(s)")
            print(f"  Other (reasoning/streaming/overhead): {total*1000 - ci_total:.0f}ms")

        # ── Reasoning category breakdown ──
        reasoning_done = [e for e in self.events if e.event_type == "reasoning_done" and e.duration_ms]
        if reasoning_done:
            print(f"\n{'─'*80}")
            print("REASONING BREAKDOWN BY CATEGORY:")
            print(f"{'─'*80}")

            category_totals: dict[str, list[float]] = {}
            for e in reasoning_done:
                cat = e.category or "uncategorized"
                category_totals.setdefault(cat, []).append(e.duration_ms)

            total_reasoning = sum(e.duration_ms for e in reasoning_done)
            addressable = 0.0

            sorted_cats = sorted(category_totals.items(), key=lambda x: sum(x[1]), reverse=True)

            print(f"\n  {'Category':<35} {'Count':>5} {'Total':>8} {'%':>6}  Addressable by")
            print(f"  {'─'*35} {'─'*5} {'─'*8} {'─'*6}  {'─'*40}")

            for cat, durations in sorted_cats:
                cat_total = sum(durations)
                pct = (cat_total / total_reasoning * 100) if total_reasoning else 0
                config = REASONING_CATEGORIES.get(cat, {})
                label = config.get("label", cat)
                addr = config.get("addressable_by", "—")

                is_addressable = cat in ("schema_discovery", "error_recovery", "data_assessment")
                if is_addressable:
                    addressable += cat_total
                    marker = " ◀ ADDRESSABLE"
                else:
                    marker = ""

                print(f"  {label:<35} {len(durations):>5} {cat_total/1000:>7.1f}s {pct:>5.1f}%  {addr}{marker}")

            irreducible = total_reasoning - addressable

            print(f"\n  {'─'*80}")
            print(f"  Total reasoning time:    {total_reasoning/1000:.1f}s")
            print(f"  Addressable:             {addressable/1000:.1f}s ({addressable/total_reasoning*100:.0f}%)"
                  f"  ← eliminated by structured data + schema")
            print(f"  Irreducible:             {irreducible/1000:.1f}s ({irreducible/total_reasoning*100:.0f}%)"
                  f"  ← actual analytical work")

            # Overall budget
            print(f"\n{'─'*80}")
            print("FULL TIME BUDGET:")
            print(f"{'─'*80}")
            print(f"  Code interpreter:              {ci_total/1000:>7.1f}s  ({ci_total/total/10:.0f}%)")
            print(f"  Reasoning (addressable):       {addressable/1000:>7.1f}s  ({addressable/total/10:.0f}%)")
            print(f"  Reasoning (irreducible):       {irreducible/1000:>7.1f}s  ({irreducible/total/10:.0f}%)")
            msg_and_overhead = total * 1000 - ci_total - total_reasoning
            print(f"  Message output + overhead:     {msg_and_overhead/1000:>7.1f}s  ({msg_and_overhead/total/10:.0f}%)")
            print(f"  {'─'*45}")
            print(f"  TOTAL:                         {total:>7.1f}s")

            total_addressable = addressable + ci_total * 0.3
            print(f"\n  Estimated addressable total:   {total_addressable/1000:.1f}s ({total_addressable/total/10:.0f}%)")
            print(f"  (addressable reasoning + ~30% of CI for file loading)")
        print()


# ---------------------------------------------------------------------------
# 3. Upload files to OpenAI for code interpreter access
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
# 4. Run the analysis
# ---------------------------------------------------------------------------

async def main():
    client = AsyncOpenAI()
    collector = TranscriptCollector()

    # Upload data files — replace these paths with your own data files
    print("Uploading data files to OpenAI...")
    file_ids = await upload_files(client, [
        "data/brand_a.xlsx",
        "data/brand_b.xlsx",
    ])
    print(f"Uploaded {len(file_ids)} files\n")

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
                summary="detailed",
            )
        ),
    )

    # Cross-file analysis prompt — replace with a prompt relevant to your data
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

    ci_count = 0
    result = Runner.run_streamed(agent, prompt)

    async for event in result.stream_events():
        etype = getattr(event, "type", "unknown")

        if etype == "raw_response_event":
            data = event.data
            data_type = getattr(data, "type", "")

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
            elif data_type == "response.reasoning_summary_part.added":
                collector.record_reasoning_start()
            elif data_type == "response.reasoning_summary_part.done":
                part = getattr(data, "part", None)
                text = getattr(part, "text", "") if part else ""
                collector.record_reasoning_done(text)

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

    final = getattr(result, "final_output", "(no output)")
    print(f"\n{'─'*80}")
    print(f"FINAL OUTPUT (truncated):")
    print(f"{'─'*80}")
    print(str(final)[:500])
    print("..." if len(str(final)) > 500 else "")

    await collector.classify_reasoning(client)
    collector.print_transcript()

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

    print("\nCleaning up uploaded files...")
    for fid in file_ids:
        await client.files.delete(fid)
        print(f"  Deleted {fid}")


if __name__ == "__main__":
    asyncio.run(main())
