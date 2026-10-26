"""Run the sample questions through the multi-agent system and write a Q&A log with citations."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from rfp_intel.agents.orchestrator import Orchestrator
from rfp_intel.config import settings


def run_qa_log(questions_file: Path, out_file: Path) -> None:
    asyncio.run(_run(questions_file, out_file))  # one event loop for all questions (async HTTP client is reused)


async def _run(questions_file: Path, out_file: Path) -> None:
    questions = json.loads(Path(questions_file).read_text(encoding="utf-8"))
    orch = Orchestrator()
    lines = ["# Sample Q&A log", "", f"Run id: `{orch.tracer.run_id}`  ", f"Model: `{settings.llm_model}`", ""]
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        ans = await orch.ask(q)
        dt = time.perf_counter() - t0
        lines += [f"## Q{i}. {q}", "", ans.answer, "", "**Citations:**"]
        lines += [f"- {c.bid_id} | {c.file} | page {c.page}" for c in ans.citations] or ["- none"]
        lines += [f"", f"_confidence {ans.confidence:.2f} · {dt:.1f}s · bids searched: {', '.join(ans.bid_ids)}_", ""]
        print(f"[{i}/{len(questions)}] {q} -> {ans.answer[:90]!r}")
    lines += ["---", f"LLM usage: `{json.dumps(orch.llm.usage.as_dict())}`"]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {out_file}")