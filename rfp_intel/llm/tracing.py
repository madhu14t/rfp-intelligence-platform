"""Structured observability: every agent step is appended to runs/<run_id>/trace.jsonl.

Each record: {ts, run_id, agent, step, input, output, tool_calls, tokens, latency_ms, status, error}
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rfp_intel.config import settings

log = logging.getLogger(__name__)


def _trim(obj: Any, limit: int = 1500) -> Any:
    """Keep traces readable: truncate long strings, recurse into containers."""
    if isinstance(obj, str):
        return obj if len(obj) <= limit else obj[:limit] + f"... [+{len(obj) - limit} chars]"
    if isinstance(obj, dict):
        return {k: _trim(v, limit) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_trim(v, limit) for v in obj[:50]]
    if hasattr(obj, "model_dump"):
        return _trim(obj.model_dump(), limit)
    return obj


class Tracer:
    def __init__(self, run_id: str | None = None, runs_dir: Path | None = None):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.dir = Path(runs_dir or settings.runs_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "trace.jsonl"
        self.records: list[dict[str, Any]] = []

    def log(self, agent: str, step: str, *, input: Any = None, output: Any = None, tool_calls: Any = None,
            tokens: int | None = None, latency_ms: float | None = None, status: str = "ok", error: str | None = None) -> None:
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": self.run_id,
            "agent": agent,
            "step": step,
            "status": status,
            "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
            "tokens": tokens,
            "input": _trim(input),
            "tool_calls": _trim(tool_calls),
            "output": _trim(output),
            "error": error,
        }
        self.records.append(rec)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        log.info("[%s] %s.%s %s (%.0f ms)", self.run_id, agent, step, status, latency_ms or 0)

    @contextmanager
    def step(self, agent: str, step: str, input: Any = None):
        """Context manager that records latency/status automatically. Set ctx['output'] / ctx['tool_calls']."""
        ctx: dict[str, Any] = {"output": None, "tool_calls": None, "tokens": None}
        t0 = time.perf_counter()
        try:
            yield ctx
        except Exception as exc:
            self.log(agent, step, input=input, output=ctx["output"], tool_calls=ctx["tool_calls"], tokens=ctx["tokens"],
                     latency_ms=(time.perf_counter() - t0) * 1000, status="error", error=repr(exc))
            raise
        self.log(agent, step, input=input, output=ctx["output"], tool_calls=ctx["tool_calls"], tokens=ctx["tokens"],
                 latency_ms=(time.perf_counter() - t0) * 1000)

    def write_summary(self, summary: dict[str, Any]) -> Path:
        p = self.dir / "summary.json"
        p.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return p
