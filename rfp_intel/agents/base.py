"""Shared base for all agents: access to the search tool, the LLM and the tracer."""
from __future__ import annotations

from rfp_intel.llm.client import GroqLLM
from rfp_intel.llm.tracing import Tracer
from rfp_intel.models import SearchResult
from rfp_intel.search.engine import SearchEngine


class Agent:
    name = "agent"

    def __init__(self, engine: SearchEngine, llm: GroqLLM, tracer: Tracer):
        self.engine = engine
        self.llm = llm
        self.tracer = tracer


def format_evidence(evidence: list[SearchResult], max_chars: int = 1400) -> tuple[str, dict[str, SearchResult]]:
    """Render evidence passages as [E1] ... blocks and return the id -> result map."""
    lines: list[str] = []
    by_id: dict[str, SearchResult] = {}
    for i, r in enumerate(evidence, start=1):
        eid = f"E{i}"
        by_id[eid] = r
        tag = f"[{eid}] file={r.file_name} | page={r.page_number} | type={r.doc_type}"
        if r.addendum_number:
            tag += f" | addendum={r.addendum_number}"
        body = r.text
        # strip the chunk header line (already summarised in the tag)
        if body.startswith("[") and "\n" in body:
            body = body.split("\n", 1)[1]
        lines.append(f"{tag}\n{body[:max_chars]}")
    return "\n\n".join(lines), by_id
