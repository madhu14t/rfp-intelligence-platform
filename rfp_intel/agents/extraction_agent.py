"""Extraction Agent(s): fill one structured field at a time using *only* retrieved evidence.

The same class serves the three specialists (Dates & Logistics, Commercial & Legal, Product & Specs) - the
group only changes the system prompt. One LLM call per field keeps prompts small (important for Groq
rate limits) and makes the validator's per-field retry loop trivial.
"""
from __future__ import annotations

import re
from typing import Any

from rfp_intel.agents.base import Agent, format_evidence
from rfp_intel.agents.fields import GROUPS, FieldSpec
from rfp_intel.llm.client import LLMError
from rfp_intel.models import ExtractionResult, FieldSource, FieldValue, SearchResult

_FORMAT_HINTS = {
    "text": "Return the value as a concise string.",
    "datetime": "Return the value as 'YYYY-MM-DD HH:MM TZ' (e.g. '2024-07-09 14:00 CST'); if no time is given use 'YYYY-MM-DD'. Put the original wording in notes.",
    "datetime_or_none": "If a meeting exists return 'YYYY-MM-DD HH:MM TZ - location - mandatory/optional'. If the documents state there is none, return the string 'None'.",
    "bool_text": "Return 'Yes - <what is required>' or 'No' or 'Not specified'. Keep it under 40 words.",
    "list": "Return a JSON array of short strings (max 12 words each, one item per document/model/part/product). Include quantities where stated. Keep the whole array under 25 items; group similar items if there are more.",
    "contact": "Return a JSON object {\"name\": ..., \"email\": ..., \"phone\": ...}. If the evidence gives a department or office (e.g. 'Procurement Services') instead of a person, use that as the name. Fill whichever of name/email/phone the evidence contains and use null only for the missing parts; never return all three as null when any contact detail is present.",
    "summary": "Write 3-6 sentences summarising what is being procured, by whom, key dates, and notable requirements.",
}

_SYSTEM = """You are the "{group}" extraction specialist in an RFP intelligence system.
You fill ONE field of a structured bid record using ONLY the evidence passages provided.

Rules:
1. Never use outside knowledge or guess. If the evidence does not contain the answer, set "value": null and
   "notes": "Not found in documents".
2. Cite evidence: "evidence_ids" lists the passage ids ([E1], [E2]...) that support the value, and "quotes" maps
   each cited id to a SHORT verbatim quote (max 200 characters) copied exactly from that passage.
3. If an addendum passage changes the value (e.g. an extended due date), prefer the addendum and say so in notes.
4. "confidence" is a number between 0 and 1 reflecting how directly the evidence states the value.
5. Output strictly JSON: {{"value": ..., "evidence_ids": [...], "quotes": {{...}}, "confidence": 0.0, "notes": "..."}}"""


def _value_strings(value: Any) -> list[str]:
    """Flatten a value (str / list / dict) into the literal strings worth looking for in the evidence."""
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(_value_strings(v))
    elif isinstance(value, list):
        for v in value:
            out.extend(_value_strings(v))
    return [x.strip() for x in out if isinstance(x, str) and len(x.strip()) >= 4]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9@]+", " ", s.lower()).strip()


def _auto_cite(value: Any, by_id: dict[str, SearchResult], max_sources: int = 3) -> list[FieldSource]:
    """Deterministic fallback citation: a passage is cited only if it literally contains one of the value's strings."""
    sources: list[FieldSource] = []
    for needle in _value_strings(value):
        n = _norm(needle)
        if len(n) < 4:
            continue
        for r in by_id.values():
            if n in _norm(r.text) and all(src.chunk_id != r.chunk_id for src in sources):
                sources.append(FieldSource(file=r.file_name, page=r.page_number, chunk_id=r.chunk_id, quote=needle[:300]))
                break
        if len(sources) >= max_sources:
            break
    return sources


class ExtractionAgent(Agent):
    def __init__(self, engine, llm, tracer, group: str):
        super().__init__(engine, llm, tracer)
        self.group = group
        self.name = f"extraction_agent[{GROUPS.get(group, group)}]"

    async def extract(self, bid_id: str, spec: FieldSpec, evidence: list[SearchResult], feedback: str | None = None,
                      context: dict[str, Any] | None = None, attempt: int = 1) -> ExtractionResult:
        ev_text, by_id = format_evidence(evidence)
        user = (
            f"Bid: {bid_id}\nField: {spec.name}\nDefinition: {spec.description}\n"
            f"Format: {_FORMAT_HINTS.get(spec.format, _FORMAT_HINTS['text'])}\n"
        )
        if context:
            user += "Already extracted fields (for context only, do not copy blindly):\n" + "\n".join(
                f"- {k}: {v}" for k, v in context.items()) + "\n"
        if feedback:
            user += f"\nPrevious attempt was rejected by the validator: {feedback}\nFix the problem.\n"
        user += f"\nEvidence passages:\n{ev_text}\n\nReturn the JSON now."

        with self.tracer.step(self.name, f"extract:{spec.name}", input={"bid_id": bid_id, "attempt": attempt,
                              "feedback": feedback, "evidence": list(by_id)}) as ctx:
            if not evidence:
                fv = FieldValue(value=None, sources=[], confidence=0.5, notes="Not found in documents (empty retrieval)")
                ctx["output"] = fv.model_dump()
                return ExtractionResult(field_name=spec.name, result=fv)
            try:
                resp = await self.llm.acomplete(
                    [{"role": "system", "content": _SYSTEM.format(group=GROUPS.get(self.group, self.group))},
                     {"role": "user", "content": user}],
                    json_mode=True, max_tokens=1800 if spec.format in ("list", "summary", "text") else 900,
                )
                ctx["tokens"] = resp.prompt_tokens + resp.completion_tokens
                data = resp.json()
            except LLMError as exc:
                fv = FieldValue(value=None, sources=[], confidence=0.0, notes=f"Extraction error: {exc}")
                ctx["output"] = fv.model_dump()
                ctx["output"]["error"] = str(exc)
                return ExtractionResult(field_name=spec.name, result=fv, evidence_chunk_ids=[r.chunk_id for r in evidence])

            fv = self._to_field_value(data, by_id)
            ctx["output"] = fv.model_dump()
            return ExtractionResult(field_name=spec.name, result=fv, evidence_chunk_ids=[r.chunk_id for r in evidence])

    @staticmethod
    def _to_field_value(data: Any, by_id: dict[str, SearchResult]) -> FieldValue:
        if isinstance(data, list):  # model returned the bare value array (usually a truncated reply)
            data = {"value": data, "evidence_ids": [], "notes": "extractor returned a bare list; citations missing"}
        if not isinstance(data, dict):
            return FieldValue(value=None, confidence=0.0, notes="Extraction error: bad JSON from extractor")
        value = data.get("value")
        # unwrap model quirks: {"value": {"value": [...]}} or [{"value": [...]}]
        for _ in range(2):
            if isinstance(value, dict) and set(value) <= {"value", "items", "list"} and value:
                value = next(iter(value.values()))
            elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict) and set(value[0]) <= {"value", "items", "list"}:
                value = next(iter(value[0].values()))
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none found", "not found", "not found in documents", "n/a"}:
            value = None
        if isinstance(value, dict) and value and all(v in (None, "", [], {}) for v in value.values()):
            value = None  # {"name": null, "email": null, "phone": null} means not found
        if isinstance(value, list) and not any(v not in (None, "", {}) for v in value):
            value = None
        quotes = data.get("quotes") or {}
        sources: list[FieldSource] = []
        for eid in data.get("evidence_ids") or []:
            r = by_id.get(str(eid))
            if r is None:
                continue
            q = quotes.get(str(eid)) if isinstance(quotes, dict) else None
            sources.append(FieldSource(file=r.file_name, page=r.page_number, chunk_id=r.chunk_id,
                                       quote=str(q)[:300] if q else None))
        if value is not None and not sources:
            sources = _auto_cite(value, by_id)  # model forgot evidence_ids: cite passages that literally contain the value
        try:
            conf = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        notes = data.get("notes")
        if value is None and not notes:
            notes = "Not found in documents"
        if value is None:
            sources = []
        return FieldValue(value=value, sources=sources, confidence=conf, notes=str(notes) if notes else None)