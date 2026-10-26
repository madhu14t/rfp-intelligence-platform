"""Validator / Critic Agent.

Two layers:
1. Deterministic checks (no LLM): every non-null value has >= 1 citation; the cited file/page exists in the bid;
   the quoted text really occurs in the cited chunk; datetime format; confidence range; null values carry a reason.
2. LLM critic (one call per bid, small model): given value + verbatim quotes, flags values the quotes do not support
   (hallucination / misreading check).

Fields that fail are returned with feedback so the orchestrator can re-run retrieval + extraction.
"""
from __future__ import annotations

import json
import re

from rfp_intel.agents.base import Agent
from rfp_intel.agents.fields import FIELD_BY_NAME
from rfp_intel.llm.client import LLMError
from rfp_intel.models import FieldValue, ValidationIssue, ValidationReport

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}( ?[A-Za-z]{2,5}| ?[+-]\d{2}:?\d{2})?)?$")
_NOT_FOUND = re.compile(r"not found|not specified|not stated|not present|not listed|not provided|not mentioned|not included|"
                        r"no .*(found|listed|provided|mentioned|specified|stated|present)|does not|do not|none of the|empty retrieval", re.I)

_CRITIC = """You are a strict validator in an RFP extraction system. For each field you get the extracted value and
the cited evidence: either short verbatim "quotes" or, for list-type fields, the full cited "passages".
Decide whether the evidence actually supports the value (the value may be a reformatting, e.g. a normalised date,
a short summary, or a list whose items each appear somewhere in the passages).
Flag a field only when the evidence does not mention the value, contradicts it, or the value adds facts that appear
nowhere in the evidence. Do not flag a list merely because it is longer than the quotes.
Return strictly JSON: {"results": {"<field>": {"supported": true|false, "reason": "<short>"}}}"""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def quote_in_text(quote: str, text: str) -> bool:
    q, t = _norm(quote), _norm(text)
    if not q:
        return False
    if q in t:
        return True
    toks = q.split()
    if len(toks) < 4:
        return False
    hit = sum(1 for tok in toks if tok in t)
    return hit / len(toks) >= 0.8


class ValidatorAgent(Agent):
    name = "validator_agent"

    def _deterministic(self, bid_id: str, fields: dict[str, FieldValue]) -> tuple[list[ValidationIssue], set[str]]:
        docs = {d["file_name"]: d for d in self.engine.store.documents_for(bid_id)}
        issues: list[ValidationIssue] = []
        not_found: set[str] = set()
        for name, fv in fields.items():
            spec = FIELD_BY_NAME.get(name)
            if not 0.0 <= fv.confidence <= 1.0:
                issues.append(ValidationIssue(field=name, severity="error", message="confidence out of range"))
            if fv.value in (None, "", [], {}):
                if fv.notes and _NOT_FOUND.search(fv.notes):
                    not_found.add(name)
                elif fv.notes and "error" in fv.notes.lower():
                    issues.append(ValidationIssue(field=name, severity="error", message=f"extraction error: {fv.notes}"))
                else:
                    issues.append(ValidationIssue(field=name, severity="error", message="null value without a reason"))
                continue
            if not fv.sources:
                issues.append(ValidationIssue(field=name, severity="error", message="value has no citation"))
                continue
            for s in fv.sources:
                d = docs.get(s.file)
                if d is None:
                    issues.append(ValidationIssue(field=name, severity="error", message=f"cited file not in bid: {s.file}"))
                    continue
                if d["pages"] and not 1 <= s.page <= d["pages"]:
                    issues.append(ValidationIssue(field=name, severity="error", message=f"cited page {s.page} outside {s.file}"))
                if s.quote and s.chunk_id:
                    chunk = self.engine.store.get_by_chunk_id(s.chunk_id)
                    if chunk and not quote_in_text(s.quote, chunk["text"]):
                        issues.append(ValidationIssue(field=name, severity="error",
                                                      message=f"quote not found in cited passage ({s.file} p.{s.page}): {s.quote[:80]!r}"))
            if spec and spec.format == "datetime" and isinstance(fv.value, str) and not _DATE_RE.match(fv.value.strip()):
                issues.append(ValidationIssue(field=name, severity="error",
                                              message=f"date not in 'YYYY-MM-DD HH:MM TZ' format: {fv.value!r}"))
            if spec and spec.format == "datetime_or_none" and isinstance(fv.value, str) and fv.value.strip().lower() != "none" \
                    and not re.search(r"\d{4}-\d{2}-\d{2}", fv.value):
                issues.append(ValidationIssue(field=name, severity="warning", message="pre-bid meeting value has no ISO date"))
            if spec and spec.format == "list" and isinstance(fv.value, str):
                issues.append(ValidationIssue(field=name, severity="warning", message="expected a list value"))
            if spec and spec.format == "contact" and not isinstance(fv.value, dict):
                issues.append(ValidationIssue(field=name, severity="warning", message="expected an object {name,email,phone}"))
        return issues, not_found

    async def _critic(self, fields: dict[str, FieldValue], candidates: list[str]) -> dict[str, dict]:
        payload = {}
        for name in candidates:
            fv = fields[name]
            spec = FIELD_BY_NAME.get(name)
            if spec and spec.format == "list":
                # a list can span a whole table/section: judge it against the cited passages, not 200-char quotes
                passages, seen = [], set()
                for src in fv.sources:
                    if src.chunk_id and src.chunk_id not in seen:
                        seen.add(src.chunk_id)
                        chunk = self.engine.store.get_by_chunk_id(src.chunk_id)
                        if chunk:
                            body = chunk["text"].split("\n", 1)[-1]
                            passages.append(f"[{src.file} p.{src.page}] {body[:1500]}")
                    if len(passages) >= 4:
                        break
                if passages:
                    payload[name] = {"value": fv.value, "passages": passages}
                continue
            quotes = [s.quote for s in fv.sources if s.quote]
            if quotes:
                payload[name] = {"value": fv.value, "quotes": quotes[:4]}
        if not payload or not self.llm.available:
            return {}
        try:
            resp = await self.llm.acomplete(
                [{"role": "system", "content": _CRITIC}, {"role": "user", "content": json.dumps(payload, default=str)}],
                json_mode=True, max_tokens=1200,
            )
            data = resp.json()
            return data.get("results", {}) if isinstance(data, dict) else {}
        except (LLMError, Exception):
            return {}

    async def validate(self, bid_id: str, fields: dict[str, FieldValue], skip_critic_for: set[str] | None = None) -> ValidationReport:
        with self.tracer.step(self.name, "validate", input={"bid_id": bid_id, "fields": list(fields)}) as ctx:
            issues, not_found = self._deterministic(bid_id, fields)
            failed = {i.field for i in issues if i.severity == "error"}
            candidates = [n for n, fv in fields.items() if fv.value not in (None, "", [], {}) and n not in failed
                          and n not in (skip_critic_for or set()) and FIELD_BY_NAME.get(n) and FIELD_BY_NAME[n].format != "summary"]
            critic = await self._critic(fields, candidates)
            for name, verdict in critic.items():
                if name in fields and isinstance(verdict, dict) and verdict.get("supported") is False:
                    issues.append(ValidationIssue(field=name, severity="error",
                                                  message=f"critic: quotes do not support value ({verdict.get('reason', '')})"))
                    failed.add(name)
            report = ValidationReport(
                passed=len([n for n in fields if n not in failed and n not in not_found]),
                failed=len(failed), not_found=len(not_found), issues=issues, failed_fields=sorted(failed),
            )
            ctx["output"] = report.model_dump()
            return report