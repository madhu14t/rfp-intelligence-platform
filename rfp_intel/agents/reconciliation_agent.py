# """Addendum Reconciliation Agent: compares the base RFP values with the addendums and applies the latest valid
# change to each field (e.g. an extended due date). Also produces a human-readable change log.

# Evidence comes from the search engine filtered to doc_type=addendum (one query per addendum-sensitive field,
# results de-duplicated), so whole documents are never pasted into the prompt.
# """
# from __future__ import annotations

# import json
# from typing import Any

# from rfp_intel.agents.base import Agent, format_evidence
# from rfp_intel.agents.fields import FIELDS
# from rfp_intel.llm.client import LLMError
# from rfp_intel.models import AddendumChange, FieldSource, FieldValue, SearchFilters, SearchResult

# _SYSTEM = """You are the Addendum Reconciliation agent of an RFP intelligence system.
# You receive (a) the current extracted values of several fields, taken mostly from the base RFP / bid page, and
# (b) passages retrieved from the addendums of the same bid.

# Task: decide, for each field, whether an addendum changes it. Addendums override the base document; the HIGHEST
# addendum number wins when several change the same field. Also list other notable changes or clarifications in the
# addendums that do not map to a field (use field name "other").

# Rules: use only the passages; quote verbatim (max 200 chars); never invent changes.
# Output strictly JSON:
# {"changes": [{"field": "<field name or 'other'>", "addendum_number": <int or null>, "evidence_id": "E3",
#               "quote": "...", "new_value": <value in the same format as the current value, or null for 'other'>,
#               "description": "<one sentence>"}]}
# If nothing changes return {"changes": []}."""


# class ReconciliationAgent(Agent):
#     name = "reconciliation_agent"

#     def _addendum_evidence(self, bid_id: str, max_chunks: int = 14) -> tuple[list[SearchResult], list[dict]]:
#         seen: set[str] = set()
#         out: list[SearchResult] = []
#         calls = []
#         for spec in FIELDS:
#             if not spec.addendum_sensitive:
#                 continue
#             resp = self.engine.search(spec.queries[0], filters=SearchFilters(bid_id=bid_id, doc_type=["addendum"]),
#                                       top_k=3, extra_queries=spec.queries[1:], rerank_results=False)
#             calls.append({"tool": "search", "field": spec.name, "filter": "addendum", "hits": len(resp.results)})
#             for r in resp.results:
#                 if r.chunk_id not in seen:
#                     seen.add(r.chunk_id)
#                     out.append(r)
#         # general "what changed" probe
#         resp = self.engine.search("purpose of this addendum change extend due date revised", filters=SearchFilters(bid_id=bid_id, doc_type=["addendum"]),
#                                   top_k=4, rerank_results=False)
#         calls.append({"tool": "search", "field": "*", "filter": "addendum", "hits": len(resp.results)})
#         for r in resp.results:
#             if r.chunk_id not in seen:
#                 seen.add(r.chunk_id)
#                 out.append(r)
#         out.sort(key=lambda r: ((r.addendum_number or 0), r.page_number))
#         return out[:max_chunks], calls

#     async def reconcile(self, bid_id: str, fields: dict[str, FieldValue]) -> tuple[dict[str, FieldValue], list[AddendumChange]]:
#         has_addendum = any(d["doc_type"] == "addendum" for d in self.engine.store.documents_for(bid_id))
#         with self.tracer.step(self.name, "reconcile", input={"bid_id": bid_id, "has_addendum": has_addendum}) as ctx:
#             if not has_addendum:
#                 ctx["output"] = {"changes": [], "reason": "no addendum documents for this bid"}
#                 return fields, []
#             evidence, calls = self._addendum_evidence(bid_id)
#             ctx["tool_calls"] = calls
#             if not evidence:
#                 ctx["output"] = {"changes": [], "reason": "no addendum passages retrieved"}
#                 return fields, []
#             ev_text, by_id = format_evidence(evidence, max_chars=1600)
#             current = {name: fv.value for name, fv in fields.items()
#                        if any(s.name == name and s.addendum_sensitive for s in FIELDS)}
#             user = ("Current field values:\n" + json.dumps(current, indent=1, default=str) +
#                     f"\n\nAddendum passages:\n{ev_text}\n\nReturn the JSON now.")
#             try:
#                 resp = await self.llm.acomplete([{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
#                                                 json_mode=True, max_tokens=1200)
#                 ctx["tokens"] = resp.prompt_tokens + resp.completion_tokens
#                 data = resp.json()
#             except LLMError as exc:
#                 ctx["output"] = {"changes": [], "error": str(exc)}
#                 return fields, []

#             changes = self._apply(data, fields, by_id)
#             ctx["output"] = [c.model_dump() for c in changes]
#             return fields, changes

#     @staticmethod
#     def _apply(data: Any, fields: dict[str, FieldValue], by_id: dict[str, SearchResult]) -> list[AddendumChange]:
#         changes: list[AddendumChange] = []
#         raw = data.get("changes", []) if isinstance(data, dict) else []
#         # apply in ascending addendum order so the highest addendum wins
#         raw = sorted((c for c in raw if isinstance(c, dict)), key=lambda c: (c.get("addendum_number") or 0))
#         for c in raw:
#             r = by_id.get(str(c.get("evidence_id")))
#             if r is None:
#                 continue
#             field = str(c.get("field", "other"))
#             new_value = c.get("new_value")
#             change = AddendumChange(
#                 field=field, addendum_number=r.addendum_number if r.addendum_number is not None else c.get("addendum_number"),
#                 file=r.file_name, page=r.page_number,
#                 original_value=fields[field].value if field in fields else None, new_value=new_value,
#                 description=str(c.get("description", "")),
#             )
#             changes.append(change)
#             if field in fields and new_value not in (None, "", "null"):
#                 fv = fields[field]
#                 original = fv.value
#                 fv.value = new_value
#                 fv.sources = [FieldSource(file=r.file_name, page=r.page_number, chunk_id=r.chunk_id,
#                                           quote=str(c.get("quote"))[:300] if c.get("quote") else None)] + fv.sources
#                 fv.confidence = max(fv.confidence, 0.85)
#                 note = f"Changed by Addendum {change.addendum_number} (original value: {original})"
#                 fv.notes = note if not fv.notes else f"{note}. {fv.notes}"
#         return changes


"""Addendum Reconciliation Agent: compares the base RFP values with the addendums and applies the latest valid
change to each field (e.g. an extended due date). Also produces a human-readable change log.

Evidence comes from the search engine filtered to doc_type=addendum (one query per addendum-sensitive field,
results de-duplicated), so whole documents are never pasted into the prompt.
"""
from __future__ import annotations

import json
from typing import Any

from rfp_intel.agents.base import Agent, format_evidence
from rfp_intel.agents.fields import FIELDS
from rfp_intel.llm.client import LLMError
from rfp_intel.models import AddendumChange, FieldSource, FieldValue, SearchFilters, SearchResult

_SYSTEM = """You are the Addendum Reconciliation agent of an RFP intelligence system.
You receive (a) the current extracted values of several fields, taken mostly from the base RFP / bid page, and
(b) passages retrieved from the addendums of the same bid.

Task: decide, for each field, whether an addendum changes it. Addendums override the base document; the HIGHEST
addendum number wins when several change the same field. Also list other notable changes or clarifications in the
addendums that do not map to a field (use field name "other").

Rules: use only the passages; quote verbatim (max 200 chars); never invent changes.
Output strictly JSON:
{"changes": [{"field": "<field name or 'other'>", "addendum_number": <int or null>, "evidence_id": "E3",
              "quote": "...", "new_value": <value in the same format as the current value, or null for 'other'>,
              "description": "<one sentence>"}]}
If nothing changes return {"changes": []}."""


def _same(a: Any, b: Any) -> bool:
    norm = lambda x: json.dumps(x, sort_keys=True, default=str).lower().replace(" ", "") if not isinstance(x, str) else x.lower().strip()
    return norm(a) == norm(b)


class ReconciliationAgent(Agent):
    name = "reconciliation_agent"

    def _addendum_evidence(self, bid_id: str, max_chunks: int = 14) -> tuple[list[SearchResult], list[dict]]:
        seen: set[str] = set()
        out: list[SearchResult] = []
        calls = []
        for spec in FIELDS:
            if not spec.addendum_sensitive:
                continue
            resp = self.engine.search(spec.queries[0], filters=SearchFilters(bid_id=bid_id, doc_type=["addendum"]),
                                      top_k=3, extra_queries=spec.queries[1:], rerank_results=False)
            calls.append({"tool": "search", "field": spec.name, "filter": "addendum", "hits": len(resp.results)})
            for r in resp.results:
                if r.chunk_id not in seen:
                    seen.add(r.chunk_id)
                    out.append(r)
        # general "what changed" probe
        resp = self.engine.search("purpose of this addendum change extend due date revised", filters=SearchFilters(bid_id=bid_id, doc_type=["addendum"]),
                                  top_k=4, rerank_results=False)
        calls.append({"tool": "search", "field": "*", "filter": "addendum", "hits": len(resp.results)})
        for r in resp.results:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                out.append(r)
        out.sort(key=lambda r: ((r.addendum_number or 0), r.page_number))
        return out[:max_chunks], calls

    async def reconcile(self, bid_id: str, fields: dict[str, FieldValue]) -> tuple[dict[str, FieldValue], list[AddendumChange]]:
        has_addendum = any(d["doc_type"] == "addendum" for d in self.engine.store.documents_for(bid_id))
        with self.tracer.step(self.name, "reconcile", input={"bid_id": bid_id, "has_addendum": has_addendum}) as ctx:
            if not has_addendum:
                ctx["output"] = {"changes": [], "reason": "no addendum documents for this bid"}
                return fields, []
            evidence, calls = self._addendum_evidence(bid_id)
            ctx["tool_calls"] = calls
            if not evidence:
                ctx["output"] = {"changes": [], "reason": "no addendum passages retrieved"}
                return fields, []
            ev_text, by_id = format_evidence(evidence, max_chars=1600)
            current = {name: fv.value for name, fv in fields.items()
                       if any(s.name == name and s.addendum_sensitive for s in FIELDS)}
            user = ("Current field values:\n" + json.dumps(current, indent=1, default=str) +
                    f"\n\nAddendum passages:\n{ev_text}\n\nReturn the JSON now.")
            try:
                resp = await self.llm.acomplete([{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
                                                json_mode=True, max_tokens=1200)
                ctx["tokens"] = resp.prompt_tokens + resp.completion_tokens
                data = resp.json()
            except LLMError as exc:
                ctx["output"] = {"changes": [], "error": str(exc)}
                return fields, []

            changes = self._apply(data, fields, by_id)
            ctx["output"] = [c.model_dump() for c in changes]
            return fields, changes

    @staticmethod
    def _apply(data: Any, fields: dict[str, FieldValue], by_id: dict[str, SearchResult]) -> list[AddendumChange]:
        changes: list[AddendumChange] = []
        raw = data.get("changes", []) if isinstance(data, dict) else []
        # apply in ascending addendum order so the highest addendum wins
        raw = sorted((c for c in raw if isinstance(c, dict)), key=lambda c: (c.get("addendum_number") or 0))
        for c in raw:
            r = by_id.get(str(c.get("evidence_id")))
            if r is None:
                continue
            field = str(c.get("field", "other"))
            new_value = c.get("new_value")
            if field in fields and new_value not in (None, "", "null") and _same(new_value, fields[field].value):
                continue  # the extractor already used the addendum value: nothing changed
            change = AddendumChange(
                field=field, addendum_number=r.addendum_number if r.addendum_number is not None else c.get("addendum_number"),
                file=r.file_name, page=r.page_number,
                original_value=fields[field].value if field in fields else None, new_value=new_value,
                description=str(c.get("description", "")),
            )
            changes.append(change)
            if field in fields and new_value not in (None, "", "null"):
                fv = fields[field]
                original = fv.value
                fv.value = new_value
                fv.sources = [FieldSource(file=r.file_name, page=r.page_number, chunk_id=r.chunk_id,
                                          quote=str(c.get("quote"))[:300] if c.get("quote") else None)] + fv.sources
                fv.confidence = max(fv.confidence, 0.85)
                note = f"Changed by Addendum {change.addendum_number} (original value: {original})"
                fv.notes = note if not fv.notes else f"{note}. {fv.notes}"
        return changes