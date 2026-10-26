# """Q&A / Report Agent: answers free-form questions with citations, builds the final JSON record and a
# human-readable summary, and (bonus) produces side-by-side bid comparisons."""
# from __future__ import annotations

# import json
# import re
# from typing import Any

# from rfp_intel.agents.base import Agent, format_evidence
# from rfp_intel.llm.client import LLMError
# from rfp_intel.models import Answer, BidRecord, Citation, SearchResult

# _QA_SYSTEM = """You are the Q&A agent of an RFP intelligence system. Answer the user's question using ONLY the evidence
# passages. Rules:
# - If the evidence does not answer the question, reply exactly "Not found in documents" (you may add what *was* found).
# - Addendums override the base RFP: if an addendum changes a value (e.g. extends a due date), report the addendum value
#   and mention the original.
# - When the question covers several bids, answer per bid.
# - Cite every factual statement with the passage ids, e.g. "...due July 9, 2024 at 2:00 PM CST [E2]".
# Return strictly JSON: {"answer": "<markdown answer with [E#] citations>", "evidence_ids": ["E2", ...], "confidence": 0.0-1.0}"""

# _COMPARE_SYSTEM = """You compare procurement bids. Using the structured records given (already extracted and cited), produce
# a concise side-by-side markdown table with one row per field and one column per bid, then 3-5 bullet points on the
# key differences (deadlines, scope, warranty, required documents, submission method). Do not invent values: if a
# record says null / Not found, write "Not found". Return markdown only."""


# class QAAgent(Agent):
#     name = "qa_agent"

#     # ------------------------------------------------------------------ question answering
#     def detect_bids(self, question: str, known_bids: list[str]) -> list[str]:
#         q = question.lower()
#         found = [b for b in known_bids if re.search(r"\b" + re.escape(b.lower()).replace(r"bid", r"bid\s?") + r"\b", q)]
#         if found:
#             return found
#         if re.search(r"\b(both|each|all|every|compare|comparison|across|versus|vs)\b", q):
#             return []  # multi-bid question -> search every bid
#         # match on distinctive words from each bid's file names (e.g. "dell laptop bid", "Dallas ISD")
#         stop = {"final", "bid", "bids", "information", "bidnet", "direct", "html", "pdf", "addendum", "affidavit", "specs",
#                 "with", "warranty", "extended", "request", "proposal", "contract", "mercury", "sourcing"}
#         for b in known_bids:
#             words: set[str] = set()
#             for d in self.engine.store.documents_for(b):
#                 words |= {w for w in re.findall(r"[a-z]{4,}", d["file_name"].lower()) if w not in stop}
#             if any(re.search(r"\b" + w + r"s?\b", q) for w in words):
#                 found.append(b)
#         return found

#     async def answer(self, question: str, evidence: list[SearchResult], bid_ids: list[str]) -> Answer:
#         ev_text, by_id = format_evidence(evidence, max_chars=1500)
#         with self.tracer.step(self.name, "answer", input={"question": question, "bid_ids": bid_ids, "evidence": list(by_id)}) as ctx:
#             if not evidence:
#                 ans = Answer(question=question, answer="Not found in documents (no relevant passages retrieved).",
#                              citations=[], confidence=0.0, bid_ids=bid_ids)
#                 ctx["output"] = ans.model_dump()
#                 return ans
#             try:
#                 resp = await self.llm.acomplete(
#                     [{"role": "system", "content": _QA_SYSTEM},
#                      {"role": "user", "content": f"Question: {question}\n\nEvidence passages:\n{ev_text}\n\nReturn the JSON now."}],
#                     json_mode=True, max_tokens=900,
#                 )
#                 ctx["tokens"] = resp.prompt_tokens + resp.completion_tokens
#                 data = resp.json()
#             except LLMError as exc:
#                 ans = Answer(question=question, answer=f"Error while answering: {exc}", citations=[], confidence=0.0, bid_ids=bid_ids)
#                 ctx["output"] = ans.model_dump()
#                 return ans
#             text = str(data.get("answer", "")).strip() or "Not found in documents"
#             cits: list[Citation] = []
#             ids = [str(e) for e in (data.get("evidence_ids") or [])]
#             # also pick up ids mentioned inline in the answer
#             ids += re.findall(r"\[(E\d+)\]", text)
#             seen: set[str] = set()
#             for eid in ids:
#                 r = by_id.get(eid)
#                 if r and r.chunk_id not in seen:
#                     seen.add(r.chunk_id)
#                     cits.append(r.citation())
#             # make the inline markers human readable: [E2] -> [2: file p.3]
#             def _sub(m: re.Match) -> str:
#                 r = by_id.get(m.group(1))
#                 return f"[{r.file_name}, p.{r.page_number}]" if r else m.group(0)
#             text = re.sub(r"\[(E\d+)\]", _sub, text)
#             try:
#                 conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
#             except (TypeError, ValueError):
#                 conf = 0.0
#             if not cits and "not found" not in text.lower():
#                 conf = min(conf, 0.3)
#                 text += "\n\n(No citation could be attached to this answer; treat with caution.)"
#             ans = Answer(question=question, answer=text, citations=cits, confidence=conf, bid_ids=bid_ids)
#             ctx["output"] = ans.model_dump()
#             return ans

#     # ------------------------------------------------------------------ reports
#     def render_summary(self, record: BidRecord) -> str:
#         lines = [f"# Bid record: {record.bid_id}", ""]
#         for name, fv in record.fields.items():
#             val = fv.value
#             if isinstance(val, (list, dict)):
#                 val = json.dumps(val, ensure_ascii=False)
#             src = "; ".join(f"{s.file} p.{s.page}" for s in fv.sources[:3]) or "-"
#             lines.append(f"- **{name}**: {val if val is not None else 'null'}  \n  confidence {fv.confidence:.2f} | sources: {src}"
#                          + (f" | notes: {fv.notes}" if fv.notes else ""))
#         if record.addendum_changes:
#             lines += ["", "## Addendum changes"]
#             for c in record.addendum_changes:
#                 lines.append(f"- Addendum {c.addendum_number} ({c.file} p.{c.page}) - {c.field}: {c.description}")
#         v = record.validation
#         lines += ["", f"## Validation: passed {v.passed}, failed {v.failed}, not found {v.not_found}"]
#         for i in v.issues:
#             lines.append(f"- [{i.severity}] {i.field}: {i.message}")
#         return "\n".join(lines)

#     async def compare(self, records: list[BidRecord]) -> str:
#         with self.tracer.step(self.name, "compare", input={"bids": [r.bid_id for r in records]}) as ctx:
#             compact: dict[str, dict[str, Any]] = {}
#             for r in records:
#                 compact[r.bid_id] = {k: (v.value if v.value is not None else "Not found") for k, v in r.fields.items()
#                                      if k != "Bid Summary"}
#             try:
#                 resp = await self.llm.acomplete(
#                     [{"role": "system", "content": _COMPARE_SYSTEM},
#                      {"role": "user", "content": json.dumps(compact, indent=1, default=str)}],
#                     max_tokens=1500,
#                 )
#                 ctx["tokens"] = resp.prompt_tokens + resp.completion_tokens
#                 ctx["output"] = resp.text[:500]
#                 return resp.text
#             except LLMError as exc:
#                 ctx["output"] = {"error": str(exc)}
#                 return f"Comparison failed: {exc}"


"""Q&A / Report Agent: answers free-form questions with citations, builds the final JSON record and a
human-readable summary, and (bonus) produces side-by-side bid comparisons."""
from __future__ import annotations

import json
import re
from typing import Any

from rfp_intel.agents.base import Agent, format_evidence
from rfp_intel.llm.client import LLMError
from rfp_intel.models import Answer, BidRecord, Citation, SearchResult

_QA_SYSTEM = """You are the Q&A agent of an RFP intelligence system. Answer the user's question using ONLY the evidence
passages. Rules:
- Reply exactly "Not found in documents" ONLY when the evidence contains nothing relevant. If it answers the question
  partially, give the partial answer and state precisely which part is missing; never start a useful answer with "Not found".
- Addendums override the base RFP: if an addendum changes a value (e.g. extends a due date), report the addendum value
  and mention the original.
- When the question covers several bids, answer per bid.
- Cite every factual statement with the passage ids, e.g. "...due July 9, 2024 at 2:00 PM CST [E2]".
Return strictly JSON: {"answer": "<markdown answer with [E#] citations>", "evidence_ids": ["E2", ...], "confidence": 0.0-1.0}"""

_COMPARE_SYSTEM = """You compare procurement bids. Using the structured records given (already extracted and cited), produce
a concise side-by-side markdown table with one row per field and one column per bid, then 3-5 bullet points on the
key differences (deadlines, scope, warranty, required documents, submission method). Do not invent values: if a
record says null / Not found, write "Not found". Return markdown only."""


class QAAgent(Agent):
    name = "qa_agent"

    # ------------------------------------------------------------------ question answering
    def detect_bids(self, question: str, known_bids: list[str]) -> list[str]:
        q = question.lower()
        found = [b for b in known_bids if re.search(r"\b" + re.escape(b.lower()).replace(r"bid", r"bid\s?") + r"\b", q)]
        if found:
            return found
        if re.search(r"\b(both|each|all|every|compare|comparison|across|versus|vs)\b", q):
            return []  # multi-bid question -> search every bid
        # match on distinctive words from each bid's file names (e.g. "dell laptop bid", "Dallas ISD")
        stop = {"final", "bid", "bids", "information", "bidnet", "direct", "html", "pdf", "addendum", "affidavit", "specs",
                "with", "warranty", "extended", "request", "proposal", "contract", "mercury", "sourcing"}
        for b in known_bids:
            words: set[str] = set()
            for d in self.engine.store.documents_for(b):
                words |= {w for w in re.findall(r"[a-z]{4,}", d["file_name"].lower()) if w not in stop}
            if any(re.search(r"\b" + w + r"s?\b", q) for w in words):
                found.append(b)
        return found

    async def answer(self, question: str, evidence: list[SearchResult], bid_ids: list[str]) -> Answer:
        ev_text, by_id = format_evidence(evidence, max_chars=1500)
        with self.tracer.step(self.name, "answer", input={"question": question, "bid_ids": bid_ids, "evidence": list(by_id)}) as ctx:
            if not evidence:
                ans = Answer(question=question, answer="Not found in documents (no relevant passages retrieved).",
                             citations=[], confidence=0.0, bid_ids=bid_ids)
                ctx["output"] = ans.model_dump()
                return ans
            try:
                resp = await self.llm.acomplete(
                    [{"role": "system", "content": _QA_SYSTEM},
                     {"role": "user", "content": f"Question: {question}\n\nEvidence passages:\n{ev_text}\n\nReturn the JSON now."}],
                    json_mode=True, max_tokens=900,
                )
                ctx["tokens"] = resp.prompt_tokens + resp.completion_tokens
                data = resp.json()
            except LLMError as exc:
                ans = Answer(question=question, answer=f"Error while answering: {exc}", citations=[], confidence=0.0, bid_ids=bid_ids)
                ctx["output"] = ans.model_dump()
                return ans
            text = str(data.get("answer", "")).strip() or "Not found in documents"
            cits: list[Citation] = []
            ids = [str(e) for e in (data.get("evidence_ids") or [])]
            # also pick up ids mentioned inline in the answer
            ids += re.findall(r"E\d+", " ".join(re.findall(r"\[((?:E\d+[,\s]*)+)\]", text)))
            seen: set[str] = set()
            for eid in ids:
                r = by_id.get(eid)
                if r and r.chunk_id not in seen:
                    seen.add(r.chunk_id)
                    cits.append(r.citation())
            # make the inline markers human readable: [E2] or [E2, E3] -> [file, p.3; file, p.4]
            def _sub(m: re.Match) -> str:
                parts = []
                for eid in re.findall(r"E\d+", m.group(1)):
                    r = by_id.get(eid)
                    if r:
                        label = f"{r.file_name}, p.{r.page_number}"
                        if label not in parts:
                            parts.append(label)
                return f"[{'; '.join(parts)}]" if parts else m.group(0)
            text = re.sub(r"\[((?:E\d+[,\s]*)+)\]", _sub, text)
            try:
                conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
            except (TypeError, ValueError):
                conf = 0.0
            if not cits and "not found" not in text.lower():
                conf = min(conf, 0.3)
                text += "\n\n(No citation could be attached to this answer; treat with caution.)"
            ans = Answer(question=question, answer=text, citations=cits, confidence=conf, bid_ids=bid_ids)
            ctx["output"] = ans.model_dump()
            return ans

    # ------------------------------------------------------------------ reports
    def render_summary(self, record: BidRecord) -> str:
        lines = [f"# Bid record: {record.bid_id}", ""]
        for name, fv in record.fields.items():
            val = fv.value
            if isinstance(val, (list, dict)):
                val = json.dumps(val, ensure_ascii=False)
            src = "; ".join(f"{s.file} p.{s.page}" for s in fv.sources[:3]) or "-"
            lines.append(f"- **{name}**: {val if val is not None else 'null'}  \n  confidence {fv.confidence:.2f} | sources: {src}"
                         + (f" | notes: {fv.notes}" if fv.notes else ""))
        if record.addendum_changes:
            lines += ["", "## Addendum changes"]
            for c in record.addendum_changes:
                lines.append(f"- Addendum {c.addendum_number} ({c.file} p.{c.page}) - {c.field}: {c.description}")
        v = record.validation
        lines += ["", f"## Validation: passed {v.passed}, failed {v.failed}, not found {v.not_found}"]
        for i in v.issues:
            lines.append(f"- [{i.severity}] {i.field}: {i.message}")
        return "\n".join(lines)

    async def compare(self, records: list[BidRecord]) -> str:
        with self.tracer.step(self.name, "compare", input={"bids": [r.bid_id for r in records]}) as ctx:
            compact: dict[str, dict[str, Any]] = {}
            for r in records:
                compact[r.bid_id] = {k: (v.value if v.value is not None else "Not found") for k, v in r.fields.items()
                                     if k != "Bid Summary"}
            try:
                resp = await self.llm.acomplete(
                    [{"role": "system", "content": _COMPARE_SYSTEM},
                     {"role": "user", "content": json.dumps(compact, indent=1, default=str)}],
                    max_tokens=1500,
                )
                ctx["tokens"] = resp.prompt_tokens + resp.completion_tokens
                ctx["output"] = resp.text[:500]
                return resp.text
            except LLMError as exc:
                ctx["output"] = {"error": str(exc)}
                return f"Comparison failed: {exc}"
