"""Agent pipeline tests with a deterministic fake LLM (no network).

Exercises: planning, parallel extraction, addendum reconciliation, validator rejection + retry loop,
null-on-failure guardrail, Q&A with citations, trace file and output persistence.
"""
from __future__ import annotations

import asyncio
import json
import re

import pytest

from rfp_intel.agents.orchestrator import Orchestrator
from rfp_intel.agents.validator_agent import quote_in_text
from rfp_intel.config import settings
from rfp_intel.llm.client import GroqLLM, LLMResponse, LLMUsage
from rfp_intel.llm.tracing import Tracer


class FakeLLM(GroqLLM):
    """Returns canned JSON that depends on which agent is calling."""

    def __init__(self):
        super().__init__(api_key="fake")
        self.usage = LLMUsage()
        self.calls: list[str] = []

    def _reply(self, messages):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        self.calls.append(system[:40])
        if "extraction specialist" in system:
            field = re.search(r"Field: (.+)", user).group(1).strip()
            evidence_ids = re.findall(r"\[(E\d+)\] file=([^|]+)\| page=(\d+)", user)
            if field == "Due Date":
                for eid, f, p in evidence_ids:
                    if "Addendum 2" in f:
                        return {"value": "2024-07-09 14:00 CST", "evidence_ids": [eid],
                                "quotes": {eid: "The new due date for this RFP will be July 9, 2024 at 2:00 PM CST"},
                                "confidence": 0.95, "notes": "Extended by Addendum 2"}
                return {"value": "2024-06-27 14:00 CST", "evidence_ids": [evidence_ids[0][0]], "quotes": {}, "confidence": 0.7, "notes": ""}
            if field == "Bid Bond Requirement":
                if "rejected by the validator" in user:
                    return {"value": None, "evidence_ids": [], "quotes": {}, "confidence": 0.8, "notes": "Not found in documents"}
                # hallucinated value with a fake quote -> validator must reject it
                return {"value": "5% bid bond", "evidence_ids": [evidence_ids[0][0]],
                        "quotes": {evidence_ids[0][0]: "a five percent bid bond is mandatory for all offerors"}, "confidence": 0.9, "notes": ""}
            if field == "Bid Number":
                for eid, f, p in evidence_ids:
                    if "JA-207652" in user:
                        return {"value": "JA-207652", "evidence_ids": [eid], "quotes": {eid: "JA-207652"}, "confidence": 0.9, "notes": ""}
            if field == "Bid Summary":
                return {"value": "Dallas ISD seeks student and staff computing devices.", "evidence_ids": [evidence_ids[0][0]],
                        "quotes": {}, "confidence": 0.8, "notes": "generated"}
            return {"value": None, "evidence_ids": [], "quotes": {}, "confidence": 0.6, "notes": "Not found in documents"}
        if "Addendum Reconciliation" in system:
            m = re.search(r"\[(E\d+)\] file=Addendum 2[^\n]*", user)
            if m and '"Due Date": "2024-06-27' in user:
                return {"changes": [{"field": "Due Date", "addendum_number": 2, "evidence_id": m.group(1),
                                     "quote": "The new due date for this RFP will be July 9, 2024 at 2:00 PM CST",
                                     "new_value": "2024-07-09 14:00 CST", "description": "Due date extended to July 9, 2024"}]}
            return {"changes": []}
        if "strict validator" in system:
            payload = json.loads(user)
            return {"results": {k: {"supported": True, "reason": "ok"} for k in payload}}
        if "relevance judge" in system:
            n = len(re.findall(r"^\[(\d+)\]", user, re.M))
            return {"scores": {str(i): 5 for i in range(n)}}
        if "Q&A agent" in system:
            return {"answer": "The due date is July 9, 2024 at 2:00 PM CST [E1].", "evidence_ids": ["E1"], "confidence": 0.9}
        return {"ok": True}

    def complete(self, messages, **kw):
        data = self._reply(messages)
        self.usage.add("fake", None, 1.0)
        return LLMResponse(text=json.dumps(data), model="fake", prompt_tokens=10, completion_tokens=5, latency_ms=1.0)

    async def acomplete(self, messages, **kw):
        return self.complete(messages, **kw)


@pytest.fixture
def orch(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "outputs_dir", tmp_path / "out")
    monkeypatch.setattr(settings, "runs_dir", tmp_path / "runs")
    return Orchestrator(engine, llm=FakeLLM(), tracer=Tracer(runs_dir=tmp_path / "runs"))


def test_quote_matching():
    assert quote_in_text("July 9, 2024 at 2:00 PM", "The new due date ... will be July 9, 2024 at 2:00 PM CST.")
    assert not quote_in_text("five percent bid bond is mandatory", "Insurance and/or bond requirements are enumerated elsewhere.")


def test_extract_bid_end_to_end(orch, tmp_path):
    record = asyncio.run(orch.extract_bid(bid_id="Bid1"))
    assert set(record.fields) >= {"Bid Number", "Due Date", "Bid Bond Requirement", "Bid Summary"}
    # addendum override applied with citation to the addendum
    due = record.fields["Due Date"]
    assert due.value == "2024-07-09 14:00 CST"
    assert any("Addendum 2" in s.file for s in due.sources)
    # hallucinated bond value was rejected, retried, and ended as null with a reason
    bond = record.fields["Bid Bond Requirement"]
    assert bond.value is None and bond.notes and "not found" in bond.notes.lower()
    assert orch.calls_for("extract:Bid Bond Requirement") >= 2
    # guardrail: every non-null value has a citation
    for name, fv in record.fields.items():
        if fv.value is not None:
            assert fv.sources, name
    assert record.validation.failed == 0
    # outputs + trace written
    assert (tmp_path / "out" / "Bid1.json").exists()
    assert (tmp_path / "out" / "Bid1_summary.md").exists()
    trace = [json.loads(l) for l in orch.tracer.path.read_text().splitlines()]
    agents = {t["agent"] for t in trace}
    assert {"orchestrator", "retrieval_agent", "validator_agent", "reconciliation_agent"} <= agents
    assert any(t["step"] == "feedback_loop" for t in trace)


def test_ask_detects_bid_and_cites(orch):
    ans = asyncio.run(orch.ask("What is the submission deadline for Bid1 after all addendums?"))
    assert ans.bid_ids == ["Bid1"]
    assert "July 9, 2024" in ans.answer
    assert ans.citations and ans.citations[0].bid_id == "Bid1"
    ans2 = asyncio.run(orch.ask("Which affidavits are required for the Dell laptop bid?"))
    assert ans2.bid_ids == ["Bid2"]


def test_detect_bids_fallback_to_all(orch):
    assert orch.qa.detect_bids("compare the warranty requirements of both bids", ["Bid1", "Bid2"]) == []


# helper used above
def _calls_for(self: Orchestrator, step: str) -> int:
    return sum(1 for r in self.tracer.records if r["step"] == step)


Orchestrator.calls_for = _calls_for  # type: ignore[attr-defined]
