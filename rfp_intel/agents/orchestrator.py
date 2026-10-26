"""Orchestrator / Planner.

Receives the user goal ("extract bid" or a question), builds an execution plan, routes sub-tasks to the agents,
runs field groups concurrently, drives the validator feedback loop (bounded retries) and decides when the job
is done. All intermediate data lives in a single ``SharedState`` object.

Framework choice: hand-rolled asyncio orchestration with Pydantic messages (see README for the reasoning).
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from rfp_intel.agents.extraction_agent import ExtractionAgent
from rfp_intel.agents.fields import FIELDS, GROUPS, FIELD_BY_NAME, FieldSpec, fields_in_group
from rfp_intel.agents.ingestion_agent import IngestionAgent
from rfp_intel.agents.qa_agent import QAAgent
from rfp_intel.agents.reconciliation_agent import ReconciliationAgent
from rfp_intel.agents.retrieval_agent import RetrievalAgent
from rfp_intel.agents.validator_agent import ValidatorAgent
from rfp_intel.config import settings
from rfp_intel.llm.client import GroqLLM, get_llm
from rfp_intel.llm.tracing import Tracer
from rfp_intel.models import Answer, BidRecord, ExecutionPlan, FieldValue, PlanStep, SharedState, ValidationReport
from rfp_intel.search.engine import SearchEngine

log = logging.getLogger(__name__)


class Orchestrator:
    name = "orchestrator"

    def __init__(self, engine: SearchEngine | None = None, llm: GroqLLM | None = None, tracer: Tracer | None = None):
        self.engine = engine or SearchEngine()
        self.llm = llm or get_llm()
        self.tracer = tracer or Tracer()
        args = (self.engine, self.llm, self.tracer)
        self.ingestion = IngestionAgent(*args)
        self.retrieval = RetrievalAgent(*args)
        self.extractors = {g: ExtractionAgent(*args, group=g) for g in GROUPS}
        self.reconciliation = ReconciliationAgent(*args)
        self.validator = ValidatorAgent(*args)
        self.qa = QAAgent(*args)
        self._sem = asyncio.Semaphore(settings.agent_concurrency)

    # ------------------------------------------------------------------ planning
    def _plan(self, goal: str, mode: str, bid_ids: list[str]) -> ExecutionPlan:
        if mode == "extract":
            steps = [PlanStep(agent="ingestion_agent", task="parse + index bid folder (skip if indexed)")]
            steps += [PlanStep(agent=f"extraction_agent[{GROUPS[g]}]", task=f"extract {len(fields_in_group(g))} fields (parallel, via retrieval_agent)")
                      for g in GROUPS if g != "summary"]
            steps += [
                PlanStep(agent="reconciliation_agent", task="apply addendum overrides, build change log"),
                PlanStep(agent="validator_agent", task=f"check citations/format/consistency; retry failures (max {settings.max_field_retries})"),
                PlanStep(agent="extraction_agent[Summary]", task="generate Bid Summary from evidence + extracted fields"),
                PlanStep(agent="qa_agent", task="write final JSON + human-readable report"),
            ]
        elif mode == "compare":
            steps = [PlanStep(agent="orchestrator", task="load or produce structured records for each bid"),
                     PlanStep(agent="qa_agent", task="side-by-side comparison")]
        else:
            steps = [PlanStep(agent="retrieval_agent", task="rewrite/expand question, hybrid search + rerank per bid"),
                     PlanStep(agent="qa_agent", task="cited answer or 'Not found in documents'")]
        plan = ExecutionPlan(goal=goal, mode=mode, steps=steps)  # type: ignore[arg-type]
        self.tracer.log(self.name, "plan", input={"goal": goal, "bid_ids": bid_ids}, output=plan.model_dump())
        return plan

    # ------------------------------------------------------------------ extraction mode
    async def _extract_field(self, state: SharedState, bid_id: str, spec: FieldSpec, feedback: str | None = None,
                             attempt: int = 1, context: dict[str, Any] | None = None) -> None:
        async with self._sem:
            try:
                evidence = await self.retrieval.aevidence_for_field(bid_id, spec, feedback=feedback)
                state.evidence[spec.name] = evidence
                result = await self.extractors[spec.group].extract(bid_id, spec, evidence, feedback=feedback,
                                                                   context=context, attempt=attempt)
                state.draft_fields[spec.name] = result.result
            except Exception as exc:  # one field must never crash the run
                log.exception("Field %s failed: %s", spec.name, exc)
                state.errors.append(f"{spec.name}: {exc}")
                state.draft_fields[spec.name] = FieldValue(value=None, confidence=0.0, notes=f"Extraction error: {exc}")

    async def extract_bid(self, folder: str | Path | None = None, bid_id: str | None = None, force_index: bool = False) -> BidRecord:
        if folder is None and bid_id is None:
            raise ValueError("Provide a bid folder or a bid_id that is already indexed")
        bid_id = bid_id or Path(folder).name
        state = SharedState(run_id=self.tracer.run_id, goal=f"extract bid {bid_id}", mode="extract", bid_ids=[bid_id])
        state.plan = self._plan(state.goal, "extract", [bid_id])

        # 1. ingestion
        if folder is not None:
            self.ingestion.run(folder, bid_id, force=force_index)
        elif bid_id not in self.engine.list_bids():
            raise ValueError(f"{bid_id} is not indexed and no folder was given")

        # 2. parallel extraction of the three specialist groups (summary comes last)
        specs = [f for f in FIELDS if f.group != "summary"]
        await asyncio.gather(*(self._extract_field(state, bid_id, s) for s in specs))

        # 3. addendum reconciliation
        state.draft_fields, state.addendum_changes = await self.reconciliation.reconcile(bid_id, state.draft_fields)
        reconciled = {c.field for c in state.addendum_changes if c.field in state.draft_fields}

        # 4. validation + feedback loop
        report = await self.validator.validate(bid_id, state.draft_fields, skip_critic_for=reconciled)
        state.validation = report
        for attempt in range(1, settings.max_field_retries + 1):
            retry = [f for f in report.failed_fields if f in FIELD_BY_NAME and f not in reconciled]
            if not retry:
                break
            self.tracer.log(self.name, "feedback_loop", input={"attempt": attempt, "fields": retry},
                            output={"issues": [i.model_dump() for i in report.issues if i.field in retry]})
            tasks = []
            for name in retry:
                state.retries[name] = state.retries.get(name, 0) + 1
                fb = "; ".join(i.message for i in report.issues if i.field == name and i.severity == "error")
                tasks.append(self._extract_field(state, bid_id, FIELD_BY_NAME[name], feedback=fb, attempt=attempt + 1))
            await asyncio.gather(*tasks)
            report = await self.validator.validate(bid_id, state.draft_fields, skip_critic_for=reconciled)
            state.validation = report
        # fields still failing after the retry budget are nulled: never ship an unsupported value
        for name in report.failed_fields:
            fv = state.draft_fields.get(name)
            if fv and fv.value is not None:
                reasons = "; ".join(i.message for i in report.issues if i.field == name)
                state.draft_fields[name] = FieldValue(value=None, sources=[], confidence=0.2,
                                                      notes=f"Not found in documents (validation failed after retries: {reasons})")
        final_report = await self.validator.validate(bid_id, state.draft_fields, skip_critic_for=set(state.draft_fields))

        # 5. summary (needs the other fields as context)
        summary_spec = FIELD_BY_NAME["Bid Summary"]
        ctx = {k: v.value for k, v in state.draft_fields.items() if v.value is not None}
        await self._extract_field(state, bid_id, summary_spec, context=ctx)
        if state.draft_fields[summary_spec.name].value is None:
            final_report.not_found += 1
        else:
            final_report.passed += 1

        # 6. final record
        ordered = {f.name: state.draft_fields.get(f.name, FieldValue(notes="Not found in documents")) for f in FIELDS}
        record = BidRecord(bid_id=bid_id, fields=ordered, addendum_changes=state.addendum_changes,
                           validation=final_report, run_id=self.tracer.run_id)
        state.final_output = record.model_dump()
        self._persist(record)
        return record

    def _persist(self, record: BidRecord) -> None:
        settings.ensure_dirs()
        out = settings.outputs_dir / f"{record.bid_id}.json"
        out.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        (settings.outputs_dir / f"{record.bid_id}_summary.md").write_text(self.qa.render_summary(record), encoding="utf-8")
        self.tracer.write_summary({"bid_id": record.bid_id, "output": str(out), "validation": record.validation.model_dump(),
                                   "llm_usage": self.llm.usage.as_dict(), "trace": str(self.tracer.path)})
        self.tracer.log(self.name, "done", output={"output_file": str(out), "validation": record.validation.model_dump(),
                                                   "llm_usage": self.llm.usage.as_dict()})

    # ------------------------------------------------------------------ question answering mode
    async def ask(self, question: str, bid_ids: list[str] | None = None, top_k: int = 12) -> Answer:
        known = self.engine.list_bids()
        targets = bid_ids or self.qa.detect_bids(question, known)
        if not targets:
            targets = known  # search everything, grouped per bid
        self._plan(question, "qa", targets)
        evidence = self.retrieval.evidence_for_question(question, targets, top_k=top_k)
        answer = await self.qa.answer(question, evidence, targets)
        answer.run_id = self.tracer.run_id
        self.tracer.log(self.name, "done", output={"answer": answer.answer[:300], "citations": len(answer.citations),
                                                   "llm_usage": self.llm.usage.as_dict()})
        return answer

    # ------------------------------------------------------------------ comparison mode (bonus)
    async def compare(self, bid_ids: list[str] | None = None) -> str:
        bid_ids = bid_ids or self.engine.list_bids()
        self._plan(f"compare {bid_ids}", "compare", bid_ids)
        records: list[BidRecord] = []
        for b in bid_ids:
            p = settings.outputs_dir / f"{b}.json"
            if p.exists():
                records.append(BidRecord.model_validate_json(p.read_text(encoding="utf-8")))
            else:
                records.append(await self.extract_bid(bid_id=b))
        md = await self.qa.compare(records)
        (settings.outputs_dir / "comparison.md").write_text(md, encoding="utf-8")
        return md


# ---------------------------------------------------------------------- sync helpers for CLI/API
def run_extract(folder: str | Path | None = None, bid_id: str | None = None, force_index: bool = False) -> BidRecord:
    return asyncio.run(Orchestrator().extract_bid(folder, bid_id, force_index))


def run_ask(question: str, bid_ids: list[str] | None = None) -> Answer:
    return asyncio.run(Orchestrator().ask(question, bid_ids))


def run_compare(bid_ids: list[str] | None = None) -> str:
    return asyncio.run(Orchestrator().compare(bid_ids))