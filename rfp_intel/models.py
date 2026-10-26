"""Pydantic models shared by ingestion, search and the agents.

Every agent-to-agent message in the system is one of these structured models.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

DocType = Literal["bid_page", "rfp", "addendum", "specs", "affidavit", "other"]


# --------------------------------------------------------------------------- ingestion
class PageContent(BaseModel):
    page_number: int
    text: str = ""
    tables: list[str] = Field(default_factory=list, description="Tables rendered as markdown")
    is_empty: bool = False
    warnings: list[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    bid_id: str
    file_name: str
    file_path: str
    doc_type: DocType
    addendum_number: Optional[int] = None
    document_date: Optional[str] = None
    title: Optional[str] = None
    pages: list[PageContent]
    parse_warnings: list[str] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


class Chunk(BaseModel):
    chunk_id: str
    bid_id: str
    file_name: str
    doc_type: DocType
    addendum_number: Optional[int] = None
    page_number: int
    section: Optional[str] = None
    is_table: bool = False
    text: str
    document_date: Optional[str] = None

    def metadata(self) -> dict[str, Any]:
        return {
            "bid_id": self.bid_id,
            "file_name": self.file_name,
            "doc_type": self.doc_type,
            "addendum_number": self.addendum_number,
            "page_number": self.page_number,
            "section": self.section,
            "is_table": self.is_table,
            "document_date": self.document_date,
        }


# --------------------------------------------------------------------------- search
class SearchFilters(BaseModel):
    bid_id: Optional[str] = None
    doc_type: Optional[list[DocType]] = None
    addendum_number: Optional[int] = None
    file_name: Optional[str] = None

    def matches(self, meta: dict[str, Any]) -> bool:
        if self.bid_id and meta.get("bid_id") != self.bid_id:
            return False
        if self.doc_type and meta.get("doc_type") not in self.doc_type:
            return False
        if self.addendum_number is not None and meta.get("addendum_number") != self.addendum_number:
            return False
        if self.file_name and self.file_name.lower() not in str(meta.get("file_name", "")).lower():
            return False
        return True


class Citation(BaseModel):
    file: str
    page: int
    bid_id: str
    chunk_id: Optional[str] = None


class SearchResult(BaseModel):
    chunk_id: str
    score: float
    text: str
    bid_id: str
    file_name: str
    doc_type: str
    addendum_number: Optional[int] = None
    page_number: int
    section: Optional[str] = None
    is_table: bool = False
    retrievers: list[str] = Field(default_factory=list, description="Which retrievers surfaced this chunk")

    def citation(self) -> Citation:
        return Citation(file=self.file_name, page=self.page_number, bid_id=self.bid_id, chunk_id=self.chunk_id)


class SearchResponse(BaseModel):
    query: str
    rewritten_queries: list[str] = Field(default_factory=list)
    filters: SearchFilters
    results: list[SearchResult]
    latency_ms: float
    mode: str


# --------------------------------------------------------------------------- agents
class FieldSource(BaseModel):
    file: str
    page: int
    chunk_id: Optional[str] = None
    quote: Optional[str] = Field(default=None, description="Short supporting quote from the chunk")


class FieldValue(BaseModel):
    value: Any = None
    sources: list[FieldSource] = Field(default_factory=list)
    confidence: float = 0.0
    notes: Optional[str] = None


class ExtractionRequest(BaseModel):
    bid_id: str
    field_name: str
    description: str
    queries: list[str]
    feedback: Optional[str] = Field(default=None, description="Validator feedback for a retry")
    attempt: int = 1


class ExtractionResult(BaseModel):
    field_name: str
    result: FieldValue
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class AddendumChange(BaseModel):
    field: str
    addendum_number: Optional[int] = None
    file: str
    page: int
    original_value: Any = None
    new_value: Any = None
    description: str


class ValidationIssue(BaseModel):
    field: str
    severity: Literal["error", "warning"]
    message: str


class ValidationReport(BaseModel):
    passed: int = 0
    failed: int = 0
    not_found: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)
    failed_fields: list[str] = Field(default_factory=list)


class BidRecord(BaseModel):
    bid_id: str
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))
    fields: dict[str, FieldValue]
    addendum_changes: list[AddendumChange] = Field(default_factory=list)
    validation: ValidationReport = Field(default_factory=ValidationReport)
    run_id: Optional[str] = None


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    bid_ids: list[str] = Field(default_factory=list)
    run_id: Optional[str] = None


class PlanStep(BaseModel):
    agent: str
    task: str
    params: dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    goal: str
    mode: Literal["extract", "qa", "compare"]
    steps: list[PlanStep]


class SharedState(BaseModel):
    """Shared memory object passed between agents during one run."""

    run_id: str
    goal: str
    mode: str
    bid_ids: list[str] = Field(default_factory=list)
    plan: Optional[ExecutionPlan] = None
    evidence: dict[str, list[SearchResult]] = Field(default_factory=dict, description="field/question -> evidence")
    draft_fields: dict[str, FieldValue] = Field(default_factory=dict)
    addendum_changes: list[AddendumChange] = Field(default_factory=list)
    validation: Optional[ValidationReport] = None
    retries: dict[str, int] = Field(default_factory=dict)
    final_output: Optional[dict[str, Any]] = None
    errors: list[str] = Field(default_factory=list)
