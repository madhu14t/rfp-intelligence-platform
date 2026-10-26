"""Section- and table-aware chunking for RFP documents.

Why this strategy (see README for the full justification):
* RFPs are organised in headed sections ("1.2 Terms", "SCOPE AND SPECIFICATIONS", "Section 4 - Scope of
  Work"). A chunk that starts at a heading keeps the context the LLM needs to interpret a value.
* Tables (bid information tables, spec/SKU tables, price schedules) are emitted as *whole* chunks so that a
  part number never gets separated from its description.
* Within a section we use a character window (~1800 chars ~ 400-450 tokens) with ~250 chars overlap, split on
  paragraph boundaries, which fits comfortably in the re-ranker/extractor context while staying specific
  enough for precise citations (every chunk carries its page number).
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

from rfp_intel.config import settings
from rfp_intel.models import Chunk, ParsedDocument

_NUMBERED_HEADING_RE = re.compile(r"^(?:section\s+)?\d{1,2}(?:\.\d{1,2}){0,2}\s*[–\-:]?\s+[A-Z][A-Za-z][^\n]{2,70}$", re.I)
MIN_SECTION_CHARS = 350  # sections smaller than this are merged with their neighbours


def _is_heading(line: str) -> bool:
    l = line.strip()
    if not l or len(l) > 95:
        return False
    if l.startswith("#"):
        return True
    if l.endswith((".", ";", ",", ":")):
        return False
    words = l.split()
    # ALL-CAPS headings with at least two real words ("SCOPE AND SPECIFICATIONS", "BID SUBMISSION INSTRUCTIONS")
    if l.isupper() and 2 <= len(words) <= 12 and sum(1 for w in words if re.fullmatch(r"[A-Z][A-Z'&/\-]{2,}", w)) >= 2:
        return True
    # Numbered headings ("1.2 Terms", "Section 4 – Scope of Work") but not table rows like "1 Tier 1 Small"
    if _NUMBERED_HEADING_RE.match(l) and len(words) <= 10 and not re.search(r"\d+\s+each\b", l, re.I):
        lead = re.match(r"^(?:section\s+)?(\d{1,2})", l, re.I)
        if lead and "." not in words[0] and int(lead.group(1)) > 30:
            return False  # "80 Calvert Street" is an address, not a heading
        return True
    return False


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split page text into (heading, body) pairs."""
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    buf: list[str] = []
    for line in text.split("\n"):
        if _is_heading(line):
            if buf and "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading = line.strip().lstrip("#").strip()
            buf = [line.strip()]
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))
    return sections


def merge_small_sections(sections: list[tuple[str | None, str]], size: int) -> list[tuple[str | None, str]]:
    """Greedily merge consecutive small sections so chunks are not fragmented by short form labels.

    A section boundary is still the preferred split point: we only merge while the combined body stays
    below the target chunk size.
    """
    merged: list[tuple[str | None, str]] = []
    for heading, body in sections:
        if merged:
            prev_heading, prev_body = merged[-1]
            if (len(prev_body) < MIN_SECTION_CHARS or len(body) < MIN_SECTION_CHARS) and len(prev_body) + len(body) <= size:
                merged[-1] = (prev_heading or heading, prev_body + "\n\n" + body)
                continue
        merged.append((heading, body))
    return merged


def window(text: str, size: int, overlap: int) -> Iterable[str]:
    """Yield overlapping windows, preferring paragraph then sentence boundaries."""
    text = text.strip()
    if len(text) <= size:
        if text:
            yield text
        return
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = text.rfind("\n\n", start + size // 2, end)
            if cut == -1:
                cut = text.rfind("\n", start + size // 2, end)
            if cut == -1:
                cut = text.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end >= n:
            break
        start = max(end - overlap, start + 1)


def _chunk_id(doc: ParsedDocument, page: int, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{doc.bid_id}|{doc.file_name}|{page}|{idx}|{text[:64]}".encode()).hexdigest()[:16]
    return f"{doc.bid_id}-{h}"


def chunk_document(doc: ParsedDocument, size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    size = size or settings.chunk_size_chars
    overlap = overlap or settings.chunk_overlap_chars
    chunks: list[Chunk] = []
    header = f"[{doc.bid_id} | {doc.file_name} | {doc.doc_type}" + (
        f" | Addendum {doc.addendum_number}]" if doc.addendum_number else "]"
    )
    for page in doc.pages:
        idx = 0
        for heading, body in merge_small_sections(split_sections(page.text), size):
            for piece in window(body, size, overlap):
                prefix = f"{header} p.{page.page_number}" + (f" | {heading}\n" if heading else "\n")
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(doc, page.page_number, idx, piece),
                        bid_id=doc.bid_id,
                        file_name=doc.file_name,
                        doc_type=doc.doc_type,
                        addendum_number=doc.addendum_number,
                        page_number=page.page_number,
                        section=heading,
                        is_table=False,
                        text=prefix + piece,
                        document_date=doc.document_date,
                    )
                )
                idx += 1
        for t in page.tables:
            # Tables are kept whole; very long tables are windowed but with the header row repeated.
            lines = t.split("\n")
            head = "\n".join(lines[:2])
            body_rows = lines[2:]
            if len(t) <= size * 2:
                parts = [t]
            else:
                parts, cur = [], []
                for row in body_rows:
                    cur.append(row)
                    if sum(len(r) for r in cur) > size:
                        parts.append(head + "\n" + "\n".join(cur))
                        cur = []
                if cur:
                    parts.append(head + "\n" + "\n".join(cur))
            for part in parts:
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(doc, page.page_number, idx, part),
                        bid_id=doc.bid_id,
                        file_name=doc.file_name,
                        doc_type=doc.doc_type,
                        addendum_number=doc.addendum_number,
                        page_number=page.page_number,
                        section="table",
                        is_table=True,
                        text=f"{header} p.{page.page_number} | TABLE\n{part}",
                        document_date=doc.document_date,
                    )
                )
                idx += 1
    return chunks
