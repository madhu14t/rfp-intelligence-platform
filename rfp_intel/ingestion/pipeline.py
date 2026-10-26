"""Ingestion pipeline: bid folder -> parsed documents -> chunks."""
from __future__ import annotations

import logging
from pathlib import Path

from rfp_intel.ingestion.chunker import chunk_document
from rfp_intel.ingestion.html_parser import parse_html
from rfp_intel.ingestion.pdf_parser import parse_pdf
from rfp_intel.models import Chunk, ParsedDocument

log = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".html", ".htm"}


def parse_file(path: Path, bid_id: str) -> ParsedDocument | None:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            return parse_pdf(path, bid_id)
        if ext in (".html", ".htm"):
            return parse_html(path, bid_id)
    except Exception as exc:  # never let one bad file kill the whole bid
        log.exception("Failed to parse %s: %s", path, exc)
        return ParsedDocument(
            bid_id=bid_id, file_name=path.name, file_path=str(path), doc_type="other", pages=[],
            parse_warnings=[f"parse failure: {exc}"],
        )
    log.info("Skipping unsupported file %s", path.name)
    return None


def ingest_folder(folder: Path, bid_id: str | None = None) -> tuple[list[ParsedDocument], list[Chunk]]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Bid folder not found: {folder}")
    bid_id = bid_id or folder.name
    docs: list[ParsedDocument] = []
    chunks: list[Chunk] = []
    for path in sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED):
        doc = parse_file(path, bid_id)
        if doc is None:
            continue
        docs.append(doc)
        doc_chunks = chunk_document(doc)
        chunks.extend(doc_chunks)
        log.info(
            "Parsed %s: type=%s addendum=%s pages=%d chunks=%d warnings=%d",
            doc.file_name, doc.doc_type, doc.addendum_number, len(doc.pages), len(doc_chunks), len(doc.parse_warnings),
        )
        for w in doc.parse_warnings:
            log.warning("  %s: %s", doc.file_name, w)
    return docs, chunks
