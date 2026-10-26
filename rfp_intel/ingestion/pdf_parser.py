"""PDF parsing with PyMuPDF: text per page, tables as markdown, graceful failure handling, optional OCR."""
from __future__ import annotations

import logging
from pathlib import Path

import pymupdf

from rfp_intel.config import settings
from rfp_intel.ingestion.cleaning import clean_page_text, remove_repeated_lines
from rfp_intel.ingestion.doc_types import detect_addendum_number, detect_doc_type, detect_document_date
from rfp_intel.models import PageContent, ParsedDocument

log = logging.getLogger(__name__)


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows if r and any(c for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    # drop columns that are empty in every row (layout artefacts of form-style PDFs)
    keep = [j for j in range(width) if any(r[j] for r in rows)]
    rows = [[r[j] for j in keep] for r in rows]
    width = len(keep)
    if width < 2:
        return ""
    header, body = rows[0], rows[1:]
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    for r in body:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def _extract_tables(page: pymupdf.Page) -> list[str]:
    tables: list[str] = []
    try:
        found = page.find_tables()
    except Exception as exc:  # pragma: no cover - depends on PyMuPDF build
        log.debug("find_tables failed on page %s: %s", page.number, exc)
        return tables
    for t in found.tables:
        try:
            rows = t.extract()
        except Exception as exc:  # pragma: no cover
            log.debug("table extract failed: %s", exc)
            continue
        # Skip degenerate 1-column "tables" (these are usually just layout boxes)
        if rows and max(len(r) for r in rows) >= 2 and len(rows) >= 2:
            md = _table_to_markdown(rows)
            if md:
                tables.append(md)
    return tables


def _ocr_page(page: pymupdf.Page) -> str:
    """Bonus: OCR a page via PyMuPDF's Tesseract integration if available."""
    try:
        tp = page.get_textpage_ocr(full=True)
        return page.get_text(textpage=tp)
    except Exception as exc:  # pragma: no cover - tesseract usually not installed
        log.warning("OCR unavailable/failed on page %s: %s", page.number + 1, exc)
        return ""


def parse_pdf(path: Path, bid_id: str) -> ParsedDocument:
    pages: list[PageContent] = []
    warnings: list[str] = []
    raw_texts: list[str] = []

    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        log.error("Unreadable PDF %s: %s", path.name, exc)
        return ParsedDocument(
            bid_id=bid_id,
            file_name=path.name,
            file_path=str(path),
            doc_type=detect_doc_type(path.name),
            pages=[],
            parse_warnings=[f"unreadable file: {exc}"],
        )

    with doc:
        for i, page in enumerate(doc):
            pno = i + 1
            page_warnings: list[str] = []
            try:
                text = page.get_text("text", sort=True) or ""
            except Exception as exc:
                text = ""
                page_warnings.append(f"text extraction failed: {exc}")
            tables = _extract_tables(page)
            if not text.strip():
                if page.get_images() and settings.ocr_enabled:
                    text = _ocr_page(page)
                    if text.strip():
                        page_warnings.append("text recovered via OCR")
                if not text.strip():
                    page_warnings.append("empty or scanned page (no extractable text)")
                    log.warning("%s page %d has no extractable text", path.name, pno)
            raw_texts.append(text)
            pages.append(PageContent(page_number=pno, text="", tables=tables, is_empty=not text.strip(), warnings=page_warnings))

    cleaned = remove_repeated_lines(raw_texts)
    for p, txt in zip(pages, cleaned):
        p.text = clean_page_text(txt)
        warnings.extend(f"p{p.page_number}: {w}" for w in p.warnings)

    first = pages[0].text if pages else ""
    doc_type = detect_doc_type(path.name, first)
    return ParsedDocument(
        bid_id=bid_id,
        file_name=path.name,
        file_path=str(path),
        doc_type=doc_type,
        addendum_number=detect_addendum_number(path.name, first) if doc_type == "addendum" else None,
        document_date=detect_document_date(first),
        title=_guess_title(first),
        pages=pages,
        parse_warnings=warnings,
    )


def _guess_title(first_page: str) -> str | None:
    for line in first_page.split("\n"):
        l = line.strip()
        if 8 <= len(l) <= 140 and not l.lower().startswith(("page", "date")):
            return l
    return None
