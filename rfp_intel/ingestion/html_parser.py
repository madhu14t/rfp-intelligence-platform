"""HTML bid-page parsing (BidNet Direct and similar procurement portals).

Strategy
1. Strip navigation / scripts / cookie banners.
2. Collect *label -> value* pairs from the portal's field widgets. Works for BidNet (`.mets-field-label` +
   `.mets-field-body`) and falls back to generic `dt/dd`, `th/td` and "Label:\\nValue" patterns, so other
   portals degrade gracefully.
3. Render the result as a markdown table ("Bid Information") plus the free-text description, which keeps
   the structure for the chunker and makes exact values (bid numbers, dates) easy to match by keyword.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from rfp_intel.ingestion.cleaning import normalise_whitespace
from rfp_intel.ingestion.doc_types import detect_document_date
from rfp_intel.models import PageContent, ParsedDocument

log = logging.getLogger(__name__)

_NOISE_TAGS = ("script", "style", "noscript", "svg", "nav", "header", "footer", "iframe", "button")
_NOISE_TEXT = ("cookies", "privacy policy", "skip to main content", "loading...", "powered by")


def _clean(s: str) -> str:
    return normalise_whitespace(re.sub(r"\s+", " ", s or ""))


def _collect_pairs(soup: BeautifulSoup) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(label: str, value: str) -> None:
        label, value = _clean(label), _clean(value)
        if not label or not value or len(value) > 4000:
            return
        key = (label.lower(), value.lower())
        if key in seen:
            return
        seen.add(key)
        pairs.append((label, value))

    # BidNet / mdf commerce widgets
    for lab in soup.select(".mets-field-label"):
        parent = lab.find_parent(class_=re.compile(r"mets-field"))
        body = parent.select_one(".mets-field-body") if parent else None
        if body:
            add(lab.get_text(" "), body.get_text(" "))

    # Generic definition lists and two-column tables
    for dt in soup.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if dd:
            add(dt.get_text(" "), dd.get_text(" "))
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) == 2:
            add(cells[0].get_text(" "), cells[1].get_text(" "))
    return pairs


def _contact_block(soup: BeautifulSoup) -> str:
    """BidNet renders contact info as a list of lines under a 'Contact Information' heading."""
    for el in soup.find_all(string=re.compile(r"contact information", re.I)):
        parent = el.find_parent(["div", "section"]) if isinstance(el, Tag) or el else None
        if parent is not None:
            container = parent.find_parent(["div", "section"]) or parent
            lines = [_clean(x) for x in container.get_text("\n").split("\n")]
            lines = [l for l in lines if l and l.lower() != "contact information"]
            if lines:
                return "\n".join(lines[:8])
    return ""


def _description(soup: BeautifulSoup) -> str:
    for lab in soup.find_all(string=re.compile(r"^\s*description\s*$", re.I)):
        parent = lab.find_parent(["div", "section"])
        if parent is not None:
            container = parent.find_parent(["div", "section"]) or parent
            txt = container.get_text("\n")
            txt = re.sub(r"^\s*description\s*$", "", txt, flags=re.I | re.M)
            txt = re.sub(r"\bsee more\b", "", txt, flags=re.I)
            return normalise_whitespace(txt)
    return ""


def parse_html(path: Path, bid_id: str) -> ParsedDocument:
    warnings: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "lxml")
    except Exception as exc:
        log.error("Unreadable HTML %s: %s", path.name, exc)
        return ParsedDocument(
            bid_id=bid_id, file_name=path.name, file_path=str(path), doc_type="bid_page", pages=[],
            parse_warnings=[f"unreadable file: {exc}"],
        )

    title = _clean(soup.title.get_text()) if soup.title else None
    pairs = _collect_pairs(soup)
    contact = _contact_block(soup)
    description = _description(soup)

    for t in soup.find_all(_NOISE_TAGS):
        t.decompose()
    main = soup.find("main") or soup.body or soup
    body_lines = [_clean(l) for l in main.get_text("\n").split("\n")]
    body_lines = [l for l in body_lines if l and not any(n in l.lower() for n in _NOISE_TEXT)]
    # de-duplicate consecutive repeated nav items
    deduped: list[str] = []
    for l in body_lines:
        if not deduped or deduped[-1] != l:
            deduped.append(l)
    body_text = "\n".join(deduped)

    sections: list[str] = []
    if title:
        sections.append(f"# {title}")
    if pairs:
        table = ["## Bid Information", "| Field | Value |", "|---|---|"]
        table += [f"| {k} | {v} |" for k, v in pairs]
        sections.append("\n".join(table))
    if contact:
        sections.append("## Contact Information\n" + contact)
    if description:
        sections.append("## Description\n" + description)
    if not pairs and not description:
        warnings.append("no structured fields found; using raw page text")
    sections.append("## Page Text\n" + body_text)

    text = "\n\n".join(sections)
    tables = ["\n".join(["| Field | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in pairs])] if pairs else []
    page = PageContent(page_number=1, text=text, tables=tables, is_empty=not text.strip(), warnings=warnings)

    date = None
    for k, v in pairs:
        if "publication" in k.lower() or "issue" in k.lower():
            date = detect_document_date(v)
            break
    return ParsedDocument(
        bid_id=bid_id, file_name=path.name, file_path=str(path), doc_type="bid_page",
        document_date=date, title=title, pages=[page], parse_warnings=warnings,
    )
