from __future__ import annotations

from pathlib import Path

import pymupdf

from rfp_intel.ingestion.chunker import _is_heading, chunk_document, merge_small_sections, split_sections, window
from rfp_intel.ingestion.cleaning import clean_page_text, fix_hyphenation, remove_repeated_lines
from rfp_intel.ingestion.doc_types import detect_addendum_number, detect_doc_type, detect_document_date
from rfp_intel.ingestion.html_parser import parse_html
from rfp_intel.ingestion.pdf_parser import parse_pdf
from rfp_intel.ingestion.pipeline import ingest_folder, parse_file

HTML = """<html><head><title>Widget Bid - Bid Information | Portal</title></head><body>
<nav>Home</nav>
<div class="mets-field mets-field-view"><span class="mets-field-label">Solicitation Number</span>
<div class="mets-field-body"><p>XX-12345</p></div></div>
<div class="dates mets-field mets-field-view"><span class="mets-field-label">Closing Date</span>
<div class="mets-field-body"><p>07/09/2024 03:00 PM EDT</p></div></div>
<div><div><span>Description</span></div><div>Buying 30 widgets. See more</div></div>
<script>var x = 1;</script></body></html>"""


def _make_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    doc.save(path)
    return path


# ------------------------------------------------------------------ doc types / metadata
def test_detect_doc_type_from_name_and_content():
    assert detect_doc_type("Addendum 2 RFP JA-1.pdf") == "addendum"
    assert detect_doc_type("Mercury_Affidavit.pdf") == "affidavit"
    assert detect_doc_type("Dell_Laptop_Specs.pdf") == "specs"
    assert detect_doc_type("PORFP_-_Dell_Laptop_Final.pdf") == "rfp"
    assert detect_doc_type("something - Bid Information - BidNet Direct.html") == "bid_page"
    assert detect_doc_type("random.pdf", "ADDENDUM No. 3\nRFP ...") == "addendum"
    assert detect_doc_type("unknown.pdf", "Purchase Order Request for Proposals") == "rfp"


def test_detect_addendum_number_and_date():
    assert detect_addendum_number("Addendum 2 RFP JA-207652.pdf") == 2
    assert detect_addendum_number("x.pdf", "ADDENDUM No. 1\nRFP") == 1
    assert detect_addendum_number("rfp.pdf", "no addendum here") is None
    assert detect_document_date("Issue Date 26-MAY-2024 08:00") == "26-MAY-2024"
    assert detect_document_date("due July 9, 2024 at 2pm") == "July 9, 2024"
    assert detect_document_date("05/24/2024") == "05/24/2024"


# ------------------------------------------------------------------ cleaning
def test_fix_hyphenation_and_whitespace():
    assert fix_hyphenation("pre-\nproposal meeting") == "pre-proposal meeting"
    assert clean_page_text("a   b\n\n\n\nc\t d") == "a b\n\nc d"


def test_remove_repeated_headers_footers():
    bodies = ["scope of work", "pricing schedule", "insurance terms", "signature page"]
    pages = [f"ACME RFP 123\n{b}\nPage {i} of 4" for i, b in enumerate(bodies, 1)]
    cleaned = remove_repeated_lines(pages, min_pages=3)
    assert all("ACME RFP 123" not in p for p in cleaned)
    assert all("Page" not in p for p in cleaned)
    assert "pricing schedule" in cleaned[1]


# ------------------------------------------------------------------ chunking
def test_heading_detection():
    assert _is_heading("SCOPE AND SPECIFICATIONS OF THE PROPOSAL")
    assert _is_heading("1.2 Terms")
    assert _is_heading("Section 4 – Scope of Work")
    assert not _is_heading("80 Calvert Street")
    assert not _is_heading("1 Tier 1 Small Student 1 Each")
    assert not _is_heading("This is a normal sentence that ends with a period.")


def test_split_merge_and_window():
    text = "INTRO SECTION\nshort intro\n\n1.2 Terms\n" + ("Initial term three years. " * 10) + "\nNEXT HEADING HERE\nmore text"
    sections = split_sections(text)
    assert [h for h, _ in sections][:2] == ["INTRO SECTION", "1.2 Terms"]
    merged = merge_small_sections(sections, size=400)
    assert len(merged) <= len(sections)
    pieces = list(window("para one. " * 60, size=200, overlap=40))
    assert len(pieces) > 1 and all(len(p) <= 200 for p in pieces)
    # overlap: the start of piece 2 appears near the end of piece 1
    assert pieces[1][:10] in pieces[0]


def test_chunk_document_keeps_metadata_and_tables(tmp_path):
    pdf = _make_pdf(tmp_path / "Addendum 1 RFP TEST-1.pdf", ["ADDENDUM No. 1\nRFP TEST-1\nThe new due date is July 9, 2024."])
    doc = parse_pdf(pdf, "BidX")
    assert doc.doc_type == "addendum" and doc.addendum_number == 1
    doc.pages[0].tables.append("| SKU | Desc |\n|---|---|\n| 400-BRFT | 256 GB SSD |")
    chunks = chunk_document(doc)
    assert chunks and all(c.bid_id == "BidX" and c.page_number == 1 and c.addendum_number == 1 for c in chunks)
    tables = [c for c in chunks if c.is_table]
    assert len(tables) == 1 and "400-BRFT" in tables[0].text
    assert "Addendum 1" in chunks[0].text  # header carries provenance


# ------------------------------------------------------------------ parsers
def test_parse_html_extracts_fields(tmp_path):
    p = tmp_path / "Widget Bid - Bid Information - BidNet Direct.html"
    p.write_text(HTML, encoding="utf-8")
    doc = parse_html(p, "BidX")
    assert doc.doc_type == "bid_page"
    text = doc.pages[0].text
    assert "| Solicitation Number | XX-12345 |" in text
    assert "07/09/2024 03:00 PM EDT" in text
    assert "var x" not in text  # scripts stripped
    assert doc.pages[0].tables


def test_parse_pdf_multiple_pages_and_empty_page(tmp_path):
    pdf = _make_pdf(tmp_path / "rfp.pdf", ["Request for Proposal 999\nDue date 01/02/2025", ""])
    doc = parse_pdf(pdf, "BidY")
    assert len(doc.pages) == 2
    assert doc.pages[1].is_empty and any("empty" in w for w in doc.parse_warnings)
    assert "Due date" in doc.pages[0].text and doc.doc_type == "rfp"


def test_unreadable_file_is_handled(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf")
    doc = parse_file(bad, "BidZ")
    assert doc is not None and doc.pages == [] and doc.parse_warnings


def test_ingest_folder_real_data():
    folder = Path(__file__).resolve().parents[1] / "data" / "Bid2"
    if not folder.exists():
        return
    docs, chunks = ingest_folder(folder)
    types = {d.file_name: d.doc_type for d in docs}
    assert types["PORFP_-_Dell_Laptop_Final.pdf"] == "rfp"
    assert types["Mercury_Affidavit.pdf"] == "affidavit"
    assert types["Dell_Laptop_Specs.pdf"] == "specs"
    assert any(d.doc_type == "bid_page" for d in docs)
    assert any("400-BRFT" in c.text for c in chunks)
    assert all(c.page_number >= 1 for c in chunks)
