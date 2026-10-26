"""Document type and metadata detection (works on file name + first page content)."""
from __future__ import annotations

import re
from typing import Optional

from rfp_intel.models import DocType

_ADDENDUM_NUM_RE = re.compile(r"addendum\s*(?:no\.?|number|#)?\s*(\d+)", re.I)
_DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),
    re.compile(r"\b(\d{1,2}-[A-Z]{3}-\d{4})\b"),
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})\b"
    ),
]


def detect_doc_type(file_name: str, first_page_text: str = "") -> DocType:
    name = file_name.lower()
    head = (first_page_text or "")[:3000].lower()

    if name.endswith((".html", ".htm")) or "bidnet" in name or "bid information" in name:
        return "bid_page"
    if "addendum" in name or re.search(r"^\s*addendum\s*no", head, re.M):
        return "addendum"
    if "affidavit" in name or ("affidavit" in head[:600] and "affirm" in head):
        return "affidavit"
    if "spec" in name or "quote" in name or ("sku" in head and "description" in head):
        return "specs"
    if any(k in name for k in ("rfp", "porfp", "rfq", "ifb", "solicitation", "final", "request for proposal")):
        return "rfp"
    if any(k in head for k in ("request for proposal", "purchase order request for proposal", "invitation for bid")):
        return "rfp"
    return "other"


def detect_addendum_number(file_name: str, first_page_text: str = "") -> Optional[int]:
    for source in (file_name, first_page_text[:1000] if first_page_text else ""):
        m = _ADDENDUM_NUM_RE.search(source)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def detect_document_date(text: str) -> Optional[str]:
    """Return the first date-looking token found in the text (best effort)."""
    for pat in _DATE_PATTERNS:
        m = pat.search(text or "")
        if m:
            return m.group(1)
    return None
