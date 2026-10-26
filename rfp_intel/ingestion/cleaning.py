"""Text cleaning utilities: header/footer removal, de-hyphenation, whitespace normalisation."""
from __future__ import annotations

import re
from collections import Counter

_WS_RE = re.compile(r"[ \t ]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_PAGE_X_OF_Y_RE = re.compile(r"^\s*(page\s*\d+\s*(of|\|)\s*\d+|\d+\s*\|\s*\d+)\s*$", re.I)
_DOTTED_LEADER_RE = re.compile(r"^[\s.…_\-]{5,}$")


def normalise_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def fix_hyphenation(text: str) -> str:
    """Join words broken across lines with a hyphen (e.g. 'pre-\\nproposal' -> 'pre-proposal')."""
    return _HYPHEN_BREAK_RE.sub(r"\1-\2", text)


def join_broken_lines(text: str) -> str:
    """Join lines that were wrapped mid-sentence (lowercase continuation), keep paragraph/list breaks."""
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        if (
            out
            and line
            and out[-1]
            and not out[-1].endswith((".", ":", ";", "!", "?"))
            and re.match(r"^[a-z(\"']", line)
            and not re.match(r"^\s*([•\-\*]|\d+[.)])\s", line)
        ):
            out[-1] = out[-1] + " " + line
        else:
            out.append(line)
    return "\n".join(out)


def remove_repeated_lines(pages: list[str], min_pages: int = 3, max_len: int = 160) -> list[str]:
    """Remove lines that repeat on many pages (running headers/footers).

    A line is considered a header/footer if (after digit-masking) it appears on >= min_pages pages
    and only in the first or last 3 lines of a page.
    """
    if len(pages) < min_pages:
        return [drop_page_markers(p) for p in pages]

    def mask(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    counter: Counter[str] = Counter()
    for page in pages:
        lines = [l for l in page.split("\n") if l.strip()]
        edge = lines[:3] + lines[-3:]
        for l in set(mask(x) for x in edge if len(x) <= max_len):
            counter[l] += 1

    repeated = {l for l, c in counter.items() if c >= min_pages}
    cleaned: list[str] = []
    for page in pages:
        kept = []
        lines = page.split("\n")
        n = len(lines)
        for i, l in enumerate(lines):
            at_edge = i < 3 or i >= n - 3
            if at_edge and l.strip() and mask(l) in repeated:
                continue
            kept.append(l)
        cleaned.append(drop_page_markers("\n".join(kept)))
    return cleaned


def drop_page_markers(text: str) -> str:
    out = []
    for l in text.split("\n"):
        if _PAGE_X_OF_Y_RE.match(l) or _DOTTED_LEADER_RE.match(l):
            continue
        out.append(l)
    return "\n".join(out)


def clean_page_text(text: str) -> str:
    text = normalise_whitespace(text)
    text = fix_hyphenation(text)
    text = join_broken_lines(text)
    return normalise_whitespace(text)
