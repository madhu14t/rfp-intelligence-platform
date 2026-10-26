"""Query understanding: rule-based synonym expansion (always) + optional LLM rewriting.

The rule-based layer is deterministic, free and covers the RFP vocabulary that matters for keyword search.
The LLM layer (QUERY_REWRITE_LLM=true) adds 2-3 paraphrases for harder natural-language questions.
"""
from __future__ import annotations

import re

from rfp_intel.llm.client import GroqLLM, LLMError

SYNONYMS: dict[str, list[str]] = {
    "deadline": ["due date", "submission deadline", "closing date", "proposal due", "solicitation due"],
    "due date": ["deadline", "closing date", "proposal due", "solicitation due", "due"],
    "closing date": ["due date", "deadline", "solicitation due"],
    "bid bond": ["bid security", "bond", "surety", "performance bond", "guarantee"],
    "bond": ["bid bond", "bid security", "surety"],
    "pre-bid": ["pre-proposal", "pre-bid meeting", "pre-proposal meeting", "pre-bid conference", "prebid conference"],
    "pre bid": ["pre-proposal meeting", "pre-bid conference", "prebid"],
    "pre-proposal": ["pre-bid", "prebid conference", "pre-proposal meeting"],
    "payment terms": ["net 30", "invoice", "invoicing", "payment", "accounts payable"],
    "payment": ["invoice", "invoicing", "net 30", "payment terms"],
    "delivery": ["delivery date", "lead time", "ship date", "deliver within", "delivery window"],
    "installation": ["install", "deployment", "imaging", "setup", "white glove", "asset tagging", "etching"],
    "warranty": ["extended warranty", "limited hardware warranty", "warranty period", "years warranty"],
    "term": ["term of contract", "contract term", "initial term", "renewal", "extension", "length of contract"],
    "contract term": ["initial term", "renewal", "length of contract", "term of bid"],
    "submission": ["submit", "submittal", "electronic", "portal", "sealed", "emma", "isupplier", "online"],
    "affidavit": ["affidavits", "certification", "form", "attestation", "mercury affidavit", "contract affidavit"],
    "documents required": ["attachments", "forms", "affidavit", "w9", "w-9", "certificate", "insurance", "form 1295"],
    "contact": ["buyer", "procurement officer", "point of contact", "poc", "email", "phone", "agency poc"],
    "manufacturer": ["oem", "authorized reseller", "mfg", "letter of authorization", "authorized"],
    "cooperative": ["co-op", "coop", "master contract", "piggyback", "interlocal", "purchasing cooperative", "state contract"],
    "model": ["model number", "model #", "sku", "part number"],
    "part number": ["sku", "part #", "part no", "model #"],
    "quantity": ["qty", "quantities", "units", "each"],
    "specification": ["specs", "specifications", "minimum requirements", "processor", "memory", "ram", "storage", "display"],
    "specs": ["specification", "minimum requirements", "processor", "ram", "storage", "display"],
    "agency": ["issuing organization", "district", "department", "office", "company name"],
    "issuing organization": ["agency", "district", "purchasing department"],
    "bid number": ["solicitation number", "rfp number", "porfp number", "reference number", "project number"],
    "title": ["solicitation title", "rfp title", "project title"],
    "addendum": ["addenda", "amendment", "change", "revised", "extended"],
    "changed": ["addendum", "amendment", "revised", "extended", "new due date"],
    "mandatory": ["required", "must", "attendance"],
}


def expand_query(query: str) -> str:
    """Return the query plus matched synonym terms (used for BM25 keyword search)."""
    q = query.lower()
    extra: list[str] = []
    for key, syns in SYNONYMS.items():
        if re.search(r"\b" + re.escape(key) + r"\b", q):
            extra.extend(s for s in syns if s not in q)
    seen: set[str] = set()
    extra = [e for e in extra if not (e in seen or seen.add(e))]
    return query if not extra else f"{query} {' '.join(extra)}"


_REWRITE_PROMPT = """You rewrite search queries for a procurement/RFP document search engine.
Given the user's question, output 2 to 3 alternative short search queries that use the vocabulary
typically found in RFPs, bid portals and addendums (e.g. "due date", "solicitation", "PORFP", "affidavit").
Keep identifiers (bid numbers, model numbers) verbatim. Return JSON: {"queries": ["...", "..."]}"""


def llm_rewrite(query: str, llm: GroqLLM, model: str | None = None) -> list[str]:
    try:
        resp = llm.complete(
            [{"role": "system", "content": _REWRITE_PROMPT}, {"role": "user", "content": query}],
            model=model, json_mode=True, max_tokens=200,
        )
        data = resp.json()
        qs = [str(x) for x in data.get("queries", []) if str(x).strip()]
        return qs[:3]
    except (LLMError, Exception):
        return []
