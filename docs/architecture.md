# Architecture

```mermaid
flowchart TB
    subgraph Ingestion["Part A - Ingestion (rfp_intel/ingestion)"]
        F[/"Bid folder<br/>HTML + PDFs (RFP, addendums, specs, affidavits)"/]
        P1["html_parser.py<br/>BeautifulSoup label/value pairs -> markdown table"]
        P2["pdf_parser.py<br/>PyMuPDF text (sorted) + find_tables -> markdown"]
        C["cleaning.py<br/>header/footer removal, de-hyphenation, whitespace"]
        D["doc_types.py<br/>bid_page | rfp | addendum(n) | specs | affidavit + date"]
        K["chunker.py<br/>section-aware windows (1800/250 chars) + whole-table chunks"]
        F --> P1 --> C --> D --> K
        F --> P2 --> C
    end

    subgraph Search["Part B - Search engine (rfp_intel/search)"]
        E["embeddings.py<br/>fastembed bge-small-en-v1.5 (384d)"]
        VS[("FAISS IndexIDMap2 / IndexFlatIP")]
        KS[("BM25Okapi<br/>identifier-aware tokenizer")]
        META[("SQLite chunks + documents<br/>bid_id, file, doc_type, addendum, page")]
        Q["query.py<br/>synonym expansion (+ optional LLM rewrite)"]
        RRF["engine.py<br/>metadata pre-filter -> RRF fusion"]
        RR["reranker.py<br/>LLM listwise rerank (llama-3.1-8b)"]
        K --> E --> VS
        K --> KS
        K --> META
        Q --> RRF
        VS --> RRF
        KS --> RRF
        META -.filters.-> RRF
        RRF --> RR
    end

    subgraph Agents["Part C - Multi-agent system (rfp_intel/agents)"]
        O["Orchestrator / Planner<br/>plan, asyncio fan-out, retry loop, SharedState"]
        IA["Ingestion Agent"]
        RA["Retrieval Agent<br/>per-field queries, doc-type + addendum passes"]
        EA1["Extraction: Dates & Logistics"]
        EA2["Extraction: Commercial & Legal"]
        EA3["Extraction: Product & Specs"]
        REC["Addendum Reconciliation Agent"]
        VAL["Validator / Critic Agent<br/>citations, quotes, formats + LLM critic"]
        QA["Q&A / Report Agent<br/>cited answers, JSON, summary, comparison"]
        O --> IA --> K
        O --> RA --> RR
        RA --> EA1 & EA2 & EA3
        EA1 & EA2 & EA3 --> REC --> VAL
        VAL -- "failed fields + feedback (max 2 retries)" --> RA
        VAL --> QA
        O --> QA
    end

    subgraph IO["Interfaces"]
        CLI["main.py CLI"]
        API["FastAPI: /index /search /ask /extract /compare"]
        UI["Streamlit UI"]
        TR[("runs/<run_id>/trace.jsonl")]
        OUT[("outputs/<bid>.json + summary.md")]
    end
    CLI & API & UI --> O
    CLI & API & UI --> RRF
    O --> TR
    QA --> OUT
```

## Message flow of one extraction run

1. **Orchestrator** writes an `ExecutionPlan` into `SharedState` and calls the **Ingestion Agent** (no-op if the bid is already indexed).
2. For the 19 content fields (grouped into 3 specialists) it launches `asyncio` tasks (bounded by `AGENT_CONCURRENCY`). Each task:
   - **Retrieval Agent** -> `SearchEngine.search()` three times: unfiltered hybrid+rerank, filtered to the field's preferred doc types, and (for addendum-sensitive fields) filtered to `doc_type=addendum`; results are merged and de-duplicated.
   - **Extraction Agent** -> one Groq call with the `[E#]` evidence passages, returns `{value, evidence_ids, quotes, confidence, notes}` (JSON mode) -> converted to a `FieldValue` with `FieldSource` citations.
3. **Addendum Reconciliation Agent** retrieves addendum passages for all addendum-sensitive fields, asks the LLM which fields change, applies the highest-numbered addendum and records `AddendumChange` entries.
4. **Validator Agent** runs deterministic checks (citation exists in bid, quote occurs in cited chunk, date format, null-with-reason) and one LLM critic call; failed fields go back to step 2 with the validator message as feedback (max `MAX_FIELD_RETRIES`). Fields still failing are nulled with a reason, never shipped unsupported.
5. The **Summary** specialist writes `Bid Summary` with the other fields as context; the **Q&A / Report Agent** renders `outputs/<bid>.json` and `<bid>_summary.md`.
6. Every step is appended to `runs/<run_id>/trace.jsonl` (input, tool calls, output, tokens, latency, status).
