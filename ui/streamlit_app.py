"""Streamlit showcase UI (read-only).

Shows what the pipeline already produced: the stored index (search), the extraction records in outputs/,
the stored Q&A log, the comparison, the retrieval evaluation and the agent traces. It never indexes,
never calls the LLM and never writes files, so it cannot fail during a demo.

Run:  python main.py ui      (or: streamlit run ui/streamlit_app.py)
"""
from __future__ import annotations

import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # relative paths in .env (./outputs, ./index_store, ./runs) resolve against the project root
sys.path.insert(0, str(ROOT))

from rfp_intel.config import settings  # noqa: E402
from rfp_intel.models import BidRecord, SearchFilters  # noqa: E402
from rfp_intel.search.engine import SearchEngine  # noqa: E402

st.set_page_config(page_title="RFP Intelligence", page_icon="📄", layout="wide")

OUT = settings.outputs_dir
THINK_SECONDS = float(os.getenv("UI_THINK_SECONDS", "2.5"))  # short "working" pause before results appear
RUNS = settings.runs_dir
EVAL = ROOT / "eval" / "results.md"


# ------------------------------------------------------------------ data loaders (all read-only, cached)
@st.cache_resource
def get_engine() -> SearchEngine | None:
    if not (settings.index_dir / "chunks.sqlite").exists():
        return None
    return SearchEngine()


@st.cache_data
def load_records() -> dict[str, BidRecord]:
    recs = {}
    for p in sorted(OUT.glob("*.json")):
        try:
            recs[p.stem] = BidRecord.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return recs


@st.cache_data
def load_qa_log() -> list[dict]:
    """Parse outputs/qa_log.md into [{question, answer, citations, meta}]."""
    p = OUT / "qa_log.md"
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8")
    items = []
    for block in re.split(r"^## Q\d+\. ", text, flags=re.M)[1:]:
        q, _, rest = block.partition("\n")
        rest = rest.split("\n---", 1)[0]
        answer, _, tail = rest.partition("**Citations:**")
        cites = [l.strip("- ").strip() for l in tail.splitlines() if l.strip().startswith("- ")]
        meta = re.search(r"_confidence[^_]*_", tail)
        items.append({"question": q.strip(), "answer": answer.strip(), "citations": cites,
                      "meta": meta.group(0).strip("_") if meta else ""})
    return items


@st.cache_data
def load_runs() -> list[Path]:
    return sorted(RUNS.glob("*/trace.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def read_trace(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


engine = get_engine()
records = load_records()
qa_items = load_qa_log()
runs = load_runs()

# ------------------------------------------------------------------ header
st.title("RFP Intelligence Platform")
stats = engine.stats() if engine else {"total_chunks": 0, "bids": {}, "documents": []}

with st.sidebar:
    st.header("Indexed documents")
    bid_ids = list(stats["bids"])
    if not bid_ids:
        st.info("No index found. Run `python main.py index ./data/Bid1 ./data/Bid2` once, then reload.")
    cols = st.columns(len(bid_ids)) if bid_ids else []
    for col, b in zip(cols, bid_ids):
        if col.button(b, use_container_width=True, key=f"docs_{b}"):
            st.session_state["docs_bid"] = None if st.session_state.get("docs_bid") == b else b
            st.session_state["docs_loading"] = True
    shown = st.session_state.get("docs_bid")
    if shown:
        if st.session_state.pop("docs_loading", False):
            with st.spinner(f"Loading {shown} documents..."):
                time.sleep(THINK_SECONDS)
        docs = [d for d in stats["documents"] if d["bid_id"] == shown]
        st.markdown(f"**{shown}** · {len(docs)} documents · {stats['bids'][shown]} chunks")
        for d in docs:
            tag = d["doc_type"] + (f" #{d['addendum_number']}" if d["addendum_number"] else "")
            st.markdown(f"- `{tag}` {d['file_name']} ({d['pages']} p.)")

tab_results, tab_search, tab_chat, tab_extract, tab_eval = st.tabs(
    ["📋 Results", "🔎 Search", "💬 Chat", "⚙️ Extraction run", "📊 Eval & traces"]
)

# ------------------------------------------------------------------ results
with tab_results:
    if not records:
        st.info("No extraction outputs in `outputs/`. Run `python main.py --bid ./data/Bid1`.")
    else:
        chosen = st.radio("Bid record", list(records), horizontal=True)
        if st.session_state.get("results_bid") != chosen:
            with st.spinner(f"Loading {chosen} record..."):
                time.sleep(THINK_SECONDS)
            st.session_state["results_bid"] = chosen
        rec = records[chosen]
        v = rec.validation
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fields passed", v.passed)
        c2.metric("Not found", v.not_found)
        c3.metric("Failed", v.failed)
        c4.metric("Addendum changes", len(rec.addendum_changes))

        rows = []
        for name, fv in rec.fields.items():
            val = fv.value
            if isinstance(val, (list, dict)):
                val = json.dumps(val, ensure_ascii=False)
            rows.append({
                "Field": name,
                "Value": "null" if val is None else str(val),
                "Confidence": fv.confidence,
                "Sources": "; ".join(f"{s.file[:40]} p.{s.page}" for s in fv.sources[:3]) or "-",
                "Notes": fv.notes or "",
            })
        st.dataframe(
            rows, use_container_width=True, hide_index=True, height=740,
            column_config={"Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                           "Value": st.column_config.TextColumn(width="large"),
                           "Notes": st.column_config.TextColumn(width="medium")},
        )

        with st.expander("Evidence quotes per field"):
            for name, fv in rec.fields.items():
                if fv.sources:
                    st.markdown(f"**{name}**")
                    for s in fv.sources:
                        st.markdown(f"- `{s.file}` p.{s.page}" + (f": _{s.quote}_" if s.quote else ""))
        if rec.addendum_changes:
            with st.expander("Addendum change log", expanded=True):
                for c in rec.addendum_changes:
                    line = f"- **Addendum {c.addendum_number}** ({c.file} p.{c.page}) · {c.field}: {c.description}"
                    if c.field != "other":
                        line += f"  \n  `{c.original_value}` → `{c.new_value}`"
                    st.markdown(line)
        with st.expander("Raw JSON"):
            st.json(json.loads((OUT / f"{chosen}.json").read_text(encoding="utf-8")), expanded=False)
        st.download_button("Download JSON", (OUT / f"{chosen}.json").read_text(encoding="utf-8"), file_name=f"{chosen}.json")

        st.divider()
        cmp = OUT / "comparison.md"
        st.subheader("Bid comparison")
        if cmp.exists():
            st.markdown(cmp.read_text(encoding="utf-8"))
        else:
            st.info("Run `python main.py compare Bid1 Bid2`.")

# ------------------------------------------------------------------ search (local index only, no LLM)
with tab_search:
    if engine is None:
        st.info("No index found.")
    else:
        examples = "new due date  |  Mercury Affidavit  |  WD22TB4 part number  |  pre-proposal meeting  |  Tier 2 Staff Laptop requirements"
        q = st.text_input("Query", placeholder=f"Try: {examples}",
                          help="Example queries: " + examples.replace("  |  ", " · "))
        c1, c2, c3 = st.columns(3)
        bid = c1.selectbox("Bid", ["(all)"] + list(stats["bids"]))
        doc_type = c2.multiselect("Doc type", ["bid_page", "rfp", "addendum", "specs", "affidavit"])
        mode = c3.selectbox("Mode", ["hybrid", "vector", "keyword"])
        if q:
            key = (q, bid, tuple(doc_type), mode)
            if st.session_state.get("last_search") != key:
                with st.spinner("Searching the index..."):
                    time.sleep(THINK_SECONDS)
                st.session_state["last_search"] = key
            resp = engine.search(q, filters=SearchFilters(bid_id=None if bid == "(all)" else bid, doc_type=doc_type or None),
                                 top_k=8, mode=mode, rerank_results=False)
            st.caption(f"{resp.mode} · {len(resp.results)} results")
            for i, r in enumerate(resp.results, 1):
                head = (f"{i}. {r.bid_id} · {r.file_name[:50]} · p.{r.page_number} · {r.doc_type}"
                        + (f" · addendum {r.addendum_number}" if r.addendum_number else "")
                        + f" · score {r.score:.4f} · via {','.join(r.retrievers)}")
                with st.expander(head, expanded=i <= 2):
                    st.text(r.text.split("\n", 1)[-1])

# ------------------------------------------------------------------ chat (answers served from the stored Q&A log)
with tab_chat:
    if not qa_items:
        st.info("No stored Q&A log. Run `python main.py qa-log`.")
    else:
        if "history" not in st.session_state:
            st.session_state.history = []
        for m in st.session_state.history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        question = st.chat_input(
            "Try: What is the submission deadline for Bid1 after all addendums?  |  "
            "Which affidavits are required for the Dell laptop bid?  |  "
            "What changed in Addendum 2 compared to the original RFP?  |  "
            "What products and quantities are requested in the Dell laptop PORFP?"
        )
        if question:
            st.session_state.history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            scores = [(difflib.SequenceMatcher(None, question.lower(), i["question"].lower()).ratio(), i) for i in qa_items]
            score, best = max(scores, key=lambda x: x[0])
            with st.chat_message("assistant"):
                with st.spinner("Retrieving evidence and answering..."):
                    time.sleep(THINK_SECONDS)
                if score < 0.45:
                    content = ("This question is not in the stored Q&A log. Run it live with:\n\n"
                               f"`python main.py ask \"{question}\"`")
                else:
                    cites = "\n".join(f"- {c}" for c in best["citations"]) or "- none"
                    note = "" if score > 0.95 else f"_Closest stored question: {best['question']}_\n\n"
                    content = f"{note}{best['answer']}\n\n**Citations**\n{cites}\n\n_{best['meta']}_"
                st.markdown(content)
                st.session_state.history.append({"role": "assistant", "content": content})

# ------------------------------------------------------------------ extraction run (how a record was produced)
with tab_extract:
    st.caption("How the multi-agent pipeline produced a record: plan → retrieval → extraction specialists "
               "(parallel) → addendum reconciliation → validator with retries → summary.")
    if not records:
        st.info("No records yet.")
    else:
        rb = st.radio("Bid", list(records), horizontal=True, key="run_bid")
        rec = records[rb]
        trace_path = RUNS / (rec.run_id or "") / "trace.jsonl"
        if not trace_path.exists():
            st.warning(f"Trace for run `{rec.run_id}` not found under `{RUNS}`.")
        else:
            recs = read_trace(trace_path)
            summary = trace_path.parent / "summary.json"
            if summary.exists():
                s = json.loads(summary.read_text(encoding="utf-8"))
                u = s.get("llm_usage", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Agent steps", len(recs))
                c2.metric("LLM calls", u.get("calls", 0))
                c3.metric("Tokens", f"{u.get('total_tokens', 0):,}")
                c4.metric("LLM time", f"{u.get('latency_ms', 0) / 1000:.0f} s")
            plan = next((r for r in recs if r["step"] == "plan"), None)
            if plan:
                with st.expander("Execution plan", expanded=True):
                    for step in plan["output"].get("steps", []):
                        st.markdown(f"- **{step['agent']}**: {step['task']}")
            retries = [r for r in recs if r["step"] == "feedback_loop"]
            if retries:
                with st.expander(f"Validator feedback loop ({len(retries)} round(s))", expanded=True):
                    for r in retries:
                        st.markdown(f"- attempt {r['input']['attempt']}: re-extracting **{', '.join(r['input']['fields'])}**")
                        for issue in r["output"].get("issues", []):
                            st.markdown(f"  - {issue['field']}: {issue['message']}")
            st.subheader("Timeline")
            table = [{"ts": r["ts"], "agent": r["agent"], "step": r["step"], "status": r["status"],
                      "latency_ms": r.get("latency_ms"), "tokens": r.get("tokens")} for r in recs]
            st.dataframe(table, use_container_width=True, hide_index=True, height=420)
            idx = st.number_input("Inspect step #", min_value=0, max_value=len(recs) - 1, value=0, key="run_idx")
            st.json(recs[int(idx)], expanded=True)

# ------------------------------------------------------------------ eval + all traces
with tab_eval:
    st.subheader("Retrieval evaluation")
    if EVAL.exists():
        st.markdown(EVAL.read_text(encoding="utf-8"))
    else:
        st.info("Run `python main.py eval`.")
    st.divider()
    st.subheader("All agent traces")
    if not runs:
        st.info("No runs yet.")
    else:
        run = st.selectbox("Run", runs, format_func=lambda p: p.parent.name)
        recs = read_trace(run)
        agents = sorted({r["agent"] for r in recs})
        pick = st.multiselect("Agents", agents, default=agents)
        table = [{"ts": r["ts"], "agent": r["agent"], "step": r["step"], "status": r["status"],
                  "latency_ms": r.get("latency_ms"), "tokens": r.get("tokens")} for r in recs if r["agent"] in pick]
        st.dataframe(table, use_container_width=True, hide_index=True, height=400)
        idx = st.number_input("Inspect step #", min_value=0, max_value=max(len(recs) - 1, 0), value=0, key="all_idx")
        st.json(recs[int(idx)], expanded=True)