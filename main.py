"""RFP Intelligence Platform - command line entry point.

Examples
  python main.py index ./data/Bid1 ./data/Bid2
  python main.py --bid ./data/Bid1                      # extract (indexes first if needed)
  python main.py extract --bid-id Bid2
  python main.py search "bid bond" --bid-id Bid1 --top-k 5 --mode hybrid
  python main.py ask "What is the submission deadline for Bid1 after all addendums?"
  python main.py compare Bid1 Bid2
  python main.py eval --configs vector keyword hybrid hybrid_rerank
  python main.py models                                  # list models available to your key/provider
  python main.py serve --port 8000
  python main.py qa-log                                  # run the 10 sample questions -> outputs/qa_log.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rfp_intel.config import settings


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "groq", "faiss", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_index(args: argparse.Namespace) -> None:
    from rfp_intel.search.engine import SearchEngine

    engine = SearchEngine()
    for folder in args.folders:
        res = engine.index_folder(folder)
        print(json.dumps(res, indent=2))
    print(json.dumps(engine.stats()["bids"], indent=2))


def cmd_search(args: argparse.Namespace) -> None:
    from rfp_intel.models import SearchFilters
    from rfp_intel.search.engine import SearchEngine

    engine = SearchEngine()
    filters = SearchFilters(bid_id=args.bid_id, doc_type=args.doc_type.split(",") if args.doc_type else None,
                            addendum_number=args.addendum)
    resp = engine.search(args.query, filters=filters, top_k=args.top_k, mode=args.mode,
                         rerank_results=None if args.rerank is None else args.rerank)
    print(f"mode={resp.mode} latency={resp.latency_ms}ms rewritten={resp.rewritten_queries}")
    for i, r in enumerate(resp.results, 1):
        print(f"\n[{i}] score={r.score} {r.bid_id} | {r.file_name} | page {r.page_number} | {r.doc_type}"
              + (f" | addendum {r.addendum_number}" if r.addendum_number else "") + f" | via {','.join(r.retrievers)}")
        body = r.text.split("\n", 1)[1] if "\n" in r.text else r.text
        print("   " + body[:600].replace("\n", "\n   "))


def cmd_extract(args: argparse.Namespace) -> None:
    from rfp_intel.agents.orchestrator import run_extract

    _require_key()
    record = run_extract(args.bid, args.bid_id, force_index=args.reindex)
    print(record.model_dump_json(indent=2))
    print(f"\nSaved to {settings.outputs_dir / (record.bid_id + '.json')}  (trace: {settings.runs_dir / record.run_id / 'trace.jsonl'})",
          file=sys.stderr)


def cmd_ask(args: argparse.Namespace) -> None:
    from rfp_intel.agents.orchestrator import run_ask

    _require_key()
    ans = run_ask(args.question, args.bid_ids or None)
    print(ans.answer)
    print("\nCitations:")
    for c in ans.citations:
        print(f"  - {c.bid_id} | {c.file} | page {c.page}")
    print(f"confidence={ans.confidence:.2f} run_id={ans.run_id}")


def cmd_compare(args: argparse.Namespace) -> None:
    from rfp_intel.agents.orchestrator import run_compare

    _require_key()
    print(run_compare(args.bid_ids or None))


def cmd_eval(args: argparse.Namespace) -> None:
    from eval.run_eval import run_eval

    run_eval(configs=args.configs, k_values=args.k, eval_file=Path(args.eval_file), out_file=Path(args.out))


def cmd_qa_log(args: argparse.Namespace) -> None:
    from eval.qa_log import run_qa_log

    _require_key()
    run_qa_log(Path(args.questions), Path(args.out))


def cmd_models(args: argparse.Namespace) -> None:
    from rfp_intel.llm.client import get_llm

    _require_key()
    print(f"Models available for provider '{settings.llm_provider}' ({settings.llm_base_url}):")
    for m in get_llm().list_models():
        print("  ", m)
    print(f"\nConfigured: LLM_MODEL={settings.llm_model}  LLM_SMALL_MODEL={settings.llm_small_model}")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("rfp_intel.api.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_ui(args: argparse.Namespace) -> None:
    import subprocess

    subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py", "--server.port", str(args.port)], check=False)


def _require_key() -> None:
    if not settings.llm_api_key:
        sys.exit(f"{settings.llm_key_env_name} is not set for LLM_PROVIDER={settings.llm_provider}. "
                 "Copy .env.example to .env and add your key.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RFP Intelligence Platform")
    p.add_argument("--bid", help="Shortcut: extract this bid folder (python main.py --bid ./Bid1)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("index", help="Parse + index one or more bid folders (incremental)")
    s.add_argument("folders", nargs="+")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("search", help="Query the search engine")
    s.add_argument("query")
    s.add_argument("--bid-id")
    s.add_argument("--doc-type", help="comma separated: bid_page,rfp,addendum,specs,affidavit")
    s.add_argument("--addendum", type=int)
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    s.add_argument("--rerank", dest="rerank", action="store_true", default=None)
    s.add_argument("--no-rerank", dest="rerank", action="store_false")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("extract", help="Extraction mode: produce the structured JSON for a bid")
    s.add_argument("--bid", help="bid folder")
    s.add_argument("--bid-id", help="already indexed bid id")
    s.add_argument("--reindex", action="store_true")
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("ask", help="Question-answering mode")
    s.add_argument("question")
    s.add_argument("--bid-ids", nargs="*")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("compare", help="Side-by-side comparison of bids (bonus)")
    s.add_argument("bid_ids", nargs="*")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("eval", help="Retrieval evaluation (Recall@k, MRR, nDCG)")
    s.add_argument("--configs", nargs="+", default=["vector", "keyword", "hybrid", "hybrid_rerank"])
    s.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    s.add_argument("--eval-file", default="eval/eval_set.json")
    s.add_argument("--out", default="eval/results.md")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("qa-log", help="Run the sample questions and write outputs/qa_log.md")
    s.add_argument("--questions", default="eval/sample_questions.json")
    s.add_argument("--out", default="outputs/qa_log.md")
    s.set_defaults(func=cmd_qa_log)

    s = sub.add_parser("models", help="List the models your LLM provider/key can use")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("serve", help="Run the REST API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("ui", help="Run the Streamlit UI (bonus)")
    s.add_argument("--port", type=int, default=8501)
    s.set_defaults(func=cmd_ui)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    settings.ensure_dirs()
    if args.bid and not args.cmd:
        args.bid_id = None
        args.reindex = False
        cmd_extract(args)
        return
    if not args.cmd:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()