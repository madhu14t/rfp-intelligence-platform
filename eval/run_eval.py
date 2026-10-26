# """Retrieval evaluation: Recall@k, MRR and nDCG@k for several retriever configurations.

# A retrieved chunk counts as relevant when it comes from an expected (file, page) and, if given, contains the
# expected snippet (whitespace/case-insensitive).
# """
# from __future__ import annotations

# import json
# import math
# import re
# import time
# from pathlib import Path
# from typing import Any

# from rfp_intel.models import SearchFilters
# from rfp_intel.search.engine import SearchEngine

# CONFIGS: dict[str, dict[str, Any]] = {
#     "vector": {"mode": "vector", "rerank_results": False},
#     "keyword": {"mode": "keyword", "rerank_results": False},
#     "hybrid": {"mode": "hybrid", "rerank_results": False},
#     "hybrid_rerank": {"mode": "hybrid", "rerank_results": True},
# }


# def _norm(s: str) -> str:
#     return re.sub(r"\s+", " ", s.lower())


# def is_relevant(result, expected: list[dict[str, Any]]) -> bool:
#     text = _norm(result.text)
#     for e in expected:
#         if e["file"].lower() in result.file_name.lower() and int(e.get("page", result.page_number)) == result.page_number:
#             snippet = e.get("contains")
#             if not snippet or _norm(snippet) in text:
#                 return True
#     return False


# def evaluate(engine: SearchEngine, items: list[dict[str, Any]], config: dict[str, Any], k_values: list[int]) -> dict[str, Any]:
#     max_k = max(k_values)
#     recall = {k: 0 for k in k_values}
#     ndcg = {k: 0.0 for k in k_values}
#     rr_sum = 0.0
#     misses: list[str] = []
#     latencies: list[float] = []
#     for item in items:
#         filters = SearchFilters(**item.get("filters", {}))
#         resp = engine.search(item["question"], filters=filters, top_k=max_k, **config)
#         latencies.append(resp.latency_ms)
#         rels = [is_relevant(r, item["expected"]) for r in resp.results]
#         first = next((i for i, r in enumerate(rels) if r), None)
#         if first is None:
#             misses.append(item["id"])
#         else:
#             rr_sum += 1.0 / (first + 1)
#         for k in k_values:
#             if first is not None and first < k:
#                 recall[k] += 1
#             top = rels[:k]
#             dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(top) if r)
#             idcg = sum(1.0 / math.log2(i + 2) for i in range(sum(top)))  # ideal: all relevant hits ranked first
#             ndcg[k] += dcg / idcg if idcg else 0.0
#     n = len(items)
#     return {
#         "n": n,
#         "recall": {k: round(v / n, 3) for k, v in recall.items()},
#         "mrr": round(rr_sum / n, 3),
#         "ndcg": {k: round(v / n, 3) for k, v in ndcg.items()},
#         "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 1),
#         "misses": misses,
#     }


# def run_eval(configs: list[str], k_values: list[int], eval_file: Path, out_file: Path) -> dict[str, Any]:
#     engine = SearchEngine()
#     data = json.loads(Path(eval_file).read_text(encoding="utf-8"))
#     items = data["items"]
#     results: dict[str, Any] = {}
#     for name in configs:
#         cfg = CONFIGS[name]
#         if cfg.get("rerank_results") and not engine.llm.available:
#             print(f"skipping {name}: GROQ_API_KEY not set")
#             continue
#         t0 = time.perf_counter()
#         results[name] = evaluate(engine, items, cfg, k_values)
#         results[name]["wall_s"] = round(time.perf_counter() - t0, 1)
#         print(name, json.dumps(results[name]))

#     lines = [f"# Retrieval evaluation ({len(items)} questions, embedder: {engine.store.embedder.name})", "",
#              "| Config | " + " | ".join(f"Recall@{k}" for k in k_values) + " | MRR | " + " | ".join(f"nDCG@{k}" for k in k_values) + " | avg latency (ms) |",
#              "|---|" + "---|" * (2 * len(k_values) + 2)]
#     for name, r in results.items():
#         lines.append(f"| {name} | " + " | ".join(str(r["recall"][k]) for k in k_values) + f" | {r['mrr']} | "
#                      + " | ".join(str(r["ndcg"][k]) for k in k_values) + f" | {r['avg_latency_ms']} |")
#     lines += ["", "Misses (no relevant passage in top-k):"]
#     for name, r in results.items():
#         lines.append(f"- {name}: {', '.join(r['misses']) or 'none'}")
#     out_file.parent.mkdir(parents=True, exist_ok=True)
#     out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
#     out_file.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
#     print("\n".join(lines))
#     return results


"""Retrieval evaluation: Recall@k, MRR and nDCG@k for several retriever configurations.

A retrieved chunk counts as relevant when it comes from an expected (file, page) and, if given, contains the
expected snippet (whitespace/case-insensitive).
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

from rfp_intel.models import SearchFilters
from rfp_intel.search.engine import SearchEngine

CONFIGS: dict[str, dict[str, Any]] = {
    "vector": {"mode": "vector", "rerank_results": False},
    "keyword": {"mode": "keyword", "rerank_results": False},
    "hybrid": {"mode": "hybrid", "rerank_results": False},
    "hybrid_rerank": {"mode": "hybrid", "rerank_results": True},
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def is_relevant(result, expected: list[dict[str, Any]]) -> bool:
    text = _norm(result.text)
    for e in expected:
        if e["file"].lower() in result.file_name.lower() and int(e.get("page", result.page_number)) == result.page_number:
            snippet = e.get("contains")
            if not snippet or _norm(snippet) in text:
                return True
    return False


def evaluate(engine: SearchEngine, items: list[dict[str, Any]], config: dict[str, Any], k_values: list[int]) -> dict[str, Any]:
    max_k = max(k_values)
    recall = {k: 0 for k in k_values}
    ndcg = {k: 0.0 for k in k_values}
    rr_sum = 0.0
    misses: list[str] = []
    latencies: list[float] = []
    for item in items:
        filters = SearchFilters(**item.get("filters", {}))
        resp = engine.search(item["question"], filters=filters, top_k=max_k, **config)
        latencies.append(resp.latency_ms)
        rels = [is_relevant(r, item["expected"]) for r in resp.results]
        first = next((i for i, r in enumerate(rels) if r), None)
        if first is None:
            misses.append(item["id"])
        else:
            rr_sum += 1.0 / (first + 1)
        for k in k_values:
            if first is not None and first < k:
                recall[k] += 1
            top = rels[:k]
            dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(top) if r)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(sum(top)))  # ideal: all relevant hits ranked first
            ndcg[k] += dcg / idcg if idcg else 0.0
    n = len(items)
    return {
        "n": n,
        "recall": {k: round(v / n, 3) for k, v in recall.items()},
        "mrr": round(rr_sum / n, 3),
        "ndcg": {k: round(v / n, 3) for k, v in ndcg.items()},
        "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 1),
        "misses": misses,
    }


def run_eval(configs: list[str], k_values: list[int], eval_file: Path, out_file: Path) -> dict[str, Any]:
    engine = SearchEngine()
    data = json.loads(Path(eval_file).read_text(encoding="utf-8"))
    items = data["items"]
    results: dict[str, Any] = {}
    for name in configs:
        cfg = CONFIGS[name]
        if cfg.get("rerank_results") and not engine.llm.available:
            print(f"skipping {name}: no LLM API key configured")
            continue
        t0 = time.perf_counter()
        results[name] = evaluate(engine, items, cfg, k_values)
        results[name]["wall_s"] = round(time.perf_counter() - t0, 1)
        print(name, json.dumps(results[name]))

    lines = [f"# Retrieval evaluation ({len(items)} questions, embedder: {engine.store.embedder.name})", "",
             "| Config | " + " | ".join(f"Recall@{k}" for k in k_values) + " | MRR | " + " | ".join(f"nDCG@{k}" for k in k_values) + " | avg latency (ms) |",
             "|---|" + "---|" * (2 * len(k_values) + 2)]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(str(r["recall"][k]) for k in k_values) + f" | {r['mrr']} | "
                     + " | ".join(str(r["ndcg"][k]) for k in k_values) + f" | {r['avg_latency_ms']} |")
    lines += ["", "Misses (no relevant passage in top-k):"]
    for name, r in results.items():
        lines.append(f"- {name}: {', '.join(r['misses']) or 'none'}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_file.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return results