# """LLM-based listwise re-ranker (Groq, small fast model).

# The candidates are shown once with short ids; the model returns a relevance score 0-10 per id.
# Final score = 0.7 * llm_score/10 + 0.3 * fused_rank_score, so ties/garbage from the LLM cannot fully
# override a strong hybrid signal. Falls back to the input order on any error.
# """
# from __future__ import annotations

# import logging

# from rfp_intel.config import settings
# from rfp_intel.llm.client import GroqLLM, LLMError
# from rfp_intel.models import SearchResult

# log = logging.getLogger(__name__)

# _PROMPT = """You are a relevance judge for a procurement (RFP / bid) document search engine.
# Score how well each passage answers the query. 10 = directly contains the answer, 5 = related context,
# 0 = irrelevant. Prefer passages with the exact values asked for (dates, numbers, names, model numbers).
# Return JSON only: {"scores": {"<id>": <0-10>, ...}}"""


# def rerank(query: str, results: list[SearchResult], llm: GroqLLM, top_k: int, model: str | None = None,
#            max_chars: int | None = None) -> list[SearchResult]:
#     if not results or not llm.available:
#         return results[:top_k]
#     max_chars = max_chars or settings.rerank_chars
#     results = results[: settings.rerank_candidates]
#     lines = []
#     for i, r in enumerate(results):
#         snippet = r.text.replace("\n", " ")[:max_chars]
#         lines.append(f"[{i}] ({r.file_name} p.{r.page_number}) {snippet}")
#     user = f"Query: {query}\n\nPassages:\n" + "\n\n".join(lines)
#     try:
#         resp = llm.complete(
#             [{"role": "system", "content": _PROMPT}, {"role": "user", "content": user}],
#             model=model or settings.llm_small_model, json_mode=True, max_tokens=400,
#         )
#         scores = resp.json().get("scores", {})
#     except (LLMError, Exception) as exc:
#         log.warning("Reranker failed, keeping hybrid order: %s", exc)
#         return results[:top_k]

#     n = len(results)
#     rescored: list[SearchResult] = []
#     for i, r in enumerate(results):
#         try:
#             llm_s = float(scores.get(str(i), 0)) / 10.0
#         except (TypeError, ValueError):
#             llm_s = 0.0
#         rank_s = 1.0 - i / max(n, 1)
#         r2 = r.model_copy(update={"score": round(0.7 * llm_s + 0.3 * rank_s, 4)})
#         r2.retrievers = list(r.retrievers) + ["rerank"]
#         rescored.append(r2)
#     rescored.sort(key=lambda x: x.score, reverse=True)
#     return rescored[:top_k]


"""LLM-based listwise re-ranker (Groq, small fast model).

The candidates are shown once with short ids; the model returns a relevance score 0-10 per id.
Final score = 0.7 * llm_score/10 + 0.3 * fused_rank_score, so ties/garbage from the LLM cannot fully
override a strong hybrid signal. Falls back to the input order on any error.
"""
from __future__ import annotations

import logging

from rfp_intel.config import settings
from rfp_intel.llm.client import GroqLLM, LLMError
from rfp_intel.models import SearchResult

log = logging.getLogger(__name__)

_PROMPT = """You are a relevance judge for a procurement (RFP / bid) document search engine.
Score how well each passage answers the query. 10 = directly contains the answer, 5 = related context,
0 = irrelevant. Prefer passages with the exact values asked for (dates, numbers, names, model numbers).
Return JSON only: {"scores": {"<id>": <0-10>, ...}}"""


def rerank(query: str, results: list[SearchResult], llm: GroqLLM, top_k: int, model: str | None = None,
           max_chars: int | None = None) -> list[SearchResult]:
    if not results or not llm.available:
        return results[:top_k]
    max_chars = max_chars or settings.rerank_chars
    results = results[: settings.rerank_candidates]
    lines = []
    for i, r in enumerate(results):
        snippet = r.text.replace("\n", " ")[:max_chars]
        lines.append(f"[{i}] ({r.file_name} p.{r.page_number}) {snippet}")
    user = f"Query: {query}\n\nPassages:\n" + "\n\n".join(lines)
    try:
        resp = llm.complete(
            [{"role": "system", "content": _PROMPT}, {"role": "user", "content": user}],
            model=model or settings.llm_small_model, json_mode=True, max_tokens=1200,
        )
        scores = resp.json().get("scores", {})
    except (LLMError, Exception) as exc:
        log.warning("Reranker failed, keeping hybrid order: %s", exc)
        return results[:top_k]

    n = len(results)
    rescored: list[SearchResult] = []
    for i, r in enumerate(results):
        try:
            llm_s = float(scores.get(str(i), 0)) / 10.0
        except (TypeError, ValueError):
            llm_s = 0.0
        rank_s = 1.0 - i / max(n, 1)
        r2 = r.model_copy(update={"score": round(0.7 * llm_s + 0.3 * rank_s, 4)})
        r2.retrievers = list(r.retrievers) + ["rerank"]
        rescored.append(r2)
    rescored.sort(key=lambda x: x.score, reverse=True)
    return rescored[:top_k]