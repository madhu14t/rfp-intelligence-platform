# """The RAG search engine: hybrid (dense + BM25) retrieval with RRF fusion, metadata filters, query
# expansion and optional LLM re-ranking. Usable on its own (CLI/API) and as the tool the agents call.
# """
# from __future__ import annotations

# import logging
# import time
# from pathlib import Path
# from typing import Literal

# from rfp_intel.config import settings
# from rfp_intel.ingestion.pipeline import ingest_folder
# from rfp_intel.llm.client import GroqLLM, get_llm
# from rfp_intel.models import SearchFilters, SearchResponse, SearchResult
# from rfp_intel.search.query import expand_query, llm_rewrite
# from rfp_intel.search.reranker import rerank
# from rfp_intel.search.store import IndexStore

# log = logging.getLogger(__name__)

# Mode = Literal["hybrid", "vector", "keyword"]


# class SearchEngine:
#     def __init__(self, store: IndexStore | None = None, llm: GroqLLM | None = None):
#         self.store = store or IndexStore()
#         self.llm = llm or get_llm()

#     # ------------------------------------------------------------------ indexing
#     def index_folder(self, folder: str | Path, bid_id: str | None = None) -> dict:
#         folder = Path(folder)
#         bid_id = bid_id or folder.name
#         t0 = time.perf_counter()
#         docs, chunks = ingest_folder(folder, bid_id)
#         n = self.store.add_bid(bid_id, chunks, docs)
#         return {
#             "bid_id": bid_id,
#             "documents": [{"file": d.file_name, "doc_type": d.doc_type, "addendum_number": d.addendum_number,
#                            "pages": len(d.pages), "warnings": d.parse_warnings} for d in docs],
#             "chunks_indexed": n,
#             "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
#         }

#     # ------------------------------------------------------------------ search
#     def search(self, query: str, *, filters: SearchFilters | None = None, top_k: int | None = None,
#                mode: Mode = "hybrid", rerank_results: bool | None = None, use_llm_rewrite: bool | None = None,
#                extra_queries: list[str] | None = None) -> SearchResponse:
#         t0 = time.perf_counter()
#         top_k = top_k or settings.default_top_k
#         filters = filters or SearchFilters()
#         pool = max(settings.candidate_pool, top_k * 3)
#         do_rerank = settings.rerank_enabled if rerank_results is None else rerank_results
#         do_rewrite = settings.query_rewrite_llm if use_llm_rewrite is None else use_llm_rewrite

#         queries = [query] + list(extra_queries or [])
#         if do_rewrite and self.llm.available:
#             queries += [q for q in llm_rewrite(query, self.llm, settings.llm_small_model) if q not in queries]
#         rewritten = queries[1:]

#         ranked_lists: list[list[int]] = []
#         hits: dict[int, SearchResult] = {}

#         def _collect(pairs: list[tuple[int, float]], source: str) -> None:
#             ids: list[int] = []
#             for iid, score in pairs:
#                 ids.append(iid)
#                 if iid not in hits:
#                     hits[iid] = self._to_result(iid, score)
#                 if source not in hits[iid].retrievers:
#                     hits[iid].retrievers.append(source)
#             ranked_lists.append(ids)

#         for q in queries:
#             if mode in ("hybrid", "vector"):
#                 _collect(self.store.vector_search(q, pool, filters), "vector")
#             if mode in ("hybrid", "keyword"):
#                 _collect(self.store.keyword_search(expand_query(q), pool, filters), "bm25")

#         fused = self._rrf(ranked_lists, settings.rrf_k)
#         results = []
#         for iid, score in fused[:pool]:
#             r = hits[iid]
#             r.score = round(score, 5)
#             results.append(r)

#         if do_rerank and mode == "hybrid" and self.llm.available and results:
#             results = rerank(query, results[: min(len(results), 20)], self.llm, top_k)
#             eff_mode = "hybrid+rerank"
#         else:
#             results = results[:top_k]
#             eff_mode = mode

#         return SearchResponse(
#             query=query, rewritten_queries=rewritten, filters=filters, results=results,
#             latency_ms=round((time.perf_counter() - t0) * 1000, 1), mode=eff_mode,
#         )

#     @staticmethod
#     def _rrf(ranked_lists: list[list[int]], k: int) -> list[tuple[int, float]]:
#         scores: dict[int, float] = {}
#         for lst in ranked_lists:
#             for rank, iid in enumerate(lst):
#                 scores[iid] = scores.get(iid, 0.0) + 1.0 / (k + rank + 1)
#         return sorted(scores.items(), key=lambda x: x[1], reverse=True)

#     def _to_result(self, iid: int, score: float) -> SearchResult:
#         m = self.store.get(iid)
#         return SearchResult(
#             chunk_id=m["chunk_id"], score=score, text=m["text"], bid_id=m["bid_id"], file_name=m["file_name"],
#             doc_type=m["doc_type"], addendum_number=m["addendum_number"], page_number=m["page_number"],
#             section=m["section"], is_table=bool(m["is_table"]),
#         )

#     # ------------------------------------------------------------------ helpers
#     def list_bids(self) -> list[str]:
#         return self.store.list_bids()

#     def stats(self) -> dict:
#         return self.store.stats()

"""The RAG search engine: hybrid (dense + BM25) retrieval with RRF fusion, metadata filters, query
expansion and optional LLM re-ranking. Usable on its own (CLI/API) and as the tool the agents call.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

from rfp_intel.config import settings
from rfp_intel.ingestion.pipeline import ingest_folder
from rfp_intel.llm.client import GroqLLM, get_llm
from rfp_intel.models import SearchFilters, SearchResponse, SearchResult
from rfp_intel.search.query import expand_query, llm_rewrite
from rfp_intel.search.reranker import rerank
from rfp_intel.search.store import IndexStore

log = logging.getLogger(__name__)

Mode = Literal["hybrid", "vector", "keyword"]


class SearchEngine:
    def __init__(self, store: IndexStore | None = None, llm: GroqLLM | None = None):
        self.store = store or IndexStore()
        self.llm = llm or get_llm()

    # ------------------------------------------------------------------ indexing
    def index_folder(self, folder: str | Path, bid_id: str | None = None) -> dict:
        folder = Path(folder)
        bid_id = bid_id or folder.name
        t0 = time.perf_counter()
        docs, chunks = ingest_folder(folder, bid_id)
        n = self.store.add_bid(bid_id, chunks, docs)
        return {
            "bid_id": bid_id,
            "documents": [{"file": d.file_name, "doc_type": d.doc_type, "addendum_number": d.addendum_number,
                           "pages": len(d.pages), "warnings": d.parse_warnings} for d in docs],
            "chunks_indexed": n,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # ------------------------------------------------------------------ search
    def search(self, query: str, *, filters: SearchFilters | None = None, top_k: int | None = None,
               mode: Mode = "hybrid", rerank_results: bool | None = None, use_llm_rewrite: bool | None = None,
               extra_queries: list[str] | None = None) -> SearchResponse:
        t0 = time.perf_counter()
        top_k = top_k or settings.default_top_k
        filters = filters or SearchFilters()
        pool = max(settings.candidate_pool, top_k * 3)
        do_rerank = settings.rerank_enabled if rerank_results is None else rerank_results
        do_rewrite = settings.query_rewrite_llm if use_llm_rewrite is None else use_llm_rewrite

        queries = [query] + list(extra_queries or [])
        if do_rewrite and self.llm.available:
            queries += [q for q in llm_rewrite(query, self.llm, settings.llm_small_model) if q not in queries]
        rewritten = queries[1:]

        ranked_lists: list[list[int]] = []
        hits: dict[int, SearchResult] = {}

        def _collect(pairs: list[tuple[int, float]], source: str) -> None:
            ids: list[int] = []
            for iid, score in pairs:
                ids.append(iid)
                if iid not in hits:
                    hits[iid] = self._to_result(iid, score)
                if source not in hits[iid].retrievers:
                    hits[iid].retrievers.append(source)
            ranked_lists.append(ids)

        for q in queries:
            if mode in ("hybrid", "vector"):
                _collect(self.store.vector_search(q, pool, filters), "vector")
            if mode in ("hybrid", "keyword"):
                _collect(self.store.keyword_search(expand_query(q), pool, filters), "bm25")

        fused = self._rrf(ranked_lists, settings.rrf_k)
        results = []
        for iid, score in fused[:pool]:
            r = hits[iid]
            r.score = round(score, 5)
            results.append(r)

        if do_rerank and mode == "hybrid" and self.llm.available and results:
            results = rerank(query, results, self.llm, top_k)
            eff_mode = "hybrid+rerank"
        else:
            results = results[:top_k]
            eff_mode = mode

        return SearchResponse(
            query=query, rewritten_queries=rewritten, filters=filters, results=results,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1), mode=eff_mode,
        )

    @staticmethod
    def _rrf(ranked_lists: list[list[int]], k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for lst in ranked_lists:
            for rank, iid in enumerate(lst):
                scores[iid] = scores.get(iid, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _to_result(self, iid: int, score: float) -> SearchResult:
        m = self.store.get(iid)
        return SearchResult(
            chunk_id=m["chunk_id"], score=score, text=m["text"], bid_id=m["bid_id"], file_name=m["file_name"],
            doc_type=m["doc_type"], addendum_number=m["addendum_number"], page_number=m["page_number"],
            section=m["section"], is_table=bool(m["is_table"]),
        )

    # ------------------------------------------------------------------ helpers
    def list_bids(self) -> list[str]:
        return self.store.list_bids()

    def stats(self) -> dict:
        return self.store.stats()