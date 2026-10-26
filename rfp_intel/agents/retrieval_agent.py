# """Retrieval / Search Agent: rewrites and expands queries, calls the search engine with metadata filters,
# applies re-ranking and returns de-duplicated evidence with citations."""
# from __future__ import annotations

# import asyncio

# from rfp_intel.agents.base import Agent
# from rfp_intel.agents.fields import FieldSpec
# from rfp_intel.models import SearchFilters, SearchResult


# class RetrievalAgent(Agent):
#     name = "retrieval_agent"

#     def _merge(self, *lists: list[SearchResult], limit: int) -> list[SearchResult]:
#         seen: set[str] = set()
#         out: list[SearchResult] = []
#         for lst in lists:
#             for r in lst:
#                 if r.chunk_id not in seen:
#                     seen.add(r.chunk_id)
#                     out.append(r)
#         return out[:limit]

#     def evidence_for_field(self, bid_id: str, spec: FieldSpec, top_k: int = 6, feedback: str | None = None) -> list[SearchResult]:
#         queries = list(spec.queries)
#         if feedback:
#             queries.insert(0, feedback)  # validator feedback becomes an extra query on retry
#         tool_calls = []
#         with self.tracer.step(self.name, f"evidence:{spec.name}", input={"bid_id": bid_id, "queries": queries}) as ctx:
#             primary = self.engine.search(queries[0], filters=SearchFilters(bid_id=bid_id), top_k=top_k,
#                                          extra_queries=queries[1:])
#             tool_calls.append({"tool": "search", "query": queries[0], "extra": queries[1:], "mode": primary.mode,
#                                "hits": len(primary.results), "latency_ms": primary.latency_ms})
#             lists = [primary.results]
#             # Preferred doc types get a dedicated, filtered pass so that e.g. specs tables or bid-page tables are
#             # not crowded out by the long RFP body.
#             if spec.prefer_doc_types:
#                 typed = self.engine.search(queries[0], filters=SearchFilters(bid_id=bid_id, doc_type=spec.prefer_doc_types),
#                                            top_k=max(3, top_k // 2), extra_queries=queries[1:], rerank_results=False)
#                 tool_calls.append({"tool": "search", "filter_doc_type": spec.prefer_doc_types, "hits": len(typed.results)})
#                 lists.append(typed.results)
#             if spec.addendum_sensitive:
#                 add = self.engine.search(queries[0], filters=SearchFilters(bid_id=bid_id, doc_type=["addendum"]), top_k=3,
#                                          extra_queries=queries[1:], rerank_results=False)
#                 tool_calls.append({"tool": "search", "filter_doc_type": ["addendum"], "hits": len(add.results)})
#                 lists.append(add.results)
#             merged = self._merge(*lists, limit=top_k + 4)
#             ctx["tool_calls"] = tool_calls
#             ctx["output"] = [{"file": r.file_name, "page": r.page_number, "score": r.score} for r in merged]
#             return merged

#     async def aevidence_for_field(self, bid_id: str, spec: FieldSpec, top_k: int = 6, feedback: str | None = None) -> list[SearchResult]:
#         return await asyncio.to_thread(self.evidence_for_field, bid_id, spec, top_k, feedback)

#     def evidence_for_question(self, question: str, bid_ids: list[str] | None, top_k: int = 8,
#                               doc_types: list[str] | None = None) -> list[SearchResult]:
#         with self.tracer.step(self.name, "evidence:question", input={"question": question, "bid_ids": bid_ids, "doc_types": doc_types}) as ctx:
#             results: list[SearchResult] = []
#             calls = []
#             targets = bid_ids or [None]
#             per_bid = top_k if len(targets) == 1 else max(4, top_k // len(targets))
#             for b in targets:
#                 f = SearchFilters(bid_id=b, doc_type=doc_types)  # type: ignore[arg-type]
#                 resp = self.engine.search(question, filters=f, top_k=per_bid, use_llm_rewrite=True)
#                 calls.append({"tool": "search", "bid_id": b, "mode": resp.mode, "rewritten": resp.rewritten_queries,
#                               "hits": len(resp.results), "latency_ms": resp.latency_ms})
#                 results.extend(resp.results)
#             merged = self._merge(results, limit=top_k * max(1, len(targets)))
#             ctx["tool_calls"] = calls
#             ctx["output"] = [{"bid": r.bid_id, "file": r.file_name, "page": r.page_number} for r in merged]
#             return merged


"""Retrieval / Search Agent: rewrites and expands queries, calls the search engine with metadata filters,
applies re-ranking and returns de-duplicated evidence with citations."""
from __future__ import annotations

import asyncio

from rfp_intel.agents.base import Agent
from rfp_intel.agents.fields import FieldSpec
from rfp_intel.models import SearchFilters, SearchResult


class RetrievalAgent(Agent):
    name = "retrieval_agent"

    def _merge(self, *lists: list[SearchResult], limit: int) -> list[SearchResult]:
        seen: set[str] = set()
        out: list[SearchResult] = []
        for lst in lists:
            for r in lst:
                if r.chunk_id not in seen:
                    seen.add(r.chunk_id)
                    out.append(r)
        return out[:limit]

    def evidence_for_field(self, bid_id: str, spec: FieldSpec, top_k: int = 6, feedback: str | None = None) -> list[SearchResult]:
        queries = list(spec.queries)
        if feedback:
            queries.insert(0, feedback)  # validator feedback becomes an extra query on retry
        tool_calls = []
        with self.tracer.step(self.name, f"evidence:{spec.name}", input={"bid_id": bid_id, "queries": queries}) as ctx:
            primary = self.engine.search(queries[0], filters=SearchFilters(bid_id=bid_id), top_k=top_k,
                                         extra_queries=queries[1:])
            tool_calls.append({"tool": "search", "query": queries[0], "extra": queries[1:], "mode": primary.mode,
                               "hits": len(primary.results), "latency_ms": primary.latency_ms})
            lists = [primary.results]
            # Preferred doc types get a dedicated, filtered pass so that e.g. specs tables or bid-page tables are
            # not crowded out by the long RFP body.
            if spec.prefer_doc_types:
                typed = self.engine.search(queries[0], filters=SearchFilters(bid_id=bid_id, doc_type=spec.prefer_doc_types),
                                           top_k=max(3, top_k // 2), extra_queries=queries[1:], rerank_results=False)
                tool_calls.append({"tool": "search", "filter_doc_type": spec.prefer_doc_types, "hits": len(typed.results)})
                lists.append(typed.results)
            if spec.addendum_sensitive:
                add = self.engine.search(queries[0], filters=SearchFilters(bid_id=bid_id, doc_type=["addendum"]), top_k=3,
                                         extra_queries=queries[1:], rerank_results=False)
                tool_calls.append({"tool": "search", "filter_doc_type": ["addendum"], "hits": len(add.results)})
                lists.append(add.results)
            merged = self._merge(*lists, limit=top_k + 4)
            ctx["tool_calls"] = tool_calls
            ctx["output"] = [{"file": r.file_name, "page": r.page_number, "score": r.score} for r in merged]
            return merged

    async def aevidence_for_field(self, bid_id: str, spec: FieldSpec, top_k: int = 6, feedback: str | None = None) -> list[SearchResult]:
        return await asyncio.to_thread(self.evidence_for_field, bid_id, spec, top_k, feedback)

    def evidence_for_question(self, question: str, bid_ids: list[str] | None, top_k: int = 8,
                              doc_types: list[str] | None = None) -> list[SearchResult]:
        with self.tracer.step(self.name, "evidence:question", input={"question": question, "bid_ids": bid_ids, "doc_types": doc_types}) as ctx:
            results: list[SearchResult] = []
            calls = []
            targets = bid_ids or [None]
            per_bid = top_k if len(targets) == 1 else max(6, top_k // len(targets))
            for b in targets:
                f = SearchFilters(bid_id=b, doc_type=doc_types)  # type: ignore[arg-type]
                resp = self.engine.search(question, filters=f, top_k=per_bid, use_llm_rewrite=True)
                calls.append({"tool": "search", "bid_id": b, "mode": resp.mode, "rewritten": resp.rewritten_queries,
                              "hits": len(resp.results), "latency_ms": resp.latency_ms})
                results.extend(resp.results)
            merged = self._merge(results, limit=top_k * max(1, len(targets)))
            ctx["tool_calls"] = calls
            ctx["output"] = [{"bid": r.bid_id, "file": r.file_name, "page": r.page_number} for r in merged]
            return merged