from __future__ import annotations

from rfp_intel.models import Chunk, SearchFilters
from rfp_intel.search.embeddings import HashEmbedder
from rfp_intel.search.engine import SearchEngine
from rfp_intel.search.query import expand_query
from rfp_intel.search.store import IndexStore, tokenize


def _chunk(cid, bid, text, doc_type="rfp", page=1, add=None, file="rfp.pdf"):
    return Chunk(chunk_id=cid, bid_id=bid, file_name=file, doc_type=doc_type, addendum_number=add, page_number=page, text=text)


def _store(tmp_path):
    store = IndexStore(tmp_path / "idx", embedder=HashEmbedder())
    store.add_bid("A", [
        _chunk("a1", "A", "Solicitation JA-207652 Student and Staff Computing Devices. Solicitation due 27-JUN-2024 14:00."),
        _chunk("a2", "A", "ADDENDUM No. 2. The new due date for this RFP will be July 9, 2024 at 2:00 PM CST.", "addendum", 1, 2, "Addendum 2.pdf"),
        _chunk("a3", "A", "Insurance and/or bond requirements are enumerated elsewhere in Contract documents.", page=17),
    ])
    store.add_bid("B", [
        _chunk("b1", "B", "Dell Thunderbolt 4 Dock WD22TB4 qty 30. Delivery within 45 days of Award.", page=3),
        _chunk("b2", "B", "256 GB, M.2 2230, TLC, Gen 4 PCIe NVMe, SSD 400-BRFT", "specs", 1, file="Dell_Laptop_Specs.pdf"),
    ])
    return store


def test_tokenizer_keeps_identifiers_and_parts():
    toks = tokenize("Part 400-BRFT and RFP JA-207652")
    assert "400-brft" in toks and "brft" in toks and "ja-207652" in toks and "207652" in toks


def test_expand_query_adds_synonyms():
    q = expand_query("what is the deadline")
    assert "due date" in q and "closing date" in q
    assert expand_query("WD22TB4") == "WD22TB4"  # no false expansion


def test_keyword_search_exact_identifier(tmp_path):
    store = _store(tmp_path)
    hits = store.keyword_search("400-BRFT", k=3)
    assert hits and store.get(hits[0][0])["chunk_id"] == "b2"


def test_metadata_filters(tmp_path):
    store = _store(tmp_path)
    eng = SearchEngine(store)
    r = eng.search("due date", filters=SearchFilters(bid_id="A"), top_k=5, rerank_results=False)
    assert r.results and all(x.bid_id == "A" for x in r.results)
    r = eng.search("due date", filters=SearchFilters(bid_id="A", doc_type=["addendum"]), top_k=5, rerank_results=False)
    assert [x.chunk_id for x in r.results] == ["a2"]
    r = eng.search("due date", filters=SearchFilters(addendum_number=2), top_k=5, rerank_results=False)
    assert [x.chunk_id for x in r.results] == ["a2"]
    r = eng.search("anything", filters=SearchFilters(bid_id="ZZZ"), top_k=5, rerank_results=False)
    assert r.results == []


def test_hybrid_fuses_both_retrievers(tmp_path):
    eng = SearchEngine(_store(tmp_path))
    r = eng.search("new due date July 9", filters=SearchFilters(bid_id="A"), top_k=3, rerank_results=False)
    assert r.results[0].chunk_id == "a2"
    assert set(r.results[0].retrievers) >= {"vector", "bm25"}
    assert r.mode == "hybrid"
    assert r.results[0].citation().page == 1 and r.results[0].citation().file == "Addendum 2.pdf"


def test_rrf_math():
    fused = SearchEngine._rrf([[1, 2, 3], [3, 1]], k=60)
    scores = dict(fused)
    assert fused[0][0] == 1  # ranked 1st and 2nd
    assert scores[1] > scores[3] > scores[2]


def test_incremental_indexing_replaces_only_that_bid(tmp_path):
    store = _store(tmp_path)
    assert store.stats()["bids"] == {"A": 3, "B": 2}
    store.add_bid("B", [_chunk("b9", "B", "Completely new content about laptops")])
    assert store.stats()["bids"] == {"A": 3, "B": 1}
    assert store.index.ntotal == 4
    store.remove_bid("A")
    assert store.list_bids() == ["B"]
    # persistence: reopen
    store2 = IndexStore(tmp_path / "idx", embedder=HashEmbedder())
    assert store2.stats()["bids"] == {"B": 1}


def test_real_bids_search_quality(engine):
    r = engine.search("new due date", filters=SearchFilters(bid_id="Bid1", doc_type=["addendum"]), top_k=1, rerank_results=False)
    assert "July 9, 2024" in r.results[0].text
    r = engine.search("400-BRFT", filters=SearchFilters(bid_id="Bid2"), top_k=1, rerank_results=False)
    assert r.results[0].file_name == "Dell_Laptop_Specs.pdf"
    r = engine.search("Mercury Affidavit required", filters=SearchFilters(bid_id="Bid2", doc_type=["rfp"]), top_k=2, rerank_results=False)
    assert any("Mercury Affidavit" in x.text for x in r.results)
    stats = engine.stats()
    assert set(stats["bids"]) == {"Bid1", "Bid2"}
