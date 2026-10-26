from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("EMBEDDING_BACKEND", "hash")  # tests never download models
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from rfp_intel.search.embeddings import HashEmbedder  # noqa: E402
from rfp_intel.search.engine import SearchEngine  # noqa: E402
from rfp_intel.search.store import IndexStore  # noqa: E402

DATA = ROOT / "data"


@pytest.fixture(scope="session")
def engine(tmp_path_factory) -> SearchEngine:
    """Engine over a temp index with the two provided bids (skipped if data folder is missing)."""
    if not (DATA / "Bid1").exists():
        pytest.skip("data/Bid1 not present")
    store = IndexStore(tmp_path_factory.mktemp("index"), embedder=HashEmbedder())
    eng = SearchEngine(store)
    eng.index_folder(DATA / "Bid1")
    eng.index_folder(DATA / "Bid2")
    return eng
