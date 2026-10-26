"""Persistent search index: FAISS vector index + SQLite chunk/metadata store + in-memory BM25.

Incremental indexing: ``add_bid`` replaces only the chunks of that bid (FAISS ``remove_ids`` + SQLite delete),
so adding a third bid folder never re-embeds Bid1/Bid2.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from rfp_intel.config import settings
from rfp_intel.models import Chunk, SearchFilters
from rfp_intel.search.embeddings import Embedder, get_embedder

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-#./]*[a-z0-9]|[a-z0-9]")


def tokenize(text: str) -> list[str]:
    """BM25 tokenizer that keeps identifiers intact *and* adds their parts (JA-207652 -> ja-207652, ja, 207652)."""
    toks: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        toks.append(tok)
        if any(ch in tok for ch in "-#./"):
            toks.extend(p for p in re.split(r"[\-#./]", tok) if p)
    return toks


class IndexStore:
    def __init__(self, index_dir: Path | None = None, embedder: Embedder | None = None):
        self.dir = Path(index_dir or settings.index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or get_embedder()
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.dir / "chunks.sqlite", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._init_db()
        self._faiss_path = self.dir / "vectors.faiss"
        self._manifest_path = self.dir / "manifest.json"
        self.index = self._load_faiss()
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[int] = []
        self._meta_cache: dict[int, dict[str, Any]] = {}
        self._rebuild_memory()

    # ------------------------------------------------------------------ persistence
    def _init_db(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                int_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE,
                bid_id TEXT, file_name TEXT, doc_type TEXT, addendum_number INTEGER,
                page_number INTEGER, section TEXT, is_table INTEGER, document_date TEXT, text TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_bid ON chunks(bid_id);
            CREATE TABLE IF NOT EXISTS documents (
                bid_id TEXT, file_name TEXT, doc_type TEXT, addendum_number INTEGER, pages INTEGER,
                document_date TEXT, warnings TEXT, PRIMARY KEY (bid_id, file_name)
            );
            """
        )
        self._db.commit()

    def _load_faiss(self) -> faiss.Index:
        if self._faiss_path.exists() and self._manifest_path.exists():
            manifest = json.loads(self._manifest_path.read_text())
            if manifest.get("embedder") == self.embedder.name and manifest.get("dim") == self.embedder.dim:
                return faiss.read_index(str(self._faiss_path))
            log.warning("Embedding backend changed (%s -> %s); existing vectors will be rebuilt on next index.",
                        manifest.get("embedder"), self.embedder.name)
            self._db.execute("DELETE FROM chunks")
            self._db.execute("DELETE FROM documents")
            self._db.commit()
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedder.dim))

    def _save(self) -> None:
        faiss.write_index(self.index, str(self._faiss_path))
        self._manifest_path.write_text(json.dumps({"embedder": self.embedder.name, "dim": self.embedder.dim,
                                                   "model": getattr(self.embedder, "model_name", None)}))
        self._db.commit()

    def _rebuild_memory(self) -> None:
        rows = self._db.execute("SELECT * FROM chunks ORDER BY int_id").fetchall()
        self._meta_cache = {r["int_id"]: dict(r) for r in rows}
        self._bm25_ids = [r["int_id"] for r in rows]
        corpus = [tokenize(r["text"]) for r in rows]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        log.info("Index loaded: %d chunks, %d vectors", len(rows), self.index.ntotal)

    # ------------------------------------------------------------------ indexing
    def add_bid(self, bid_id: str, chunks: list[Chunk], documents: Iterable[Any] = ()) -> int:
        with self._lock:
            self.remove_bid(bid_id, save=False)
            if not chunks:
                self._save()
                self._rebuild_memory()
                return 0
            cur = self._db.cursor()
            int_ids: list[int] = []
            for c in chunks:
                cur.execute(
                    "INSERT INTO chunks (chunk_id, bid_id, file_name, doc_type, addendum_number, page_number, section, is_table, document_date, text)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (c.chunk_id, c.bid_id, c.file_name, c.doc_type, c.addendum_number, c.page_number, c.section,
                     int(c.is_table), c.document_date, c.text),
                )
                int_ids.append(cur.lastrowid)
            for d in documents:
                cur.execute(
                    "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?)",
                    (d.bid_id, d.file_name, d.doc_type, d.addendum_number, len(d.pages), d.document_date,
                     json.dumps(d.parse_warnings)),
                )
            vecs = self.embedder.embed_documents([c.text for c in chunks])
            self.index.add_with_ids(vecs.astype(np.float32), np.asarray(int_ids, dtype=np.int64))
            self._save()
            self._rebuild_memory()
            return len(chunks)

    def remove_bid(self, bid_id: str, save: bool = True) -> int:
        with self._lock:
            rows = self._db.execute("SELECT int_id FROM chunks WHERE bid_id=?", (bid_id,)).fetchall()
            ids = np.asarray([r["int_id"] for r in rows], dtype=np.int64)
            if len(ids):
                self.index.remove_ids(ids)
                self._db.execute("DELETE FROM chunks WHERE bid_id=?", (bid_id,))
                self._db.execute("DELETE FROM documents WHERE bid_id=?", (bid_id,))
            if save:
                self._save()
                self._rebuild_memory()
            return len(ids)

    # ------------------------------------------------------------------ querying
    def _allowed_ids(self, filters: SearchFilters | None) -> list[int] | None:
        if filters is None or not any([filters.bid_id, filters.doc_type, filters.addendum_number is not None, filters.file_name]):
            return None
        return [i for i, m in self._meta_cache.items() if filters.matches(m)]

    def vector_search(self, query: str, k: int, filters: SearchFilters | None = None) -> list[tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        q = self.embedder.embed_query(query).astype(np.float32)[None, :]
        allowed = self._allowed_ids(filters)
        if allowed is None:
            scores, ids = self.index.search(q, min(k, self.index.ntotal))
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
        if not allowed:
            return []
        sel = faiss.IDSelectorBatch(np.asarray(allowed, dtype=np.int64))
        params = faiss.SearchParameters(sel=sel)
        scores, ids = self.index.search(q, min(k, len(allowed)), params=params)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def keyword_search(self, query: str, k: int, filters: SearchFilters | None = None) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        allowed = self._allowed_ids(filters)
        allowed_set = set(allowed) if allowed is not None else None
        order = np.argsort(-scores)
        out: list[tuple[int, float]] = []
        for pos in order:
            iid = self._bm25_ids[pos]
            if scores[pos] <= 0:
                break
            if allowed_set is not None and iid not in allowed_set:
                continue
            out.append((iid, float(scores[pos])))
            if len(out) >= k:
                break
        return out

    def get(self, int_id: int) -> dict[str, Any]:
        return self._meta_cache[int_id]

    def get_by_chunk_id(self, chunk_id: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ introspection
    def stats(self) -> dict[str, Any]:
        bids = self._db.execute("SELECT bid_id, COUNT(*) c FROM chunks GROUP BY bid_id").fetchall()
        docs = self._db.execute("SELECT * FROM documents ORDER BY bid_id, file_name").fetchall()
        return {
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "total_chunks": self.index.ntotal,
            "bids": {r["bid_id"]: r["c"] for r in bids},
            "documents": [dict(d) for d in docs],
        }

    def list_bids(self) -> list[str]:
        return [r[0] for r in self._db.execute("SELECT DISTINCT bid_id FROM chunks ORDER BY bid_id")]

    def documents_for(self, bid_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._db.execute("SELECT * FROM documents WHERE bid_id=? ORDER BY file_name", (bid_id,))]

    def close(self) -> None:
        self._db.close()
