"""Pluggable embedding backends.

* ``fastembed`` (default): local ONNX sentence embeddings (BAAI/bge-small-en-v1.5, 384-d). No GPU, no API key.
* ``hash``: dependency-free hashed character-n-gram embeddings. Used by the unit tests and as an automatic
  fallback when the fastembed model cannot be downloaded (offline machines). Quality is lower than bge but
  the whole pipeline (hybrid search, agents) keeps working.
"""
from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod

import numpy as np

from rfp_intel.config import settings

log = logging.getLogger(__name__)


class Embedder(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...


class HashEmbedder(Embedder):
    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        text = text.lower()
        tokens = re.findall(r"[a-z0-9][a-z0-9\-#./]*", text)
        feats: list[str] = []
        for tok in tokens:
            feats.append("w:" + tok)
            padded = f" {tok} "
            feats.extend("g:" + padded[i : i + 3] for i in range(len(padded) - 2))
        for i in range(len(tokens) - 1):
            feats.append("b:" + tokens[i] + "_" + tokens[i + 1])
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            sign = 1.0 if (h >> 1) & 1 else -1.0
            v[h % self.dim] += sign
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class FastEmbedEmbedder(Embedder):
    name = "fastembed"

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # imported lazily so the hash backend has no heavy deps

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))
        # bge models recommend a query instruction prefix
        self._query_prefix = "Represent this sentence for searching relevant passages: " if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = np.asarray(list(self._model.embed(texts, batch_size=32)), dtype=np.float32)
        return _normalise(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        vec = np.asarray(next(iter(self._model.query_embed([self._query_prefix + text]))), dtype=np.float32)
        return _normalise(vec[None, :])[0]


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


_cache: dict[str, Embedder] = {}


def get_embedder(backend: str | None = None, model: str | None = None) -> Embedder:
    backend = (backend or settings.embedding_backend).lower()
    model = model or settings.embedding_model
    key = f"{backend}:{model}"
    if key in _cache:
        return _cache[key]
    if backend == "fastembed":
        try:
            emb: Embedder = FastEmbedEmbedder(model)
        except Exception as exc:  # offline / model download failure -> keep the system usable
            log.warning("fastembed backend unavailable (%s). Falling back to hash embeddings.", exc)
            emb = HashEmbedder()
    elif backend == "hash":
        emb = HashEmbedder()
    else:
        raise ValueError(f"Unknown EMBEDDING_BACKEND={backend!r} (expected 'fastembed' or 'hash')")
    _cache[key] = emb
    return emb
