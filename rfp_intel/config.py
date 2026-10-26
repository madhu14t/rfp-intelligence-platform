"""Central configuration. Everything comes from environment variables (or a .env file).

No secrets or model names are hard-coded anywhere else in the code base.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env from the current working directory, if present


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# Provider presets: every provider here speaks the OpenAI chat-completions protocol.
# Select with LLM_PROVIDER (groq | gemini | nvidia | chutes | openai | openrouter | ollama); override any value with LLM_BASE_URL / LLM_MODEL / LLM_SMALL_MODEL / LLM_API_KEY.
PROVIDERS: dict[str, dict[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
        "small_model": "openai/gpt-oss-20b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-3-flash-preview",
        "small_model": "gemini-3.1-flash-lite",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
        "small_model": "gpt-4o-mini",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
        "model": "meta/llama-3.3-70b-instruct",
        "small_model": "meta/llama-3.1-8b-instruct",
    },
    "chutes": {
        "base_url": "https://llm.chutes.ai/v1",
        "key_env": "CHUTES_API_KEY",
        "model": "deepseek-ai/DeepSeek-V3.2",
        "small_model": "google/gemma-4-31b-it",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "small_model": "meta-llama/llama-3.1-8b-instruct",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": "OLLAMA_API_KEY",  # not needed by Ollama; any non-empty value works
        "model": "llama3.1",
        "small_model": "llama3.1",
    },
}


def _provider() -> dict[str, str]:
    name = _env("LLM_PROVIDER", "groq").lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER={name!r}; choose one of {sorted(PROVIDERS)}")
    return PROVIDERS[name]


def _api_key() -> str:
    p = _provider()
    key = os.getenv("LLM_API_KEY") or os.getenv(p["key_env"]) or ""
    if not key and _env("LLM_PROVIDER", "groq").lower() == "ollama":
        key = "ollama"
    return key


@dataclass
class Settings:
    # --- LLM (provider-agnostic, OpenAI-compatible) ---
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "groq").lower())
    llm_api_key: str = field(default_factory=_api_key)
    llm_key_env_name: str = field(default_factory=lambda: _provider()["key_env"])
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", _provider()["base_url"]))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", _provider()["model"]))
    llm_small_model: str = field(default_factory=lambda: _env("LLM_SMALL_MODEL", _provider()["small_model"]))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 90.0))
    llm_max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 6))

    @property
    def groq_api_key(self) -> str:  # backwards-compatible alias
        return self.llm_api_key

    # --- Embeddings ---
    # "fastembed" (default, local ONNX model) or "hash" (dependency-free fallback used in tests/offline)
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "fastembed"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))

    # --- Index / storage ---
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", "./index_store")))
    runs_dir: Path = field(default_factory=lambda: Path(_env("RUNS_DIR", "./runs")))
    outputs_dir: Path = field(default_factory=lambda: Path(_env("OUTPUTS_DIR", "./outputs")))

    # --- Chunking ---
    chunk_size_chars: int = field(default_factory=lambda: _env_int("CHUNK_SIZE_CHARS", 1800))
    chunk_overlap_chars: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_CHARS", 250))

    # --- Retrieval ---
    default_top_k: int = field(default_factory=lambda: _env_int("DEFAULT_TOP_K", 8))
    candidate_pool: int = field(default_factory=lambda: _env_int("CANDIDATE_POOL", 30))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    rerank_enabled: bool = field(default_factory=lambda: _env("RERANK_ENABLED", "true").lower() == "true")
    query_rewrite_llm: bool = field(default_factory=lambda: _env("QUERY_REWRITE_LLM", "false").lower() == "true")

    # --- Agents ---
    max_field_retries: int = field(default_factory=lambda: _env_int("MAX_FIELD_RETRIES", 2))
    # 2 keeps free-tier token-per-minute limits (Groq: 8k TPM) survivable; raise to 4-6 on paid tiers / Gemini
    agent_concurrency: int = field(default_factory=lambda: _env_int("AGENT_CONCURRENCY", 2))
    # rerank payload: candidates shown to the LLM judge and chars per candidate (keeps each call ~1.5k tokens)
    rerank_candidates: int = field(default_factory=lambda: _env_int("RERANK_CANDIDATES", 10))
    rerank_chars: int = field(default_factory=lambda: _env_int("RERANK_CHARS", 350))

    # --- OCR (bonus) ---
    ocr_enabled: bool = field(default_factory=lambda: _env("OCR_ENABLED", "false").lower() == "true")

    def ensure_dirs(self) -> None:
        for d in (self.index_dir, self.runs_dir, self.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()