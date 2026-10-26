"""Provider-agnostic LLM client (OpenAI-compatible chat completions).

Works with Groq, Google Gemini, OpenAI, Ollama or any OpenAI-compatible endpoint - selected with
LLM_PROVIDER in .env (see config.py). Features: retries that honour rate-limit "retry after" hints, fail-fast on
model-not-found / bad key, JSON-mode fallback, lenient JSON parsing, token + latency accounting.
Both sync and async entry points are provided; the agents use the async one so field groups run concurrently.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from rfp_intel.config import settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)

    def add(self, model: str, usage: Any, latency_ms: float) -> None:
        self.calls += 1
        self.latency_ms += latency_ms
        self.by_model[model] = self.by_model.get(model, 0) + 1
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "by_model": self.by_model,
        }


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float

    def json(self) -> Any:
        return parse_json(self.text)


_BARE_EID_RE = re.compile(r'(?<=[\[,:\s])(E\d+)(?=\s*[,\]}:])')


def _repair(text: str) -> str:
    """Fix the common small-model mistakes: bare evidence ids (E8 -> "E8"), trailing commas."""
    text = _BARE_EID_RE.sub(r'"\1"', text)
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return text


def parse_json(text: str) -> Any:
    """Parse JSON from an LLM reply, tolerating <thought> blocks, code fences, prose and small syntax slips."""
    text = text.strip()
    # thinking models (Gemma, Gemini lite) may prepend their reasoning
    text = re.sub(r"<(thought|think|thinking|reasoning)>.*?</\1>", "", text, flags=re.S).strip()
    text = re.sub(r"^<(thought|think|thinking|reasoning)>.*?(?=\{|\[|```)", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    for attempt in (text, _repair(text)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    # scan every '{' / '[' and keep the LAST complete JSON value (reasoning text may contain stray braces)
    decoder = json.JSONDecoder()
    best: tuple[int, Any] | None = None  # (span length, value): the largest complete value wins
    for attempt in (text, _repair(text)):
        i = 0
        while i < len(attempt):
            if attempt[i] in "{[":
                try:
                    obj, end = decoder.raw_decode(attempt[i:])
                except json.JSONDecodeError:
                    i += 1
                    continue
                # prefer dicts over lists (a bare list is usually the inner "value" of a truncated object)
                score = end + (10_000 if isinstance(obj, dict) else 0)
                if isinstance(obj, (dict, list)) and (best is None or score > best[0]):
                    best = (score, obj)
                i += max(end, 1)  # skip past this value so nested objects are not picked over the outer one
            else:
                i += 1
    if best is not None:
        return best[1]
    raise LLMError(f"Could not parse JSON from LLM output: {text[:200]!r}")


_RETRY_AFTER_RE = re.compile(r"try again in\s*([\d.]+)\s*(ms|s|m)", re.I)


def _classify(exc: Exception) -> tuple[str, float | None]:
    """Return (kind, wait_seconds). kind in {fatal, rate_limit, json_mode, transient}."""
    msg = str(exc)
    status = getattr(exc, "status_code", None)
    if "model_not_found" in msg or "does not exist" in msg or "not found for API version" in msg or status == 404:
        return "fatal", None
    if status in (401, 403) or "invalid_api_key" in msg or "API key not valid" in msg or "PERMISSION_DENIED" in msg:
        return "fatal", None
    if status == 402 or "Payment Required" in msg or "account balance" in msg or "insufficient_quota" in msg:
        return "fatal", None  # no credits left on the provider account: retrying cannot help
    if "json_validate_failed" in msg or "response_format" in msg:
        return "json_mode", None
    if status == 429 or "rate_limit" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
        # daily quota exhausted (Gemini free tier: "requests per day", retryDelay of hours) -> pointless to retry
        delay = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)s", msg)
        if "per day" in msg.lower() or (delay and int(delay.group(1)) > 600):
            return "fatal", None
        m = _RETRY_AFTER_RE.search(msg)
        wait = None
        if m:
            val, unit = float(m.group(1)), m.group(2).lower()
            wait = val / 1000 if unit == "ms" else val * 60 if unit == "m" else val
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if wait is None and headers is not None and headers.get("retry-after"):
            try:
                wait = float(headers["retry-after"])
            except ValueError:
                pass
        return "rate_limit", wait
    return "transient", None


def _fatal_message(exc: Exception) -> str:
    msg = str(exc)
    if "402" in msg or "Payment Required" in msg or "account balance" in msg or "insufficient_quota" in msg:
        return (f"{_short(exc)}\nThe '{settings.llm_provider}' account has no credits/quota left. Top it up or switch "
                "LLM_PROVIDER in .env (e.g. gemini or groq).")
    if "per day" in msg.lower() or "RESOURCE_EXHAUSTED" in msg:
        return (f"{_short(exc)}\nDaily free-tier quota for model '{settings.llm_model}' / '{settings.llm_small_model}' is used up. "
                "Switch LLM_MODEL / LLM_SMALL_MODEL in .env to another model from `python main.py models` "
                "(quotas are per model), set RERANK_ENABLED=false to halve the calls, or wait for the reset.")
    if "model_not_found" in msg or "does not exist" in msg or "not found" in msg.lower():
        return (f"{msg}\nThe model is not available for provider '{settings.llm_provider}'. Run `python main.py models` "
                "to list the models your key can use and set LLM_MODEL / LLM_SMALL_MODEL in .env.")
    return f"{msg}\nCheck {settings.llm_key_env_name} in .env (provider: {settings.llm_provider})."


class LLMClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or settings.llm_api_key
        self.base_url = base_url or settings.llm_base_url
        self.usage = LLMUsage()
        self._sync = None
        self._async = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _client(self):
        if self._sync is None:
            from openai import OpenAI

            self._sync = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=settings.llm_timeout_s, max_retries=0)
        return self._sync

    def _aclient(self):
        from openai import AsyncOpenAI

        loop = asyncio.get_running_loop()
        if self._async is None or getattr(self, "_async_loop", None) is not loop:
            # a new event loop (e.g. repeated asyncio.run() calls from Streamlit/CLI) needs a fresh async client
            self._async = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=settings.llm_timeout_s, max_retries=0)
            self._async_loop = loop
        return self._async

    def list_models(self) -> list[str]:
        """Model ids available to this key/provider (use: python main.py models)."""
        data = self._client().models.list()
        return sorted(str(m.id).replace("models/", "") for m in data.data)

    def _kwargs(self, messages, model, json_mode, max_tokens, temperature):
        kw: dict[str, Any] = {
            "model": model or settings.llm_model,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        return kw

    def _plan_retry(self, exc: Exception, attempt: int, kw: dict[str, Any]) -> float:
        """Decide how to react to an error. Returns seconds to wait; raises for fatal errors."""
        kind, wait = _classify(exc)
        if kind == "fatal":
            raise LLMError(_fatal_message(exc)) from exc
        if kind == "json_mode" and "response_format" in kw:
            kw.pop("response_format")  # retry without strict JSON mode; parse_json is lenient
            if "json" not in kw["messages"][-1]["content"].lower():
                kw["messages"][-1]["content"] += "\nReturn valid JSON only."
            return 1.0
        if kind == "rate_limit":
            return min(max(wait or 0, 5.0) + 1.0, 90.0)
        return min(2 ** attempt, 20)

    def complete(self, messages: list[dict[str, str]], *, model: str | None = None, json_mode: bool = False,
                 max_tokens: int = 1500, temperature: float | None = None) -> LLMResponse:
        if not self.available:
            raise LLMError(f"{settings.llm_key_env_name} is not set")
        kw = self._kwargs(messages, model, json_mode, max_tokens, temperature)
        last_exc: Exception | None = None
        for attempt in range(1, settings.llm_max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client().chat.completions.create(**kw)
                return self._wrap(resp, kw["model"], t0)
            except LLMError:
                raise
            except Exception as exc:
                last_exc = exc
                wait = self._plan_retry(exc, attempt, kw)
                log.warning("LLM call failed (attempt %d/%d): %s. Retrying in %.1fs", attempt, settings.llm_max_retries, _short(exc), wait)
                time.sleep(wait)
        raise LLMError(f"LLM call failed after retries: {last_exc}")

    async def acomplete(self, messages: list[dict[str, str]], *, model: str | None = None, json_mode: bool = False,
                        max_tokens: int = 1500, temperature: float | None = None) -> LLMResponse:
        if not self.available:
            raise LLMError(f"{settings.llm_key_env_name} is not set")
        kw = self._kwargs(messages, model, json_mode, max_tokens, temperature)
        last_exc: Exception | None = None
        for attempt in range(1, settings.llm_max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = await self._aclient().chat.completions.create(**kw)
                return self._wrap(resp, kw["model"], t0)
            except LLMError:
                raise
            except Exception as exc:
                last_exc = exc
                wait = self._plan_retry(exc, attempt, kw)
                log.warning("LLM call failed (attempt %d/%d): %s. Retrying in %.1fs", attempt, settings.llm_max_retries, _short(exc), wait)
                await asyncio.sleep(wait)
        raise LLMError(f"LLM call failed after retries: {last_exc}")

    def _wrap(self, resp: Any, model: str, t0: float) -> LLMResponse:
        latency = (time.perf_counter() - t0) * 1000
        usage = getattr(resp, "usage", None)
        self.usage.add(model, usage, latency)
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency,
        )


def _short(exc: Exception) -> str:
    s = str(exc)
    return s if len(s) <= 220 else s[:220] + "..."


# Backwards-compatible name used across the code base and tests
GroqLLM = LLMClient

_default: LLMClient | None = None


def get_llm() -> LLMClient:
    global _default
    if _default is None:
        _default = LLMClient()
    return _default