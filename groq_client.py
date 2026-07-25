"""
Production Groq client with correct retry semantics per Groq docs:
  - 429, 498, 500, 502, 503  -> retryable (rotating keys, exponential backoff + jitter)
  - 400, 401, 403, 404, 413, 422, 424  -> NON-retryable (raise immediately)
  - Network errors (APIConnectionError, APITimeoutError) -> retryable
"""
from __future__ import annotations
import os
import time
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv
import groq
from groq import Groq
load_dotenv()


# ---- Config --------------------------------------------------------------

GROQ_KEYS = [k for k in [
    *(os.getenv(f"GROQ_API_KEY_{i}") for i in range(1, 21)),
    os.getenv("GROQ_API_KEY"),
] if k]

DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

MODEL_REGISTRY = {
    "orchestrator": os.getenv("ORCHESTRATOR_MODEL", "openai/gpt-oss-120b"),
    "orchestrator_fallback": os.getenv("ORCHESTRATOR_FALLBACK_MODEL", "llama-3.3-70b-versatile"),
    "investigator": os.getenv("INVESTIGATOR_MODEL", "qwen/qwen3.6-27b"),
    "investigator_fallback": os.getenv("INVESTIGATOR_FALLBACK_MODEL", "openai/gpt-oss-20b"),
    "investigator_tertiary": os.getenv("INVESTIGATOR_TERTIARY_MODEL", "openai/gpt-oss-safeguard-20b"),
}

RETRYABLE_STATUS = {429, 498, 500, 502, 503}
NON_RETRYABLE_STATUS = {400, 401, 403, 404, 413, 422, 424}

BASE_BACKOFF = 1.5
MAX_BACKOFF = 30.0
MAX_ATTEMPTS_PER_KEY = 2


# ---- Per-window stats ----------------------------------------------------

@dataclass
class WindowStats:
    active_model: Optional[str] = None
    api_key_index: Optional[int] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rate_limit_hits: int = 0
    successful_requests: int = 0
    keys_used: list[int] = field(default_factory=list)


# ---- Key manager ---------------------------------------------------------

class APIKeyManager:
    """Thread-safe round-robin key rotation."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("No Groq API keys configured.")
        self._keys = keys
        self._idx = 0
        self._lock = threading.Lock()

    def next_key(self) -> tuple[int, str]:
        with self._lock:
            idx = self._idx
            self._idx = (self._idx + 1) % len(self._keys)
            return idx + 1, self._keys[idx]

    @property
    def size(self) -> int:
        return len(self._keys)


# ---- Client --------------------------------------------------------------

class ProductionGroqClient:
    """Drop-in replacement for `groq.Groq` with rotation, retries, stats."""

    def __init__(self, api_keys: Optional[list[str]] = None):
        self._key_mgr = APIKeyManager(api_keys or GROQ_KEYS)
        self._clients: dict[int, Groq] = {}
        self._clients_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._window_stats = WindowStats()
        self._global_rate_limit_hits = 0
        
        # CHANGED: track (key_index, model) instead of just key_index
        self._tpd_exhausted: set[tuple[int, str]] = set()
        self._tpd_lock = threading.Lock()

        self.chat = _ChatRouter(self)

    # ---- Stats -----------------------------------------------------------

    def reset_window_stats(self) -> None:
        with self._stats_lock:
            self._window_stats = WindowStats()

    def get_window_stats(self) -> WindowStats:
        with self._stats_lock:
            return WindowStats(
                active_model=self._window_stats.active_model,
                api_key_index=self._window_stats.api_key_index,
                prompt_tokens=self._window_stats.prompt_tokens,
                completion_tokens=self._window_stats.completion_tokens,
                rate_limit_hits=self._window_stats.rate_limit_hits,
                successful_requests=self._window_stats.successful_requests,
                keys_used=list(self._window_stats.keys_used),
            )

    def _record_success(self, model: str, key_index: int, usage: Optional[Any] = None) -> None:
        with self._stats_lock:
            self._window_stats.active_model = model
            self._window_stats.api_key_index = key_index
            self._window_stats.successful_requests += 1
            if usage:
                self._window_stats.prompt_tokens += getattr(usage, 'prompt_tokens', 0)
                self._window_stats.completion_tokens += getattr(usage, 'completion_tokens', 0)
            self._window_stats.keys_used.append(key_index)

    def _record_rate_limit(self, key_index: int) -> None:
        with self._stats_lock:
            self._window_stats.rate_limit_hits += 1
            self._global_rate_limit_hits += 1
            self._window_stats.keys_used.append(key_index)

    # ---- Client cache ----------------------------------------------------

    def _get_client(self, key_index: int, api_key: str) -> Groq:
        with self._clients_lock:
            if key_index not in self._clients:
                self._clients[key_index] = Groq(api_key=api_key, max_retries=0)
            return self._clients[key_index]

    # ---- Backoff ---------------------------------------------------------

    @staticmethod
    def _backoff(attempt: int) -> float:
        base = min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF)
        return random.uniform(0.1, base)

    # ---- Core dispatch ---------------------------------------------------

    def _create_completion(self, **kwargs) -> Any:
        model = kwargs.get("model", DEFAULT_MODEL)
        key_attempts: dict[int, int] = {}
        total_calls = 0
        last_error: Optional[Exception] = None

        # CHANGED: while loop that only counts real API attempts, not skipped keys
        while total_calls < self._key_mgr.size * MAX_ATTEMPTS_PER_KEY:
            key_index, api_key = self._key_mgr.next_key()

            # CHANGED: skip only if THIS key + THIS model is TPD-exhausted
            if (key_index, model) in self._tpd_exhausted:
                with self._tpd_lock:
                    exhausted_for_model = {k for k, m in self._tpd_exhausted if m == model}
                    if len(exhausted_for_model) >= self._key_mgr.size:
                        raise RuntimeError(f"All API keys have exhausted their daily token limits (TPD) for model {model}.")
                continue  # rotate to next key

            key_attempts[key_index] = key_attempts.get(key_index, 0) + 1
            total_calls += 1

            # If this key has been tried enough times, check if we're done
            if key_attempts[key_index] > MAX_ATTEMPTS_PER_KEY:
                with self._tpd_lock:
                    exhausted_for_model = {k for k, m in self._tpd_exhausted if m == model}
                active_keys = [k for k in range(1, self._key_mgr.size + 1) if k not in exhausted_for_model]
                if active_keys and all(key_attempts.get(k, 0) >= MAX_ATTEMPTS_PER_KEY for k in active_keys):
                    break
                continue

            client = self._get_client(key_index, api_key)

            try:
                response = client.chat.completions.create(**kwargs)
                self._record_success(model, key_index, getattr(response, 'usage', None))
                return response

            except groq.RateLimitError as e:
                last_error = e
                self._record_rate_limit(key_index)

                error_body = str(e).lower()
                if "tokens per day" in error_body or "tpd" in error_body:
                    with self._tpd_lock:
                        # CHANGED: blacklist the pair, not just the key
                        self._tpd_exhausted.add((key_index, model))
                    print(f"[GroqClient] TPD exhausted on key {key_index} for model {model}. Blacklisted for session.")
                    with self._tpd_lock:
                        exhausted_for_model = {k for k, m in self._tpd_exhausted if m == model}
                    if len(exhausted_for_model) >= self._key_mgr.size:
                        raise RuntimeError(f"All API keys have exhausted their daily token limits (TPD) for model {model}.") from e
                    continue

                print(f"[GroqClient] 429/498 RPM on key {key_index}; rotating+backoff.")
                time.sleep(self._backoff(key_attempts[key_index] - 1))
                continue

            except (groq.APIConnectionError, groq.APITimeoutError) as e:
                last_error = e
                print(f"[GroqClient] Network error on key {key_index}: {type(e).__name__}; retrying...")
                time.sleep(self._backoff(key_attempts[key_index] - 1))
                continue

            except groq.APIStatusError as e:
                if e.status_code in RETRYABLE_STATUS:
                    last_error = e
                    print(f"[GroqClient] Retryable {e.status_code} on key {key_index}; backoff...")
                    time.sleep(self._backoff(key_attempts[key_index] - 1))
                    continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError(
            f"Exhausted all Groq attempts for model {model}. "
            f"TPD-exhausted pairs for this model: {[(k,m) for k,m in self._tpd_exhausted if m == model]}."
        )


class _CompletionsRouter:
    def __init__(self, owner: ProductionGroqClient):
        self._owner = owner

    def create(self, **kwargs) -> Any:
        return self._owner._create_completion(**kwargs)


class _ChatRouter:
    def __init__(self, owner: ProductionGroqClient):
        self.completions = _CompletionsRouter(owner)