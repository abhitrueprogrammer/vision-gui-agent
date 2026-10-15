from __future__ import annotations

import os
import random
import socket
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import time
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv

T = TypeVar("T")

# Every Gemini request is counted under the purpose active when it was made
# (policy, refine, goal_compiler, ...), so reported model cost includes calls
# made outside the policy. The context variable is copied into
# asyncio.to_thread workers, so threaded calls are attributed correctly.
CALL_PURPOSE: ContextVar[str] = ContextVar("gemini_call_purpose", default="other")
CALL_COUNTS: Counter[str] = Counter()


@contextmanager
def call_purpose(purpose: str):
    token = CALL_PURPOSE.set(purpose)
    try: yield
    finally: CALL_PURPOSE.reset(token)

# A dated model, not a "-latest" alias, so reported results stay reproducible.
DEFAULT_MODEL = "gemini-3.5-flash-lite"


def configured_gemini_keys(env_path: Path = Path(".env")) -> list[str]:
    """Return configured Gemini keys, including the optional second slot."""
    keys = []
    if env_path.is_file():
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if line.startswith("#"):
                line = line[1:].strip()
            if line.startswith("GEMINI_API_KEY=") or line.startswith("GEMINI_API_KEY_"):
                value = line.partition("=")[2].strip().strip('"').strip("'")
                if value:
                    keys.append(value)
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2"):
        if (value := os.getenv(name)) and value not in keys:
            keys.append(value)
    return keys


def _overloaded(error: Exception) -> bool:
    """A temporary server-side failure (overload, internal error, deadline) worth waiting out."""
    message = str(error).upper()
    return any(marker in message for marker in ("503", "502", "BAD GATEWAY", "UNAVAILABLE", "HIGH DEMAND", "OVERLOADED", "500 INTERNAL", "DEADLINE_EXCEEDED"))


def _quota_exhausted(error: Exception) -> bool:
    message = str(error).upper()
    return "RESOURCE_EXHAUSTED" in message or "429" in message


def _temporary_dns_failure(error: BaseException) -> bool:
    # SDK/httpx wrappers retain the resolver error in their exception chain.
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, socket.gaierror) and error.errno == socket.EAI_AGAIN:
            return True
        error = error.__cause__ or error.__context__
    return False


# The resolved model behind the last response (e.g. a pinned name's exact build), for provenance.
LAST_MODEL_VERSION: dict[str, str | None] = {"value": None}


class RateLimiter:
    """Requests- and tokens-per-minute budget shared by every process using one SQLite file.

    Each call is charged before it is made with the running average token cost
    of calls for the same purpose, then corrected from the response's usage
    metadata, so concurrent shards stay under one quota instead of colliding.
    """

    def __init__(self, path: Path, rpm: int, tpm: int, initial_estimate: int = 1800) -> None:
        import sqlite3
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rpm, self.tpm = rpm, tpm
        self.connection = sqlite3.connect(path, timeout=10, check_same_thread=False, isolation_level=None)
        self.connection.execute("PRAGMA busy_timeout=10000")
        for attempt in range(50):
            try:
                self.connection.execute("PRAGMA journal_mode=WAL"); break
            except sqlite3.OperationalError:
                if attempt == 49: raise
                time.sleep(.02 * (attempt + 1))
        self.connection.execute("CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY, at REAL NOT NULL, tokens INTEGER NOT NULL)")
        self.average: dict[str, float] = {}
        self.initial_estimate = initial_estimate

    def acquire(self, purpose: str = "other", now: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep) -> int:
        estimate = int(self.average.get(purpose, self.initial_estimate))
        while True:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                current = now()
                self.connection.execute("DELETE FROM calls WHERE at < ?", (current - 60,))
                count, tokens, oldest = self.connection.execute("SELECT COUNT(*), COALESCE(SUM(tokens),0), MIN(at) FROM calls").fetchone()
                if count < self.rpm and tokens + estimate <= self.tpm:
                    row = self.connection.execute("INSERT INTO calls(at, tokens) VALUES(?,?)", (current, estimate)).lastrowid
                    self.connection.execute("COMMIT")
                    return row
                self.connection.execute("COMMIT")
            except BaseException:
                self.connection.execute("ROLLBACK"); raise
            sleep(max(.05, min(1.0, (oldest or current) + 60 - current)))

    def record(self, row: int, purpose: str, tokens: int | None) -> None:
        if not tokens: return
        self.connection.execute("UPDATE calls SET tokens=? WHERE id=?", (tokens, row))
        previous = self.average.get(purpose)
        self.average[purpose] = tokens if previous is None else .8 * previous + .2 * tokens


_LIMITER: dict[str, RateLimiter | None] = {}


def shared_limiter() -> RateLimiter | None:
    """The limiter the harness configured through VGA_RATE_DB / VGA_RPM / VGA_TPM, if any."""
    path = os.getenv("VGA_RATE_DB")
    if not path: return None
    if path not in _LIMITER:
        _LIMITER[path] = RateLimiter(Path(path), int(os.getenv("VGA_RPM", "60")), int(os.getenv("VGA_TPM", "250000")))
    return _LIMITER[path]


class GeminiClientPool:
    """Use the next configured key when Gemini reports quota exhaustion."""

    def __init__(self, key_slot: int | None = None) -> None:
        load_dotenv()
        keys = configured_gemini_keys()
        if key_slot is not None:
            if not 1 <= key_slot <= len(keys):
                raise RuntimeError(f"GEMINI key slot {key_slot} is unavailable")
            keys = [keys[key_slot - 1]]
        if not keys:
            raise RuntimeError("GEMINI_API_KEY is required")
        from google import genai
        from google.genai import types
        self._genai, self.types, self._keys, self._index = genai, types, keys, 0
        self.client = self._new_client()

    def _new_client(self) -> Any:
        return self._genai.Client(api_key=self._keys[self._index], http_options=self.types.HttpOptions(timeout=60_000))

    # Once every key reports quota exhaustion, wait with growing, jittered delays for at most this long.
    BACKOFF_LIMIT_S = 300.0

    def generate(self, request: Callable[[Any], T], sleep: Callable[[float], None] | None = None) -> T:
        sleep = sleep or time.sleep  # looked up per call, so tests can patch time.sleep
        purpose = CALL_PURPOSE.get()
        CALL_COUNTS[purpose] += 1
        limiter = shared_limiter()
        dns_retries, waited, delay = 0, 0.0, 1.0
        while True:
            overloaded = False
            for attempt in range(len(self._keys)):
                if overloaded: break
                while True:
                    row = limiter.acquire(purpose) if limiter else None
                    try:
                        result = request(self.client)
                    except Exception as error:
                        if _temporary_dns_failure(error) and dns_retries < 2:
                            pause = .5 * (2 ** dns_retries)
                            dns_retries += 1
                            logging.getLogger(__name__).warning(
                                "Temporary Gemini DNS failure; retry %s/2 in %.1fs", dns_retries, pause)
                            sleep(pause)
                            continue
                        if _overloaded(error):  # the same key will do once the server recovers
                            overloaded = True
                            break
                        if not _quota_exhausted(error): raise
                        if attempt + 1 < len(self._keys):
                            self._index = (self._index + 1) % len(self._keys)
                            self.client = self._new_client()
                        break
                    usage = getattr(result, "usage_metadata", None)
                    if limiter and row is not None: limiter.record(row, purpose, getattr(usage, "total_token_count", None))
                    if getattr(result, "model_version", None): LAST_MODEL_VERSION["value"] = result.model_version
                    return result
            if waited >= self.BACKOFF_LIMIT_S:
                raise RuntimeError("Gemini quota exhausted or unavailable after backoff (RESOURCE_EXHAUSTED/UNAVAILABLE)")
            pause = min(delay, 60.0, self.BACKOFF_LIMIT_S - waited) * (.5 + random.random())
            logging.getLogger(__name__).warning("Gemini overloaded or out of quota; backing off %.1fs", pause)
            sleep(pause)
            waited, delay = waited + pause, delay * 2
