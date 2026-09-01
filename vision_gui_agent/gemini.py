from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv

T = TypeVar("T")


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


def _quota_exhausted(error: Exception) -> bool:
    message = str(error).upper()
    return "RESOURCE_EXHAUSTED" in message or "429" in message


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

    def generate(self, request: Callable[[Any], T]) -> T:
        for attempt in range(len(self._keys)):
            try:
                return request(self.client)
            except Exception as error:
                if not _quota_exhausted(error) or attempt + 1 == len(self._keys):
                    raise
                self._index = (self._index + 1) % len(self._keys)
                self.client = self._new_client()
        raise AssertionError("unreachable")
