"""On-disk cache of screenshot detections, shared across processes.

Detection (OCR plus the OmniParser detector) is deterministic for identical
pixels and dominates step time, and benchmark screens repeat heavily across
runs. Entries are keyed by the screenshot's bytes, the grounder's identity and
a pipeline version, so a change to the detection code never serves stale
results. Gemini refinement is never cached here.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from .models import Element, json_value

_TUPLE_FIELDS = ("context_bounds", "label_bounds", "proposal_sources", "uncertainties", "state_observed")


def _element(raw: dict) -> Element:
    values = dict(raw)
    for name in _TUPLE_FIELDS:
        if values.get(name) is not None: values[name] = tuple(values[name])
    return Element(**values)


class DetectionCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Several shards share one file: WAL lets readers run beside a writer,
        # and the busy timeout waits out short write locks instead of failing.
        self.connection = sqlite3.connect(path, timeout=5, check_same_thread=False)
        self.connection.execute("PRAGMA busy_timeout=5000")
        # Switching to WAL ignores the busy timeout, so concurrent openers of a
        # fresh file can collide; retry that one statement briefly.
        for attempt in range(50):
            try:
                self.connection.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as error:
                if "locked" not in str(error) or attempt == 49: raise
                time.sleep(.02 * (attempt + 1))
        self.connection.execute("CREATE TABLE IF NOT EXISTS detections(key TEXT PRIMARY KEY, elements_json TEXT NOT NULL, label TEXT)")
        self.connection.commit()
        self.hits = self.misses = 0

    @staticmethod
    def key(screenshot: Path, grounder: str) -> str:
        return hashlib.sha256(grounder.encode() + b"\0" + screenshot.read_bytes()).hexdigest()

    def get(self, key: str) -> tuple[list[Element], str | None] | None:
        row = self.connection.execute("SELECT elements_json, label FROM detections WHERE key=?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return [_element(item) for item in json.loads(row[0])], row[1]

    def put(self, key: str, elements: list[Element], label: str | None) -> None:
        payload = json.dumps([json_value(asdict(element)) for element in elements])
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO detections(key, elements_json, label) VALUES(?,?,?)", (key, payload, label))

    def get_json(self, key: str) -> object | None:
        row = self.connection.execute("SELECT elements_json FROM detections WHERE key=?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put_json(self, key: str, value: object) -> None:
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO detections(key, elements_json, label) VALUES(?,?,NULL)", (key, json.dumps(value)))

    def close(self) -> None:
        self.connection.close()
