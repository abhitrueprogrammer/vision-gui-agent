from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vision_gui_agent.models import Element, Observation
from vision_gui_agent.logging_store import RunLogger
from vision_gui_agent.models import ActionDecision


class ArtifactSerializationTests(unittest.TestCase):
    def test_ocr_scalars_are_json_serializable(self) -> None:
        element = Element(1, "", "input", "Name", "", "", "input", np.float32(1), np.float32(2), np.float32(3), np.float32(4), label_bounds=(np.float32(1), np.float32(2), np.float32(3), np.float32(4)))
        payload = Observation("", "", [element], "", "Form").to_dict()
        self.assertEqual(json.loads(json.dumps(payload))["elements"][0]["x"], 1.0)

    def test_logger_accepts_ocr_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(Path(directory) / "runs-v2.sqlite3")
            logger.start_run("run", "goal", "test")
            observation = Observation("", "", [Element(1, "", "input", "Name", "", "", "input", np.float32(1), 2, 3, 4)], "", "Form")
            logger.log("run", 0, None, None, ActionDecision("fill", 1, text="Ada"), True, observation, {})
            self.assertEqual(logger.connection.execute("SELECT COUNT(*) FROM transitions").fetchone()[0], 1)
            logger.close()


if __name__ == "__main__":
    unittest.main()
