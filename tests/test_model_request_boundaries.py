import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from vision_gui_agent.decision import GeminiPolicy
from vision_gui_agent.models import Element, Observation
from vision_gui_agent.perception import GeminiVisualGrounder


def configured(cls, generate):
    instance = object.__new__(cls)
    instance.model = 'fixture'
    instance.last_response = None
    instance.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    instance.types = SimpleNamespace(Part=SimpleNamespace(from_bytes=lambda **kw: kw),
        GenerateContentConfig=lambda **kw: kw, AutomaticFunctionCallingConfig=lambda **kw: kw)
    return instance


class ModelRequestBoundaryTests(unittest.TestCase):
    def test_refinement_serializes_numpy_ocr_candidates(self):
        requests = []
        def generate(**request):
            requests.append(request)
            return SimpleNamespace(text='{"found":true,"actionable":true,"kind":"button","x":20,"y":20,"width":40,"height":30}')
        grounder = configured(GeminiVisualGrounder, generate)
        selected = Element(1, '', 'button', 'Search', '', '', 'button', 20, 20, 40, 30)
        ocr = Element(2, '', 'text', 'Heading', '', '', 'text',
                      np.float32(80), np.float32(10), np.float32(60), np.float32(20), actionable=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'screen.png'
            Image.new('RGB', (200, 100), 'white').save(path)
            result = asyncio.run(grounder.refine(path, selected, elements=[selected, ocr]))
        self.assertIsNotNone(result)
        self.assertIn('"box": [80.0, 10.0, 60.0, 20.0]', requests[0]['contents'][-1])

    def test_policy_serializes_numpy_source_scores(self):
        requests = []
        def generate(**request):
            requests.append(request)
            return SimpleNamespace(text='{"action":"click","element_id":1}')
        policy = configured(GeminiPolicy, generate)
        target = Element(1, '', 'button', 'Search', '', '', 'button', 20, 20, 40, 30,
                         recognition_confidence=np.float32(.9))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'screen.png'
            Image.new('RGB', (200, 100), 'white').save(path)
            asyncio.run(policy.decide('Search', Observation(str(path), str(path), [target], '', ''), {}, []))
        prompt = json.loads(requests[0]['contents'][-1])
        self.assertAlmostEqual(prompt['elements'][0]['source_scores']['recognition'], .9, places=6)

    def test_network_failure_does_not_retain_previous_response(self):
        def generate(**request):
            raise ConnectionError('Temporary failure in name resolution')
        for method in ('compile_goal', 'decide'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                policy = configured(GeminiPolicy, generate)
                policy.last_response = 'previous successful response'
                path = Path(tmp) / 'screen.png'
                Image.new('RGB', (200, 100), 'white').save(path)
                with self.assertRaises(ConnectionError):
                    if method == 'compile_goal':
                        asyncio.run(policy.compile_goal('Search'))
                    else:
                        asyncio.run(policy.decide('Search', Observation(str(path), str(path), [], '', ''), {}, []))
                self.assertIsNone(policy.last_response)
