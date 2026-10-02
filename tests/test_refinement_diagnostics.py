import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from vision_gui_agent.models import ActionDecision, Element
from vision_gui_agent.perception import GeminiVisualGrounder, GroundingAbstention


class RefinementDiagnosticsTests(unittest.TestCase):
    def refine_error(self, generate):
        grounder = object.__new__(GeminiVisualGrounder)
        grounder.model = 'test-model'
        grounder._generate = generate
        target = Element(7, '', 'button', 'private label', '', '', 'button', 20, 20, 40, 30)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'screen.png'
            Image.new('RGB', (200, 100), 'white').save(path)
            with self.assertRaises(GroundingAbstention) as raised:
                asyncio.run(grounder.refine(path, target, decision=ActionDecision('click', 7)))
        return str(raised.exception)

    def test_request_error_has_status_context_without_raw_exception(self):
        class ClientError(Exception):
            code = 429
        def generate(request):
            raise ClientError('secret-api-key and private response body')
        message = self.refine_error(generate)
        for detail in ('stage=generate', 'ClientError', 'http_status=429', 'rate limit', 'target_id=7', 'action=click', 'model=test-model'):
            self.assertIn(detail, message)
        for private in ('secret-api-key', 'private response', 'private label'):
            self.assertNotIn(private, message)

    def test_invalid_json_reports_position_without_response_contents(self):
        message = self.refine_error(lambda request: SimpleNamespace(text='private non-JSON response'))
        for detail in ('stage=parse', 'JSONDecodeError', 'line=1', 'column=1'):
            self.assertIn(detail, message)
        self.assertNotIn('private non-JSON', message)

    def test_empty_response_is_distinct_from_request_failure(self):
        message = self.refine_error(lambda request: SimpleNamespace(text=None))
        self.assertIn('stage=response', message)
        self.assertIn('empty text response', message)

    def test_negative_refinement_records_specific_rejection_without_raw_response(self):
        import json
        from dataclasses import replace
        from vision_gui_agent.perception import OmniParserVisualGrounder
        target = Element(7, '', 'other', 'private label', '', '', 'other', 20, 20, 40, 30,
                         proposal_sources=('detector',))
        valid = dict(found=True, actionable=True, kind='button', x=20, y=20, width=40, height=30)
        cases = [
            ({'found':False, 'reason':'private response'}, 'not_found_or_ambiguous'),
            ({'found':'true'}, 'invalid_found_flag'),
            (dict(valid, actionable=False), 'actionability_unconfirmed'),
            (dict(valid, x=-1), 'bounds_outside_crop'),
            (dict(valid, width='bad'), 'invalid_bounds'),
            (dict(valid, width=1), 'invalid_box_size'),
            (valid, 'instance_unconfirmed'),
        ]
        grounder = object.__new__(GeminiVisualGrounder)
        grounder.model = 'fixture'
        wrapper = object.__new__(OmniParserVisualGrounder)
        wrapper.refiner = grounder
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'screen.png'
            Image.new('RGB', (200, 100), 'white').save(path)
            for response, reason in cases:
                grounder._generate = lambda request: SimpleNamespace(text=json.dumps(response))
                with self.subTest(reason=reason):
                    result = asyncio.run(wrapper.refine(path, target, elements=[target, replace(target, id=8)]))
                    self.assertIsNone(result)
                    detail = getattr(wrapper, 'last_refinement_rejection', '')
                    self.assertIn(reason, detail)
                    self.assertIn('target_id=7', detail)
                    self.assertNotIn('private', detail)
            grounder._generate = lambda request: SimpleNamespace(text=json.dumps(dict(valid, instance_confirmed=True)))
            self.assertIsNotNone(asyncio.run(wrapper.refine(path, target)))
            self.assertIsNone(wrapper.last_refinement_rejection)
