"""Replay cached local detector/OCR proposals through a supplied baseline and current fusion.

Run offline from the repository root with PYTHONPATH=.; the baseline is a trusted
Python perception module (for example `git show b3ca1c4:vision_gui_agent/perception.py`).
No Gemini calls are made. Model loading uses the ordinary grounder configuration.
"""
import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from vision_gui_agent.models import json_value
from vision_gui_agent.perception import OmniParserVisualGrounder
from vision_gui_agent.perception_validation import _inside


async def validate(baseline_path, corpus):
    spec = importlib.util.spec_from_file_location('vision_gui_agent.perception_baseline', baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    local = OmniParserVisualGrounder()
    rows = []
    for case in json.loads((corpus / 'manifest.json').read_text())['cases']:
        path = corpus / (case['id'] + '.png')
        with Image.open(path).convert('RGB') as image:
            predictions = local.detector.predict(source=image, conf=.25, imgsz=max(image.size))
        ocr = local.ocr(str(path), return_word_box=True)
        targets = [t for t in case['targets'] if t['visible']]
        for name, cls in [('before', baseline.OmniParserVisualGrounder), ('after', OmniParserVisualGrounder)]:
            elements = await cls(SimpleNamespace(predict=lambda **kw: predictions), lambda *a, **kw: ocr).detect(path)
            safe = lambda e, t: _inside((e.x+e.width/2, e.y+e.height/2), t['safe_region'])
            rows.append({'case': case['id'], 'version': name, 'visible_targets': len(targets),
                'actionable_safe_point_hits': sum(any(e.actionable and safe(e,t) for e in elements) for t in targets),
                'false_actionable_proposals': sum(e.actionable and not any(safe(e,t) for t in targets) for e in elements),
                'proposals': [{'id': e.id, 'text': e.text, 'kind': e.tag, 'actionable': e.actionable,
                               'box': [e.x,e.y,e.width,e.height]} for e in elements]})
    return {'track': 'local_detector_ocr_corpus', 'weights': str(local.detector.model_path),
            'scope': 'Real cached OmniParser/RapidOCR proposals replayed identically through before/after fusion; no semantic model or planning evaluation',
            'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=Path('tests/fixtures/perception'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(asyncio.run(validate(args.baseline, args.corpus)), indent=2, default=json_value))
