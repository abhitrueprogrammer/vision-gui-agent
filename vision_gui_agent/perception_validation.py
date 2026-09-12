"""Offline proposal ablations on an explicitly annotated screenshot corpus.

Recorded detector/OCR fixtures isolate fusion; they do not measure detector or
language-model accuracy. Oracle target/box comparisons never reach the agent.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from .perception import OmniParserVisualGrounder, refinement_reasons, selection_tiles, model_image
from .models import Element, Observation, ActionDecision
from PIL import Image
from io import BytesIO


def _detector(boxes):
    return SimpleNamespace(predict=lambda **_: [SimpleNamespace(boxes=SimpleNamespace(
        xyxy=SimpleNamespace(tolist=lambda: boxes), conf=SimpleNamespace(tolist=lambda: [.98]*len(boxes))))])


def _ocr(records):
    return [([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],(label,.99)) for label,(x,y,w,h) in records]


def _inside(point, box):
    x,y,w,h=box
    return x <= point[0] <= x+w and y <= point[1] <= y+h


def _iou(left, right):
    x,y,w,h=left; a,b,c,d=right
    area=max(0,min(x+w,a+c)-max(x,a))*max(0,min(y+h,b+d)-max(y,b))
    return area/(w*h+c*d-area) if w*h+c*d-area else 0


async def validate(corpus: Path) -> dict:
    manifest=json.loads((corpus/'manifest.json').read_text())
    results=[]
    gate_rows=[]
    for case in manifest['cases']:
        # Hold oracle candidate identity/box fixed. This tests escalation coverage,
        # not whether a policy/model selects or refines the right instance.
        targets=[t for t in case['targets'] if t['visible']]
        candidates=[Element(i+1,'',t['kind'],t['label'],'','',t['kind'],*t['box'],
                    proposal_sources=('oracle',),semantic_confirmed=True) for i,t in enumerate(targets)]
        path=corpus/(case['id']+'.png')
        obs=Observation(str(path),str(path),candidates,'','')
        for target,element in zip(targets,candidates):
            action='set_checked' if target['kind']=='checkbox' else 'fill' if target['kind']=='input' else 'click'
            reasons=refinement_reasons(element,candidates,ActionDecision(action,element.id))
            ambiguous=(sum(t['label']==target['label'] for t in targets)>1 or min(target['box'][2:])<24
                       or action in {'fill','set_checked'})
            gate_rows.append({'case':case['id'],'instance':target['instance'],'oracle_needs_refinement':ambiguous,
                'old_gate':not element.actionable or element.confidence<.7,'new_gate':bool(reasons),
                'reasons':reasons})
        with Image.open(BytesIO(model_image(str(path),max_width=None))) as native:
            native_size=native.size
        with Image.open(path) as original:
            assert native_size == original.size
        selection_tiles(obs)  # Exercise the same native crop contract used by policy.
        for mode in ('detector_only','ocr_only','fusion'):
            records=_ocr(case['ocr']) if mode != 'detector_only' else []
            grounder=OmniParserVisualGrounder(_detector(case['detector'] if mode != 'ocr_only' else []),lambda *a,**k: records)
            elements=await grounder.detect(corpus/(case['id']+'.png'))
            targets=[t for t in case['targets'] if t['visible']]
            box=lambda e:(e.x,e.y,e.width,e.height)
            safe=lambda e,t:_inside((e.x+e.width/2,e.y+e.height/2),t['safe_region'])
            results.append({'case':case['id'],'mode':mode,'visible_targets':len(targets),
                'geometric_iou_hits':sum(any(_iou(box(e),t['box'])>=.5 for e in elements) for t in targets),
                'actionable_safe_point_hits':sum(any(e.actionable and safe(e,t) for e in elements) for t in targets),
                'false_actionable_proposals':sum(e.actionable and not any(safe(e,t) for t in targets) for e in elements),
                'wrong_typed_proposals':sum(e.tag not in {'other','text','menuitem'} and any(safe(e,t) and e.tag != t['kind'] for t in targets) for e in elements),
                'proposals':len(elements)})
    totals={}
    for mode in ('detector_only','ocr_only','fusion'):
        rows=[r for r in results if r['mode']==mode]
        totals[mode]={key:sum(r[key] for r in rows) for key in ('visible_targets','geometric_iou_hits','actionable_safe_point_hits','false_actionable_proposals','wrong_typed_proposals','proposals')}
    return {'track':'recorded_proposal_ablation','scope':manifest['source'],'totals':totals,'results':results,'oracle_fixed_candidate_gates':gate_rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus',type=Path)
    args=parser.parse_args()
    print(json.dumps(asyncio.run(validate(args.corpus)),indent=2))

if __name__=='__main__':main()
