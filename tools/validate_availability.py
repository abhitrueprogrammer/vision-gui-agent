"""Validate the pixel disabled-control check against DOM-labelled fixtures.

For every labelled control, the OCR text boxes whose centres fall inside it are
measured with ``label_contrast``; the control is predicted disabled when its
clearest label is below ``DISABLED_CONTRAST``. Reports precision and recall on
disabled controls per fixture set. Labels are evaluator-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from vision_gui_agent.perception import DISABLED_CONTRAST, OmniParserVisualGrounder, label_contrast

ROOT = Path(__file__).resolve().parents[1] / "tests/fixtures/perception"
SETS = ("lab_availability", "lab_apps", "framework_styles", "real_app")


def _vlm_verdicts(record: dict, path: Path, vlm, cache) -> dict[int, bool | None]:
    import asyncio
    from vision_gui_agent.models import Element
    controls = [Element(index, "", "button", control["text"], "", "", "button", *control["box"])
                for index, control in enumerate(record["controls"], 1)]
    return asyncio.run(vlm.availability(path, controls, cache))


def evaluate(set_name: str, ocr, mode: str = "pixel", vlm=None, cache=None) -> dict:
    records = json.loads((ROOT / set_name / "labels.json").read_text())
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "unread": 0}
    mistakes = []
    for record in records:
        verdicts = _vlm_verdicts(record, ROOT / set_name / record["image"], vlm, cache) if mode != "pixel" else {}
        path = ROOT / set_name / record["image"]
        with Image.open(path) as source:
            image = source.convert("RGB")
        boxes = [OmniParserVisualGrounder._box(item) for item in OmniParserVisualGrounder._records(ocr(str(path)))]
        for index, control in enumerate(record["controls"], 1):
            x, y, width, height = control["box"]
            inside = [box for box in boxes if x <= (box[0] + box[2]) / 2 <= x + width and y <= (box[1] + box[3]) / 2 <= y + height]
            contrasts = [value for box in inside if (value := label_contrast(image, box)) is not None]
            pixel = max(contrasts) < DISABLED_CONTRAST if contrasts else None
            vlm_disabled = None if verdicts.get(index) is None else not verdicts[index]
            predicted = {"pixel": pixel, "vlm": vlm_disabled,
                         "combined": None if pixel is None and vlm_disabled is None else bool(pixel) or bool(vlm_disabled)}[mode]
            if predicted is None:
                counts["unread"] += 1
                continue
            contrasts = contrasts or [0.0]
            key = ("tp" if predicted else "fn") if control["disabled"] else ("fp" if predicted else "tn")
            counts[key] += 1
            if key in {"fp", "fn"}:
                mistakes.append({"image": record["image"], "text": control["text"], "kind": key, "contrast": round(max(contrasts), 2)})
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else None
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else None
    return {"set": set_name, **counts, "precision": precision, "recall": recall, "mistakes": mistakes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--mode", choices=("pixel", "vlm", "combined"), default="pixel",
                        help="pixel contrast only, Gemini availability only, or disabled if either says so")
    parser.add_argument("--cache", default="artifacts/cache/detections.sqlite3")
    args = parser.parse_args()
    from rapidocr import RapidOCR
    ocr = RapidOCR(params={"Det.box_thresh": .35})
    vlm = cache = None
    if args.mode != "pixel":
        from vision_gui_agent.detection_cache import DetectionCache
        from vision_gui_agent.perception import GeminiVisualGrounder
        vlm, cache = GeminiVisualGrounder(), DetectionCache(Path(args.cache))
    reports = [evaluate(name, ocr, args.mode, vlm, cache) for name in SETS if (ROOT / name / "labels.json").exists()]
    print(f"mode={args.mode} threshold={DISABLED_CONTRAST}")
    totals = {key: sum(report[key] for report in reports) for key in ("tp", "fp", "fn", "tn", "unread")}
    totals["precision"] = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else None
    totals["recall"] = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else None
    if args.json:
        print(json.dumps({"threshold": DISABLED_CONTRAST, "sets": reports, "total": totals}, indent=2))
        return
    for report in reports + [{"set": "TOTAL", **totals, "mistakes": []}]:
        p, r = report["precision"], report["recall"]
        print(f"{report['set']:18} tp={report['tp']:3} fp={report['fp']:3} fn={report['fn']:3} tn={report['tn']:3} "
              f"unread={report['unread']:3} precision={'n/a' if p is None else f'{p:.3f}'} recall={'n/a' if r is None else f'{r:.3f}'}")
        for mistake in report["mistakes"][:12]:
            print(f"    {mistake['kind']} {mistake['contrast']:5.2f} {mistake['text']!r:28} {mistake['image']}")


if __name__ == "__main__":
    main()
