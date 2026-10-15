"""Capture labelled screenshots for validating disabled-control detection.

Labels come from the DOM and are evaluator-only: the agent never reads them.
Each screenshot gets a sidecar list of visible controls with their text, box
and whether the page marks them disabled (``disabled``, ``aria-disabled`` or a
``disabled`` class). Sources: a local Gitea instance (real-app fixtures) and the
current Visual Function Lab (lab fixtures).
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import Page, async_playwright

ROOT = Path(__file__).resolve().parents[1] / "tests/fixtures/perception"
VIEWPORT = {"width": 1440, "height": 1000}

LABEL_CONTROLS = """() => {
  const nodes = document.querySelectorAll('button, .ui.button, a.button, input[type=submit], input[type=button], [role=button]');
  const out = [];
  for (const el of nodes) {
    const r = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    if (r.width < 8 || r.height < 8 || style.visibility === 'hidden' || style.display === 'none') continue;
    if (r.bottom < 0 || r.top > innerHeight || r.right < 0 || r.left > innerWidth) continue;
    const text = (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().replace(/\\s+/g, ' ');
    if (!text) continue;
    // A nested button inherits a disabled wrapper (Gitea puts .disabled on the outer .ui.button).
    if (el.parentElement && el.parentElement.closest('.ui.button') && el.matches('.ui.button .ui.button, .ui.button button')) continue;
    const disabled = el.disabled === true || el.getAttribute('aria-disabled') === 'true' || !!el.closest('.disabled,[disabled]');
    out.push({text, box: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)], disabled});
  }
  return out;
}"""


async def _save(page: Page, directory: Path, name: str) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    # Gitea keeps an event stream open, so "networkidle" never fires.
    await page.wait_for_load_state("load")
    await page.wait_for_timeout(400)
    await page.screenshot(path=str(directory / f"{name}.png"))
    controls = await page.evaluate(LABEL_CONTROLS)
    return {"id": name, "image": f"{name}.png", "url": page.url, "controls": controls}


async def _gitea_login(page: Page, base: str, user: str, password: str) -> None:
    await page.goto(f"{base}/user/login")
    await page.fill("#user_name", user)
    await page.fill("#password", password)
    await page.click("form button.ui.primary.button")
    await page.wait_for_load_state("load")


async def capture_gitea(base: str, out: Path) -> list[dict]:
    pages = {
        "owner": ["/owner/launch-plan", "/owner/launch-plan/issues", "/owner/launch-plan/issues/1",
                  "/owner/launch-plan/issues/new", "/owner/launch-plan/labels", "/owner/launch-plan/milestones",
                  "/owner/launch-plan/settings", "/owner/launch-plan/settings/collaboration",
                  "/owner/launch-plan/releases/new", "/owner/launch-plan/compare/main...main", "/repo/create",
                  "/user/settings", "/owner/launch-plan/pulls/5", "/owner/launch-plan/pulls/6",
                  "/owner/launch-plan/settings/branches"],
        "viewer": ["/owner/launch-plan", "/owner/launch-plan/issues", "/owner/launch-plan/issues/1",
                   "/owner/launch-plan/labels", "/owner/launch-plan/milestones", "/user/settings",
                   "/owner/launch-plan/pulls/5", "/owner/launch-plan/pulls/6"],
    }
    pages["maintainer"] = ["/owner/launch-plan/pulls/5", "/owner/launch-plan/pulls/6", "/owner/launch-plan",
                           "/owner/launch-plan/issues/1", "/owner/launch-plan/issues/new"]
    passwords = {"owner": "ownerpass123", "viewer": "viewerpass123", "maintainer": "maintainerpass123"}
    records = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        for user, paths in pages.items():
            context = await browser.new_context(viewport=VIEWPORT)
            page = await context.new_page()
            await _gitea_login(page, base, user, passwords[user])
            for path in paths:
                await page.goto(base + path)
                name = f"gitea_{user}_" + (path.strip("/").replace("/", "_").replace(".", "_") or "home")
                records.append(await _save(page, out, name))
            await context.close()
        await browser.close()
    return records


async def capture_lab(out: Path) -> list[dict]:
    """Screens from every app and layout, captured along the exploration and unavailable tasks."""
    from vision_gui_agent.visual_function_lab import LAYOUTS, TASK_SPLIT, TASKS, parse_entry
    from vision_gui_agent.visual_function_lab_server import serve_visual_function_lab
    server = serve_visual_function_lab(0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    evaluator = server.RequestHandlerClass.evaluator
    tasks = [TASKS[parse_entry(entry)[0]] for group in ("exploration", "unavailable") for entry in TASK_SPLIT[group]]
    records, seen = [], set()
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page(viewport=VIEWPORT)
            for layout in LAYOUTS:
                for task in tasks:
                    evaluator.reset(task.initial_state, layout, task.app)
                    for action in task.actions if task.expected_effective else task.setup + (task.target,):
                        section = task.spec.actions[action].section
                        key = (layout, task.app, section, json.dumps(evaluator.state, sort_keys=True, default=str))
                        if key not in seen:
                            seen.add(key)
                            await page.goto(f"{base}/{task.app}?tab={section}")
                            records.append(await _save(page, out, f"lab_{len(records):03d}_{task.app}_{layout}_{section}"))
                        evaluator.act(action)
            await browser.close()
    finally:
        server.shutdown(); server.server_close()
    return records


FRAMEWORKS = {
    # name: (stylesheet, {variant: enabled-button HTML with {label}}); disabled copies add the framework's disabled marker.
    "bootstrap": ("https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css", {
        "primary": '<button class="btn btn-primary"{d}>{label}</button>',
        "secondary": '<button class="btn btn-secondary"{d}>{label}</button>',
        "outline": '<button class="btn btn-outline-primary"{d}>{label}</button>',
        "link": '<button class="btn btn-link"{d}>{label}</button>',
        "light": '<button class="btn btn-light"{d}>{label}</button>'}),
    "bulma": ("https://cdn.jsdelivr.net/npm/bulma@1.0.2/css/bulma.min.css", {
        "primary": '<button class="button is-primary"{d}>{label}</button>',
        "default": '<button class="button"{d}>{label}</button>',
        "outline": '<button class="button is-link is-outlined"{d}>{label}</button>',
        "text": '<button class="button is-text"{d}>{label}</button>',
        "dark": '<button class="button is-dark"{d}>{label}</button>'}),
    "fomantic": ("https://cdn.jsdelivr.net/npm/fomantic-ui@2.9.3/dist/semantic.min.css", {
        "primary": '<button class="ui primary button{c}"{d}>{label}</button>',
        "default": '<button class="ui button{c}"{d}>{label}</button>',
        "basic": '<button class="ui basic button{c}"{d}>{label}</button>',
        "red": '<button class="ui red button{c}"{d}>{label}</button>',
        "basic_primary": '<button class="ui basic primary button{c}"{d}>{label}</button>'}),
    "pico": ("https://cdn.jsdelivr.net/npm/@picocss/pico@2.0.6/css/pico.min.css", {
        "primary": '<button{d}>{label}</button>',
        "secondary": '<button class="secondary"{d}>{label}</button>',
        "contrast": '<button class="contrast"{d}>{label}</button>',
        "outline": '<button class="outline"{d}>{label}</button>',
        "outline_secondary": '<button class="outline secondary"{d}>{label}</button>'}),
}
LABELS = ("Save changes", "Export report", "Approve", "Send invite", "Archive", "Continue")


async def capture_frameworks(out: Path) -> list[dict]:
    """Render each framework's button variants enabled and disabled, plus never-flag distractors."""
    records = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        page = await browser.new_page(viewport=VIEWPORT)
        for name, (stylesheet, variants) in FRAMEWORKS.items():
            for page_index in range(2):
                rows = []
                for row, (variant, template) in enumerate(variants.items()):
                    cells = []
                    for column, label in enumerate(LABELS):
                        # Alternate which half is disabled so each page mixes both states in every variant.
                        disabled = (column + row + page_index) % 2 == 1
                        cells.append(template.format(label=label, d=" disabled" if disabled else "", c=" disabled" if disabled else ""))
                    rows.append(f'<div style="display:flex;gap:16px;align-items:center;margin:14px 0">'
                                f'<span class="muted-label" style="width:120px;color:#9aa1a9">{variant}</span>{"".join(cells)}</div>')
                html = (f'<!doctype html><html><head><link rel="stylesheet" href="{stylesheet}"></head>'
                        f'<body style="padding:32px"><h3>{name} buttons</h3>'
                        f'<input placeholder="Search projects" style="width:320px;margin:8px 0">'
                        f'<p style="color:#a0a6ad">Last updated 3 days ago</p>{"".join(rows)}</body></html>')
                (out / f"framework_{name}_{page_index}.html").parent.mkdir(parents=True, exist_ok=True)
                (out / f"framework_{name}_{page_index}.html").write_text(html)
                await page.set_content(html, wait_until="load")
                records.append(await _save(page, out, f"framework_{name}_{page_index}"))
        await browser.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gitea", default="http://127.0.0.1:3300", help="base URL of the seeded Gitea instance")
    parser.add_argument("--skip-gitea", action="store_true")
    args = parser.parse_args()
    # lab_availability holds the original single-app screens used by the first validation round.
    for name, coroutine in (("lab_apps", capture_lab(ROOT / "lab_apps")),
                            ("framework_styles", capture_frameworks(ROOT / "framework_styles")),
                            ("real_app", None if args.skip_gitea else capture_gitea(args.gitea, ROOT / "real_app"))):
        if coroutine is None: continue
        records = asyncio.run(coroutine)
        (ROOT / name / "labels.json").write_text(json.dumps(records, indent=1) + "\n")
        disabled = sum(control["disabled"] for record in records for control in record["controls"])
        total = sum(len(record["controls"]) for record in records)
        print(f"{name}: {len(records)} screenshots, {total} controls, {disabled} disabled")


if __name__ == "__main__":
    main()
