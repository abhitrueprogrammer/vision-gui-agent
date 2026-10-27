"""Locally hosted, screenshot-only Visual Function Lab apps.

Serves each benchmark app at /<app>?tab=<section>. The rendered HTML carries
only ordinary user-facing labels and messages, never predicate names,
booleans, evaluator state or semantic action ids. Controls post an opaque
token; only the server maps it back to an action. Every server instance owns
its own evaluator, so several servers (shards, ground-truth rendering) never
share state.
"""
from __future__ import annotations

import argparse
import json
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlencode, urlparse

from .visual_function_lab import (APPS, DEFAULT_APP, LAYOUTS, PROTOCOL_VERSION, TASKS, AppSpec, VisualFunctionLabEvaluator,
                                  action_token, TOKEN_ACTIONS)

# The original single-app URL keeps working for older tools and notes.
ALIASES = {"fullsuite": "workspace"}


class _Handler(BaseHTTPRequestHandler):
    evaluator = VisualFunctionLabEvaluator()

    def log_message(self, *_): pass

    def _send(self, body: str, content_type: str = "text/html") -> None:
        encoded = body.encode(); self.send_response(200); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(encoded))); self.end_headers(); self.wfile.write(encoded)

    def _redirect(self, location: str) -> None:
        self.send_response(302); self.send_header("Location", location); self.send_header("Content-Length", "0"); self.end_headers()

    def _send_download(self, body: bytes, filename: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    @staticmethod
    def _app_id(path: str) -> str | None:
        name = path.strip("/").split("/")[0]
        name = ALIASES.get(name, name)
        return name if name in APPS else None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        first = lambda key, default=None: query.get(key, [default])[0]
        if parsed.path == "/reset":
            try:
                self.evaluator.reset(first("state", "blank"), first("layout", "classic"), first("app", self.evaluator.app_id))
            except ValueError as error:
                self.send_response(400); self.end_headers(); self.wfile.write(str(error).encode()); return
            self._send(json.dumps({"ok": True}), "application/json"); return
        if parsed.path == "/score":
            task = first("task")
            if task is None:
                self._send(json.dumps(self.evaluator.visible_state()), "application/json"); return
            if task not in TASKS:
                self.send_response(404); self.end_headers(); return
            # Evaluator-side outcome for a live agent run; never expose this to the agent.
            report = self.evaluator.score_agent(task, first("declared", "0") == "1")
            report.update(protocol=PROTOCOL_VERSION, symbolic=self.evaluator.score_symbolic(task), layout=self.evaluator.layout,
                          state=self.evaluator.visible_state())
            self._send(json.dumps(report), "application/json"); return
        if parsed.path == "/admin":  # the controls moved to the index page; old links still work
            self._redirect("/"); return
        for app in APPS.values():
            for action, (url, filename) in app.downloads.items():
                if parsed.path == url:
                    self._send_download(_launch_brief_pdf(), filename); return
        app_id = self._app_id(parsed.path)
        if app_id is not None:
            app = APPS[app_id]
            tab = first("tab")
            # One canonical URL per section, so a section has exactly one scope.
            if tab not in dict(app.sections) or parsed.path.strip("/") != app_id:
                self._redirect(f"/{app_id}?" + urlencode({"tab": tab if tab in dict(app.sections) else app.sections[0][0]})); return
            if self.evaluator.app_id != app_id:
                self.evaluator.reset("blank", self.evaluator.layout, app_id)
            self._send(self.app_page(app, tab)); return
        self._send(self.index_page())

    def do_POST(self) -> None:
        app_id = self._app_id(urlparse(self.path).path)
        if app_id is None:
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
        target = TOKEN_ACTIONS.get(payload.get("token", ""))
        if target is None or target[0] != self.evaluator.app_id:
            self.send_response(400); self.end_headers(); return
        action = target[1]
        effective = self.evaluator.act(action)
        response = {"effective": effective}
        if effective and action in self.evaluator.app.downloads:
            response["download"], response["filename"] = self.evaluator.app.downloads[action]
        self._send(json.dumps(response), "application/json")

    # -- rendering -----------------------------------------------------
    def _control(self, app: AppSpec, name: str) -> str:
        spec, state = app.actions[name], self.evaluator.state
        # Error-style controls stay clickable and explain themselves when used too early.
        blocked = spec.style == "disabled" and not app.available(state, name)
        return (f'<button class="control {spec.tone}" {"disabled" if blocked else ""} '
                f'onclick="act(\'{action_token(app.id, name)}\')">{escape(spec.label)}</button>')

    def _section(self, app: AppSpec, tab: str) -> str:
        state = self.evaluator.state
        holds = lambda when: all(state.get(key) == value for key, value in when.items())
        headings = [spec.evidence for spec in app.actions.values() if spec.section == tab and spec.heading
                    and holds(next(({key: value} for key, value in spec.effects.items() if value not in (False, None)), {}))]
        heading = headings[0] if headings else dict(app.sections)[tab]
        lines = [status.text for status in app.all_statuses() if status.section == tab and holds(status.when) and status.text != heading]
        parts = [f"<h2>{escape(heading)}</h2>"]
        if tab in app.static: parts.append(f'<p class="meta">{escape(app.static[tab])}</p>')
        parts += [f'<p class="status">{escape(text)}</p>' for text in lines]
        controls = [self._control(app, name) for name, spec in app.actions.items() if spec.section == tab and app.rendered(state, name)]
        if self.evaluator.layout == "reordered": controls.reverse()
        if controls: parts.append(f'<div class="toolbar">{"".join(controls)}</div>')
        error = self.evaluator.take_error(tab)
        if error: parts.append(f'<div class="toast error">{escape(error)}</div>')
        return "".join(parts)

    def app_page(self, app: AppSpec, tab: str) -> str:
        layout = self.evaluator.layout
        nav = "".join(f'<a class="nav-item{" active" if key == tab else ""}" href="/{app.id}?tab={key}">{escape(label)}</a>'
                      for key, label in app.sections)
        return f'''<!doctype html><title>{escape(app.title)}</title><style>{_CSS}</style>
<header class="topbar"><span class="brand">{escape(app.title)}</span></header>
<div class="app {layout}"><nav class="sidebar">{nav}</nav><main class="content">{self._section(app, tab)}</main></div>
<script>
async function act(token){{const result=await (await fetch('/{app.id}',{{method:'POST',body:JSON.stringify({{token}})}})).json();
if(result.download){{const link=document.createElement('a');link.href=result.download;link.download=result.filename;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>location.reload(),500)}}
else{{location.reload()}}}}
</script>'''

    def index_page(self) -> str:
        """The apps and the evaluator controls for people running the benchmark; agents start on the app pages."""
        app = self.evaluator.app
        links = "".join(f'<li><a href="/{item.id}">{escape(item.title)}</a></li>' for item in APPS.values())
        buttons = "".join(f'<button onclick="reset(\'{name}\')">{name.title()}</button>' for name in LAYOUTS)
        return ('<!doctype html><title>Visual Function Lab</title><style>body{font:18px sans-serif;margin:3rem;color:#1f2937}'
                'a{color:#2563eb}li{margin:.4rem 0}button{margin:.25rem;padding:.7rem 1rem}</style>'
                f'<h1>Visual Function Lab</h1><ul>{links}</ul><p>Opening a different app starts it from a clean state.</p>'
                f'<h2>Layout</h2><p>Current: <strong>{self.evaluator.layout}</strong> (all apps). '
                f'A button sets the layout and resets <strong>{escape(app.title)}</strong>, the app last opened.</p>{buttons}'
                f"<script>async function reset(layout){{await fetch('/reset?state=blank&app={app.id}&layout='+layout);location.reload()}}</script>")


_CSS = '''
body{margin:0;font:16px sans-serif;color:#1f2937;background:#f8fafc}
.topbar{display:flex;align-items:center;padding:.9rem 1.5rem;background:#111827;color:#f9fafb}
.topbar .brand{font-weight:600;font-size:17px}
.control{padding:.6rem 1rem;border:1px solid #334155;background:#fff;color:#1f2937;font-size:15px;min-height:44px;min-width:120px;margin-right:.6rem}
.control.primary{background:#1d4ed8;border-color:#1e40af;color:#fff}
.control:disabled{opacity:.35;filter:grayscale(1)}
.meta{color:#64748b;font-size:14px;margin:.2rem 0 1rem}
.status{color:#334155;margin:.45rem 0}
.toolbar{margin-top:1rem}
.toast{margin-top:1rem;padding:.8rem 1rem;background:#dcfce7;border:1px solid #16a34a;display:inline-block}
.toast.error{background:#fee2e2;border-color:#dc2626;color:#7f1d1d}
.app.classic{display:flex;min-height:calc(100vh - 56px)}
.app.classic .sidebar{width:220px;flex:none;background:#0f172a;padding:2rem 0}
.app.classic .sidebar .nav-item{display:block;padding:.8rem 1.5rem;color:#cbd5e1;text-decoration:none}
.app.classic .content{flex:1;padding:2.5rem}
.app.compact{display:block}
.app.compact .sidebar{display:flex;background:#0f172a;padding:0}
.app.compact .sidebar .nav-item{padding:1rem 1.2rem;color:#cbd5e1;text-decoration:none}
.app.compact .content{padding:1.5rem}
.app.reordered{display:flex;flex-direction:row-reverse;min-height:calc(100vh - 56px)}
.app.reordered .sidebar{width:260px;flex:none;background:#1e293b;padding:3rem 0}
.app.reordered .sidebar .nav-item{display:block;padding:1.1rem 2rem;color:#e2e8f0;text-decoration:none}
.app.reordered .content{flex:1;padding:3.5rem 4rem}
.sidebar .nav-item.active{background:#1e293b;color:#fff}
'''


def _launch_brief_pdf() -> bytes:
    """Return a small, valid PDF artifact for the export workflow."""
    stream = b"BT /F1 22 Tf 72 720 Td (Launch Brief) Tj 0 -36 Td /F1 12 Tf (Q3 go-to-market plan and rollout timeline.) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(output)); output.extend(f"{number} 0 obj\n".encode()); output.extend(body); output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]: output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(output)


def serve_visual_function_lab(port: int = 4200) -> ThreadingHTTPServer:
    """Start a lab server with its own evaluator on a background thread."""
    handler = type("LabHandler", (_Handler,), {"evaluator": VisualFunctionLabEvaluator()})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local Visual Function Lab apps")
    parser.add_argument("--port", type=int, default=4200)
    args = parser.parse_args()
    handler = type("LabHandler", (_Handler,), {"evaluator": VisualFunctionLabEvaluator()})
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Visual Function Lab listening at http://127.0.0.1:{args.port}/ (apps: {', '.join(APPS)}, default {DEFAULT_APP})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__": main()
