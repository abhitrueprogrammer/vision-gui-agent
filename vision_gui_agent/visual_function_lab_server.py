"""Locally hosted, screenshot-only Visual Function Lab browser benchmark.

Serves a realistic project-workspace dashboard at /fullsuite. The rendered
HTML and visible text carry only ordinary user-facing labels and feedback --
never predicate names, booleans, evaluator state, or semantic action IDs.
Client controls post an opaque token; only the server maps a token back to
an evaluator action.
"""
from __future__ import annotations

import json
import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlparse

from .visual_function_lab import ACTION_TOKENS, ACTIONS, LAYOUTS, SIDEBAR, TOKEN_ACTIONS, VisualFunctionLabEvaluator


class _Handler(BaseHTTPRequestHandler):
    evaluator = VisualFunctionLabEvaluator()

    def log_message(self, *_): pass
    def _send(self, body: str, content_type: str = "text/html") -> None:
        encoded = body.encode(); self.send_response(200); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(encoded))); self.end_headers(); self.wfile.write(encoded)

    def _send_download(self, body: bytes, filename: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/reset":
            self.evaluator.reset(query.get("state", ["blank"])[0], query.get("layout", ["classic"])[0]); self._send(json.dumps({"ok": True}), "application/json"); return
        if parsed.path == "/score":
            self._send(json.dumps(self.evaluator.visible_state()), "application/json"); return
        if parsed.path == "/admin":
            self._send(self.admin_page()); return
        if parsed.path == "/downloads/launch-brief.pdf":
            self._send_download(_launch_brief_pdf(), "Launch brief.pdf"); return
        if parsed.path == "/fullsuite":
            self._send(self.fullsuite_page(query.get("tab", ["documents"])[0])); return
        self._send(self.index_page())

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/fullsuite":
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
        action = TOKEN_ACTIONS.get(payload.get("token", ""))
        if action is None:
            self.send_response(400); self.end_headers(); return
        effective = self.evaluator.act(action)
        response = {"effective": effective}
        if effective and action == "confirm_export":
            response["download"] = "/downloads/launch-brief.pdf"
        self._send(json.dumps(response), "application/json")

    # -- rendering -----------------------------------------------------
    def _available(self, name: str) -> bool:
        spec = ACTIONS[name]
        return all(self.evaluator.state.get(key) == value for key, value in spec.preconditions.items())

    def _control(self, name: str, *, disabled: bool | None = None) -> str:
        spec = ACTIONS[name]
        blocked = not self._available(name) if disabled is None else disabled
        return (f'<button class="control" {"disabled" if blocked else ""} '
                f'onclick="act(\'{ACTION_TOKENS[name]}\')">{escape(spec.label)}</button>')

    def index_page(self) -> str:
        return ('<!doctype html><title>Project Workspace</title><style>body{font:16px sans-serif;margin:4rem;color:#1f2937}'
                'a{color:#2563eb}</style><h1>Project Workspace</h1>'
                '<p><a href="/fullsuite">Open the project workspace</a></p>')

    def _documents_tab(self) -> str:
        state = self.evaluator.state
        if not state.get("document_open"):
            return (
                '<h2>Documents</h2>'
                '<div class="entry-card">'
                '<div class="entry-meta">Product launch overview &middot; Updated 2 days ago &middot; Morgan Reyes</div>'
                f'{self._control("open_document")}</div>'
            )
        parts = [
            '<h2>Editing Launch Brief</h2>'
            '<div class="entry-meta">Product launch overview &middot; Last edited 2 days ago</div>'
            f'<div class="toolbar">{self._control("open_export_modal")}</div>'
            '<p class="preview">Launch Brief outlines the go-to-market plan, target audience, and rollout timeline for the Q3 release.</p>'
        ]
        if state.get("export_completed"):
            parts.append('<div class="toast">Launch brief.pdf is ready</div>')
        if state.get("export_modal_visible"):
            selected = state.get("export_format") == "pdf"
            parts.append(
                '<div class="modal"><h3>Export Launch Brief</h3>'
                '<p class="hint">Choose a file format for this document.</p>'
                f'<div>{self._control("select_pdf_format")}</div>'
                + ('<div class="hint">PDF document selected</div>' if selected else '')
                + f'<div>{self._control("confirm_export")}</div></div>'
            )
        return "".join(parts)

    def _data_tab(self) -> str:
        state = self.evaluator.state
        if not state.get("dataset_open"):
            return (
                '<h2>Data</h2>'
                '<div class="entry-card">'
                '<div class="entry-meta">Revenue forecast by region &middot; Updated today &middot; 2 regions</div>'
                f'{self._control("open_dataset")}</div>'
            )
        selected = state.get("region_selected")
        return (
            '<h2>Viewing Q3 Forecast</h2>'
            '<div class="entry-meta">Revenue forecast by region &middot; Updated today</div>'
            '<table class="regions"><tr><th>Region</th><th>Projected revenue</th></tr>'
            f'<tr><td>{self._control("select_region")}</td><td>$482,000</td></tr>'
            '<tr><td class="muted">East region</td><td>$317,000</td></tr>'
            '</table>'
            + ('<div class="hint">West region selected</div>' if selected else '')
            + f'<div class="toolbar">{self._control("generate_report")}</div>'
        )

    def _reports_tab(self) -> str:
        if self.evaluator.state.get("report_generated"):
            return '<h2>Reports</h2><div class="card">Q3 Forecast report created</div>'
        return '<h2>Reports</h2><p class="muted">No reports yet. Generate one from the Data section.</p>'

    def _settings_tab(self) -> str:
        state = self.evaluator.state
        if not state.get("settings_open"):
            return (
                '<h2>Settings</h2>'
                '<div class="entry-card">'
                '<div class="entry-meta">Approval, reviewers, and access for this project</div>'
                f'{self._control("open_settings")}</div>'
            )
        if not state.get("authenticated"):
            return (
                '<h2>Project controls</h2>'
                '<div class="entry-card"><div class="entry-title">Sign in required</div>'
                '<div class="entry-meta">Project controls are restricted to signed-in members.</div>'
                f'{self._control("authenticate")}</div>'
            )
        parts = [
            '<h2>Project controls</h2>',
            '<p class="hint">Signed in as Taylor Brooks</p>',
            f'<div class="toolbar">{self._control("enable_approval")}</div>',
        ]
        if state.get("approval_enabled"):
            parts.append('<p class="hint">Approval workflow enabled</p>')
            parts.append(f'<div class="toolbar">{self._control("select_reviewers")}</div>')
            if state.get("reviewers_selected"):
                parts.append('<p class="hint">Reviewers: Alex Kim, Jordan Lee</p>')
            parts.append(f'<div class="toolbar">{self._control("save_settings")}</div>')
            if state.get("settings_saved"):
                parts.append('<div class="toast">Approval workflow saved</div>')
        return "".join(parts)

    def fullsuite_page(self, tab: str) -> str:
        # Only the selected section is ever rendered -- an ordinary dashboard
        # never shows every section at once. Sidebar links are real
        # navigation (a normal <a href>); clicking one changes the page like
        # any real website, no click-token needed for that.
        tab = tab if tab in dict(SIDEBAR) else "documents"
        layout = self.evaluator.layout
        content = {
            "overview": '<h2>Overview</h2><p>Welcome to the project workspace. Use the sidebar to open documents, review data, or manage project settings.</p>',
            "documents": self._documents_tab(), "data": self._data_tab(),
            "reports": self._reports_tab(), "settings": self._settings_tab(),
        }[tab]
        nav = "".join(f'<a class="nav-item{" active" if key == tab else ""}" href="/fullsuite?tab={key}">{label}</a>' for key, label in SIDEBAR)
        return f'''<!doctype html><title>Project Workspace</title><style>{_CSS}</style>
<header class="topbar"><span class="brand">Acme Project Workspace</span>
<button class="reset" onclick="resetWorkspace()">Reset workspace</button></header>
<div class="app {layout}"><nav class="sidebar">{nav}</nav><main class="content">{content}</main></div>
<script>
async function act(token){{const result=await (await fetch('/fullsuite',{{method:'POST',body:JSON.stringify({{token}})}})).json();
if(result.download){{const link=document.createElement('a');link.href=result.download;link.download='Launch brief.pdf';document.body.appendChild(link);link.click();link.remove();setTimeout(()=>location.reload(),500)}}
else{{location.reload()}}}}
async function resetWorkspace(){{await fetch('/reset?state=blank&layout={layout}');location.href='/fullsuite'}}
</script>'''

    def admin_page(self) -> str:
        layout = self.evaluator.layout
        buttons = "".join(f'<button onclick="reset(\'{name}\')">{name.title()}</button>' for name in LAYOUTS)
        return (f'<!doctype html><title>Visual Function Lab controls</title><style>body{{font:18px sans-serif;margin:3rem}}'
                f'button{{margin:.25rem;padding:.7rem 1rem}}</style><h1>Visual Function Lab controls</h1>'
                f'<p>Current layout: <strong>{layout}</strong></p>{buttons}<p><a href="/fullsuite">Open workspace</a></p>'
                f'<script>async function reset(layout){{await fetch(\'/reset?state=blank&layout=\'+layout);location.reload()}}</script>')


_CSS = '''
body{margin:0;font:16px sans-serif;color:#1f2937;background:#f8fafc}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:.9rem 1.5rem;background:#111827;color:#f9fafb}
.topbar .brand{font-weight:600;font-size:17px}
.topbar .reset{padding:.5rem .9rem;background:#1f2937;color:#e5e7eb;border:1px solid #374151;border-radius:4px}
.control{padding:.6rem 1rem;border:1px solid #334155;background:#fff;font-size:15px;min-height:44px;min-width:120px}
.control:disabled{opacity:.35;filter:grayscale(1)}
.entry-card{border:1px solid #cbd5e1;background:#fff;padding:1rem 1.2rem;max-width:420px;border-radius:6px}
.entry-title{font-size:18px;font-weight:600;margin-bottom:.3rem}
.entry-meta{color:#64748b;font-size:14px;margin-bottom:.8rem}
.preview{max-width:520px;color:#334155;line-height:1.5}
.toast,.card{margin-top:1rem;padding:.8rem 1rem;background:#dcfce7;border:1px solid #16a34a;display:inline-block}
.hint{color:#334155;display:block;margin:.5rem 0}
.muted{color:#94a3b8}
.modal{margin-top:1rem;padding:1.2rem;border:2px solid #334155;background:#fff;max-width:360px}
.regions{border-collapse:collapse;margin-top:.5rem}
.regions th,.regions td{padding:.6rem 1rem;text-align:left;border-bottom:1px solid #e2e8f0}
.toolbar{margin-top:.8rem}
.app.classic{display:flex;min-height:calc(100vh - 56px)}
.app.classic .sidebar{width:220px;flex:none;background:#0f172a;padding:2rem 0}
.app.classic .sidebar .nav-item{display:block;padding:.8rem 1.5rem;color:#cbd5e1;text-decoration:none}
.app.classic .sidebar .nav-item.active{background:#1e293b;color:#fff}
.app.classic .content{flex:1;padding:2.5rem}
.app.compact{display:block}
.app.compact .sidebar{display:flex;background:#0f172a;padding:0}
.app.compact .sidebar .nav-item{padding:1rem 1.2rem;color:#cbd5e1;text-decoration:none}
.app.compact .sidebar .nav-item.active{background:#1e293b;color:#fff}
.app.compact .content{padding:1.5rem}
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
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local Visual Function Lab benchmark")
    parser.add_argument("--port", type=int, default=4200)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"Visual Function Lab listening at http://127.0.0.1:{args.port}/fullsuite")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__": main()
