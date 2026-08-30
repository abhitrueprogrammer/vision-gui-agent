"""Locally hosted, screenshot-only Visual Function Lab browser benchmark."""
from __future__ import annotations

import json
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlparse

from .visual_function_lab import ACTIONS, BUTTON_COLORS, STATUS_COLORS, VisualFunctionLabEvaluator


class _Handler(BaseHTTPRequestHandler):
    evaluator = VisualFunctionLabEvaluator()

    def log_message(self, *_): pass
    def _send(self, body: str, content_type: str = "text/html") -> None:
        encoded = body.encode(); self.send_response(200); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(encoded))); self.end_headers(); self.wfile.write(encoded)
    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        if self.path.startswith("/reset"):
            self.evaluator.reset(query.get("state", ["blank"])[0], query.get("layout", ["classic"])[0]); self._send(json.dumps({"ok": True}), "application/json"); return
        if self.path.startswith("/score"):
            self._send(json.dumps(self.evaluator.visible_state()), "application/json"); return
        self._send(self.page())
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
        self._send(json.dumps({"effective": self.evaluator.act(payload["action"])}), "application/json")
    def page(self) -> str:
        values = self.evaluator.visible_state(); layout = self.evaluator.layout
        status = "<br>".join(
            f'<span class="state-chip" style="background:{STATUS_COLORS[f"{key.replace('_', ' ')}: {str(value).lower()}"]}"></span>'
            f"{key.replace('_', ' ')}: {str(value).lower()}" for key, value in sorted(values.items())
        ) or "Ready"
        groups = []
        for workspace in ("document", "data", "account"):
            buttons = "".join(
                f'<button {"" if all(self.evaluator.state.get(key) == value for key, value in spec.preconditions.items()) else "disabled"} '
                f'style="background:{BUTTON_COLORS[name]}" onclick="act(\'{name}\')">{spec.label}</button>'
                for name, spec in ACTIONS.items() if spec.workspace == workspace
            )
            groups.append(f"<section><h2>{workspace.title()} workspace</h2>{buttons}</section>")
        return f'''<!doctype html><title>Visual Function Lab</title><style>body{{font:18px sans-serif;margin:3rem}}main{{display:block}}button{{padding:.8rem 1rem;border:1px solid #334155;color:#111;min-width:190px;min-height:52px;font-size:16px}}button:disabled{{opacity:.35;filter:grayscale(1)}}.compact button{{font-size:14px;min-width:165px}}.high_contrast{{background:#000;color:#fff}}#state{{padding:1rem;border:2px solid currentColor;min-height:22px}}.state-chip{{display:inline-block;width:14px;height:14px;margin-right:7px;vertical-align:middle}}section{{margin-top:1.5rem;display:grid;grid-template-columns:repeat(3,max-content);gap:14px 18px}}section h2{{grid-column:1/-1;margin:0}}</style><main class="{layout}"><h1>Visual Function Lab</h1><p id="state">{status}</p>{''.join(groups)}</main><script>async function act(action){{await fetch('/',{{method:'POST',body:JSON.stringify({{action}})}});location.reload()}}</script>'''


def serve_visual_function_lab(port: int = 4200) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local Visual Function Lab benchmark")
    parser.add_argument("--port", type=int, default=4200)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"Visual Function Lab listening at http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__": main()
