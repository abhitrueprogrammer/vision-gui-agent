from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import data

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=HERE / "templates")


def duration(milliseconds: float | None) -> str:
    if milliseconds is None:
        return "—"
    seconds = milliseconds / 1000
    return f"{milliseconds:.0f} ms" if seconds < 1 else f"{seconds:.1f} s" if seconds < 120 else f"{seconds / 60:.1f} min"


templates.env.filters["duration"] = duration


class RootRegistry:
    """Discovered artifact roots, rescanned on a TTL and immediately on a lookup miss."""

    def __init__(self, base: Path, ttl: float = 15.0) -> None:
        self.base, self.ttl = base, ttl
        self._roots: dict[str, data.Root] = {}
        self._scanned_at = float("-inf")

    def rescan(self) -> None:
        self._roots = {root.slug: root for root in data.discover_roots(self.base)}
        self._scanned_at = time.monotonic()

    def all(self) -> list[data.Root]:
        if time.monotonic() - self._scanned_at > self.ttl:
            self.rescan()
        return list(self._roots.values())

    def get(self, slug: str) -> data.Root | None:
        self.all()
        if slug not in self._roots:
            self.rescan()
        return self._roots.get(slug)


class GraphResponse(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    total_nodes: int
    total_edges: int
    filtered: bool
    run_missing: bool = False


class MetricsResponse(BaseModel):
    summary: dict[str, Any] | None
    by_model: list[dict[str, Any]]
    statuses: dict[str, int]
    verification: dict[str, int]
    decision_origins: dict[str, int]
    decision_sources: dict[str, int]
    latency: list[dict[str, Any]]


def create_app(base: Path, ttl: float = 15.0) -> FastAPI:
    app = FastAPI(title="Vision GUI Agent viewer", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.registry = registry = RootRegistry(base, ttl)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.exception_handler(sqlite3.OperationalError)
    async def database_busy(request: Request, error: sqlite3.OperationalError):
        # The agent holds a write lock while committing a step; htmx retries on its next poll.
        if "locked" in str(error) or "busy" in str(error):
            return PlainTextResponse("Database busy, retry shortly.", status_code=503, headers={"Retry-After": "2"})
        raise error

    def get_root(root: str) -> data.Root:
        found = registry.get(root)
        if found is None:
            raise HTTPException(404, "Unknown artifact root")
        return found

    def render(request: Request, name: str, **context: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, {"roots": registry.all(), **context})

    @app.get("/", response_class=HTMLResponse)
    def runs_page(request: Request, root: str | None = None, status: str | None = None, model: str | None = None,
                  q: str | None = None, rescan: bool = False, auto: bool = False):
        if rescan:
            registry.rescan()
        roots = registry.all()
        selected = registry.get(root) if root else next((item for item in roots if item.slug != data.TOP_LEVEL_SLUG), None) or (roots[0] if roots else None)
        if root and selected is None:
            raise HTTPException(404, "Unknown artifact root")
        runs = data.list_runs(selected, status or None, model or None, q or None) if selected else []
        options = data.filter_options(selected) if selected else {"status": [], "model": []}
        context = {"root": selected, "runs": runs, "options": options, "auto": auto,
                   "filters": {"status": status or "", "model": model or "", "q": q or ""}}
        if request.headers.get("HX-Target") == "runs-table":
            return render(request, "_runs_table.html", **context)
        return render(request, "runs.html", base=registry.base, **context)

    @app.get("/runs/{root}/{run_id}", response_class=HTMLResponse)
    def run_page(request: Request, run_id: str, step: int = 0, root: data.Root = Depends(get_root)):
        run = data.run_detail(root, run_id)
        if run is None:
            raise HTTPException(404, "Unknown run")
        steps = run["steps"]
        current = steps[min(max(step, 0), len(steps) - 1)] if steps else None
        return render(request, "run.html", root=root, run=run, current=current)

    @app.get("/graph/{root}", response_class=HTMLResponse)
    def graph_page(request: Request, run: str | None = None, root: data.Root = Depends(get_root)):
        return render(request, "graph.html", root=root, run_id=run)

    @app.get("/api/graph/{root}", response_model=GraphResponse)
    def graph_api(run: str | None = None, all: bool = False, expand: bool = False, root: data.Root = Depends(get_root)):
        graph = data.graph_elements(root, run, show_all=all, expand=expand)
        if graph is None:
            raise HTTPException(404, "No state graph in this artifact root")
        return graph

    @app.get("/metrics/{root}", response_class=HTMLResponse)
    def metrics_page(request: Request, root: data.Root = Depends(get_root)):
        return render(request, "metrics.html", root=root, metrics=data.run_metrics(root))

    @app.get("/api/metrics/{root}", response_model=MetricsResponse)
    def metrics_api(root: data.Root = Depends(get_root)):
        return data.run_metrics(root)

    def send(root: data.Root, kind: str, run_id: str | None, name: str) -> FileResponse:
        path = data.resolve_artifact(root, kind, run_id, name)
        if path is None:
            raise HTTPException(404, "Artifact not found")
        disposition = "attachment" if path.suffix.lower() == ".pdf" else "inline"
        return FileResponse(path, filename=path.name, content_disposition_type=disposition,
                            headers={"Cache-Control": "max-age=86400"})

    @app.get("/file/{root}/run/{run_id}/{name}")
    def run_file(run_id: str, name: str, root: data.Root = Depends(get_root)):
        return send(root, "run", run_id, name)

    @app.get("/file/{root}/dl/{run_id}/{name}")
    def download_file(run_id: str, name: str, root: data.Root = Depends(get_root)):
        return send(root, "dl", run_id, name)

    @app.get("/file/{root}/snap/{name}")
    def snapshot_file(name: str, root: data.Root = Depends(get_root)):
        return send(root, "snap", None, name)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException):
        if request.url.path.startswith(("/api/", "/file/")):
            return JSONResponse({"detail": error.detail}, status_code=error.status_code, headers=error.headers)
        return templates.TemplateResponse(request, "error.html", {"roots": registry.all(), "error": error},
                                          status_code=error.status_code)

    return app
