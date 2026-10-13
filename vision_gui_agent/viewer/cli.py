from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse recorded agent runs. Read-only: never modifies artifacts.")
    parser.add_argument("--artifacts", default="artifacts", help="directory searched for runs-v2.sqlite3 roots")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; screenshots may contain sensitive data, so keep this local")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the viewer in the default browser")
    args = parser.parse_args()
    artifacts = Path(args.artifacts)
    if not artifacts.is_dir():
        parser.error(f"artifact directory not found: {artifacts}")

    import uvicorn
    from .app import create_app

    url = f"http://{args.host}:{args.port}/"
    print(f"Serving {artifacts.resolve()} at {url}")
    if args.open:
        threading.Timer(1.0, webbrowser.open, (url,)).start()
    uvicorn.run(create_app(artifacts), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
