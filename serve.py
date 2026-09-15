#!/usr/bin/env python3
"""Serve Embodied Fly Lab on localhost with no external dependencies."""

from __future__ import annotations

import argparse
import functools
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Embodied Fly Lab locally")
    parser.add_argument("--port", type=int, default=0, help="port; 0 selects a free port")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    handler = functools.partial(NoCacheHandler, directory=ROOT)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Embodied Fly Lab: {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
