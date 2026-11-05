#!/usr/bin/env python3
"""
Runs the FastAPI backend API for local development.
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Trader API (FastAPI)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", args.host, "--port", str(args.port)]
    if args.reload:
        command.append("--reload")

    # Keep startup transparent for operators:
    print(f"API starting: {' '.join(command)}")
    from subprocess import run

    run(command, check=True)


if __name__ == "__main__":
    main()
