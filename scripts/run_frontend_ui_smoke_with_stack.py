#!/usr/bin/env python3
"""Run backend+frontend stack and execute browser-based UI smoke test.

Stack is started by run_dev.py, then Playwright smoke test is invoked.
Everything is shut down cleanly in the end.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOST = "127.0.0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stack and execute Playwright UI smoke test automatically."
    )
    parser.add_argument("--api-host", default=HOST, help="Backend bind host")
    parser.add_argument("--api-port", default=8010, type=int, help="Backend port")
    parser.add_argument("--front-host", default=HOST, help="Frontend bind host")
    parser.add_argument("--front-port", default=3010, type=int, help="Frontend port")
    parser.add_argument(
        "--ready-timeout",
        default=20,
        type=float,
        help="Hazır olma bekleme süresi (saniye)",
    )
    parser.add_argument(
        "--check-interval",
        default=0.5,
        type=float,
        help="Hazır olma kontrol aralığı (saniye)",
    )
    parser.add_argument(
        "--shutdown-timeout",
        default=5,
        type=float,
        help="Durdurma için graceful bekleme (saniye)",
    )
    return parser.parse_args()


def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[SMOKE-UI-STACK] {now} {message}")


def _wait_for_url(
    url: str,
    timeout_seconds: float,
    check_interval: float,
    proc: subprocess.Popen | None = None,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(check_interval)
    return False


def _run_stack(
    api_host: str,
    api_port: int,
    front_host: str,
    front_port: int,
    ready_timeout: float,
    check_interval: float,
    shutdown_timeout: float,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-u",
        "run_dev.py",
        "--no-npm-install-check",
        "--api-host",
        api_host,
        "--api-port",
        str(api_port),
        "--front-host",
        front_host,
        "--front-port",
        str(front_port),
        "--ready-timeout",
        str(int(ready_timeout)),
        "--check-interval",
        str(check_interval),
        "--shutdown-timeout",
        str(shutdown_timeout),
    ]
    log(f"run_dev başlatılıyor: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT)

    api_url = f"http://{api_host}:{api_port}/api/v1/health"
    front_url = f"http://{front_host}:{front_port}"

    if not _wait_for_url(api_url, ready_timeout, check_interval, proc=proc):
        raise RuntimeError("Backend hazır olamadı.")

    if not _wait_for_url(front_url, ready_timeout, check_interval, proc=proc):
        raise RuntimeError("Frontend hazır olamadı.")

    return proc


def _terminate(proc: subprocess.Popen | None, timeout: float) -> None:
    if proc is None or proc.poll() is not None:
        return

    try:
        proc.send_signal(signal.SIGINT)
    except Exception:
        proc.terminate()

    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def main() -> int:
    args = parse_args()

    if shutil.which("node") is None:
        log("node bulunamadı. Playwright e2e çalıştırılamaz.")
        return 1

    if not (PROJECT_ROOT / "frontend").exists():
        log("frontend dizini bulunamadı.")
        return 1

    proc: subprocess.Popen | None = None

    try:
        proc = _run_stack(
            api_host=args.api_host,
            api_port=args.api_port,
            front_host=args.front_host,
            front_port=args.front_port,
            ready_timeout=args.ready_timeout,
            check_interval=args.check_interval,
            shutdown_timeout=args.shutdown_timeout,
        )

        api_base = f"http://{args.api_host}:{args.api_port}"
        front_base = f"http://{args.front_host}:{args.front_port}"
        cmd = [
            "node",
            "scripts/run_frontend_ui_smoke.mjs",
            "--api-base",
            api_base,
            "--front-base",
            front_base,
            "--timeout",
            str(int(args.ready_timeout)),
        ]
        log(f"UI smoke senaryosu çalıştırılıyor: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            log(f"UI smoke başarısız: exit={result.returncode}")
            return result.returncode
        log("UI smoke başarılı, stack güvenli şekilde durduruluyor.")
        return 0
    except urllib.error.URLError as exc:
        log(f"Hazır olma kontrolü sırasında bağlantı hatası: {exc}")
        return 1
    except Exception as exc:
        log(f"Hata: {exc}")
        return 1
    finally:
        if proc is not None and proc.poll() is None:
            log("Stack durduruluyor.")
        _terminate(proc, timeout=args.shutdown_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
