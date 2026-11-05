#!/usr/bin/env python3
"""
Run backend and Next.js cockpit together for local development.
"""

from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Trader app in local dev mode.")
    parser.add_argument("--api-host", default="127.0.0.1", help="Backend bind host")
    parser.add_argument("--api-port", default=8000, type=int, help="Backend port")
    parser.add_argument("--front-host", default="127.0.0.1", help="Frontend bind host")
    parser.add_argument("--front-port", default=3000, type=int, help="Frontend port")
    parser.add_argument(
        "--no-npm-install-check",
        action="store_true",
        help="Skip frontend dependency check",
    )
    parser.add_argument(
        "--install-frontend-deps",
        action="store_true",
        help="Run npm install automatically when frontend/node_modules is missing",
    )
    parser.add_argument(
        "--ready-timeout",
        default=30,
        type=int,
        help="Süreçlerin hazır olması için bekleme süresi (saniye)",
    )
    parser.add_argument(
        "--check-interval",
        default=1.0,
        type=float,
        help="Ready-check aralığı (saniye)",
    )
    parser.add_argument(
        "--shutdown-timeout",
        default=5,
        type=float,
        help="Kapanışta graceful bekleme süresi (saniye)",
    )
    return parser.parse_args()


def _log(level: str, message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{level}] {now} {message}")


def _wait_for_url(
    url: str,
    timeout_seconds: int,
    check_interval: float,
    watched_processes: list[tuple[subprocess.Popen, str]] | None = None,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass

        if watched_processes is not None:
            for proc, name in watched_processes:
                if proc and proc.poll() is not None:
                    raise RuntimeError(f"{name} beklenmedik şekilde kapandı (code={proc.returncode}).")

        time.sleep(check_interval)
    return False


def _run_sync_command(command: list[str], cwd: Path | None = None) -> None:
    _log("INFO", f"Komut çalıştırılıyor: {' '.join(command)}")
    subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=True,
    )


def _run_command(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    kwargs = {"cwd": str(cwd) if cwd else None}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["preexec_fn"] = os.setsid
    if env is not None:
        kwargs["env"] = env

    return subprocess.Popen(command, **{k: v for k, v in kwargs.items() if v is not None})


def _terminate(processes: list[subprocess.Popen], timeout: float) -> None:
    if not processes:
        return

    _log("INFO", f"Kapanma başladı (graceful timeout={timeout:.1f}s)")

    for proc in processes:
        if proc and proc.poll() is None:
            try:
                if os.name != "nt":
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
            except ProcessLookupError:
                continue
            except Exception:
                proc.terminate()

    for proc in processes:
        if proc and proc.poll() is None:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    if os.name != "nt":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                except Exception:
                    continue


def _assert_process_alive(process: subprocess.Popen, name: str) -> None:
    if process.poll() is not None:
        raise RuntimeError(
            f"{name} hemen çıktı (pid={process.pid}, code={process.returncode}). "
            "Başlangıç başarısız olabilir."
        )


def main() -> None:
    args = parse_args()
    return_code = 0

    if shutil.which("npm") is None:
        _log("ERROR", "npm bulunamadı. Node.js kurulu değilse yükleyin.")
        sys.exit(1)

    if not args.no_npm_install_check and not (FRONTEND_DIR / "node_modules").exists():
        if args.install_frontend_deps:
            _log("INFO", "frontend/node_modules bulunamadı, otomatik olarak npm install çalıştırılıyor.")
            _run_sync_command(["npm", "install"], cwd=FRONTEND_DIR)
        else:
            _log("WARN", "frontend/node_modules bulunamadı, `npm install` gerekiyor.")
            print("Çalıştırın: cd frontend && npm install")
            print("Veya bu scripti `--install-frontend-deps` ile çalıştırın.")
            sys.exit(1)

    if not (FRONTEND_DIR / "package.json").exists():
        _log("ERROR", "frontend/package.json bulunamadı.")
        sys.exit(1)

    api_cmd = [
        sys.executable,
        "run_api.py",
        "--host",
        args.api_host,
        "--port",
        str(args.api_port),
        "--reload",
    ]
    frontend_cmd = [
        "npm",
        "run",
        "dev",
        "--",
        "--hostname",
        args.front_host,
        "--port",
        str(args.front_port),
    ]
    frontend_env = os.environ.copy()
    frontend_env["NEXT_PUBLIC_API_BASE"] = f"http://{args.api_host}:{args.api_port}"

    backend_proc = frontend_proc = None
    procs = []

    try:
        _log("INFO", "Backend başlatılıyor: run_api.py")
        backend_proc = _run_command(api_cmd, cwd=ROOT)
        procs.append(backend_proc)
        _assert_process_alive(backend_proc, "Backend")

        health_url = f"http://{args.api_host}:{args.api_port}/api/v1/health"
        _log("INFO", f"Backend sağlık kontrolü: {health_url}")
        if not _wait_for_url(
            health_url,
            timeout_seconds=args.ready_timeout,
            check_interval=args.check_interval,
            watched_processes=[(backend_proc, "Backend")],
        ):
            _log("ERROR", "Backend sağlık kontrolü zaman aşımına uğradı.")
            raise RuntimeError("Backend did not become ready in time.")

        _log("INFO", "Frontend başlatılıyor: npm run dev")
        frontend_proc = _run_command(frontend_cmd, cwd=FRONTEND_DIR, env=frontend_env)
        procs.append(frontend_proc)
        _assert_process_alive(frontend_proc, "Frontend")

        cockpit_url = f"http://{args.front_host}:{args.front_port}"
        _log("INFO", f"Cockpit health kontrolü: {cockpit_url}")
        if not _wait_for_url(
            cockpit_url,
            timeout_seconds=args.ready_timeout,
            check_interval=args.check_interval,
            watched_processes=[(backend_proc, "Backend"), (frontend_proc, "Frontend")],
        ):
            _log("ERROR", "Cockpit sağlık kontrolü zaman aşımına uğradı.")
            raise RuntimeError("Cockpit did not become ready in time.")

        _log(
            "READY",
            "Backend: http://{}:{} | Cockpit: http://{}:{}".format(
                args.api_host,
                args.api_port,
                args.front_host,
                args.front_port,
            ),
        )
        print("Çıkmak için Ctrl+C")

        while True:
            for proc in [backend_proc, frontend_proc]:
                if proc and proc.poll() is not None:
                    _log("STOP", f"Process exited early: pid={proc.pid}, code={proc.returncode}")
                    raise RuntimeError("Alt süreç erken sonlandı.")
            time.sleep(1)

    except KeyboardInterrupt:
        return_code = 0
        _log("INFO", "Kullanıcı tarafından durduruldu.")
    except Exception as exc:
        _log("ERROR", str(exc))
        return_code = 1
    finally:
        _terminate([p for p in procs if p], timeout=args.shutdown_timeout)
        if return_code != 0:
            sys.exit(return_code)


if __name__ == "__main__":
    main()
