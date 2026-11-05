#!/usr/bin/env python3
"""Smoke check for local dev orchestration (backend + Next.js cockpit)."""

from __future__ import annotations

import argparse
import json
import importlib.util
import signal
import socket
import subprocess
import sys
import re
from typing import Any, Dict, Tuple
import time
import urllib.request
from datetime import datetime
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short run_dev smoke check.")
    parser.add_argument("--api-port", default=8010, type=int)
    parser.add_argument("--front-port", default=3010, type=int)
    parser.add_argument("--duration", default=2, type=int, help="Çalışma süresi (saniye)")
    parser.add_argument("--ready-timeout", default=20, type=int)
    parser.add_argument("--check-interval", default=0.5, type=float)
    parser.add_argument("--shutdown-timeout", default=5, type=float)
    return parser.parse_args()


def _log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[SMOKE] {now} {message}")


def _wait_for_url(url: str, timeout_seconds: float, check_interval: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(check_interval)
    return False


def _request_json(
    method: str,
    url: str,
    payload: Dict[str, Any] | None = None,
    headers: Dict[str, str] | None = None,
) -> Tuple[int, Dict[str, Any]]:
    data = None
    request_headers: Dict[str, str] = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        data = body
        request_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, method=method.upper(), headers=request_headers)
    with urllib.request.urlopen(req, timeout=3) as response:
        status = response.status
        payload_text = response.read().decode("utf-8")
        payload_json: Dict[str, Any] = json.loads(payload_text) if payload_text else {}
        return status, payload_json


def _request_text(url: str, timeout_seconds: int = 3) -> Tuple[int, str, str]:
    req = urllib.request.Request(url, method="GET", headers={"Accept": "text/html,*/*;q=0.8"})
    with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
        status = response.status
        content_type = response.headers.get_content_type()
        payload_bytes = response.read()
        if "text" in content_type:
            payload_text = payload_bytes.decode("utf-8")
        else:
            payload_text = payload_bytes.decode("utf-8", errors="ignore")
        return status, content_type, payload_text


def _extract_chunk_urls(html: str) -> list[str]:
    script_srcs = re.findall(
        r'<script[^>]+src=(["\'])([^"\']+)\1',
        html,
        flags=re.IGNORECASE,
    )
    candidates = [src for _, src in script_srcs]

    attr_free_matches = re.findall(r'"([^"]*?_next/static/chunks[^"]*page[^"]*\.js[^"]*)"', html)
    attr_free_matches.extend(
        re.findall(r"'([^']*?_next/static/chunks[^']*page[^']*\.js[^']*)'", html)
    )
    candidates.extend(attr_free_matches)

    normalized: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in normalized:
            normalized.append(candidate)

    if not normalized:
        return []

    prioritized = sorted(
        normalized,
        key=lambda value: ("/_next/static/chunks/app/page" in value, "/_next/static/chunks" in value, "page" in value),
        reverse=True,
    )
    return prioritized


def _resolve_frontend_url(front_url: str, chunk_url: str) -> str:
    if chunk_url.startswith("http://") or chunk_url.startswith("https://"):
        return chunk_url
    if chunk_url.startswith("//"):
        return ("https:" if front_url.startswith("https://") else "http:") + chunk_url
    if chunk_url.startswith("/"):
        return f"{front_url.rstrip('/')}{chunk_url}"
    return f"{front_url.rstrip('/')}/{chunk_url}"


def _verify_frontend_contract(front_url: str, api_base: str) -> bool:
    status, content_type, html = _request_text(front_url)
    if status != 200:
        _log(f"Frontend home check başarısız: status={status}")
        return False
    if "html" not in content_type.lower():
        _log(f"Frontend content-type beklenen html değil: {content_type}")
        return False
    root_markers = (
        "Trader Cockpit (MVP)",
        "<title>Trader Cockpit</title>",
        "<h2>Genel",
        "<h2>Giriş",
    )
    missing = [marker for marker in root_markers if marker not in html]
    if missing:
        _log(
            "Frontend kontrat kontrolü: beklenen UI marker'ları bulunamadı: "
            + ", ".join(missing[:5])
        )
        return False

    chunk_expected_markers = (
        "Trader Cockpit (MVP)",
        "Genel",
        "Watchlist",
        "Teknik analiz",
        "Paper portf",
        "RSI tarama",
    )

    chunk_urls = [url for url in _extract_chunk_urls(html) if "page" in url]
    fallback_attempted = False
    fallback_url = "/_next/static/chunks/app/page.js"
    if not any(url.startswith(("/_next/static/chunks/app/page", "/_next/static/chunks/app/page-")) for url in chunk_urls):
        fallback_attempted = True
        chunk_urls.append(fallback_url)

    for chunk_url in chunk_urls:
        resolved_url = _resolve_frontend_url(front_url, chunk_url)
        try:
            chunk_status, chunk_content_type, chunk_body = _request_text(resolved_url)
        except Exception as exc:
            if fallback_attempted and chunk_url == fallback_url:
                _log(
                    "Frontend kontrat kontrolü: page.js fallback URL'sinden içerik alınamadı "
                    f"(hata: {exc})."
                )
                return False
            _log(
                "Frontend kontrat kontrolü: app/page.js adayı okunamadı "
                f"(url={resolved_url}, hata={exc})."
            )
            continue

        if chunk_status != 200:
            if fallback_attempted and chunk_url == fallback_url:
                _log(
                    "Frontend kontrat kontrolü: app/page.js fallback URL'sinden içerik alınamadı "
                    f"(status={chunk_status})."
                )
                return False
            _log(
                "Frontend kontrat kontrolü: app/page.js adayı çekilemedi "
                f"(url={resolved_url}, status={chunk_status})."
            )
            continue

        if "javascript" not in chunk_content_type.lower():
            if fallback_attempted and chunk_url == fallback_url:
                _log(
                    "Frontend kontrat kontrolü: app/page.js fallback URL content-type beklenmedik "
                    f"({chunk_content_type})."
                )
                return False
            _log(
                "Frontend kontrat kontrolü: app/page.js adayı content-type beklenmedik "
                f"({resolved_url}, {chunk_content_type})."
            )
            continue

        missing_chunk_markers = [marker for marker in chunk_expected_markers if marker not in chunk_body]
        if missing_chunk_markers:
            _log(
                "Frontend kontrat kontrolü: app/page.js içinde beklenen UI marker'ları bulunamadı: "
                + ", ".join(missing_chunk_markers[:5])
            )
            if not (fallback_attempted and chunk_url == fallback_url):
                continue
            return False

        if api_base and api_base not in chunk_body:
            if fallback_attempted and chunk_url == fallback_url:
                _log(
                    "Frontend kontrat kontrolü: app/page.js fallback içeriğinde beklenen API base değeri "
                    f"görünmedi ({api_base})."
                )
                return False
            _log(
                "Frontend kontrat kontrolü: app/page.js adayı içinde beklenen API base değeri görünmedi "
                f"({resolved_url}, {api_base})."
            )
            continue

        if chunk_url != fallback_url:
            _log(f"Frontend kontrat kontrolü: chunk URL match={resolved_url}")
        return True

    _log("Frontend kontrat kontrolü: app/page.js chunk URL'i doğrulandı veya API base ile eşleşmedi.")
    return False


def _verify_backend_contract(api_host: str, api_port: int) -> bool:
    base = f"http://{api_host}:{api_port}"
    test_email = "smoke_test@example.com"
    test_symbol = "SMOKE01.IS"

    status, health_payload = _request_json("GET", f"{base}/api/v1/health")
    if status != 200:
        _log(f"Health check başarısız: status={status}")
        return False
    if health_payload.get("meta", {}).get("correlation_id") is None:
        _log("Health payload contract: correlation_id eksik.")
        return False
    if health_payload.get("data", {}).get("status") != "ok":
        _log("Health payload contract: status ok değil.")
        return False

    status, invite_payload = _request_json(
        "POST",
        f"{base}/api/v1/auth/invite",
        {"email": test_email},
    )
    if status != 200:
        _log(f"Auth invite başarısız: status={status}")
        return False
    invite_token = invite_payload.get("data", {}).get("magic_link_token")
    if not isinstance(invite_token, str) or not invite_token:
        _log("Invite token bulunamadı.")
        return False

    status, magic_payload = _request_json(
        "POST",
        f"{base}/api/v1/auth/magic",
        {"token": invite_token},
    )
    if status != 200:
        _log(f"Magic token tüketimi başarısız: status={status}")
        return False

    session_token = magic_payload.get("data", {}).get("session_token")
    if not isinstance(session_token, str) or not session_token:
        _log("Session token bulunamadı.")
        return False

    auth_headers = {"Authorization": f"Bearer {session_token}"}
    status, symbols_payload = _request_json("GET", f"{base}/api/v1/watchlist", headers=auth_headers)
    if status != 200:
        _log(f"Watchlist endpoint başarısız: status={status}")
        return False
    if not isinstance(symbols_payload.get("data", {}).get("symbols"), list):
        _log("Watchlist payload contract: symbols alanı list değil.")
        return False

    status, universe_payload = _request_json("GET", f"{base}/api/v1/symbols", headers=auth_headers)
    if status != 200:
        _log(f"Symbol universe endpoint başarısız: status={status}")
        return False
    symbols_universe = universe_payload.get("data", {}).get("symbols")
    if not isinstance(symbols_universe, list) or not symbols_universe:
        _log("Symbol universe payload contract: symbols alanı boş veya list değil.")
        return False
    if not any(entry.get("symbol") == test_symbol for entry in symbols_universe if isinstance(entry, dict)):
        _log("Symbol universe akışı: test sembolü universe içinde bulunamadı; test sembolü için fallback seçiliyor.")

    status, add_symbol_payload = _request_json(
        "POST",
        f"{base}/api/v1/watchlist/{test_symbol}/add",
        headers=auth_headers,
    )
    if status != 200:
        _log(f"Watchlist ekleme akışı başarısız: status={status}")
        return False
    watchlist_symbols = add_symbol_payload.get("data", {}).get("symbols")
    if test_symbol not in (watchlist_symbols or []):
        _log("Watchlist ekleme akışı: test sembolü listede görünmüyor.")
        return False

    status, symbols_payload = _request_json("GET", f"{base}/api/v1/watchlist", headers=auth_headers)
    if status != 200:
        _log(f"Watchlist tekrar okuma başarısız: status={status}")
        return False
    symbols = symbols_payload.get("data", {}).get("symbols")
    if test_symbol not in (symbols or []):
        _log("Watchlist eklenen sembol doğrulaması: GET sonrası sembol listede yok.")
        return False

    status, set_symbol_payload = _request_json(
        "PUT",
        f"{base}/api/v1/watchlist",
        headers=auth_headers,
        payload={"symbols": [test_symbol]},
    )
    if status != 200:
        _log(f"Watchlist set endpoint başarısız: status={status}")
        return False
    if set_symbol_payload.get("data", {}).get("symbols") != [test_symbol]:
        _log("Watchlist set akışı: Beklenen tek-satır set doğrulaması başarısız.")
        return False

    status, removed_symbols_payload = _request_json(
        "POST",
        f"{base}/api/v1/watchlist/{test_symbol}/remove",
        headers=auth_headers,
    )
    if status != 200:
        _log(f"Watchlist silme akışı başarısız: status={status}")
        return False
    if test_symbol in (removed_symbols_payload.get("data", {}).get("symbols") or []):
        _log("Watchlist silme akışı: test sembolü kaldırılmış olmamalı.")
        return False

    status, portfolio_payload = _request_json("GET", f"{base}/api/v1/portfolio", headers=auth_headers)
    if status != 200:
        _log(f"Portfolio endpoint başarısız: status={status}")
        return False
    if not isinstance(portfolio_payload.get("data", {}).get("holdings"), list):
        _log("Portfolio payload contract: holdings alanı list değil.")
        return False

    status, upsert_payload = _request_json(
        "POST",
        f"{base}/api/v1/portfolio",
        headers=auth_headers,
        payload={
            "symbol": test_symbol,
            "quantity": 3,
            "average_cost": 45.5,
        },
    )
    if status != 200:
        _log(f"Portfolio ekleme akışı başarısız: status={status}")
        return False
    holding = upsert_payload.get("data", {}).get("holding")
    if not isinstance(holding, dict) or holding.get("symbol") != test_symbol:
        _log("Portfolio ekleme akışı: Holding response beklenen sembolle dönmüyor.")
        return False
    if holding.get("quantity") != 3:
        _log(f"Portfolio ekleme akışı: İlk upsert miktarı beklenen 3 olmadı ({holding.get('quantity')}).")
        return False
    if holding.get("average_cost") != 45.5:
        _log(
            "Portfolio ekleme akışı: İlk upsert ortalama maliyet beklenen 45.5 olmadı "
            f"({holding.get('average_cost')})."
        )
        return False

    status, upsert_again_payload = _request_json(
        "POST",
        f"{base}/api/v1/portfolio",
        headers=auth_headers,
        payload={
            "symbol": test_symbol,
            "quantity": 1,
            "average_cost": 55,
        },
    )
    if status != 200:
        _log(f"Portfolio tekrar ekleme akışı başarısız: status={status}")
        return False
    upsert_again_holding = upsert_again_payload.get("data", {}).get("holding")
    if not isinstance(upsert_again_holding, dict) or upsert_again_holding.get("symbol") != test_symbol:
        _log("Portfolio tekrar ekleme akışı: Holding response beklenen sembolle dönmüyor.")
        return False
    if upsert_again_holding.get("quantity") != 4:
        _log(
            "Portfolio tekrar ekleme akışı: İkinci upsert sonrası beklenen 4 adet bulunamadı "
            f"({upsert_again_holding.get('quantity')})."
        )
        return False
    if upsert_again_holding.get("average_cost") != 47.875:
        _log(
            "Portfolio tekrar ekleme akışı: Ağırlıklı ortalama maliyet hesaplaması beklenen değerde değil "
            f"({upsert_again_holding.get('average_cost')})."
        )
        return False

    status, portfolio_after_add_payload = _request_json("GET", f"{base}/api/v1/portfolio", headers=auth_headers)
    if status != 200:
        _log(f"Portfolio tekrar okuma başarısız: status={status}")
        return False
    holdings_after_add = portfolio_after_add_payload.get("data", {}).get("holdings")
    symbols_after_add = [row.get("symbol") for row in holdings_after_add or []]
    if test_symbol not in symbols_after_add:
        _log("Portfolio ekleme doğrulaması: GET sonrası test sembolü bulunamadı.")
        return False

    status, delete_payload = _request_json(
        "DELETE",
        f"{base}/api/v1/portfolio/{test_symbol}",
        headers=auth_headers,
    )
    if status != 200:
        _log(f"Portfolio silme akışı başarısız: status={status}")
        return False
    if delete_payload.get("data", {}).get("removed") is not True:
        _log("Portfolio silme akışı: Beklenen removed=True dönüşü alınamadı.")
        return False

    status, portfolio_after_delete_payload = _request_json("GET", f"{base}/api/v1/portfolio", headers=auth_headers)
    if status != 200:
        _log(f"Portfolio silme sonrası okuma başarısız: status={status}")
        return False
    holdings_after_delete = portfolio_after_delete_payload.get("data", {}).get("holdings")
    symbols_after_delete = [row.get("symbol") for row in holdings_after_delete or []]
    if test_symbol in symbols_after_delete:
        _log("Portfolio silme doğrulaması: Sembol silinmiş görünmüyor.")
        return False

    return True


def _is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) != 0


def _has_dependency(module_name: str) -> bool:
    has_module = importlib.util.find_spec(module_name) is not None
    if not has_module:
        _log(f"Eksik Python bağımlılığı: {module_name}")
    return has_module


def _terminate(proc: subprocess.Popen, timeout: float) -> None:
    if proc.poll() is not None:
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
    host = "127.0.0.1"

    missing = [name for name in ("pandas", "fastapi", "uvicorn") if not _has_dependency(name)]
    if missing:
        _log(f"Eksik bağımlılıklar: {', '.join(missing)}")
        _log("Çözüm: python3 -m pip install -r requirements.txt")
        return 1

    for port in (args.api_port, args.front_port):
        if not _is_port_free(host, port):
            _log(f"Port {host}:{port} kullanımda görünüyor, önce kapatın.")
            return 1

    cmd = [
        sys.executable,
        "-u",
        "run_dev.py",
        "--no-npm-install-check",
        "--api-host",
        host,
        "--api-port",
        str(args.api_port),
        "--front-host",
        host,
        "--front-port",
        str(args.front_port),
        "--ready-timeout",
        str(args.ready_timeout),
        "--check-interval",
        str(args.check_interval),
        "--shutdown-timeout",
        str(args.shutdown_timeout),
    ]

    _log(f"run_dev smoke başlatılıyor: python {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT)

    try:
        backend_url = f"http://{host}:{args.api_port}/api/v1/health"
        front_url = f"http://{host}:{args.front_port}"

        if not _wait_for_url(backend_url, args.ready_timeout, args.check_interval):
            _log("Backend hazır olamadı.")
            return 1

        if not _wait_for_url(front_url, args.ready_timeout, args.check_interval):
            _log("Frontend hazır olamadı.")
            return 1

        if not _verify_backend_contract(host, args.api_port):
            _log("Backend API kontrat kontrolü başarısız.")
            return 1

        if not _verify_frontend_contract(front_url, f"http://{host}:{args.api_port}"):
            _log("Frontend kontrat kontrolü başarısız.")
            return 1

        _log("Backend ve frontend hazır, kısa süre çalışıyor...")
        time.sleep(max(0, args.duration))

        if proc.poll() is not None:
            _log(f"run_dev beklenmedik şekilde erken çıktı: code={proc.returncode}")
            return 1

        _log("Smoke tamamlandı, durduruluyor.")
        return 0
    finally:
        _terminate(proc, timeout=args.shutdown_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
