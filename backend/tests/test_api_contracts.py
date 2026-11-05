from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.providers.market_data import MarketDataResult
from backend.app import routers
from backend.app.services.config import get_default_universe
from backend.app.services.portfolio import PortfolioService
from backend.app.services.security import AuthService
from backend.app.services.watchlist import WatchlistService


class _DummyTechnicalAnalyzer:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.indicators: dict[str, pd.Series] = {}

    def add_indicator(self, indicator_name: str) -> None:
        self.indicators[indicator_name] = pd.Series(
            [1.0] * len(self.data),
            index=self.data.index,
        )


class _DummyMarketDataProvider:
    def __init__(self, result: MarketDataResult) -> None:
        self._result = result
        self.calls: list[tuple[str, str, str]] = []

    def get_ohlcv(self, _symbol: str, *args, **kwargs) -> MarketDataResult:
        period = kwargs.get("period", "6mo")
        interval = kwargs.get("interval", "1d")
        self.calls.append((_symbol, period, interval))
        return self._result


class _NaNTechnicalAnalyzer:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.indicators: dict[str, pd.Series] = {}

    def add_indicator(self, indicator_name: str) -> None:
        _ = indicator_name
        self.indicators["rsi"] = pd.Series([float("nan")] * len(self.data), index=self.data.index)


class _FaultyMarketDataProvider:
    def __init__(self, result: MarketDataResult, failing_symbol: str) -> None:
        self._result = result
        self._failing_symbol = failing_symbol
        self.calls: list[tuple[str, str, str]] = []

    def get_ohlcv(self, symbol: str, *args, **kwargs) -> MarketDataResult:
        period = kwargs.get("period", "6mo")
        interval = kwargs.get("interval", "1d")
        self.calls.append((symbol, period, interval))
        if symbol == self._failing_symbol:
            raise RuntimeError(f"Market data fetch failed for {symbol}")
        return self._result


class TestApiContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self._orig_market_data = routers._market_data
        self._orig_analyzer = routers.TechnicalAnalyzer
        self._orig_auth = routers._auth
        self._orig_watchlist = routers._watchlist
        self._orig_portfolio = routers._portfolio

        self._fixture = self._build_fixture()
        self._market_data = _DummyMarketDataProvider(self._fixture)
        routers._market_data = self._market_data
        routers.TechnicalAnalyzer = _DummyTechnicalAnalyzer
        routers._auth = AuthService()
        routers._watchlist = WatchlistService([entry["symbol"] for entry in get_default_universe(3)])
        routers._portfolio = PortfolioService()

    def tearDown(self) -> None:
        routers._market_data = self._orig_market_data
        routers.TechnicalAnalyzer = self._orig_analyzer
        routers._auth = self._orig_auth
        routers._watchlist = self._orig_watchlist
        routers._portfolio = self._orig_portfolio

    def _build_fixture(self) -> MarketDataResult:
        now = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
        index = pd.date_range(now - timedelta(days=30), periods=40, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "Open": [100.0 + i for i in range(40)],
                "High": [101.0 + i for i in range(40)],
                "Low": [99.0 + i for i in range(40)],
                "Close": [100.0 + i for i in range(40)],
                "Volume": [1000 + i for i in range(40)],
            },
            index=index,
        )
        return MarketDataResult(
            symbol="THYAO.IS",
            provider="mock",
            fetched_at=now,
            as_of=index[-1].to_pydatetime(),
            staleness_seconds=10.0,
            df=df,
        )

    def _auth_headers(self) -> dict[str, str]:
        invite = self.client.post("/api/v1/auth/invite", json={"email": "test@example.com"}).json()["data"]["magic_link_token"]
        session = self.client.post("/api/v1/auth/magic", json={"token": invite}).json()["data"]
        return {"Authorization": f"Bearer {session['session_token']}"}

    def test_health_contract(self) -> None:
        response = self.client.get("/api/v1/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["data"]["status"], "ok")
        self.assertIn("correlation_id", payload["meta"])

    def test_watchlist_requires_auth(self) -> None:
        response = self.client.get("/api/v1/watchlist")
        self.assertEqual(response.status_code, 401)

        response = self.client.get("/api/v1/watchlist", headers=self._auth_headers())
        self.assertEqual(response.status_code, 200)
        self.assertIn("symbols", response.json()["data"])

    def test_endpoints_require_auth(self) -> None:
        response = self.client.get("/api/v1/analysis/THYAO.IS")
        self.assertEqual(response.status_code, 401)

        response = self.client.post("/api/v1/screeners/rsi", json={"symbols": ["THYAO.IS"]})
        self.assertEqual(response.status_code, 401)

        response = self.client.get("/api/v1/portfolio")
        self.assertEqual(response.status_code, 401)

    def test_analysis_contract_is_deterministic(self) -> None:
        response = self.client.get(
            "/api/v1/analysis/thyao.is",
            headers=self._auth_headers(),
        )
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertEqual(payload["data"]["symbol"], "THYAO.IS")
        self.assertEqual(payload["data"]["indicators"]["ema_21"], 1.0)
        self.assertEqual(payload["meta"]["source"], "mock")
        self.assertEqual(payload["meta"]["symbol"], "THYAO.IS")
        self.assertEqual(payload["meta"]["fetched_at"], self._fixture.fetched_at.isoformat())
        self.assertEqual(payload["meta"]["as_of"], self._fixture.as_of.isoformat())
        self.assertEqual(payload["meta"]["staleness_seconds"], 10.0)
        provider = self._market_data
        self.assertEqual(provider.calls[-1], ("THYAO.IS", "6mo", "1d"))

    def test_analysis_invalid_indicator(self) -> None:
        response = self.client.get(
            "/api/v1/analysis/THYAO.IS?indicators=not_exists",
            headers=self._auth_headers(),
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["error_code"], "invalid_indicator")

    def test_analysis_with_real_calculator(self) -> None:
        routers.TechnicalAnalyzer = self._orig_analyzer
        self._fixture.df.loc[:, ["Open", "Close"]] = 120.0
        self._fixture.df.loc[:, "High"] = 121.0
        self._fixture.df.loc[:, "Low"] = 119.0
        response = self.client.get(
            "/api/v1/analysis/THYAO.IS",
            headers=self._auth_headers(),
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["data"]["price"]["close"], 120.0)
        self.assertEqual(payload["data"]["indicators"]["ema_21"], 120.0)
        self.assertEqual(payload["data"]["indicators"]["macd"], 0.0)
        self.assertAlmostEqual(payload["data"]["indicators"]["vwema_5"], 120.0)
        self.assertEqual(payload["meta"]["source"], "mock")

    def test_rsi_screener_with_real_calculator_includes_boundary(self) -> None:
        routers.TechnicalAnalyzer = self._orig_analyzer
        headers = self._auth_headers()
        for bounds, count in (({"rsi_min": 100.0, "rsi_max": 100.0}, 1),
                              ({"rsi_max": 99.0}, 0)):
            with self.subTest(bounds=bounds):
                response = self.client.post(
                    "/api/v1/screeners/rsi",
                    headers=headers,
                    json={"symbols": ["THYAO.IS"], **bounds},
                )
                self.assertEqual(response.status_code, 200)
                data = response.json()["data"]
                self.assertEqual(data["count"], count)
                self.assertEqual(data["warnings"], [])
                self.assertEqual(data["skipped_symbols"], [])
                if count:
                    self.assertEqual(data["results"][0]["rsi"], 100.0)

    def test_rsi_screener_with_real_calculator_reports_short_history(self) -> None:
        routers.TechnicalAnalyzer = self._orig_analyzer
        self._fixture.df = self._fixture.df.iloc[:21].copy()
        self._fixture.as_of = self._fixture.df.index[-1].to_pydatetime()
        response = self.client.post(
            "/api/v1/screeners/rsi",
            headers=self._auth_headers(),
            json={"symbols": ["THYAO.IS"]},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["results"], [])
        self.assertEqual(data["skipped_symbols"], ["THYAO.IS"])
        self.assertEqual(data["warnings"][0]["code"], "insufficient_ohlcv_for_rsi")

    def test_rsi_screener_filters_range_and_returns_contract(self) -> None:
        token = self._auth_headers()

        response = self.client.post(
            "/api/v1/screeners/rsi",
            headers=token,
            json={"symbols": ["THYAO.IS", "GARAN.IS"], "rsi_min": 0.0},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["data"]["count"], 2)
        first = payload["data"]["results"][0]
        self.assertEqual(first["source"], "mock")
        self.assertEqual(first["fetched_at"], self._fixture.fetched_at.isoformat())
        self.assertEqual(first["staleness_seconds"], 10.0)
        self.assertEqual(first["as_of"], self._fixture.as_of.isoformat())

        response = self.client.post(
            "/api/v1/screeners/rsi",
            headers=token,
            json={"symbols": ["THYAO.IS", "GARAN.IS"], "rsi_min": 2.0},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["count"], 0)

    def test_rsi_screener_skips_missing_symbol_data(self) -> None:
        token = self._auth_headers()
        original_provider = routers._market_data
        routers._market_data = _FaultyMarketDataProvider(self._fixture, failing_symbol="KOZAL.IS")

        try:
            response = self.client.post(
                "/api/v1/screeners/rsi",
                headers=token,
                json={"symbols": ["THYAO.IS", "KOZAL.IS"], "rsi_min": 0.0},
            )
        finally:
            routers._market_data = original_provider

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["data"]["count"], 1)
        self.assertEqual(payload["data"]["results"][0]["symbol"], "THYAO.IS")
        self.assertEqual(payload["data"]["skipped_symbols"], ["KOZAL.IS"])
        self.assertEqual(
            payload["data"]["warnings"],
            [
                {
                    "symbol": "KOZAL.IS",
                    "code": "market_data_unavailable",
                    "reason": "Market data fetch failed for KOZAL.IS",
                }
            ],
        )

    def test_rsi_screener_reports_insufficient_ohlcv_for_rsi(self) -> None:
        token = self._auth_headers()
        original_analyzer = routers.TechnicalAnalyzer
        routers.TechnicalAnalyzer = _NaNTechnicalAnalyzer

        try:
            response = self.client.post(
                "/api/v1/screeners/rsi",
                headers=token,
                json={"symbols": ["THYAO.IS"], "rsi_min": 0.0},
            )
        finally:
            routers.TechnicalAnalyzer = original_analyzer

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["data"]["count"], 0)
        self.assertEqual(payload["data"]["skipped_symbols"], ["THYAO.IS"])
        self.assertEqual(
            payload["data"]["warnings"],
            [
                {
                    "symbol": "THYAO.IS",
                    "code": "insufficient_ohlcv_for_rsi",
                    "reason": "RSI hesaplanamadı; veri yetersiz veya geçersiz.",
                }
            ],
        )

    def test_rsi_screener_invalid_inputs(self) -> None:
        token = self._auth_headers()

        response = self.client.post(
            "/api/v1/screeners/rsi",
            headers=token,
            json={"symbols": ["THYAO.IS"], "rsi_min": 80.0, "rsi_max": 20.0},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["error_code"], "invalid_input")

        response = self.client.post(
            "/api/v1/screeners/rsi",
            headers=token,
            json={"symbols": "not_a_list"},
        )
        self.assertEqual(response.status_code, 422)

    def test_portfolio_input_validation_is_strict(self) -> None:
        headers = self._auth_headers()

        response = self.client.post(
            "/api/v1/portfolio",
            headers=headers,
            json={"symbol": "KCHOL.IS", "quantity": 0, "average_cost": 120},
        )
        self.assertEqual(response.status_code, 422)

        response = self.client.post(
            "/api/v1/portfolio",
            headers=headers,
            json={"symbol": "KCHOL.IS", "quantity": 1, "average_cost": -1},
        )
        self.assertEqual(response.status_code, 422)

    def test_portfolio_crud_contract(self) -> None:
        headers = self._auth_headers()

        add_response = self.client.post(
            "/api/v1/portfolio",
            headers=headers,
            json={"symbol": "KCHOL.IS", "quantity": 2, "average_cost": 120},
        )
        self.assertEqual(add_response.status_code, 200)

        list_response = self.client.get("/api/v1/portfolio", headers=headers)
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(len(list_response.json()["data"]["holdings"]), 1)

        remove_response = self.client.delete("/api/v1/portfolio/KCHOL.IS", headers=headers)
        self.assertEqual(remove_response.status_code, 200)
        self.assertTrue(remove_response.json()["data"]["removed"])

        missing = self.client.delete("/api/v1/portfolio/KCHOL.IS", headers=headers)
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
