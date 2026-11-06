"""ADR-0009 HTTP contract through the real router, analyzer and auth service."""

from datetime import timedelta
import ipaddress
import socket
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app import routers
from backend.app.main import app
from backend.app.providers.market_data import MarketDataResult
from backend.app.services.security import AuthService
from backend.tests.test_momentum_indicators import bars, range_expiry_bars
from modules.technical_analysis import TechnicalAnalyzer


class _FixtureProvider:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def get_ohlcv(self, symbol, *, period, interval):
        self.calls.append((symbol, period, interval))
        return self.result


class TestStochasticApi(unittest.TestCase):
    def setUp(self):
        # Windows asyncio needs a loopback self-pipe; external IO stays forbidden.
        original_connect = socket.socket.connect
        original_connect_ex = socket.socket.connect_ex

        def is_local(address):
            try:
                return ipaddress.ip_address(address[0]).is_loopback
            except (ValueError, TypeError, IndexError):
                return False

        def connect(sock, address):
            if not is_local(address):
                raise AssertionError("External socket forbidden")
            return original_connect(sock, address)

        def connect_ex(sock, address):
            if not is_local(address):
                raise AssertionError("External socket forbidden")
            return original_connect_ex(sock, address)

        self.enterContext(patch.object(socket.socket, "connect", connect))
        self.enterContext(patch.object(socket.socket, "connect_ex", connect_ex))
        self.enterContext(patch.object(socket, "create_connection",
                                     side_effect=AssertionError("External socket forbidden")))

        data = range_expiry_bars()
        as_of = data.index[-1].to_pydatetime()
        self.result = MarketDataResult(
            symbol="THYAO.IS", provider="test-fixture", df=data,
            as_of=as_of, fetched_at=as_of + timedelta(hours=1), staleness_seconds=3600,
        )
        self.provider = _FixtureProvider(self.result)
        self.enterContext(patch.object(routers, "_market_data", self.provider))
        self.enterContext(patch.object(routers, "TechnicalAnalyzer", TechnicalAnalyzer))
        self.enterContext(patch.object(routers, "_auth", AuthService()))
        # A regression produces an assertion on HTTP500, rather than KeyError.
        self.client = self.enterContext(TestClient(app, raise_server_exceptions=False))
        invite = self.client.post("/api/v1/auth/invite", json={"email": "fixture@example.com"})
        self.assertEqual(invite.status_code, 200)
        magic = self.client.post("/api/v1/auth/magic", json={
            "token": invite.json()["data"]["magic_link_token"],
        })
        self.assertEqual(magic.status_code, 200)
        self.headers = {"Authorization": "Bearer " + magic.json()["data"]["session_token"]}

    def use_data(self, data):
        self.result.df = data
        self.result.as_of = data.index[-1].to_pydatetime()
        self.result.fetched_at = self.result.as_of + timedelta(hours=1)

    def test_stoch_returns_last_k_under_only_the_requested_scalar_key(self):
        response = self.client.get("/api/v1/analysis/THYAO.IS",
                                   params={"indicators": "stoch"}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        # Last three K values are100,25,75; D=200/3 must not be returned.
        self.assertEqual(response.json()["data"]["indicators"], {"stoch": 75.0})

    def test_stoch_keeps_real_zero_instead_of_null(self):
        self.use_data(bars([(120, 80, 100)] * 19 + [(120, 80, 80)]))
        response = self.client.get("/api/v1/analysis/THYAO.IS",
                                   params={"indicators": "stoch"}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["indicators"], {"stoch": 0.0})

    def test_stoch_keeps_warmup_as_null_under_the_requested_key(self):
        # Controlled provider can return short data; provider policy is unchanged.
        self.use_data(range_expiry_bars().iloc[:13])
        response = self.client.get("/api/v1/analysis/THYAO.IS",
                                   params={"indicators": "stoch"}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["indicators"], {"stoch": None})

    def test_stoch_keeps_undefined_flat_range_as_null(self):
        self.use_data(bars([(10, 10, 10)] * 20))
        response = self.client.get("/api/v1/analysis/THYAO.IS",
                                   params={"indicators": "stoch"}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["indicators"], {"stoch": None})

    def test_mixed_selection_preserves_other_series_price_and_metadata(self):
        response = self.client.get("/api/v1/analysis/thyao.is", params={
            "period": "3mo", "interval": "1h",
            "indicators": " stoch ,williams_r,cci,macd",
        }, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["data"], {
            "symbol": "THYAO.IS", "period": "3mo", "interval": "1h",
            "as_of": "2026-01-02T10:00:00+00:00",
            "price": {"open": 100.0, "high": 120.0, "low": 80.0,
                      "close": 110.0, "volume": 117.0},
            # Eighteen bars: CCI20 and MACD26 have not warmed up.
            "indicators": {"stoch": 75.0, "williams_r": -25.0, "cci": None, "macd": None},
        })
        self.assertEqual(payload["meta"]["symbol"], "THYAO.IS")
        self.assertEqual(payload["meta"]["source"], "test-fixture")
        self.assertEqual(payload["meta"]["as_of"], "2026-01-02T10:00:00+00:00")
        self.assertEqual(payload["meta"]["fetched_at"], "2026-01-02T11:00:00+00:00")
        self.assertEqual(payload["meta"]["staleness_seconds"], 3600)
        self.assertIsInstance(payload["meta"]["correlation_id"], str)
        self.assertTrue(payload["meta"]["correlation_id"])
        self.assertEqual(self.provider.calls, [("THYAO.IS", "3mo", "1h")])

    def test_k_and_d_query_selectors_remain_unsupported(self):
        for name in ("stoch_k", "stoch_d", "stoch,stoch_d"):
            with self.subTest(name=name):
                response = self.client.get("/api/v1/analysis/THYAO.IS",
                                           params={"indicators": name}, headers=self.headers)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["detail"]["error_code"], "invalid_indicator")

    def test_stoch_requires_a_valid_session(self):
        for headers, error_code in (({}, "missing_token"),
                                    ({"Authorization": "Bearer invalid-fixture"}, "invalid_session")):
            with self.subTest(error_code=error_code):
                response = self.client.get("/api/v1/analysis/THYAO.IS",
                                           params={"indicators": "stoch"}, headers=headers)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.json()["detail"]["error_code"], error_code)
        self.assertEqual(self.provider.calls, [])


if __name__ == "__main__":
    unittest.main()
