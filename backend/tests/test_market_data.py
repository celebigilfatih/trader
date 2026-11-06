from __future__ import annotations

from datetime import datetime, timezone
import unittest
from unittest.mock import Mock, call, patch

import pandas as pd
import requests

from backend.app.providers.market_data import MarketDataProvider, MarketDataResult


class _DataFetcher:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def get_stock_data(self, *args, **kwargs):
        _ = args
        value = self._responses[self.calls]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return value


def _fixture_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 104.0],
            "Volume": [1000, 1100],
        },
        index=pd.to_datetime(["2026-08-01", "2026-08-02"]),
    )


class TestMarketDataProvider(unittest.TestCase):
    def setUp(self):
        network = patch("socket.socket.connect", side_effect=AssertionError("Unexpected network"))
        network.start()
        self.addCleanup(network.stop)

    def test_get_ohlcv_returns_contract_and_metadata(self) -> None:
        fetcher = _DataFetcher([_fixture_df()])
        provider = MarketDataProvider(fetcher=fetcher, max_retries=0)

        result = provider.get_ohlcv("THYAO.IS")

        self.assertEqual(fetcher.calls, 1)
        self.assertEqual(result.symbol, "THYAO.IS")
        self.assertEqual(result.provider, "yahoo_finance")
        contract = result.contract()
        self.assertEqual(contract.symbol, "THYAO.IS")
        self.assertEqual(contract.source, "yahoo_finance")
        self.assertGreaterEqual(contract.staleness_seconds, 0.0)

    def test_retry_then_success(self) -> None:
        fetcher = _DataFetcher([RuntimeError("temporary fail"), _fixture_df()])
        provider = MarketDataProvider(fetcher=fetcher, max_retries=1, retry_backoff=0.0)

        result = provider.get_ohlcv("GARAN.IS")

        self.assertEqual(fetcher.calls, 2)
        self.assertEqual(result.symbol, "GARAN.IS")

    def test_configured_timeout_is_forwarded_to_fetcher(self):
        fetcher = Mock()
        fetcher.get_stock_data.return_value = _fixture_df()
        provider = MarketDataProvider(fetcher=fetcher, timeout_seconds=3.25)
        provider.get_ohlcv("THYAO.IS", period="1y", interval="1h")
        fetcher.get_stock_data.assert_called_once_with(
            "THYAO.IS", period="1y", interval="1h", timeout_seconds=3.25)

    def test_default_budget_recovers_after_two_failures_with_exact_backoff(self):
        fetcher = Mock()
        fetcher.get_stock_data.side_effect = [TimeoutError("Fixture timeout"),
                                              RuntimeError("Fixture connection failure"), _fixture_df()]
        provider = MarketDataProvider(fetcher=fetcher)
        with patch("backend.app.providers.market_data.time.sleep") as sleep:
            result = provider.get_ohlcv("THYAO.IS")
        self.assertEqual(result.symbol, "THYAO.IS")
        self.assertEqual(fetcher.get_stock_data.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(1.5), call(3.0)])

    def test_timeout_exhaustion_has_no_fourth_attempt_or_final_sleep(self):
        fetcher = Mock()
        fetcher.get_stock_data.side_effect = TimeoutError("Fixture timeout")
        with patch("backend.app.providers.market_data.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "Fixture timeout"):
                MarketDataProvider(fetcher=fetcher).get_ohlcv("THYAO.IS")
        self.assertEqual(fetcher.get_stock_data.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(1.5), call(3.0)])

    def test_none_and_empty_results_can_recover_without_fabricated_prices(self):
        fetcher = Mock()
        expected = _fixture_df()
        fetcher.get_stock_data.side_effect = [None, pd.DataFrame(), expected]
        with patch("backend.app.providers.market_data.time.sleep") as sleep:
            result = MarketDataProvider(fetcher=fetcher).get_ohlcv("THYAO.IS")
        pd.testing.assert_frame_equal(result.df, expected)
        self.assertEqual(fetcher.get_stock_data.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(1.5), call(3.0)])

    def test_zero_retry_budget_attempts_once_without_sleep(self):
        for failure in (None, TimeoutError("Fixture timeout")):
            with self.subTest(failure=failure):
                fetcher = Mock()
                fetcher.get_stock_data.side_effect = [failure]
                with patch("backend.app.providers.market_data.time.sleep") as sleep:
                    with self.assertRaises(RuntimeError):
                        MarketDataProvider(fetcher=fetcher, max_retries=0).get_ohlcv("THYAO.IS")
                self.assertEqual(fetcher.get_stock_data.call_count, 1)
                sleep.assert_not_called()

    def test_success_does_not_retry_or_sleep(self):
        fetcher = Mock()
        fetcher.get_stock_data.return_value = _fixture_df()
        with patch("backend.app.providers.market_data.time.sleep") as sleep:
            MarketDataProvider(fetcher=fetcher).get_ohlcv("THYAO.IS")
        self.assertEqual(fetcher.get_stock_data.call_count, 1)
        sleep.assert_not_called()

    def test_fail_when_no_data_after_retries(self) -> None:
        fetcher = _DataFetcher([None, pd.DataFrame()])
        provider = MarketDataProvider(fetcher=fetcher, max_retries=1, retry_backoff=0.0)

        with self.assertRaises(RuntimeError) as err:
            provider.get_ohlcv("AKBNK.IS")

        self.assertIn("Market data fetch failed for AKBNK.IS", str(err.exception))
        self.assertEqual(fetcher.calls, 2)

    def test_success_diagnostics_measure_whole_call_with_monotonic_clock(self):
        provider = MarketDataProvider(fetcher=_DataFetcher([_fixture_df()]))
        with patch("backend.app.providers.market_data.time.monotonic", side_effect=[10.0, 12.5]):
            provider.get_ohlcv("THYAO.IS")
        self.assertEqual(provider.last_fetch_diagnostics, [{
            "source": "yahoo_finance", "status": "success", "attempts": 1,
            "elapsed_seconds": 2.5, "error_code": None,
        }])

    def test_recovered_diagnostics_count_attempts_and_clear_error(self):
        provider = MarketDataProvider(fetcher=_DataFetcher([
            TimeoutError("private fixture detail"), None, _fixture_df()]))
        with patch("backend.app.providers.market_data.time.sleep") as sleep, patch(
            "backend.app.providers.market_data.time.monotonic", side_effect=[20.0, 25.0]
        ):
            provider.get_ohlcv("THYAO.IS")
        self.assertEqual(sleep.call_args_list, [call(1.5), call(3.0)])
        self.assertEqual(provider.last_fetch_diagnostics, [{
            "source": "yahoo_finance", "status": "success", "attempts": 3,
            "elapsed_seconds": 5.0, "error_code": None,
        }])

    def test_failed_diagnostics_classify_only_observed_failure(self):
        failures = [
            (TimeoutError("private fixture detail"), "timeout"),
            (requests.exceptions.ReadTimeout("private fixture detail"), "timeout"),
            (ConnectionError("private fixture detail"), "connection_error"),
            (requests.exceptions.ConnectionError("private fixture detail"), "connection_error"),
            (RuntimeError("private fixture detail"), "fetch_failed"),
            (None, "no_data"), (pd.DataFrame(), "no_data"),
        ]
        for failure, code in failures:
            with self.subTest(code=code, failure_type=type(failure).__name__):
                provider = MarketDataProvider(fetcher=_DataFetcher([failure] * 3))
                with patch("backend.app.providers.market_data.time.sleep"), patch(
                    "backend.app.providers.market_data.time.monotonic", side_effect=[5.0, 11.0]
                ), self.assertRaises(RuntimeError):
                    provider.get_ohlcv("THYAO.IS")
                self.assertEqual(provider.last_fetch_diagnostics, [{
                    "source": "yahoo_finance", "status": "failed", "attempts": 3,
                    "elapsed_seconds": 6.0, "error_code": code,
                }])

    def test_new_call_resets_diagnostics_without_mutating_old_snapshot(self):
        fetcher = Mock()
        provider = MarketDataProvider(fetcher=fetcher, max_retries=0)
        self.assertEqual(provider.last_fetch_diagnostics, [])
        fetcher.get_stock_data.return_value = _fixture_df()
        provider.get_ohlcv("THYAO.IS")
        previous = provider.last_fetch_diagnostics
        previous_record = dict(previous[0])

        def fail_after_checking_reset(*args, **kwargs):
            self.assertEqual(provider.last_fetch_diagnostics, [])
            raise TimeoutError("private fixture detail")

        fetcher.get_stock_data.side_effect = fail_after_checking_reset
        with self.assertRaises(RuntimeError):
            provider.get_ohlcv("GARAN.IS")
        self.assertEqual(previous, [previous_record])
        self.assertIsNot(previous, provider.last_fetch_diagnostics)
        self.assertEqual(provider.last_fetch_diagnostics[0]["error_code"], "timeout")
        self.assertGreaterEqual(provider.last_fetch_diagnostics[0]["elapsed_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
