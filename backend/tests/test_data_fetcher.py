"""Offline Yahoo adapter checks; the real adapter talks only to a Ticker double."""
import unittest
from unittest.mock import Mock, call, patch

import pandas as pd

from backend.app.providers.market_data import MarketDataProvider
from modules.data_fetcher import BISTDataFetcher


def prices():
    return pd.DataFrame({
        "Open": [100.0 + i for i in range(20)],
        "High": [102.0 + i for i in range(20)],
        "Low": [99.0 + i for i in range(20)],
        "Close": [101.0 + i for i in range(20)],
        "Volume": [1000] * 20,
        "Dividends": [0.0] * 20,
        "Stock Splits": [0.0] * 20,
    }, index=pd.date_range("2026-01-01", periods=20, tz="UTC"))


class TestDataFetcherTimeout(unittest.TestCase):
    def setUp(self):
        network = patch("socket.socket.connect", side_effect=AssertionError("Unexpected network"))
        network.start()
        self.addCleanup(network.stop)
        self.ticker = Mock()
        ticker_class = patch("modules.data_fetcher.yf.Ticker", return_value=self.ticker)
        self.ticker_class = ticker_class.start()
        self.addCleanup(ticker_class.stop)

    def test_explicit_timeout_reaches_history_and_preserves_ohlcv(self):
        frame = prices()
        original = frame.copy(deep=True)
        self.ticker.history.return_value = frame
        result = BISTDataFetcher().get_stock_data(
            "THYAO.IS", period="6mo", interval="1d", timeout_seconds=3.25)
        self.ticker_class.assert_called_once_with("THYAO.IS")
        self.ticker.history.assert_called_once_with(
            period="6mo", interval="1d", auto_adjust=False, prepost=False,
            actions=True, timeout=3.25)
        pd.testing.assert_frame_equal(result, original.drop(columns=["Dividends", "Stock Splits"]))
        pd.testing.assert_frame_equal(frame, original)

    def test_legacy_call_omits_timeout_override(self):
        self.ticker.history.return_value = prices()
        self.assertIsNotNone(BISTDataFetcher().get_stock_data("THYAO.IS"))
        self.ticker.history.assert_called_once_with(
            period="1y", interval="1d", auto_adjust=False, prepost=False, actions=True)

    def test_history_timeout_keeps_existing_none_failure_contract(self):
        self.ticker.history.side_effect = TimeoutError("Fixture timeout")
        result = BISTDataFetcher().get_stock_data("THYAO.IS", timeout_seconds=3.25)
        self.assertIsNone(result)
        self.assertEqual(self.ticker.history.call_args.kwargs["timeout"], 3.25)

    def test_empty_history_does_not_fabricate_ohlcv(self):
        self.ticker.history.return_value = pd.DataFrame()
        self.assertIsNone(BISTDataFetcher().get_stock_data("THYAO.IS"))
        self.assertEqual(self.ticker.history.call_count, 1)

    def test_real_provider_adapter_chain_retries_and_forwards_default_timeout(self):
        frame = prices()
        self.ticker.history.side_effect = [TimeoutError("Fixture timeout"), frame]
        provider = MarketDataProvider(fetcher=BISTDataFetcher(), max_retries=1)
        with patch("backend.app.providers.market_data.time.sleep") as sleep:
            result = provider.get_ohlcv("THYAO.IS")
        self.assertEqual(self.ticker.history.call_count, 2)
        for history_call in self.ticker.history.call_args_list:
            self.assertEqual(history_call.kwargs["timeout"], 8.0)
        self.assertEqual(sleep.call_args_list, [call(1.5)])
        self.assertEqual(result.provider, "yahoo_finance")
        self.assertEqual(result.as_of.isoformat(), "2026-01-20T00:00:00+00:00")
        pd.testing.assert_frame_equal(result.df, frame.drop(columns=["Dividends", "Stock Splits"]))


if __name__ == "__main__":
    unittest.main()
