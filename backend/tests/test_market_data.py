from __future__ import annotations

from datetime import datetime, timezone
import unittest

import pandas as pd

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

    def test_fail_when_no_data_after_retries(self) -> None:
        fetcher = _DataFetcher([None, pd.DataFrame()])
        provider = MarketDataProvider(fetcher=fetcher, max_retries=1, retry_backoff=0.0)

        with self.assertRaises(RuntimeError) as err:
            provider.get_ohlcv("AKBNK.IS")

        self.assertIn("Market data fetch failed for AKBNK.IS", str(err.exception))
        self.assertEqual(fetcher.calls, 2)


if __name__ == "__main__":
    unittest.main()
