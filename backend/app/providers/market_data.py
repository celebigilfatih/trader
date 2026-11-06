from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Optional

import pandas as pd
import requests

from modules.data_fetcher import BISTDataFetcher
from ..core.contract import DataContract


@dataclass
class MarketDataResult:
    symbol: str
    provider: str
    fetched_at: datetime
    as_of: datetime
    staleness_seconds: float
    df: pd.DataFrame

    def contract(self) -> DataContract:
        return DataContract(
            symbol=self.symbol,
            source=self.provider,
            fetched_at=self.fetched_at,
            as_of=self.as_of,
            staleness_seconds=self.staleness_seconds,
        )


class MarketDataProvider:
    def __init__(
        self,
        timeout_seconds: float = 8.0,
        max_retries: int = 2,
        retry_backoff: float = 1.5,
        fetcher: Optional[object] = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._fetcher = fetcher or BISTDataFetcher()
        self.last_fetch_diagnostics: list[dict] = []

    def _record_diagnostic(self, started: float, attempts: int, status: str, error_code: Optional[str]) -> None:
        self.last_fetch_diagnostics = [{
            "source": "yahoo_finance",
            "status": status,
            "attempts": attempts,
            "elapsed_seconds": max(0.0, time.monotonic() - started),
            "error_code": error_code,
        }]

    def get_ohlcv(self, symbol: str, period: str = "6mo", interval: str = "1d") -> MarketDataResult:
        self.last_fetch_diagnostics = []
        started = time.monotonic()
        last_error = None
        attempts = 0
        error_code = "fetch_failed"
        for attempt in range(self.max_retries + 1):
            attempts += 1
            error_code = None
            try:
                df = self._fetcher.get_stock_data(
                    symbol, period=period, interval=interval,
                    timeout_seconds=self.timeout_seconds,
                )
                fetched_at = datetime.now(timezone.utc)
                if df is None or df.empty:
                    error_code = "no_data"
                    raise ValueError("No data from provider")

                as_of = pd.to_datetime(df.index[-1]).to_pydatetime()
                if as_of.tzinfo is None:
                    as_of = as_of.replace(tzinfo=timezone.utc)
                staleness = (fetched_at - as_of).total_seconds()
                result = MarketDataResult(
                    symbol=symbol,
                    provider="yahoo_finance",
                    fetched_at=fetched_at,
                    as_of=as_of,
                    staleness_seconds=max(0.0, staleness),
                    df=df,
                )
                self._record_diagnostic(started, attempt + 1, "success", None)
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if error_code is None:
                    if isinstance(exc, (TimeoutError, requests.exceptions.Timeout)):
                        error_code = "timeout"
                    elif isinstance(exc, (ConnectionError, requests.exceptions.ConnectionError)):
                        error_code = "connection_error"
                    else:
                        error_code = "fetch_failed"
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff * (attempt + 1))
        self._record_diagnostic(started, attempts, "failed", error_code)
        raise RuntimeError(f"Market data fetch failed for {symbol}: {last_error}")
