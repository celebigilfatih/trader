from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Optional

import pandas as pd

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

    def get_ohlcv(self, symbol: str, period: str = "6mo", interval: str = "1d") -> MarketDataResult:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                start = time.time()
                df = self._fetcher.get_stock_data(symbol, period=period, interval=interval)
                fetched_at = datetime.now(timezone.utc)
                if df is None or df.empty:
                    raise ValueError("No data from provider")

                as_of = pd.to_datetime(df.index[-1]).to_pydatetime()
                if as_of.tzinfo is None:
                    as_of = as_of.replace(tzinfo=timezone.utc)
                staleness = (fetched_at - as_of).total_seconds()
                return MarketDataResult(
                    symbol=symbol,
                    provider="yahoo_finance",
                    fetched_at=fetched_at,
                    as_of=as_of,
                    staleness_seconds=max(0.0, staleness),
                    df=df,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff * (attempt + 1))
        raise RuntimeError(f"Market data fetch failed for {symbol}: {last_error}")
