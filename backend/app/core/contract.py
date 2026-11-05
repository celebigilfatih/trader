from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def new_correlation_id() -> str:
    return uuid4().hex


@dataclass
class DataContract:
    symbol: str
    source: str
    fetched_at: datetime
    staleness_seconds: Optional[float]
    as_of: Optional[datetime]

    def meta(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "staleness_seconds": self.staleness_seconds,
        }


def ok_response(data: Any, contract: Optional[DataContract] = None, correlation_id: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "data": data,
        "meta": {
            "correlation_id": correlation_id or new_correlation_id(),
        },
    }
    if contract is not None:
        payload["meta"].update(contract.meta())
    return payload


def error_response(error_code: str, message: str, correlation_id: Optional[str] = None) -> Dict[str, str]:
    return {
        "error_code": error_code,
        "message": message,
        "correlation_id": correlation_id or new_correlation_id(),
    }
