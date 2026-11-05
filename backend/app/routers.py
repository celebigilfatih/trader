from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from .core.contract import error_response, now_utc, ok_response
from .providers.market_data import MarketDataProvider
from .services.config import DEFAULT_INDICATORS, DEFAULT_INTERVAL, DEFAULT_PERIOD, get_default_universe
from .services.portfolio import Holding, PortfolioService
from .services.security import AuthService
from .services.watchlist import WatchlistService
from modules.config import INDICATORS_CONFIG
from modules.technical_analysis import TechnicalAnalyzer


router = APIRouter()
_market_data = MarketDataProvider()
_auth = AuthService()
_watchlist = WatchlistService([entry["symbol"] for entry in get_default_universe(30)])
_portfolio = PortfolioService()


def _require_auth(authorization: Optional[str] = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail=error_response("missing_token", "Authorization token required"),
        )
    token = authorization.removeprefix("Bearer ").strip()
    email = _auth.validate_session(token)
    if not email:
        raise HTTPException(
            status_code=401,
            detail=error_response("invalid_session", "Invalid or expired session"),
        )
    return email


def _safe_float(value: object) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_symbols(raw_symbols: List[str]) -> List[str]:
    return [symbol.strip().upper() for symbol in raw_symbols if symbol and symbol.strip()]


class InviteRequest(BaseModel):
    email: str = Field(..., min_length=3)


class MagicConsumeRequest(BaseModel):
    token: str = Field(..., min_length=12)


class WatchlistUpdateRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)


class PortfolioUpsertRequest(BaseModel):
    symbol: str
    quantity: float = Field(..., gt=0)
    average_cost: float = Field(..., ge=0)


class ScreenerRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    interval: str = DEFAULT_INTERVAL
    period: str = DEFAULT_PERIOD
    rsi_min: Optional[float] = None
    rsi_max: Optional[float] = None


@router.get("/api/v1/health")
def health() -> Dict[str, object]:
    return ok_response(
        {
            "status": "ok",
            "service": "trader-api",
            "timestamp": now_utc().isoformat(),
        },
        correlation_id="health-check",
    )


@router.post("/api/v1/auth/invite")
def create_invite(payload: InviteRequest) -> Dict[str, object]:
    token = _auth.create_invite(payload.email)
    return ok_response(
        {
            "email": payload.email.strip().lower(),
            "magic_link_token": token,
            "note": "Demo mode: use returned token with consume endpoint",
        }
    )


@router.post("/api/v1/auth/magic")
def consume_magic(payload: MagicConsumeRequest) -> Dict[str, object]:
    session = _auth.consume_magic_link(payload.token)
    if session is None:
        raise HTTPException(
            status_code=400,
            detail=error_response("invalid_magic_token", "Token invalid or expired"),
        )
    return ok_response(
        {
            "email": session.email,
            "session_token": session.token,
            "expires_at": session.expires_at.isoformat(),
        }
    )


@router.get("/api/v1/symbols")
def symbols(email: str = Depends(_require_auth)) -> Dict[str, object]:
    _ = email
    return ok_response(
        {
            "symbols": get_default_universe(100),
        }
    )


@router.get("/api/v1/watchlist")
def list_watchlist(email: str = Depends(_require_auth)) -> Dict[str, object]:
    return ok_response({"symbols": _watchlist.list_for_user(email)}, correlation_id="watchlist-list")


@router.put("/api/v1/watchlist")
def set_watchlist(payload: WatchlistUpdateRequest, email: str = Depends(_require_auth)) -> Dict[str, object]:
    symbols = _watchlist.set_for_user(email, _normalized_symbols(payload.symbols))
    return ok_response({"symbols": symbols})


@router.post("/api/v1/watchlist/{symbol}/add")
def add_watchlist_symbol(symbol: str, email: str = Depends(_require_auth)) -> Dict[str, object]:
    symbols = _watchlist.add_symbol(email, symbol.strip().upper())
    return ok_response({"symbols": symbols})


@router.post("/api/v1/watchlist/{symbol}/remove")
def remove_watchlist_symbol(symbol: str, email: str = Depends(_require_auth)) -> Dict[str, object]:
    symbols = _watchlist.remove_symbol(email, symbol.strip().upper())
    return ok_response({"symbols": symbols})


@router.get("/api/v1/analysis/{symbol}")
def analyze_symbol(
    symbol: str,
    period: str = Query(default=DEFAULT_PERIOD),
    interval: str = Query(default=DEFAULT_INTERVAL),
    indicators: Optional[str] = Query(default=None),
    email: str = Depends(_require_auth),
) -> Dict[str, object]:
    _ = email
    normalized_symbol = symbol.strip().upper()
    data = _market_data.get_ohlcv(normalized_symbol, period=period, interval=interval)

    requested_indicators = (
        [name.strip() for name in indicators.split(",") if name.strip()]
        if indicators
        else list(DEFAULT_INDICATORS)
    )
    available = [name for name in requested_indicators if name in INDICATORS_CONFIG]
    unsupported = [name for name in requested_indicators if name not in INDICATORS_CONFIG]

    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=error_response(
                "invalid_indicator",
                f"Unsupported indicators: {', '.join(unsupported)}",
            ),
        )

    analyzer = TechnicalAnalyzer(data.df)
    for indicator in available:
        analyzer.add_indicator(indicator)

    latest_row = analyzer.data.iloc[-1]
    latest_indicators: Dict[str, Optional[float]] = {}
    for indicator_name in available:
        latest_indicators[indicator_name] = _safe_float(analyzer.indicators[indicator_name].iloc[-1])

    payload = {
        "symbol": normalized_symbol,
        "period": period,
        "interval": interval,
        "as_of": data.as_of.isoformat(),
        "price": {
            "open": _safe_float(latest_row["Open"]),
            "high": _safe_float(latest_row["High"]),
            "low": _safe_float(latest_row["Low"]),
            "close": _safe_float(latest_row["Close"]),
            "volume": _safe_float(latest_row["Volume"]),
        },
        "indicators": latest_indicators,
    }
    return ok_response(payload, contract=data.contract())


@router.post("/api/v1/screeners/rsi")
def rsi_screener(payload: ScreenerRequest, email: str = Depends(_require_auth)) -> Dict[str, object]:
    _ = email
    requested_symbols = (
        _normalized_symbols(payload.symbols)
        if payload.symbols
        else [entry["symbol"] for entry in get_default_universe(25)]
    )
    if not requested_symbols:
        raise HTTPException(
            status_code=400,
            detail=error_response("invalid_input", "No symbols provided"),
        )
    if payload.rsi_min is not None and payload.rsi_max is not None and payload.rsi_min > payload.rsi_max:
        raise HTTPException(
            status_code=400,
            detail=error_response("invalid_input", "rsi_min cannot be greater than rsi_max"),
        )

    results = []
    skipped_symbols: List[str] = []
    skipped_items: List[Dict[str, str]] = []
    for symbol in requested_symbols:
        try:
            data = _market_data.get_ohlcv(symbol, period=payload.period, interval=payload.interval)
        except RuntimeError as error:
            skipped_symbols.append(symbol)
            skipped_items.append(
                {
                    "symbol": symbol,
                    "code": "market_data_unavailable",
                    "reason": str(error),
                }
            )
            continue
        analyzer = TechnicalAnalyzer(data.df)
        analyzer.add_indicator("rsi")
        rsi_value = _safe_float(analyzer.indicators["rsi"].iloc[-1])
        if rsi_value is None:
            skipped_symbols.append(symbol)
            skipped_items.append(
                {
                    "symbol": symbol,
                    "code": "insufficient_ohlcv_for_rsi",
                    "reason": "RSI hesaplanamadı; veri yetersiz veya geçersiz.",
                }
            )
            continue

        if payload.rsi_min is not None and rsi_value < payload.rsi_min:
            continue
        if payload.rsi_max is not None and rsi_value > payload.rsi_max:
            continue

        results.append(
            {
                "symbol": symbol,
                "rsi": rsi_value,
                "price": _safe_float(data.df["Close"].iloc[-1]),
                "as_of": data.as_of.isoformat(),
                "source": data.provider,
                "fetched_at": data.fetched_at.isoformat(),
                "staleness_seconds": data.staleness_seconds,
            }
        )

    results = sorted(results, key=lambda row: row["rsi"])
    return ok_response(
        {
            "count": len(results),
            "results": results,
            "skipped_symbols": skipped_symbols,
            "warnings": skipped_items,
        }
    )


@router.get("/api/v1/portfolio")
def list_portfolio(email: str = Depends(_require_auth)) -> Dict[str, object]:
    return ok_response({"holdings": [_serialize_holding(h) for h in _portfolio.list(email)]})


def _serialize_holding(holding: Holding) -> Dict[str, object]:
    return {
        "symbol": holding.symbol,
        "quantity": holding.quantity,
        "average_cost": holding.average_cost,
        "updated_at": holding.updated_at,
    }


@router.post("/api/v1/portfolio")
def upsert_holding(payload: PortfolioUpsertRequest, email: str = Depends(_require_auth)) -> Dict[str, object]:
    holding = _portfolio.upsert(
        email,
        payload.symbol.strip().upper(),
        payload.quantity,
        payload.average_cost,
    )
    return ok_response({"holding": _serialize_holding(holding)})


@router.delete("/api/v1/portfolio/{symbol}")
def delete_holding(symbol: str, email: str = Depends(_require_auth)) -> Dict[str, object]:
    removed = _portfolio.remove(email, symbol.strip().upper())
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=error_response("not_found", "Holding not found"),
        )
    return ok_response({"removed": True, "symbol": symbol})
