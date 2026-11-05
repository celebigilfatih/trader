from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from modules.config import BIST_SYMBOLS


DEFAULT_INDICATORS = ["ema_21", "rsi", "macd", "vwema_5"]
DEFAULT_PERIOD = "6mo"
DEFAULT_INTERVAL = "1d"


def get_default_universe(limit: int = 100) -> List[Dict[str, str]]:
    universe = [{"symbol": k, "name": v} for k, v in BIST_SYMBOLS.items()]
    return universe[:limit]
