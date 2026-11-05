from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, List, Optional


@dataclass
class Holding:
    symbol: str
    quantity: float
    average_cost: float
    updated_at: str


class PortfolioService:
    def __init__(self) -> None:
        self._holdings: Dict[str, List[Holding]] = {}
        self._lock = Lock()

    def list(self, email: str) -> List[Holding]:
        with self._lock:
            return list(self._holdings.get(email, []))

    def upsert(self, email: str, symbol: str, quantity: float, average_cost: float) -> Holding:
        with self._lock:
            if quantity <= 0 or average_cost < 0:
                raise ValueError("quantity > 0 and average_cost >= 0 required")
            current = self._holdings.setdefault(email, [])
            now = datetime.now(timezone.utc).isoformat()
            for idx, existing in enumerate(current):
                if existing.symbol == symbol:
                    if existing.quantity == 0:
                        weighted_cost = average_cost
                    else:
                        weighted_cost = ((existing.average_cost * existing.quantity) + (average_cost * quantity)) / (existing.quantity + quantity)
                    current[idx] = Holding(symbol=symbol, quantity=existing.quantity + quantity, average_cost=weighted_cost, updated_at=now)
                    return current[idx]
            holding = Holding(symbol=symbol, quantity=quantity, average_cost=average_cost, updated_at=now)
            current.append(holding)
            return holding

    def remove(self, email: str, symbol: str) -> bool:
        with self._lock:
            current = self._holdings.get(email, [])
            new_list = [h for h in current if h.symbol != symbol]
            removed = len(new_list) != len(current)
            self._holdings[email] = new_list
            return removed

    def clear(self, email: str) -> None:
        with self._lock:
            self._holdings[email] = []
