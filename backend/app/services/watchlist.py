from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Dict, List


@dataclass
class Watchlist:
    owner: str
    symbols: List[str]


class WatchlistService:
    def __init__(self, defaults: List[str]):
        self._defaults = defaults
        self._store: Dict[str, Watchlist] = {}
        self._lock = RLock()

    def list_for_user(self, email: str) -> List[str]:
        with self._lock:
            wl = self._store.get(email)
            if wl:
                return wl.symbols
            return list(self._defaults)

    def set_for_user(self, email: str, symbols: List[str]) -> List[str]:
        with self._lock:
            self._store[email] = Watchlist(owner=email, symbols=list(dict.fromkeys([s for s in symbols if s.strip()])))
            return list(self._store[email].symbols)

    def add_symbol(self, email: str, symbol: str) -> List[str]:
        with self._lock:
            current = self.list_for_user(email)
            if symbol not in current:
                current.append(symbol)
            self._store[email] = Watchlist(owner=email, symbols=current)
            return current

    def remove_symbol(self, email: str, symbol: str) -> List[str]:
        with self._lock:
            current = self.list_for_user(email)
            current = [s for s in current if s != symbol]
            self._store[email] = Watchlist(owner=email, symbols=current)
            return current
