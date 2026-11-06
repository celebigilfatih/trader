"""Invalid DayTrader inputs fail before level calculation or record mutation.

The scan fixture runs the real analyzer and trader; only market-data fetching
is replaced. It proves symbol isolation, not provider or trading performance.
"""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules.day_trader import DayTrader
from modules.technical_analysis import TechnicalAnalyzer


def candles(closes, half_range=1.0):
    return pd.DataFrame(
        {"Open": closes, "High": [c + half_range for c in closes],
         "Low": [c - half_range for c in closes], "Close": closes,
         "Volume": [1000] * len(closes)},
        index=pd.date_range("2026-01-01", periods=len(closes), freq="D"),
    )


class TestDayTraderValidation(unittest.TestCase):
    def setUp(self):
        self.io_guards = []
        for target in ("socket.socket.connect", "socket.create_connection",
                       "requests.sessions.Session.request", "smtplib.SMTP",
                       "smtplib.SMTP_SSL"):
            guard = patch(target, side_effect=AssertionError("Unexpected external I/O"))
            self.io_guards.append(guard.start())
            self.addCleanup(guard.stop)
        self.trader = DayTrader()
        self.addCleanup(self.trader.data_fetcher.session.close)

    def assert_invalid(self, operation, message):
        # Assertion failures also diagnose legacy IndexError/NaN-return behavior.
        try:
            operation()
        except Exception as error:
            self.assertIsInstance(error, ValueError)
            self.assertRegex(str(error), message)
        else:
            self.fail("Invalid input returned normally instead of raising ValueError")

    def test_atr_rejects_empty_or_insufficient_requested_period(self):
        for count, period in ((0, 14), (1, 14), (13, 14), (3, 4), (14, 15)):
            with self.subTest(count=count, period=period):
                data = candles([100] * count)
                original = data.copy(deep=True)
                self.assert_invalid(
                    lambda: self.trader._calculate_atr(data, period),
                    rf"ATR.*{period}",
                )
                pd.testing.assert_frame_equal(data, original)

    def test_atr_accepts_exact_requested_period_and_preserves_zero_range(self):
        self.assertEqual(self.trader._calculate_atr(candles([100] * 3), 3), 2.0)
        self.assertEqual(self.trader._calculate_atr(candles([100] * 14)), 2.0)
        self.assertEqual(self.trader._calculate_atr(candles([100] * 14, 0)), 0.0)

    def test_entry_rejects_empty_and_short_data_before_indicator_work(self):
        for count in (0, 1, 13):
            with self.subTest(count=count):
                data = candles([100] * count)
                analyzer = TechnicalAnalyzer(data)
                with patch.object(analyzer, "add_indicator", wraps=analyzer.add_indicator) as add:
                    self.assert_invalid(
                        lambda: self.trader._calculate_entry_exit_points(data, analyzer),
                        r"ATR.*14",
                    )
                    add.assert_not_called()
                self.assertEqual(analyzer.indicators, {})

    def test_entry_rejects_nonpositive_or_nonfinite_current_price_before_indicators(self):
        for price in (0, -1, np.nan, np.inf, -np.inf):
            with self.subTest(price=price):
                data = candles([100.0] * 40)
                data.loc[data.index[-1], "Close"] = price
                analyzer = TechnicalAnalyzer(data)
                with patch.object(analyzer, "add_indicator", wraps=analyzer.add_indicator) as add:
                    self.assert_invalid(
                        lambda: self.trader._calculate_entry_exit_points(data, analyzer),
                        r"[Pp]rice.*positive.*finite|[Pp]rice.*finite.*positive",
                    )
                    add.assert_not_called()

    def test_entry_rejects_zero_range_and_missing_ranges_without_hold_fallback(self):
        zero_range = candles([100.0] * 40, 0)
        missing_ranges = candles([100.0] * 40)
        missing_ranges[["High", "Low"]] = np.nan
        for data in (zero_range, missing_ranges):
            with self.subTest(zero_range=data is zero_range):
                analyzer = TechnicalAnalyzer(data)
                with patch.object(analyzer, "add_indicator", wraps=analyzer.add_indicator) as add:
                    self.assert_invalid(
                        lambda: self.trader._calculate_entry_exit_points(data, analyzer),
                        r"ATR.*positive.*finite|ATR.*finite.*positive",
                    )
                    add.assert_not_called()

    def test_entry_rejects_all_invalid_atr_results_before_buy_sell_or_hold_levels(self):
        # These exact-period consumer checks cover nonfinite helper outputs even
        # when pandas rolling happens to collapse infinity into NaN.
        for closes in ([100] * 40 + [99, 101], [100] * 40 + [101, 99],
                       list(range(100, 142))):
            data = candles(closes)
            for atr in (0, -1, np.nan, np.inf, -np.inf):
                with self.subTest(last_close=closes[-1], atr=atr):
                    analyzer = TechnicalAnalyzer(data)
                    with patch.object(self.trader, "_calculate_atr", return_value=atr):
                        self.assert_invalid(
                            lambda: self.trader._calculate_entry_exit_points(data, analyzer),
                            r"ATR.*positive.*finite|ATR.*finite.*positive",
                        )
                    self.assertEqual(analyzer.indicators, {})

    def test_invalid_add_direction_cannot_replace_active_record_or_read_clock(self):
        self.trader.add_position("EXISTING", 100, "BUY", 95, 110, 2)
        active = copy.deepcopy(self.trader.active_positions)
        self.trader.trading_history.append({"sentinel": "prior history"})
        history = copy.deepcopy(self.trader.trading_history)
        for symbol in ("NEW", "EXISTING"):
            for direction in ("HOLD", "buy", "", None):
                with self.subTest(symbol=symbol, direction=direction):
                    with patch("modules.day_trader.datetime") as clock:
                        self.assert_invalid(
                            lambda: self.trader.add_position(symbol, 100, direction, 95, 110, 2),
                            r"BUY.*SELL",
                        )
                        clock.now.assert_not_called()
                    self.assertEqual(self.trader.active_positions, active)
                    self.assertEqual(self.trader.trading_history, history)

    def test_invalid_stored_direction_cannot_partially_close_record_or_read_clock(self):
        self.trader.trading_history.append({"sentinel": "prior history"})
        for direction in ("HOLD", "sell", "", None):
            with self.subTest(direction=direction):
                self.trader.add_position("BAD", 100, "BUY", 95, 110, 2)
                record = self.trader.active_positions["BAD"]
                record["type"] = direction  # Simulate preexisting invalid state.
                active = copy.deepcopy(self.trader.active_positions)
                history = copy.deepcopy(self.trader.trading_history)
                with patch("modules.day_trader.datetime") as clock:
                    self.assert_invalid(
                        lambda: self.trader.close_position("BAD", 105, "manual"),
                        r"BUY.*SELL",
                    )
                    clock.now.assert_not_called()
                self.assertIs(self.trader.active_positions["BAD"], record)
                self.assertEqual(self.trader.active_positions, active)
                self.assertEqual(self.trader.trading_history, history)

    def test_missing_close_stays_noop_even_with_unrelated_invalid_record(self):
        self.trader.active_positions["BAD"] = {"type": "INVALID"}
        with patch("modules.day_trader.datetime") as clock:
            self.trader.close_position("MISSING", 105, "manual")
            clock.now.assert_not_called()
        self.assertEqual(self.trader.active_positions, {"BAD": {"type": "INVALID"}})
        self.assertEqual(self.trader.trading_history, [])

    def test_real_scan_skips_valueerror_symbol_and_keeps_later_valid_opportunity(self):
        fixtures = {"INVALID": candles([100] * 60, 0),
                    "VALID": candles([100] * 58 + [99, 101])}
        fetched = []
        rejected = []

        def fetch(symbol, period, interval):
            self.assertEqual((period, interval), ("5d", "15m"))
            fetched.append(symbol)
            return fixtures[symbol].copy(deep=True)

        calculate = self.trader._calculate_entry_exit_points

        def observe_validation(data, analyzer):
            try:
                return calculate(data, analyzer)
            except ValueError as error:
                rejected.append(str(error))
                raise

        with patch("modules.day_trader.BIST_SYMBOLS", {"INVALID": "Bad", "VALID": "Good"}), \
                patch.object(self.trader.data_fetcher, "get_stock_data", side_effect=fetch), \
                patch.object(self.trader, "_calculate_entry_exit_points", side_effect=observe_validation):
            opportunities = self.trader.scan_intraday_opportunities("15m")

        self.assertEqual(fetched, ["INVALID", "VALID"])
        self.assertEqual(len(rejected), 1)
        self.assertRegex(rejected[0], r"ATR")
        self.assertEqual(len(opportunities), 1)
        result = opportunities[0]
        self.assertEqual((result["symbol"], result["name"], result["signal"]),
                         ("VALID", "Good", "BUY"))
        self.assertEqual((result["entry_price"], result["stop_loss"], result["take_profit"]),
                         (101.202, 97.893, 107.214))
        self.assertEqual((result["risk_reward"], result["confidence"], result["timeframe"]),
                         (1.82, 45, "15m"))
        self.assertEqual(self.trader.active_positions, {})
        self.assertEqual(self.trader.trading_history, [])
        for guard in self.io_guards:
            guard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
