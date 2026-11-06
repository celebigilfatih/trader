"""Offline checks of DayTrader's local calculations and synthetic position records.

Expected numbers come from hand-counted ranges, prices and quantities. These
checks characterize the prototype, not a profitable strategy or broker ledger.
"""

import copy
from datetime import datetime
import unittest
from unittest.mock import patch

import pandas as pd

from modules.day_trader import DayTrader
from modules.technical_analysis import TechnicalAnalyzer


def candles(closes, volumes=None, half_range=1.0):
    return pd.DataFrame(
        {"Open": closes, "High": [c + half_range for c in closes],
         "Low": [c - half_range for c in closes], "Close": closes,
         "Volume": volumes if volumes is not None else [1000] * len(closes)},
        index=pd.date_range("2026-01-01", periods=len(closes), freq="D"),
    )


class FixedClock:
    current = datetime(2026, 1, 10, 10, 0)

    @classmethod
    def now(cls):
        return cls.current


class RuleIndicatorInputs:
    """Explicit series for DayTrader rule boundaries, not an indicator-chain test."""

    def __init__(self, **values):
        defaults = {"rsi": [50, 50], "macd": [0, 0],
                    "macd_signal": [1, 1], "ema_5": [100, 100],
                    "ema_21": [100, 100], "vwap": [99, 99],
                    "bb_lower": [90, 90], "bb_upper": [110, 110]}
        defaults.update(values)
        self.indicators = {name: pd.Series(series) for name, series in defaults.items()}

    def add_indicator(self, name):
        # Values above deliberately isolate the consumer's score boundaries.
        pass


class TestDayTraderRisk(unittest.TestCase):
    def setUp(self):
        # Constructors remain real: BISTDataFetcher merely creates a Session.
        # Any accidental external I/O during these local operations is a failure.
        for target in ("socket.socket.connect", "socket.create_connection",
                       "requests.sessions.Session.request", "smtplib.SMTP",
                       "smtplib.SMTP_SSL", "builtins.open"):
            guard = patch(target, side_effect=AssertionError("Unexpected external I/O"))
            guard.start()
            self.addCleanup(guard.stop)
        self.trader = DayTrader()
        self.addCleanup(self.trader.data_fetcher.session.close)
        FixedClock.current = datetime(2026, 1, 10, 10, 0)
        clock = patch("modules.day_trader.datetime", FixedClock)
        clock.start()
        self.addCleanup(clock.stop)

    def test_atr_uses_both_previous_close_gaps_and_only_last_fourteen_ranges(self):
        data = candles([100, 100, 109, 101] + [100] * 11)
        data.iloc[0, data.columns.get_loc("High")] = 200
        data.iloc[0, data.columns.get_loc("Low")] = 0
        # The oldest width 200 is excluded. Gap up TR=10, gap down TR=9;
        # the twelve other ranges are 2, so final ATR=(10+9+12*2)/14.
        original = data.copy(deep=True)
        self.assertAlmostEqual(self.trader._calculate_atr(data), 43 / 14)
        pd.testing.assert_frame_equal(data, original)

    def test_atr_respects_custom_period_instead_of_default_or_wilder_smoothing(self):
        data = candles([100] * 14 + [109, 101, 100])
        # Last three true ranges are 10, 9, 2: their simple average is 7.
        self.assertEqual(self.trader._calculate_atr(data, period=3), 7.0)

    def test_atr_percent_scales_range_by_current_price(self):
        data = candles([250.0] * 20, half_range=3.0)
        self.assertEqual(self.trader._calculate_atr(data), 6.0)
        self.assertAlmostEqual(self.trader._calculate_atr_percent(data), 2.4)

    def test_volume_ratio_includes_current_bar_and_excludes_old_outlier(self):
        data = candles([100] * 22, [1000000] + [100] * 20 + [300])
        # Last 20 volumes: nineteen 100s plus 300 => mean 110, ratio 30/11.
        self.assertAlmostEqual(self.trader._calculate_volume_ratio(data), 30 / 11)
        self.assertEqual(self.trader._calculate_volume_ratio(candles([100] * 22)), 1.0)

    def test_computed_buy_chain_uses_atr_and_entry_offset_in_risk_reward(self):
        data = candles([100] * 40 + [99, 101])
        analyzer = TechnicalAnalyzer(data)
        result = self.trader._calculate_entry_exit_points(data, analyzer)
        # Last TR=3, preceding thirteen TR=2 => ATR=29/14.
        # Crosses contribute +2 each, VWAP +1, upper BB -2 => score +3.
        # Stop=101-1.5*29/14; target=101+3*29/14; entry=101*1.002.
        self.assertEqual(result, {
            "signal": "BUY", "entry_price": 101.202, "stop_loss": 97.893,
            "take_profit": 107.214, "risk_reward": 1.82, "confidence": 45,
            "signal_score": 3, "reasons": ["MACD Bullish Cross",
                "EMA5 > EMA21 Cross", "Above VWAP", "BB Upper Touch"],
        })
        self.assertAlmostEqual(analyzer.indicators["rsi"].iloc[-1], 67.6923076923077)
        self.assertEqual(analyzer.indicators["vwap"].iloc[-1], 100.0)
        self.assertLess(result["risk_reward"], 2.0)

    def test_computed_sell_chain_reverses_stop_target_and_entry_offset(self):
        data = candles([100] * 40 + [101, 99])
        result = self.trader._calculate_entry_exit_points(data, TechnicalAnalyzer(data))
        # Mirror chain: -2 MACD -2 EMA -1 VWAP +2 lower BB = -3.
        # Stop=99+1.5*29/14; target=99-3*29/14; entry=99*.998.
        self.assertEqual(result, {
            "signal": "SELL", "entry_price": 98.802, "stop_loss": 102.107,
            "take_profit": 92.786, "risk_reward": 1.82, "confidence": 45,
            "signal_score": -3, "reasons": ["MACD Bearish Cross",
                "EMA5 < EMA21 Cross", "Below VWAP", "BB Lower Touch"],
        })

    def test_computed_hold_has_current_price_for_all_levels_and_zero_reward(self):
        data = candles(list(range(100, 140)))
        result = self.trader._calculate_entry_exit_points(data, TechnicalAnalyzer(data))
        # Persistent trend produces no final-bar cross, no band touch: VWAP +1.
        self.assertEqual(result, {
            "signal": "HOLD", "entry_price": 139, "stop_loss": 139,
            "take_profit": 139, "risk_reward": 0, "confidence": 15,
            "signal_score": 1, "reasons": ["Above VWAP"],
        })

    def test_levels_round_to_three_decimals_after_unrounded_risk_computation(self):
        # Scale BUY by .0123456: entry=1.2493994112, stop=1.20854605714,
        # target=1.32362468571. Rounded-price R:R would be .075/.040=1.875,
        # whereas using full precision before output rounding still gives 1.82.
        data = candles([1.23456] * 40 + [1.2222144, 1.2469056], half_range=.0123456)
        result = self.trader._calculate_entry_exit_points(data, TechnicalAnalyzer(data))
        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["entry_price"], 1.249)
        self.assertEqual(result["stop_loss"], 1.209)
        self.assertEqual(result["take_profit"], 1.324)
        self.assertEqual(result["risk_reward"], 1.82)

    def test_explicit_rule_scores_include_two_and_minus_two_but_hold_at_one(self):
        data = candles([100] * 40)
        for inputs, score, signal in (
            ({"rsi": [30, 29], "vwap": [101, 101]}, 2, "BUY"),
            ({"rsi": [70, 71]}, -2, "SELL"),
            ({}, 1, "HOLD"),
            ({"vwap": [101, 101]}, -1, "HOLD"),
        ):
            with self.subTest(score=score):
                result = self.trader._calculate_entry_exit_points(data, RuleIndicatorInputs(**inputs))
                self.assertEqual(result["signal_score"], score)
                self.assertEqual(result["signal"], signal)
                self.assertEqual(result["confidence"], 30 if abs(score) == 2 else 15)

    def test_explicit_agreeing_rules_cap_confidence_in_both_directions(self):
        data = candles([100] * 40)
        for inputs, score, signal in (
            ({"rsi": [30, 29], "macd": [0, 2], "ema_5": [99, 101],
              "bb_lower": [100, 100]}, 10, "BUY"),
            ({"rsi": [70, 71], "macd": [2, 0], "ema_5": [101, 99],
              "vwap": [101, 101], "bb_upper": [100, 100]}, -10, "SELL"),
        ):
            with self.subTest(signal=signal):
                result = self.trader._calculate_entry_exit_points(data, RuleIndicatorInputs(**inputs))
                self.assertEqual(result["signal_score"], score)
                self.assertEqual(result["signal"], signal)
                self.assertEqual(result["confidence"], 100)

    def test_calculation_preserves_input_and_repeat_results_are_independent(self):
        data = candles([100] * 40 + [99, 101])
        original = data.copy(deep=True)
        analyzer = TechnicalAnalyzer(data)
        first = self.trader._calculate_entry_exit_points(data, analyzer)
        first["reasons"].append("caller change")
        second = self.trader._calculate_entry_exit_points(data, analyzer)
        self.assertNotIn("caller change", second["reasons"])
        self.assertEqual(second["signal"], "BUY")
        pd.testing.assert_frame_equal(data, original)
        pd.testing.assert_frame_equal(analyzer.data, original)
        self.assertEqual(self.trader.active_positions, {})
        self.assertEqual(self.trader.trading_history, [])

    def test_open_records_preserve_direction_quantity_levels_and_entry_time(self):
        self.trader.add_position("SYNTH-A", 125.5, "BUY", 121, 134, 3.5)
        FixedClock.current = datetime(2026, 1, 10, 10, 5)
        self.trader.add_position("SYNTH-B", 200, "SELL", 205, 190, 2)
        self.assertEqual(self.trader.get_active_positions(), [
            {"symbol": "SYNTH-A", "entry_price": 125.5, "type": "BUY",
             "stop_loss": 121, "take_profit": 134, "size": 3.5,
             "entry_time": datetime(2026, 1, 10, 10, 0), "status": "ACTIVE"},
            {"symbol": "SYNTH-B", "entry_price": 200, "type": "SELL",
             "stop_loss": 205, "take_profit": 190, "size": 2,
             "entry_time": datetime(2026, 1, 10, 10, 5), "status": "ACTIVE"},
        ])
        self.assertEqual(self.trader.trading_history, [])

    def test_buy_close_applies_quantity_to_profit_loss_and_breakeven(self):
        for count, (exit_price, quantity, expected) in enumerate(
                ((130, 3.5, 15.75), (120, 2, -11), (125.5, 4, 0)), start=1):
            with self.subTest(exit_price=exit_price, quantity=quantity):
                trader = self.trader
                trader.add_position("SYNTH-A", 125.5, "BUY", 121, 134, quantity)
                trader.close_position("SYNTH-A", exit_price, "manual")
                self.assertEqual(len(trader.trading_history), count)
                self.assertEqual(trader.trading_history[-1]["pnl"], expected)
                self.assertEqual(trader.active_positions, {})
        self.assertEqual(len(self.trader.trading_history), 3)

    def test_sell_close_reverses_price_difference_and_applies_quantity(self):
        for count, (exit_price, quantity, expected) in enumerate(
                ((190, 2.5, 25), (205, 3, -15), (200, 4, 0)), start=1):
            with self.subTest(exit_price=exit_price, quantity=quantity):
                self.trader.add_position("SYNTH-B", 200, "SELL", 205, 190, quantity)
                self.trader.close_position("SYNTH-B", exit_price, "manual")
                self.assertEqual(len(self.trader.trading_history), count)
                self.assertEqual(self.trader.trading_history[-1]["pnl"], expected)
                self.assertEqual(self.trader.active_positions, {})
        self.assertEqual(len(self.trader.trading_history), 3)

    def test_close_moves_only_selected_record_and_retains_times_and_reason(self):
        self.trader.add_position("SYNTH-A", 125.5, "BUY", 121, 134, 3.5)
        self.trader.add_position("SYNTH-B", 200, "SELL", 205, 190, 2)
        survivor = copy.deepcopy(self.trader.active_positions["SYNTH-B"])
        FixedClock.current = datetime(2026, 1, 10, 11, 30)
        self.trader.close_position("SYNTH-A", 130, "target reached")
        self.assertEqual(self.trader.active_positions, {"SYNTH-B": survivor})
        self.assertEqual(self.trader.trading_history, [{
            "symbol": "SYNTH-A", "entry_price": 125.5, "type": "BUY",
            "stop_loss": 121, "take_profit": 134, "size": 3.5,
            "entry_time": datetime(2026, 1, 10, 10, 0), "status": "CLOSED",
            "exit_price": 130, "exit_time": datetime(2026, 1, 10, 11, 30),
            "close_reason": "target reached", "pnl": 15.75,
        }])

    def test_closed_history_is_snapshot_when_symbol_is_opened_again(self):
        self.trader.add_position("SYNTH-A", 100, "BUY", 95, 110, 2)
        prior_record = self.trader.active_positions["SYNTH-A"]
        self.trader.close_position("SYNTH-A", 105, "manual")
        prior_record["entry_price"] = -999
        self.assertEqual(len(self.trader.trading_history), 1)
        self.assertEqual(self.trader.trading_history[0]["entry_price"], 100)
        self.trader.add_position("SYNTH-A", 110, "SELL", 115, 100, 3)
        self.assertEqual(self.trader.trading_history[0]["pnl"], 10)
        self.assertEqual(self.trader.trading_history[0]["status"], "CLOSED")
        self.assertEqual(self.trader.active_positions["SYNTH-A"]["status"], "ACTIVE")
        self.assertNotIn("exit_price", self.trader.active_positions["SYNTH-A"])

    def test_closing_nonexistent_or_already_closed_symbol_changes_nothing(self):
        self.trader.add_position("SYNTH-A", 100, "BUY", 95, 110, 2)
        self.trader.add_position("SYNTH-B", 200, "SELL", 205, 190, 3)
        self.trader.close_position("SYNTH-A", 105, "manual")
        active = copy.deepcopy(self.trader.active_positions)
        history = copy.deepcopy(self.trader.trading_history)
        for symbol in ("UNKNOWN", "SYNTH-A"):
            self.trader.close_position(symbol, 999, "ignored")
            self.assertEqual(self.trader.active_positions, active)
            self.assertEqual(self.trader.trading_history, history)


if __name__ == "__main__":
    unittest.main()
