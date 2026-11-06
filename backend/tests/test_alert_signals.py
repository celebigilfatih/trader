"""Offline checks for real signal rules, without SMTP or wall-clock outputs."""

import unittest

import pandas as pd

from modules.alert_system import AlertSystem
from modules.technical_analysis import TechnicalAnalyzer


def analyzer_with(closes=None, volumes=None, **indicators):
    """Use real analyzer data/accessors; rule inputs are explicit synthetic series."""
    closes = [100.0] * 20 if closes is None else closes
    data = pd.DataFrame({
        "Open": closes,
        "High": [price + 1 for price in closes],
        "Low": [price - 1 for price in closes],
        "Close": closes,
        "Volume": [1000] * len(closes) if volumes is None else volumes,
    }, index=pd.date_range("2026-01-01", periods=len(closes)))
    analyzer = TechnicalAnalyzer(data)
    analyzer.indicators = {name: pd.Series(values, dtype=float)
                           for name, values in indicators.items()}
    return analyzer


class TestAlertSignals(unittest.TestCase):
    def setUp(self):
        self.alerts = AlertSystem()

    def test_rsi_extremes_thresholds_neutral_and_missing(self):
        for value, expected in [(None, None), (float("nan"), None),
                                (0, "AL"), (29.99, "AL"), (30, "AL"),
                                (30.01, None), (50, None), (69.99, None),
                                (70, "SAT"), (70.01, "SAT"), (100, "SAT")]:
            with self.subTest(value=value):
                self.assertEqual(self.alerts._rsi_signal(value), expected)

    def test_line_crossings_include_previous_equality_but_not_current_equality(self):
        for method, first_name, second_name in [
            ("_macd_signal", "macd", "macd_signal"),
            ("_moving_average_signal", "ema_21", "ema_50"),
        ]:
            for first, second, expected in [
                ([1, 3], [2, 2], "AL"), ([3, 1], [2, 2], "SAT"),
                ([2, 3], [2, 2], "AL"), ([2, 1], [2, 2], "SAT"),
                ([1, 2], [2, 2], None), ([3, 2], [2, 2], None),
                ([3, 4], [2, 2], None), ([1, 0], [2, 2], None),
            ]:
                with self.subTest(method=method, first=first, second=second):
                    analyzer = analyzer_with(**{first_name: first, second_name: second})
                    self.assertEqual(getattr(self.alerts, method)(analyzer), expected)

    def test_crossings_require_two_valid_observations_of_each_line(self):
        nan = float("nan")
        for method, first, second in [
            ("_macd_signal", "macd", "macd_signal"),
            ("_moving_average_signal", "ema_21", "ema_50"),
        ]:
            for series in [{}, {first: [1, 3]}, {second: [2, 2]},
                           {first: [nan, 3], second: [2, 2]},
                           {first: [1, 3], second: [nan, 2]}]:
                with self.subTest(method=method, series=series):
                    self.assertIsNone(getattr(self.alerts, method)(analyzer_with(**series)))

    def test_trend_signals_require_direction_change(self):
        for method, name in [("_supertrend_signal", "supertrend_trend"),
                             ("_ott_signal", "ott_trend")]:
            for series, expected in [([-1, 1], "AL"), ([1, -1], "SAT"),
                                     ([1, 1], None), ([-1, -1], None),
                                     ([1], None), ([float("nan"), 1], None)]:
                with self.subTest(method=method, series=series):
                    self.assertEqual(getattr(self.alerts, method)(
                        analyzer_with(**{name: series})), expected)
            with self.subTest(method=method, missing=True):
                self.assertIsNone(getattr(self.alerts, method)(analyzer_with()))

    def test_bollinger_tolerance_boundaries(self):
        for price, expected in [(101.99, "AL"), (102, "AL"), (102.01, None),
                                (117.59, None), (117.6, "SAT"), (117.61, "SAT")]:
            with self.subTest(price=price):
                analyzer = analyzer_with(closes=[price] * 20,
                                         bb_lower=[100], bb_upper=[120])
                self.assertEqual(self.alerts._bollinger_signal(analyzer), expected)

    def test_bollinger_missing_or_unavailable_bands_have_no_signal(self):
        for series in [{}, {"bb_upper": [120]}, {"bb_lower": [100]},
                       {"bb_upper": [float("nan")], "bb_lower": [100]},
                       {"bb_upper": [120], "bb_lower": [float("nan")]}]:
            with self.subTest(series=series):
                self.assertIsNone(self.alerts._bollinger_signal(analyzer_with(**series)))

    def test_volume_signal_requires_strict_spike_and_price_movement(self):
        # 19*900 + 1900 = 19000; last volume is exactly 2*20-bar mean.
        for volume, price, expected in [(1900, 103, None), (1901, 103, "AL"),
                                        (1901, 97, "SAT"), (1901, 102, None),
                                        (1901, 98, None), (1901, 100, None)]:
            with self.subTest(volume=volume, price=price):
                analyzer = analyzer_with(closes=[100] * 19 + [price],
                                         volumes=[900] * 19 + [volume])
                self.assertEqual(self.alerts._volume_signal(analyzer), expected)

    def test_generate_signal_combines_real_rules_by_majority(self):
        # RSI + upward MACD vote AL, SuperTrend votes SAT; then reverse it.
        for series, expected in [
            ({"rsi": [30], "macd": [1, 3], "macd_signal": [2, 2],
              "supertrend_trend": [1, -1]}, "AL"),
            ({"rsi": [70], "macd": [3, 1], "macd_signal": [2, 2],
              "supertrend_trend": [-1, 1]}, "SAT"),
            ({"rsi": [30], "supertrend_trend": [1, -1]}, "BEKLE"),
            ({}, "BEKLE"),
        ]:
            with self.subTest(series=series):
                self.assertEqual(self.alerts.generate_signal(analyzer_with(**series)), expected)

    def test_each_rule_reaches_public_signal_generation(self):
        for series, expected in [
            ({"rsi": [0]}, "AL"), ({"rsi": [100]}, "SAT"),
            ({"macd": [1, 3], "macd_signal": [2, 2]}, "AL"),
            ({"ema_21": [3, 1], "ema_50": [2, 2]}, "SAT"),
            ({"supertrend_trend": [-1, 1]}, "AL"),
            ({"ott_trend": [1, -1]}, "SAT"),
            ({"bb_lower": [100], "bb_upper": [120]}, "AL"),
        ]:
            with self.subTest(series=series):
                self.assertEqual(self.alerts.generate_signal(analyzer_with(**series)), expected)
        self.assertEqual(self.alerts.generate_signal(analyzer_with(
            closes=[100] * 19 + [103], volumes=[900] * 19 + [1901])), "AL")

    def test_real_rsi_calculation_flows_into_signal(self):
        for prices, expected in [(list(range(100, 140)), "SAT"),
                                 (list(range(140, 100, -1)), "AL")]:
            with self.subTest(expected=expected):
                analyzer = analyzer_with(closes=prices)
                analyzer.add_indicator("rsi")
                self.assertEqual(self.alerts.generate_signal(analyzer), expected)

    def test_signal_generation_preserves_data_and_indicators(self):
        analyzer = analyzer_with(rsi=[30], macd=[1, 3], macd_signal=[2, 2])
        original_data = analyzer.data.copy(deep=True)
        original_indicators = {name: series.copy(deep=True)
                               for name, series in analyzer.indicators.items()}
        self.alerts.generate_signal(analyzer)
        pd.testing.assert_frame_equal(analyzer.data, original_data)
        self.assertEqual(set(analyzer.indicators), set(original_indicators))
        for name, series in original_indicators.items():
            pd.testing.assert_series_equal(analyzer.indicators[name], series)


if __name__ == "__main__":
    unittest.main()
