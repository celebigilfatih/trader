"""Real OTT producer/consumer integration for Accepted ADR-0006.

Only OTT is computed; constant volume keeps unrelated volume scoring neutral.
Literal expected directions and prices are independent of the implementation.
"""

import unittest

import pandas as pd

from modules.alert_system import AlertSystem
from modules.technical_analysis import TechnicalAnalyzer


def ohlcv(closes):
    return pd.DataFrame({
        "Open": closes,
        "High": [value + 1 for value in closes],
        "Low": [value - 1 for value in closes],
        "Close": closes,
        "Volume": [1000] * len(closes),
    }, index=pd.date_range("2026-01-01", periods=len(closes)))


class TestOttIntegration(unittest.TestCase):
    def setUp(self):
        self.alerts = AlertSystem()

    def real_ott(self, closes):
        analyzer = TechnicalAnalyzer(ohlcv(closes))
        analyzer.add_indicator("ott")
        # Missing producer output must fail here, before any reader silently
        # returns None or a direct lookup raises KeyError.
        self.assertIn("ott_trend", analyzer.indicators)
        return analyzer

    def test_literal_reversals_reach_the_real_crossover_reader(self):
        cases = [
            ([100.0, 110.0, 90.0], [100.0, 105.995, 96.72],
             [float("nan"), 1.0, -1.0], "SAT"),
            ([100.0, 90.0, 110.0], [100.0, 93.795, 103.15076923076923],
             [float("nan"), -1.0, 1.0], "AL"),
        ]
        for closes, prices, directions, signal in cases:
            with self.subTest(closes=closes):
                analyzer = self.real_ott(closes)
                pd.testing.assert_series_equal(
                    analyzer.indicators["ott"],
                    pd.Series(prices, index=analyzer.data.index),
                    check_names=False, rtol=1e-12, atol=1e-12)
                pd.testing.assert_series_equal(
                    analyzer.indicators["ott_trend"],
                    pd.Series(directions, index=analyzer.data.index),
                    check_names=False)
                self.assertEqual(self.alerts._ott_signal(analyzer), signal)

    def test_first_computed_direction_is_not_a_crossover(self):
        for closes, direction in (([100.0, 110.0], 1.0),
                                  ([100.0, 90.0], -1.0)):
            with self.subTest(closes=closes):
                analyzer = self.real_ott(closes)
                self.assertTrue(pd.isna(analyzer.indicators["ott_trend"].iloc[0]))
                self.assertEqual(analyzer.indicators["ott_trend"].iloc[1], direction)
                self.assertIsNone(self.alerts._ott_signal(analyzer))

    def test_unchanged_directions_do_not_create_a_crossover(self):
        for closes, direction in (([100.0, 110.0, 120.0], 1.0),
                                  ([100.0, 90.0, 80.0], -1.0),
                                  ([100.0, 100.0, 100.0], -1.0)):
            with self.subTest(closes=closes):
                analyzer = self.real_ott(closes)
                self.assertEqual(analyzer.indicators["ott_trend"].iloc[1:].tolist(),
                                 [direction, direction])
                self.assertIsNone(self.alerts._ott_signal(analyzer))

    def test_empty_and_single_observation_have_no_crossover(self):
        # Position recommendation separately requires at least two OHLCV rows.
        for closes in ([], [100.0]):
            with self.subTest(closes=closes):
                analyzer = self.real_ott(closes)
                self.assertTrue(analyzer.indicators["ott_trend"].dropna().empty)
                self.assertIsNone(self.alerts._ott_signal(analyzer))

    def test_real_last_direction_adds_only_the_existing_ott_position_weight(self):
        cases = [
            ([100.0, 110.0], 1, 0, "ZAYIF AL", ["OTT Pozitif"], []),
            ([100.0, 90.0], 0, 1, "ZAYIF SAT", [], ["OTT Negatif"]),
            ([100.0, 110.0, 90.0], 0, 1, "ZAYIF SAT", [], ["OTT Negatif"]),
            ([100.0, 90.0, 110.0], 1, 0, "ZAYIF AL", ["OTT Pozitif"], []),
        ]
        for closes, bull, bear, recommendation, bull_signals, bear_signals in cases:
            with self.subTest(closes=closes):
                analyzer = self.real_ott(closes)
                self.assertEqual(set(analyzer.indicators),
                                 {"ott", "ott_signal", "ott_trend"})
                result = self.alerts.generate_position_recommendation(analyzer)
                self.assertEqual(result["bull_score"], bull)
                self.assertEqual(result["bear_score"], bear)
                self.assertEqual(result["total_score"], bull - bear)
                self.assertEqual(result["recommendation"], recommendation)
                self.assertEqual(result["bull_signals"], bull_signals)
                self.assertEqual(result["bear_signals"], bear_signals)
                self.assertEqual(result["confidence"], 10)

    def test_repeated_real_readers_preserve_producer_output_and_caller(self):
        data = ohlcv([100.0, 110.0, 90.0])
        original = data.copy(deep=True)
        analyzer = TechnicalAnalyzer(data)
        analyzer.add_indicator("ott")
        self.assertIn("ott_trend", analyzer.indicators)
        calculated_data = analyzer.data.copy(deep=True)
        outputs = {name: values.copy(deep=True)
                   for name, values in analyzer.indicators.items()}
        first = self.alerts.generate_position_recommendation(analyzer)
        for _ in range(2):
            analyzer.add_indicator("ott")
            self.assertEqual(self.alerts._ott_signal(analyzer), "SAT")
            self.assertEqual(self.alerts.generate_position_recommendation(analyzer), first)
            for name, expected in outputs.items():
                pd.testing.assert_series_equal(analyzer.indicators[name], expected)
        self.assertEqual(first["total_score"], -1)
        pd.testing.assert_frame_equal(analyzer.data, calculated_data)
        pd.testing.assert_frame_equal(data, original)


if __name__ == "__main__":
    unittest.main()
