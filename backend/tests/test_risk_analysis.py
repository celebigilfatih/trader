"""Offline characterization of heuristic risk rules, not investment guidance."""

import unittest

import pandas as pd

from modules.alert_system import AlertSystem
from modules.technical_analysis import TechnicalAnalyzer


def risk_analyzer(closes=None, volumes=None, low=90.0, high=110.0, **indicators):
    """Real analyzer; explicit scalar series isolate the risk consumer boundary."""
    closes = [100.0] * 21 if closes is None else closes
    data = pd.DataFrame({
        "Open": closes, "High": [max(high, close + 1) for close in closes],
        "Low": [min(low, close - 1) for close in closes], "Close": closes,
        "Volume": [1000] * len(closes) if volumes is None else volumes,
    }, index=pd.date_range("2026-01-01", periods=len(closes)))
    analyzer = TechnicalAnalyzer(data)
    analyzer.indicators = {name: pd.Series(value, index=data.index, dtype=float)
                           for name, value in indicators.items()}
    return analyzer


class TestRiskAnalysis(unittest.TestCase):
    def setUp(self):
        self.alerts = AlertSystem()

    def analyze(self, **kwargs):
        return self.alerts.generate_comprehensive_risk_analysis(risk_analyzer(**kwargs))

    def test_neutral_prices_without_indicators_have_no_added_risk(self):
        result = self.analyze()
        self.assertEqual(result["risk_score"], 0)
        self.assertEqual(result["risk_level"], "MİNİMAL RİSK")
        self.assertEqual(set(result["risk_factors"]),
                         {"low_volatility", "normal_volume", "neutral_position"})
        self.assertTrue(result["recommendations"])

    def test_rsi_rule_boundaries_add_only_the_documented_weights(self):
        # Zero truthiness is a separately recorded defect, not endorsed here.
        for rsi, score, factor in ((20, 1, "rsi_oversold"), (20.01, 0, "rsi_normal"),
                                    (59.99, 0, "rsi_normal"), (60, .5, "rsi_high"),
                                    (79.99, .5, "rsi_high"), (80, 2, "rsi_overbought"),
                                    (100, 2, "rsi_overbought")):
            with self.subTest(rsi=rsi):
                result = self.analyze(rsi=rsi)
                self.assertEqual(result["risk_score"], score)
                self.assertIn(factor, result["risk_factors"])

    def test_negative_macd_momentum_requires_strict_histogram_threshold(self):
        for macd, signal, score, factor in ((-.2, -.1, 0, "weak_momentum"),
                                             (-.21, -.1, 1.5, "strong_negative_momentum"),
                                             (.2, .1, 0, "weak_momentum"),
                                             (.21, .1, 0, "strong_positive_momentum")):
            with self.subTest(macd=macd, signal=signal):
                result = self.analyze(macd=macd, macd_signal=signal)
                self.assertEqual(result["risk_score"], score)
                self.assertIn(factor, result["risk_factors"])

    def test_ema_distance_thresholds_penalize_only_downtrends(self):
        for ema, score, factor in ((98, 0, "sideways_trend"),
                                    (97.99, 1, "medium_downtrend"),
                                    (95, 1, "medium_downtrend"),
                                    (94.99, 2, "strong_downtrend"),
                                    (102, 0, "sideways_trend"),
                                    (102.01, 0, "medium_uptrend"),
                                    (105, 0, "medium_uptrend"),
                                    (105.01, 0, "strong_uptrend")):
            with self.subTest(ema=ema):
                result = self.analyze(ema_21=ema, ema_50=100)
                self.assertEqual(result["risk_score"], score)
                self.assertIn(factor, result["risk_factors"])

    def test_support_and_resistance_use_strict_two_percent_and_support_precedence(self):
        for low, high, score, factor in ((98, 110, 0, "neutral_position"),
                                         (98.01, 110, 2, "near_support"),
                                         (90, 102, 0, "neutral_position"),
                                         (90, 101.99, 1, "near_resistance"),
                                         (99, 101, 2, "near_support")):
            with self.subTest(low=low, high=high):
                result = self.analyze(low=low, high=high)
                self.assertEqual(result["risk_score"], score)
                self.assertIn(factor, result["risk_factors"])
                if factor == "near_support":
                    self.assertNotIn("near_resistance", result["risk_factors"])

    def test_volume_average_includes_current_bar_at_high_and_low_boundaries(self):
        # At high boundary: (19*900+1900)/20=950, current/mean=2 exactly.
        # At low boundary: (19*3900+1900)/20=3800, ratio=.5 exactly.
        cases = [(900, 1899, 0, "normal_volume"), (900, 1900, 0, "normal_volume"),
                 (900, 1901, 2, "high_volume_decline"),
                 (3900, 1899, 1, "low_volume"), (3900, 1900, 0, "normal_volume"),
                 (3900, 1901, 0, "normal_volume")]
        for previous, current, score, factor in cases:
            with self.subTest(previous=previous, current=current):
                result = self.analyze(closes=[100.0] * 20 + [97.0],
                                      volumes=[previous] * 20 + [current])
                self.assertEqual(result["risk_score"], score)
                self.assertIn(factor, result["risk_factors"])

    def test_high_volume_decline_requires_more_than_two_percent_price_drop(self):
        for close, score in ((98.0, 0), (97.99, 2)):
            with self.subTest(close=close):
                result = self.analyze(closes=[100.0] * 20 + [close],
                                      volumes=[900] * 20 + [1901])
                self.assertEqual(result["risk_score"], score)
                self.assertEqual("high_volume_decline" in result["risk_factors"], score == 2)

    def test_volatility_buckets_use_annualized_sample_standard_deviation(self):
        # 20 returns: ten a=(peak-100)/100 and ten b=(100-peak)/peak.
        # Sample variance=5*(a-b)^2/19; annualized std=abs(a-b)*sqrt(1260/19).
        # Peaks 101/102/104 give about 16.2%/32.3%/63.9%, independently derived.
        for peak, score, factor, percent in ((101, 0, "low_volatility", "16.2"),
                                              (101.5, 0, "low_volatility", "24.2"),
                                              (101.6, 1, "medium_volatility", "25.9"),
                                              (102, 1, "medium_volatility", "32.3"),
                                              (102.4, 1, "medium_volatility", "38.6"),
                                              (102.5, 2, "high_volatility", "40.2"),
                                              (104, 2, "high_volatility", "63.9")):
            with self.subTest(peak=peak):
                result = self.analyze(closes=[100.0, peak] * 10 + [100.0])
                self.assertEqual(result["risk_score"], score)
                self.assertIn(factor, result["risk_factors"])
                self.assertIn("%" + percent, result["risk_factors"][factor])

    def test_volatility_needs_twenty_returns_and_excludes_older_shock(self):
        short = self.analyze(closes=[100.0, 104.0] * 10)
        self.assertEqual(short["risk_score"], 0)
        self.assertFalse(any("volatility" in factor for factor in short["risk_factors"]))
        enough = self.analyze(closes=[100.0, 104.0] * 10 + [100.0])
        self.assertEqual(enough["risk_score"], 2)
        self.assertIn("high_volatility", enough["risk_factors"])
        older = self.analyze(closes=[50.0] + [100.0] * 21)
        self.assertEqual(older["risk_score"], 0)
        self.assertIn("low_volatility", older["risk_factors"])
        # One .04 return and nineteen zeroes: sample std=.04/sqrt(20).
        # This asymmetric case distinguishes a 19-return window from 20.
        final_shock = self.analyze(closes=[100.0] * 20 + [104.0])
        self.assertEqual(final_shock["risk_score"], 0)
        self.assertIn("%14.2", final_shock["risk_factors"]["low_volatility"])

    def test_aggregated_tiers_drive_position_and_stop_target_amounts(self):
        # Separate risk components give values immediately below and at 2/4/7.
        cases = [
            ({}, 1000, 90, 0, "MİNİMAL RİSK", "Portföyün %75-100'ü (Minimal risk)", "₺93.00", "₺121.00"),
            ({"macd": -.3, "macd_signal": -.1}, 1000, 90, 1.5, "MİNİMAL RİSK", "Portföyün %75-100'ü (Minimal risk)", "₺93.00", "₺121.00"),
            ({"rsi": 80}, 1000, 90, 2, "DÜŞÜK RİSK", "Portföyün %50-75'i (Düşük risk)", "₺95.00", "₺112.50"),
            ({"rsi": 80, "macd": -.3, "macd_signal": -.1}, 1000, 90, 3.5, "DÜŞÜK RİSK", "Portföyün %50-75'i (Düşük risk)", "₺95.00", "₺112.50"),
            ({"rsi": 80, "ema_21": 94, "ema_50": 100}, 1000, 90, 4, "ORTA RİSK", "Portföyün %25-50'si (Orta risk)", "₺96.50", "₺107.00"),
            ({"rsi": 80, "ema_21": 94, "ema_50": 100, "macd": -.3, "macd_signal": -.1}, 100, 90, 6.5, "ORTA RİSK", "Portföyün %25-50'si (Orta risk)", "₺96.50", "₺107.00"),
            ({"rsi": 80, "ema_21": 94, "ema_50": 100}, 100, 99, 7, "YÜKSEK RİSK", "Portföyün %15-25'i (Yüksek risk)", "₺98.00", "₺103.00"),
        ]
        for indicators, last_volume, low, score, level, size, stop, target in cases:
            with self.subTest(score=score):
                result = self.analyze(volumes=[1000] * 20 + [last_volume], low=low, **indicators)
                self.assertEqual(result["risk_score"], score)
                self.assertEqual(result["risk_level"], level)
                self.assertEqual(result["position_sizing"], size)
                # Only financial amounts are pinned; current duplicated '%' copy is not endorsed.
                self.assertEqual(result["stop_loss_suggestion"].split()[0], stop)
                self.assertEqual(result["take_profit_suggestion"].split()[0], target)

    def test_stop_and_target_round_non_hundred_prices(self):
        result = self.analyze(closes=[123.456] * 21, low=100, high=150)
        self.assertEqual(result["risk_score"], 0)
        # 123.456*.93=114.81408, 123.456*1.21=149.38176.
        self.assertEqual(result["stop_loss_suggestion"].split()[0], "₺114.81")
        self.assertEqual(result["take_profit_suggestion"].split()[0], "₺149.38")

    def test_real_indicator_chain_has_manual_two_point_risk_total(self):
        closes = list(range(100, 140))
        data = pd.DataFrame({"Open": closes, "High": [p + 10 for p in closes],
                             "Low": [p - 10 for p in closes], "Close": closes,
                             "Volume": 1000}, index=pd.date_range("2026-01-01", periods=40))
        analyzer = TechnicalAnalyzer(data)
        for indicator in ("rsi", "macd", "ema_21", "ema_50"):
            analyzer.add_indicator(indicator)
        result = self.alerts.generate_comprehensive_risk_analysis(analyzer)
        # Rising closes: RSI=100 adds 2; upward EMA/MACD and ordinary volume add 0.
        # Last 20 bars span support110/resistance149; close139 is far from both.
        self.assertEqual(result["risk_score"], 2)
        self.assertEqual(result["risk_level"], "DÜŞÜK RİSK")
        self.assertIn("rsi_overbought", result["risk_factors"])
        self.assertIn("neutral_position", result["risk_factors"])

    def test_inputs_and_previous_results_are_independent_across_calls(self):
        analyzer = risk_analyzer(rsi=80, macd=-.3, macd_signal=-.1)
        original = analyzer.data.copy(deep=True)
        indicators = {name: series.copy(deep=True) for name, series in analyzer.indicators.items()}
        first = self.alerts.generate_comprehensive_risk_analysis(analyzer)
        first["risk_factors"]["synthetic_sentinel"] = "sentinel"
        first["recommendations"].append("synthetic_sentinel")
        second = self.alerts.generate_comprehensive_risk_analysis(analyzer)
        self.assertEqual(second["risk_score"], 3.5)
        self.assertNotIn("synthetic_sentinel", second["risk_factors"])
        self.assertNotIn("synthetic_sentinel", second["recommendations"])
        pd.testing.assert_frame_equal(analyzer.data, original)
        for name, series in indicators.items():
            pd.testing.assert_series_equal(analyzer.indicators[name], series)


if __name__ == "__main__":
    unittest.main()
