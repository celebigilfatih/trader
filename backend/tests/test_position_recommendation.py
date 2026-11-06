"""Offline characterization of position rules; scores are heuristic, not odds."""

import unittest

import pandas as pd

from modules.alert_system import AlertSystem
from modules.technical_analysis import TechnicalAnalyzer


def position_analyzer(indicators=None, closes=None, volumes=None):
    """Real analyzer/accessor with explicit last-bar inputs for isolated rules."""
    closes = [100.0] * 21 if closes is None else closes
    data = pd.DataFrame({
        "Open": closes,
        "High": [value + 1 for value in closes],
        "Low": [value - 1 for value in closes],
        "Close": closes,
        "Volume": [100] * len(closes) if volumes is None else volumes,
    }, index=pd.date_range("2026-01-01", periods=len(closes)))
    analyzer = TechnicalAnalyzer(data)
    for name, value in (indicators or {}).items():
        analyzer.indicators[name] = pd.Series(
            [value] * len(closes), index=data.index, dtype=float)
    return analyzer


class TestPositionRecommendation(unittest.TestCase):
    def setUp(self):
        self.alerts = AlertSystem()

    def assert_scores(self, result, bull, bear):
        self.assertEqual(result["bull_score"], bull)
        self.assertEqual(result["bear_score"], bear)
        self.assertEqual(result["total_score"], bull - bear)

    def test_no_indicators_yields_hold_without_fabricated_signals(self):
        result = self.alerts.generate_position_recommendation(position_analyzer())
        self.assert_scores(result, 0, 0)
        self.assertEqual(result["recommendation"], "TUT")
        self.assertEqual(result["position_strength"], "NÖTR")
        self.assertEqual(result["recommendation_color"], "#ffa500")
        self.assertEqual(result["confidence"], 0)
        for field in ("bull_signals", "bear_signals", "technical_details", "risk_warnings"):
            self.assertEqual(result[field], [])

    def test_rsi_inclusive_band_edges_and_missing_values(self):
        # Zero truthiness is a separately reported defect, not a golden value.
        cases = [
            (None, 0, 0, None), (float("nan"), 0, 0, None),
            (29.99, 2, 0, "RSI Aşırı Satım"), (30, 2, 0, "RSI Aşırı Satım"),
            (30.01, 1, 0, "RSI Düşük"), (40, 1, 0, "RSI Düşük"),
            (40.01, 0, 0, None), (59.99, 0, 0, None),
            (60, 0, 1, "RSI Yüksek"), (69.99, 0, 1, "RSI Yüksek"),
            (70, 0, 2, "RSI Aşırı Alım"), (70.01, 0, 2, "RSI Aşırı Alım"),
        ]
        for value, bull, bear, signal in cases:
            with self.subTest(rsi=value):
                indicators = {} if value is None else {"rsi": value}
                result = self.alerts.generate_position_recommendation(position_analyzer(indicators))
                self.assert_scores(result, bull, bear)
                self.assertEqual(result["bull_signals"] + result["bear_signals"],
                                 [] if signal is None else [signal])
                if value in (40.01, 59.99):
                    self.assertIn("Nötr bölge", result["technical_details"][0])

    def test_macd_sign_and_line_order_determine_weight_without_crossing(self):
        for macd, signal, bull, bear, label in [
            (2, 1, 2, 0, "MACD Pozitif Kesişim"),
            (-1, -2, 1, 0, "MACD Yükseliş"),
            (-2, -1, 0, 2, "MACD Negatif Kesişim"),
            (1, 2, 0, 1, "MACD Düşüş"),
            (1, 1, 0, 0, None), (-1, -1, 0, 0, None),
            (float("nan"), 1, 0, 0, None), (1, float("nan"), 0, 0, None),
        ]:
            with self.subTest(macd=macd, signal=signal):
                result = self.alerts.generate_position_recommendation(position_analyzer(
                    {"macd": macd, "macd_signal": signal}))
                self.assert_scores(result, bull, bear)
                self.assertEqual(result["bull_signals"] + result["bear_signals"],
                                 [] if label is None else [label])
        for indicators in ({"macd": 2}, {"macd_signal": 1}):
            with self.subTest(missing=indicators):
                self.assert_scores(self.alerts.generate_position_recommendation(
                    position_analyzer(indicators)), 0, 0)

    def test_ema_price_order_weights_and_equalities(self):
        for short, long, bull, bear, label in [
            (90, 80, 2, 0, "Güçlü Yükseliş Trendi"),
            (90, 110, 1, 0, "Kısa Vadeli Yükseliş"),
            (110, 120, 0, 2, "Güçlü Düşüş Trendi"),
            (110, 90, 0, 1, "Kısa Vadeli Düşüş"),
            (90, 90, 1, 0, "Kısa Vadeli Yükseliş"),
            (110, 110, 0, 1, "Kısa Vadeli Düşüş"),
            (100, 90, 0, 0, None), (100, 110, 0, 0, None),
        ]:
            with self.subTest(short=short, long=long):
                result = self.alerts.generate_position_recommendation(position_analyzer(
                    {"ema_21": short, "ema_50": long}))
                self.assert_scores(result, bull, bear)
                self.assertEqual(result["bull_signals"] + result["bear_signals"],
                                 [] if label is None else [label])

    def test_ma200_uses_one_and_half_points_and_includes_equality_in_bear_branch(self):
        for ma, bull, bear, label in [
            (99.99, 1.5, 0, "Uzun Vadeli Yükseliş"),
            (100, 0, 1.5, "Uzun Vadeli Düşüş"),
            (100.01, 0, 1.5, "Uzun Vadeli Düşüş"),
        ]:
            with self.subTest(ma=ma):
                result = self.alerts.generate_position_recommendation(
                    position_analyzer({"ma_200": ma}))
                self.assert_scores(result, bull, bear)
                self.assertEqual(result["bull_signals"] + result["bear_signals"], [label])

    def test_supertrend_and_ott_read_last_direction_without_requiring_a_crossing(self):
        for name, weight, label in [
            ("supertrend_trend", 1.5, "SuperTrend"), ("ott_trend", 1, "OTT"),
        ]:
            for direction, bull, bear, signal in [
                (1, weight, 0, label + " Pozitif"),
                (-1, 0, weight, label + " Negatif"), (0, 0, 0, None),
            ]:
                with self.subTest(name=name, direction=direction):
                    result = self.alerts.generate_position_recommendation(
                        position_analyzer({name: direction}))
                    self.assert_scores(result, bull, bear)
                    self.assertEqual(result["bull_signals"] + result["bear_signals"],
                                     [] if signal is None else [signal])

    def test_bollinger_extremes_are_inclusive_and_middle_has_half_weight(self):
        for price, bull, bear, label in [
            (89.99, 1.5, 0, "Bollinger Alt Bandında"),
            (90, 1.5, 0, "Bollinger Alt Bandında"),
            (90.01, 0, 0.5, None), (100, 0, 0.5, None),
            (100.01, 0.5, 0, None), (109.99, 0.5, 0, None),
            (110, 0, 1.5, "Bollinger Üst Bandında"),
            (110.01, 0, 1.5, "Bollinger Üst Bandında"),
        ]:
            with self.subTest(price=price):
                result = self.alerts.generate_position_recommendation(position_analyzer(
                    {"bb_lower": 90, "bb_middle": 100, "bb_upper": 110},
                    closes=[price] * 21))
                self.assert_scores(result, bull, bear)
                self.assertEqual(result["bull_signals"] + result["bear_signals"],
                                 [] if label is None else [label])

    def test_volume_requires_strict_ratio_and_price_edges_uses_latest_twenty_bars(self):
        # 19*37 + 57 = 760: mean 38, ratio exactly 1.5. The oldest spike is excluded.
        for volume, price, bull, bear, label in [
            (56, 103, 0, 0, None), (57, 103, 0, 0, None),
            (57.05, 103, 1, 0, "Yüksek Hacimle Yükseliş"),
            (58, 103, 1, 0, "Yüksek Hacimle Yükseliş"),
            (58, 97, 0, 1, "Yüksek Hacimle Düşüş"),
            (58, 102, 0, 0, None), (58, 98, 0, 0, None),
            (58, 100, 0, 0, None),
        ]:
            with self.subTest(volume=volume, price=price):
                result = self.alerts.generate_position_recommendation(position_analyzer(
                    closes=[100] * 20 + [price], volumes=[99999] + [37] * 19 + [volume]))
                self.assert_scores(result, bull, bear)
                self.assertEqual(result["bull_signals"] + result["bear_signals"],
                                 [] if label is None else [label])

    def test_vwap_two_percent_edges_are_strict_in_both_directions(self):
        for price, bull, bear, label in [
            (102.01, 1, 0, "VWAP Üstünde"), (102, 0, 0, None),
            (101.99, 0, 0, None), (98.01, 0, 0, None),
            (98, 0, 0, None), (97.99, 0, 1, "VWAP Altında"),
        ]:
            with self.subTest(price=price):
                result = self.alerts.generate_position_recommendation(position_analyzer(
                    {"vwap": 100}, closes=[price] * 21))
                self.assert_scores(result, bull, bear)
                self.assertEqual(result["bull_signals"] + result["bear_signals"],
                                 [] if label is None else [label])

    def test_total_score_decisions_strength_colors_and_confidence_boundaries(self):
        # These literal totals sum independently controlled rule weights.
        cases = [
            ({"bb_lower": 80, "bb_middle": 90, "bb_upper": 120}, 0.5,
             "ZAYIF AL ZAYIF", "ÇOK ZAYIF", "#b8f5b8", 5),
            ({"ott_trend": 1}, 1, "ZAYIF AL", "ZAYIF", "#90ee90", 10),
            ({"ma_200": 90}, 1.5, "ZAYIF AL", "ZAYIF", "#90ee90", 15),
            ({"rsi": 30}, 2, "AL", "ORTA", "#32cd32", 20),
            ({"rsi": 30, "ma_200": 90}, 3.5, "AL", "ORTA", "#32cd32", 35),
            ({"rsi": 30, "macd": 2, "macd_signal": 1}, 4,
             "GÜÇLÜ AL", "GÜÇLÜ", "#00ff00", 40),
            ({"bb_lower": 80, "bb_middle": 110, "bb_upper": 120}, -0.5,
             "TUT", "NÖTR", "#ffa500", 5),
            ({"ott_trend": -1}, -1, "ZAYIF SAT", "ZAYIF", "#ff6347", 10),
            ({"ma_200": 110}, -1.5, "ZAYIF SAT", "ZAYIF", "#ff6347", 15),
            ({"rsi": 70}, -2, "SAT", "ORTA", "#ff4500", 20),
            ({"rsi": 70, "ma_200": 110}, -3.5, "SAT", "ORTA", "#ff4500", 35),
            ({"rsi": 70, "macd": -2, "macd_signal": -1}, -4,
             "GÜÇLÜ SAT", "GÜÇLÜ", "#ff0000", 40),
        ]
        for indicators, total, recommendation, strength, color, confidence in cases:
            with self.subTest(total=total):
                result = self.alerts.generate_position_recommendation(position_analyzer(indicators))
                self.assertEqual(result["total_score"], total)
                self.assertEqual(result["recommendation"], recommendation)
                self.assertEqual(result["position_strength"], strength)
                self.assertEqual(result["recommendation_color"], color)
                self.assertEqual(result["confidence"], confidence)

    def test_full_directional_totals_and_confidence_cap(self):
        for indicators, bull, bear, recommendation in [
            ({"rsi": 30, "macd": 2, "macd_signal": 1, "ema_21": 90,
              "ema_50": 80, "ma_200": 80, "supertrend_trend": 1, "ott_trend": 1,
              "bb_lower": 100, "bb_middle": 110, "bb_upper": 120, "vwap": 90},
             12.5, 0, "GÜÇLÜ AL"),
            ({"rsi": 70, "macd": -2, "macd_signal": -1, "ema_21": 110,
              "ema_50": 120, "ma_200": 120, "supertrend_trend": -1, "ott_trend": -1,
              "bb_lower": 80, "bb_middle": 90, "bb_upper": 100, "vwap": 110},
             0, 12.5, "GÜÇLÜ SAT"),
        ]:
            with self.subTest(recommendation=recommendation):
                result = self.alerts.generate_position_recommendation(position_analyzer(indicators))
                self.assert_scores(result, bull, bear)
                self.assertEqual(result["recommendation"], recommendation)
                self.assertEqual(result["confidence"], 100)
                signals = result["bull_signals"] + result["bear_signals"]
                self.assertEqual(len(signals), 8)
                self.assertEqual(len(result["technical_details"]), 8)

    def test_valid_sizing_branches(self):
        # Strong-SAT sizing is reproduced separately, not endorsed by this test.
        for indicators, size in [
            ({}, "Mevcut pozisyonu koru"),
            ({"rsi": 30}, "Portföyün %50-75'i"),
            ({"rsi": 70}, "Pozisyonun %75-100'ünü sat"),
            ({"ott_trend": 1}, "Portföyün %25-50'si"),
            ({"ott_trend": -1}, "Portföyün %25-50'si"),
            ({"bb_lower": 80, "bb_middle": 90, "bb_upper": 120}, "Portföyün %10-25'i"),
            ({"rsi": 30, "macd": 2, "macd_signal": 1}, "Portföyün %75-100'ü"),
        ]:
            with self.subTest(indicators=indicators):
                result = self.alerts.generate_position_recommendation(position_analyzer(indicators))
                self.assertEqual(result["position_size"], size)

    def test_high_bear_and_mixed_warning_thresholds_use_gross_scores(self):
        high = "⚠️ Yüksek düşüş riski - Dikkatli olun"
        mixed = "⚠️ Karışık sinyaller - Aşamalı pozisyon alın"
        for indicators, bull, bear, warnings in [
            ({"macd": -2, "macd_signal": -1, "ma_200": 110,
              "supertrend_trend": -1}, 0, 5, []),
            ({"macd": -2, "macd_signal": -1, "ma_200": 110,
              "supertrend_trend": -1, "ott_trend": -1}, 0, 6, [high]),
            ({"rsi": 30, "macd": 2, "macd_signal": 1, "ema_21": 90,
              "ema_50": 80, "ma_200": 110, "supertrend_trend": -1}, 6, 3, []),
            ({"rsi": 30, "macd": -1, "macd_signal": -2, "ema_21": 90,
              "ema_50": 80, "ma_200": 110, "supertrend_trend": -1,
              "ott_trend": -1}, 5, 4, []),
            ({"rsi": 30, "macd": 2, "macd_signal": 1, "ema_21": 90,
              "ema_50": 80, "ma_200": 110, "supertrend_trend": -1,
              "ott_trend": -1}, 6, 4, [mixed]),
            ({"rsi": 30, "macd": 2, "macd_signal": 1, "ema_21": 90,
              "ema_50": 80, "ma_200": 110, "supertrend_trend": -1,
              "ott_trend": -1, "bb_lower": 80, "bb_middle": 90,
              "bb_upper": 100}, 6, 5.5, [high, mixed]),
        ]:
            with self.subTest(bull=bull, bear=bear):
                result = self.alerts.generate_position_recommendation(position_analyzer(indicators))
                self.assert_scores(result, bull, bear)
                self.assertEqual(result["risk_warnings"], warnings)

    def test_low_volume_warning_is_strict_and_does_not_change_scores(self):
        # 19*39 + 19 = 760: mean 38, last ratio exactly 0.5.
        for volume, warnings in [(18, ["⚠️ Düşük hacim - Breakout bekleyin"]),
                                 (19, []), (20, [])]:
            with self.subTest(volume=volume):
                result = self.alerts.generate_position_recommendation(position_analyzer(
                    {"rsi": 30}, volumes=[99999] + [39] * 19 + [volume]))
                self.assert_scores(result, 2, 0)
                self.assertEqual(result["risk_warnings"], warnings)

    def test_preserves_data_indicator_series_and_independent_return_lists(self):
        analyzer = position_analyzer({"rsi": 30, "supertrend_trend": 1})
        analyzer.indicators["rsi"].iloc[0] = 70
        source = analyzer.data.copy(deep=True)
        indicators = {name: series.copy(deep=True) for name, series in analyzer.indicators.items()}
        first = self.alerts.generate_position_recommendation(analyzer)
        second = self.alerts.generate_position_recommendation(analyzer)
        self.assert_scores(first, 3.5, 0)
        self.assertEqual(first, second)
        for key in ("bull_signals", "bear_signals", "technical_details", "risk_warnings"):
            self.assertIsNot(first[key], second[key])
            first[key].append("caller mutation")
            self.assertNotIn("caller mutation", second[key])
        pd.testing.assert_frame_equal(analyzer.data, source)
        self.assertEqual(set(analyzer.indicators), set(indicators))
        for name, series in indicators.items():
            pd.testing.assert_series_equal(analyzer.indicators[name], series)
        self.assertEqual(analyzer.signals, {})
        self.assertEqual(self.alerts.alert_history, [])
        self.assertEqual(self.alerts.last_alerts, {})

    def test_real_computed_indicator_chain_reaches_position_recommendation(self):
        # Linear prices 100..339: RSI=100, SMA200=239.5, cumulative VWAP=219.5.
        # Last BB20 mean=329.5, population variance=33.25, so price is between
        # middle and upper. Positive MACD/ordered EMAs/ST contribute 2+2+1.5;
        # MA200/BB/VWAP add 1.5+0.5+1, RSI subtracts 2: total 6.5.
        analyzer = position_analyzer(closes=list(range(100, 340)))
        original = analyzer.data.copy(deep=True)
        for name in ("rsi", "macd", "ema_21", "ema_50", "ma_200",
                     "bollinger", "supertrend", "vwap"):
            analyzer.add_indicator(name)
        self.assertEqual(analyzer.indicators["rsi"].iloc[-1], 100)
        self.assertEqual(analyzer.indicators["ma_200"].iloc[-1], 239.5)
        self.assertEqual(analyzer.indicators["vwap"].iloc[-1], 219.5)
        result = self.alerts.generate_position_recommendation(analyzer)
        self.assert_scores(result, 8.5, 2)
        self.assertEqual(result["recommendation"], "GÜÇLÜ AL")
        self.assertEqual(result["confidence"], 65)
        self.assertEqual(result["bull_signals"], [
            "MACD Pozitif Kesişim", "Güçlü Yükseliş Trendi", "Uzun Vadeli Yükseliş",
            "SuperTrend Pozitif", "VWAP Üstünde",
        ])
        self.assertEqual(result["bear_signals"], ["RSI Aşırı Alım"])
        self.assertEqual(result["risk_warnings"], [])
        pd.testing.assert_frame_equal(analyzer.data, original)


if __name__ == "__main__":
    unittest.main()
