"""Offline regressions for zero indicators and directional position sizing."""

import unittest
from unittest.mock import patch

import pandas as pd

from modules.alert_system import AlertSystem
from modules.technical_analysis import TechnicalAnalyzer


def analyzer_with(indicators=None, closes=None):
    """Use real data and accessor; scalar series isolate individual rule inputs."""
    closes = [100.0] * 21 if closes is None else closes
    data = pd.DataFrame({
        "Open": closes,
        "High": [price + 10 for price in closes],
        "Low": [price - 10 for price in closes],
        "Close": closes,
        "Volume": [1000] * len(closes),
    }, index=pd.date_range("2026-01-01", periods=len(closes)))
    analyzer = TechnicalAnalyzer(data)
    analyzer.indicators = {
        name: pd.Series(value, index=data.index, dtype=float)
        for name, value in (indicators or {}).items()
    }
    return analyzer


class TestRiskPositionFixes(unittest.TestCase):
    def setUp(self):
        self.alerts = AlertSystem()

    def test_zero_rsi_keeps_existing_oversold_risk_weight(self):
        result = self.alerts.generate_comprehensive_risk_analysis(
            analyzer_with({"rsi": 0}))
        self.assertEqual(result["risk_score"], 1)
        self.assertIn("rsi_oversold", result["risk_factors"])
        self.assertTrue(any("RSI aşırı satım" in item
                            for item in result["recommendations"]))

    def test_zero_rsi_keeps_existing_oversold_position_weight(self):
        result = self.alerts.generate_position_recommendation(
            analyzer_with({"rsi": 0}))
        self.assertEqual(result["bull_score"], 2)
        self.assertEqual(result["bear_score"], 0)
        self.assertEqual(result["total_score"], 2)
        self.assertEqual(result["bull_signals"], ["RSI Aşırı Satım"])
        self.assertEqual(result["recommendation"], "AL")

    def test_macd_zero_line_and_signal_keep_risk_sign_rules(self):
        for macd, signal, score, factor in [
            (0, 0, 0, "weak_momentum"),
            (0, -.2, 0, "weak_momentum"),
            (0, .2, 0, "weak_momentum"),
            (.2, 0, 0, "strong_positive_momentum"),
            (-.2, 0, 1.5, "strong_negative_momentum"),
            (2, 1, 0, "strong_positive_momentum"),
            (-2, -1, 1.5, "strong_negative_momentum"),
        ]:
            with self.subTest(macd=macd, signal=signal):
                result = self.alerts.generate_comprehensive_risk_analysis(
                    analyzer_with({"macd": macd, "macd_signal": signal}))
                self.assertEqual(result["risk_score"], score)
                self.assertIn(factor, result["risk_factors"])

    def test_macd_zero_line_and_signal_keep_position_sign_weights(self):
        for macd, signal, bull, bear, label in [
            (0, 0, 0, 0, None),
            (0, -1, 1, 0, "MACD Yükseliş"),
            (0, 1, 0, 1, "MACD Düşüş"),
            (1, 0, 2, 0, "MACD Pozitif Kesişim"),
            (-1, 0, 0, 2, "MACD Negatif Kesişim"),
            (-1, -2, 1, 0, "MACD Yükseliş"),
            (1, 2, 0, 1, "MACD Düşüş"),
        ]:
            with self.subTest(macd=macd, signal=signal):
                result = self.alerts.generate_position_recommendation(
                    analyzer_with({"macd": macd, "macd_signal": signal}))
                self.assertEqual(result["bull_score"], bull)
                self.assertEqual(result["bear_score"], bear)
                self.assertEqual(result["total_score"], bull - bear)
                self.assertEqual(result["bull_signals"] + result["bear_signals"],
                                 [] if label is None else [label])
                self.assertEqual(len(result["technical_details"]),
                                 0 if label is None else 1)

    def test_none_and_nan_rsi_do_not_create_risk_or_position_evidence(self):
        # The real accessor filters missing values. Supplying its boundary
        # directly also verifies that each consumer rejects explicit None/NaN.
        for missing in (None, float("nan")):
            with self.subTest(missing=missing):
                analyzer = analyzer_with()
                with patch.object(analyzer, "get_latest_indicators",
                                  return_value={"rsi": missing}):
                    risk = self.alerts.generate_comprehensive_risk_analysis(analyzer)
                    position = self.alerts.generate_position_recommendation(analyzer)
                self.assertEqual(risk["risk_score"], 0)
                self.assertFalse(any(key.startswith("rsi_")
                                     for key in risk["risk_factors"]))
                self.assertEqual(position["total_score"], 0)
                self.assertEqual(position["bull_signals"], [])
                self.assertEqual(position["bear_signals"], [])
                self.assertEqual(position["technical_details"], [])

    def test_missing_macd_line_or_signal_does_not_create_evidence(self):
        for missing in (None, float("nan")):
            for macd, signal in ((missing, 1), (missing, 0),
                                 (1, missing), (0, missing), (missing, missing)):
                with self.subTest(macd=macd, signal=signal):
                    analyzer = analyzer_with()
                    with patch.object(analyzer, "get_latest_indicators",
                                      return_value={"macd": macd, "macd_signal": signal}):
                        risk = self.alerts.generate_comprehensive_risk_analysis(analyzer)
                        position = self.alerts.generate_position_recommendation(analyzer)
                    self.assertEqual(risk["risk_score"], 0)
                    self.assertFalse(any("momentum" in key
                                         for key in risk["risk_factors"]))
                    self.assertEqual(position["total_score"], 0)
                    self.assertEqual(position["bull_signals"], [])
                    self.assertEqual(position["bear_signals"], [])
                    self.assertEqual(position["technical_details"], [])

    def test_real_descending_prices_produce_zero_rsi_in_both_consumers(self):
        # Every price change is -1, so average gain is zero and RSI is zero.
        # Low volatility, normal volume and support ten points away add no risk.
        analyzer = analyzer_with(closes=list(range(340, 100, -1)))
        original = analyzer.data.copy(deep=True)
        analyzer.add_indicator("rsi")
        self.assertEqual(analyzer.indicators["rsi"].iloc[-1], 0)
        risk = self.alerts.generate_comprehensive_risk_analysis(analyzer)
        position = self.alerts.generate_position_recommendation(analyzer)
        self.assertEqual(risk["risk_score"], 1)
        self.assertIn("rsi_oversold", risk["risk_factors"])
        self.assertEqual(position["bull_score"], 2)
        self.assertEqual(position["bear_score"], 0)
        self.assertEqual(position["bull_signals"], ["RSI Aşırı Satım"])
        pd.testing.assert_frame_equal(analyzer.data, original)

    def test_strong_sell_uses_existing_sell_size_and_strong_buy_keeps_buy_size(self):
        for indicators, total, recommendation, size in [
            ({"rsi": 70, "macd": -2, "macd_signal": -1}, -4,
             "GÜÇLÜ SAT", "Pozisyonun %75-100'ünü sat"),
            ({"rsi": 70, "macd": -2, "macd_signal": -1, "ma_200": 110}, -5.5,
             "GÜÇLÜ SAT", "Pozisyonun %75-100'ünü sat"),
            ({"rsi": 30, "macd": 2, "macd_signal": 1}, 4,
             "GÜÇLÜ AL", "Portföyün %75-100'ü"),
        ]:
            with self.subTest(total=total):
                result = self.alerts.generate_position_recommendation(
                    analyzer_with(indicators))
                self.assertEqual(result["total_score"], total)
                self.assertEqual(result["recommendation"], recommendation)
                self.assertEqual(result["position_size"], size)


if __name__ == "__main__":
    unittest.main()
