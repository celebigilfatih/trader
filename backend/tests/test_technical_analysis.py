"""Offline regression checks for real calculations, not analyzer doubles."""

import unittest

import pandas as pd

from modules.technical_analysis import TechnicalAnalyzer


def ohlcv(closes, volumes=None):
    """Synthetic daily prices; no user data, randomness or provider calls."""
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [value + 1 for value in closes],
            "Low": [value - 1 for value in closes],
            "Close": closes,
            "Volume": volumes if volumes is not None else [1000] * len(closes),
        },
        index=pd.date_range("2026-01-01", periods=len(closes), freq="D"),
    )


class TestTechnicalAnalysis(unittest.TestCase):
    def test_missing_ohlcv_column_is_rejected(self):
        for column in ("Open", "High", "Low", "Close", "Volume"):
            with self.subTest(column=column):
                with self.assertRaises(ValueError):
                    TechnicalAnalyzer(ohlcv([10, 20]).drop(columns=column))

    def test_unknown_indicator_is_rejected(self):
        analyzer = TechnicalAnalyzer(ohlcv([10, 20]))
        with self.assertRaises(ValueError):
            analyzer.add_indicator("unknown")
        self.assertEqual(analyzer.indicators, {})

    def test_ema_uses_weighted_history(self):
        analyzer = TechnicalAnalyzer(ohlcv([10, 20, 30]))
        analyzer.add_indicator("ema_5")
        # span=5: weights 4/9, 2/3, 1 => weighted mean 430/19.
        self.assertAlmostEqual(analyzer.get_latest_indicators()["ema_5"], 430 / 19)

    def test_sma_excludes_prices_outside_its_window(self):
        analyzer = TechnicalAnalyzer(ohlcv(list(range(1, 202))))
        analyzer.add_indicator("ma_200")
        self.assertTrue(analyzer.indicators["ma_200"].iloc[:199].isna().all())
        self.assertEqual(analyzer.indicators["ma_200"].iloc[199], 100.5)
        self.assertEqual(analyzer.get_latest_indicators()["ma_200"], 101.5)

    def test_vwma_weights_price_by_volume(self):
        analyzer = TechnicalAnalyzer(ohlcv([10, 20, 30, 40, 50], [1, 1, 1, 1, 6]))
        analyzer.add_indicator("vwma_5")
        self.assertEqual(analyzer.get_latest_indicators()["vwma_5"], 40.0)

    def test_rsi_distinguishes_rising_and_falling_prices(self):
        for closes, expected in ((list(range(100, 140)), 100.0),
                                 (list(range(140, 100, -1)), 0.0)):
            with self.subTest(expected=expected):
                analyzer = TechnicalAnalyzer(ohlcv(closes))
                analyzer.add_indicator("rsi")
                self.assertEqual(analyzer.get_latest_indicators()["rsi"], expected)

    def test_rsi_smooths_both_gains_and_losses(self):
        analyzer = TechnicalAnalyzer(ohlcv([10] * 21 + [12, 11]))
        analyzer.add_indicator("rsi")
        # Configured period=22: gain/loss ratio (2*21/484)/(1/22) = 21/11.
        self.assertAlmostEqual(analyzer.get_latest_indicators()["rsi"], 65.625)

    def test_rsi_requires_its_warmup_window(self):
        for count, available in ((21, False), (22, True)):
            with self.subTest(count=count):
                analyzer = TechnicalAnalyzer(ohlcv(list(range(100, 100 + count))))
                analyzer.add_indicator("rsi")
                self.assertEqual("rsi" in analyzer.get_latest_indicators(), available)

    def test_macd_is_zero_for_constant_prices(self):
        analyzer = TechnicalAnalyzer(ohlcv([120.0] * 40))
        analyzer.add_indicator("macd")
        latest = analyzer.get_latest_indicators()
        for name in ("macd", "macd_signal", "macd_histogram"):
            self.assertEqual(latest[name], 0.0)

    def test_bollinger_bands_converge_for_constant_prices(self):
        analyzer = TechnicalAnalyzer(ohlcv([120.0] * 40))
        analyzer.add_indicator("bollinger")
        latest = analyzer.get_latest_indicators()
        for name in ("bb_upper", "bb_middle", "bb_lower"):
            self.assertEqual(latest[name], 120.0)

    def test_support_resistance_uses_only_requested_lookback(self):
        analyzer = TechnicalAnalyzer(ohlcv([100, 10, 20, 30]))
        self.assertEqual(analyzer.calculate_support_resistance(lookback=2), (19, 31))

    def test_calculations_do_not_mutate_input(self):
        data = ohlcv(list(range(100, 140)))
        original = data.copy(deep=True)
        analyzer = TechnicalAnalyzer(data)
        for name in ("ema_21", "rsi", "macd", "vwema_5"):
            analyzer.add_indicator(name)
        pd.testing.assert_frame_equal(data, original)


if __name__ == "__main__":
    unittest.main()
