"""Offline regression checks for real calculations, not analyzer doubles."""

import unittest
from math import sqrt

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

    def test_macd_tracks_impulse_and_reversion_with_twelve_and_twenty_six_periods(self):
        # Independent recursive EMA oracle: 25 closes at 100, then a +/-26 shock.
        # At shock+k: M = 4*(11/13)**k - (52/27)*(25/27)**k.
        # Literal values below were derived with exact rational arithmetic,
        # without TechnicalAnalyzer, ta, pandas ewm or production config reads.
        expected = {
            25: 56 / 27,
            26: 1.60135063838768,
            33: 0.0106078469903389,
            34: -0.0740284025344887,
            35: -0.139495225922493,
        }
        for direction in (1, -1):
            data = ohlcv([100] * 25 + [100 + 26 * direction] + [100] * 10)
            analyzer = TechnicalAnalyzer(data)
            analyzer.add_indicator("macd")
            line = analyzer.indicators["macd"]
            pd.testing.assert_index_equal(line.index, data.index)
            for position, value in expected.items():
                with self.subTest(direction=direction, position=position):
                    self.assertAlmostEqual(line.iloc[position], direction * value, places=11)

    def test_macd_signal_and_histogram_smooth_the_reversing_impulse(self):
        # For k>=1, S[k]=4/5*S[k-1]+M[k]/5, seeded with S[0]=56/27.
        # Signal is exposed only after 9 valid MACD values. Histogram is M-S.
        expected = {
            33: (0.686439486124713, -0.675831639134374),
            34: (0.534345908392872, -0.608374310927361),
            35: (0.399577681529799, -0.539072907452293),
        }
        for direction in (1, -1):
            data = ohlcv([100] * 25 + [100 + 26 * direction] + [100] * 10)
            analyzer = TechnicalAnalyzer(data)
            analyzer.add_indicator("macd")
            for name, offset in (("macd_signal", 0), ("macd_histogram", 1)):
                series = analyzer.indicators[name]
                pd.testing.assert_index_equal(series.index, data.index)
                for position, values in expected.items():
                    with self.subTest(direction=direction, name=name, position=position):
                        self.assertAlmostEqual(series.iloc[position], direction * values[offset], places=11)

    def test_macd_exposes_line_then_signal_only_after_their_warmups(self):
        for count, available in (
            (25, set()),
            (26, {"macd"}),
            (33, {"macd"}),
            (34, {"macd", "macd_signal", "macd_histogram"}),
        ):
            with self.subTest(count=count):
                analyzer = TechnicalAnalyzer(ohlcv(list(range(100, 100 + count))))
                analyzer.add_indicator("macd")
                self.assertEqual(set(analyzer.get_latest_indicators()), available)
                for name, first_position in (("macd", 25), ("macd_signal", 33), ("macd_histogram", 33)):
                    series = analyzer.indicators[name]
                    self.assertTrue(series.iloc[:first_position].isna().all())
                    self.assertTrue(series.iloc[first_position:].notna().all())

    def test_bollinger_bands_converge_for_constant_prices(self):
        analyzer = TechnicalAnalyzer(ohlcv([120.0] * 40))
        analyzer.add_indicator("bollinger")
        latest = analyzer.get_latest_indicators()
        for name in ("bb_upper", "bb_middle", "bb_lower"):
            self.assertEqual(latest[name], 120.0)

    def test_bollinger_uses_last_twenty_prices_and_population_spread(self):
        data = ohlcv([1000] + list(range(1, 22)))
        analyzer = TechnicalAnalyzer(data)
        analyzer.add_indicator("bollinger")
        # Position 19: mean=1190/20=59.5; population variance=46583.25.
        # Position 20: closes 1..20, mean=10.5; variance=33.25.
        # Position 21: closes 2..21, mean=11.5; variance remains 33.25.
        for position, middle, half_width in ((19, 59.5, sqrt(186333)),
                                             (20, 10.5, sqrt(133)),
                                             (21, 11.5, sqrt(133))):
            for name, value in (("bb_upper", middle + half_width),
                                ("bb_middle", middle),
                                ("bb_lower", middle - half_width)):
                with self.subTest(position=position, name=name):
                    series = analyzer.indicators[name]
                    pd.testing.assert_index_equal(series.index, data.index)
                    self.assertAlmostEqual(series.iloc[position], value, places=9)

    def test_bollinger_requires_twenty_prices_without_filling_earlier_bands(self):
        for count in (19, 20):
            with self.subTest(count=count):
                analyzer = TechnicalAnalyzer(ohlcv(list(range(1, count + 1))))
                analyzer.add_indicator("bollinger")
                expected = {"bb_upper", "bb_middle", "bb_lower"} if count == 20 else set()
                self.assertEqual(set(analyzer.get_latest_indicators()), expected)
                for series in analyzer.indicators.values():
                    self.assertTrue(series.iloc[:19].isna().all())
                    if count == 20:
                        self.assertFalse(pd.isna(series.iloc[19]))

    def test_support_resistance_uses_only_requested_lookback(self):
        analyzer = TechnicalAnalyzer(ohlcv([100, 10, 20, 30]))
        self.assertEqual(analyzer.calculate_support_resistance(lookback=2), (19, 31))

    def test_calculations_do_not_mutate_input(self):
        data = ohlcv(list(range(100, 140)))
        original = data.copy(deep=True)
        analyzer = TechnicalAnalyzer(data)
        for name in ("ema_21", "rsi", "macd", "bollinger", "vwema_5"):
            analyzer.add_indicator(name)
        pd.testing.assert_frame_equal(data, original)


if __name__ == "__main__":
    unittest.main()
