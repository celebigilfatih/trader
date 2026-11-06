"""Real momentum calculations with hand-derived OHLCV oracles, offline."""

import unittest
from unittest.mock import patch

import pandas as pd

from modules.config import INDICATORS_CONFIG
from modules.technical_analysis import TechnicalAnalyzer


def bars(rows):
    """Rows are explicit (high, low, close), including asymmetric candles."""
    return pd.DataFrame(
        {
            "Open": [(high + low) / 2 for high, low, _ in rows],
            "High": [high for high, _, _ in rows],
            "Low": [low for _, low, _ in rows],
            "Close": [close for _, _, close in rows],
            "Volume": [100 + i for i in range(len(rows))],
        },
        index=pd.date_range(
            "2026-01-01", periods=len(rows), freq="2h", tz="UTC", name="bar"
        ),
    )


def range_expiry_bars():
    # At position13 the 200/0 outlier is still in the 14-bar window;
    # at position14 it leaves. Close extrema differ from high/low extrema.
    return bars([(200, 0, 100)] + [(120, 80, 100)] * 12
                + [(120, 80, close) for close in (80, 100, 120, 90, 110)])


def cci_impulse_bars():
    # TP=(H+L+C)/3: nineteen 10s, then12,8, then twenty 10s.
    # The first impulse closes at8, so substituting Close for TP reverses it.
    return bars([(12, 8, 10)] * 19 + [(20, 8, 8), (12, 4, 8)]
                + [(12, 8, 10)] * 20)


class TestMomentumIndicators(unittest.TestCase):
    def setUp(self):
        # Only IO is replaced: analyzer, pandas and ta remain real.
        for target in ("socket.socket.connect", "socket.socket.connect_ex",
                       "socket.create_connection"):
            self.enterContext(patch(target, side_effect=AssertionError("Network forbidden")))

    def test_stochastic_uses_high_low_and_expires_the_old_extremes(self):
        analyzer = TechnicalAnalyzer(range_expiry_bars())
        analyzer.add_indicator("stoch")
        for position, expected in ((13, 40), (14, 50), (15, 100), (16, 25), (17, 75)):
            with self.subTest(position=position):
                self.assertAlmostEqual(analyzer.indicators["stoch_k"].iloc[position], expected)
        self.assertEqual(analyzer.get_latest_indicators()["stoch_k"], 75)

    def test_stochastic_d_is_the_trailing_three_k_mean(self):
        analyzer = TechnicalAnalyzer(range_expiry_bars())
        analyzer.add_indicator("stoch")
        # K=40,50,100,25,75; D must drop40, then50, rather than accumulate.
        for position, expected in ((15, 190 / 3), (16, 175 / 3), (17, 200 / 3)):
            with self.subTest(position=position):
                self.assertAlmostEqual(analyzer.indicators["stoch_d"].iloc[position], expected)
        latest = analyzer.get_latest_indicators()
        self.assertIn("stoch_d", latest)
        self.assertAlmostEqual(latest["stoch_d"], 200 / 3)

    def test_stochastic_waits_for_fourteen_bars_then_three_valid_k_values(self):
        for count, available in ((0, set()), (1, set()), (13, set()),
                                 (14, {"stoch_k"}), (15, {"stoch_k"}),
                                 (16, {"stoch_k", "stoch_d"})):
            with self.subTest(count=count):
                analyzer = TechnicalAnalyzer(range_expiry_bars().iloc[:count])
                analyzer.add_indicator("stoch")
                self.assertEqual(set(analyzer.get_latest_indicators()), available)
                for key, first in (("stoch_k", 13), ("stoch_d", 15)):
                    series = analyzer.indicators[key]
                    self.assertTrue(series.iloc[:first].isna().all())
                    self.assertTrue(series.iloc[first:].notna().all())

    def test_stochastic_keeps_zero_and_both_range_endpoints(self):
        for close, expected in ((80, 0), (120, 100), (100, 50)):
            with self.subTest(close=close):
                analyzer = TechnicalAnalyzer(bars([(120, 80, 100)] * 13 + [(120, 80, close)]))
                analyzer.add_indicator("stoch")
                self.assertEqual(analyzer.get_latest_indicators(), {"stoch_k": expected})

    def test_williams_uses_high_low_and_expires_the_old_extremes(self):
        analyzer = TechnicalAnalyzer(range_expiry_bars())
        analyzer.add_indicator("williams_r")
        for position, expected in ((13, -60), (14, -50), (15, 0), (16, -75), (17, -25)):
            with self.subTest(position=position):
                self.assertAlmostEqual(analyzer.indicators["williams_r"].iloc[position], expected)
        self.assertEqual(analyzer.get_latest_indicators(), {"williams_r": -25})

    def test_williams_keeps_zero_and_negative_range_endpoints(self):
        for close, expected in ((80, -100), (120, 0), (100, -50)):
            with self.subTest(close=close):
                analyzer = TechnicalAnalyzer(bars([(120, 80, 100)] * 13 + [(120, 80, close)]))
                analyzer.add_indicator("williams_r")
                self.assertEqual(analyzer.get_latest_indicators(), {"williams_r": expected})

    def test_williams_requires_fourteen_bars_without_filling_warmup(self):
        for count in (0, 1, 13, 14):
            with self.subTest(count=count):
                analyzer = TechnicalAnalyzer(range_expiry_bars().iloc[:count])
                analyzer.add_indicator("williams_r")
                self.assertEqual(analyzer.get_latest_indicators(),
                                 {"williams_r": -60} if count == 14 else {})
                self.assertTrue(analyzer.indicators["williams_r"].iloc[:13].isna().all())

    def test_cci_uses_typical_price_mean_absolute_deviation_and_scale(self):
        analyzer = TechnicalAnalyzer(cci_impulse_bars())
        analyzer.add_indicator("cci")
        # At19: mean10.1, MAD0.19, numerator1.9 => 1.9/(.015*.19).
        # At20: mean10, MAD0.2, numerator-2; at21 numerator0 is valid.
        for position, expected in ((19, 2000 / 3), (20, -2000 / 3), (21, 0)):
            with self.subTest(position=position):
                self.assertAlmostEqual(analyzer.indicators["cci"].iloc[position], expected)
        prefix = TechnicalAnalyzer(cci_impulse_bars().iloc[:22])
        prefix.add_indicator("cci")
        self.assertEqual(prefix.get_latest_indicators(), {"cci": 0})

    def test_cci_expires_each_impulse_after_twenty_bars(self):
        analyzer = TechnicalAnalyzer(cci_impulse_bars())
        analyzer.add_indicator("cci")
        # At38 both impulses remain, mean10. At39 only8 remains:
        # mean9.9, MAD.19, numerator.1 => 2000/57. At40 MAD is zero.
        self.assertEqual(analyzer.indicators["cci"].iloc[38], 0)
        self.assertAlmostEqual(analyzer.indicators["cci"].iloc[39], 2000 / 57)
        self.assertTrue(pd.isna(analyzer.indicators["cci"].iloc[40]))
        self.assertEqual(analyzer.get_latest_indicators(), {})

    def test_cci_requires_twenty_bars_without_filling_warmup(self):
        for count in (0, 1, 19, 20):
            with self.subTest(count=count):
                analyzer = TechnicalAnalyzer(cci_impulse_bars().iloc[:count])
                analyzer.add_indicator("cci")
                self.assertTrue(analyzer.indicators["cci"].iloc[:19].isna().all())
                if count == 20:
                    latest = analyzer.get_latest_indicators()
                    self.assertIn("cci", latest)
                    self.assertAlmostEqual(latest["cci"], 2000 / 3)
                else:
                    self.assertEqual(analyzer.get_latest_indicators(), {})

    def test_flat_undefined_windows_do_not_fabricate_finite_indicators(self):
        for name in ("stoch", "williams_r", "cci"):
            with self.subTest(name=name):
                analyzer = TechnicalAnalyzer(bars([(10, 10, 10)] * 22))
                analyzer.add_indicator(name)
                self.assertTrue(all(series.isna().all() for series in analyzer.indicators.values()))
                self.assertEqual(analyzer.get_latest_indicators(), {})

    def test_indicators_recover_after_a_flat_window_without_filling_earlier_values(self):
        data = bars([(10, 10, 10)] * 20 + [(12, 8, 11), (10, 10, 10), (10, 10, 10)])
        analyzer = TechnicalAnalyzer(data)
        for name in ("stoch", "williams_r", "cci"):
            analyzer.add_indicator(name)
        for key, expected in (("stoch_k", 75), ("williams_r", -25), ("cci", 2000 / 3)):
            with self.subTest(key=key):
                self.assertTrue(analyzer.indicators[key].iloc[:20].isna().all())
                self.assertAlmostEqual(analyzer.indicators[key].iloc[20], expected)
        self.assertTrue(analyzer.indicators["stoch_d"].iloc[:22].isna().all())
        latest = analyzer.get_latest_indicators()
        self.assertIn("stoch_d", latest)
        self.assertAlmostEqual(latest["stoch_d"], 175 / 3)

    def test_configured_stochastic_periods_drive_k_and_d(self):
        data = bars([(20, 0, 10), (12, 4, 8), (14, 6, 10), (16, 8, 12), (18, 10, 14)])
        with patch.dict(INDICATORS_CONFIG["stoch"], {"k_period": 3, "d_period": 2}):
            analyzer = TechnicalAnalyzer(data)
            analyzer.add_indicator("stoch")
        for position, expected in ((2, 50), (3, 200 / 3), (4, 200 / 3)):
            self.assertAlmostEqual(analyzer.indicators["stoch_k"].iloc[position], expected)
        self.assertTrue(analyzer.indicators["stoch_k"].iloc[:2].isna().all())
        self.assertTrue(analyzer.indicators["stoch_d"].iloc[:3].isna().all())
        self.assertAlmostEqual(analyzer.indicators["stoch_d"].iloc[3], 175 / 3)
        self.assertAlmostEqual(analyzer.get_latest_indicators()["stoch_d"], 200 / 3)

    def test_configured_williams_period_drives_extreme_expiry(self):
        data = bars([(20, 0, 10), (12, 4, 8), (14, 6, 10), (16, 8, 12), (18, 10, 14)])
        with patch.dict(INDICATORS_CONFIG["williams_r"], {"period": 3}):
            analyzer = TechnicalAnalyzer(data)
            analyzer.add_indicator("williams_r")
        self.assertTrue(analyzer.indicators["williams_r"].iloc[:2].isna().all())
        for position, expected in ((2, -50), (3, -100 / 3), (4, -100 / 3)):
            self.assertAlmostEqual(analyzer.indicators["williams_r"].iloc[position], expected)

    def test_configured_cci_period_controls_typical_price_window(self):
        data = bars([(20, 0, 10), (12, 4, 8), (14, 6, 10), (16, 8, 12), (18, 10, 14)])
        with patch.dict(INDICATORS_CONFIG["cci"], {"period": 3}):
            analyzer = TechnicalAnalyzer(data)
            analyzer.add_indicator("cci")
        self.assertTrue(analyzer.indicators["cci"].iloc[:2].isna().all())
        for position, expected in ((2, 50), (3, 100), (4, 100)):
            self.assertAlmostEqual(analyzer.indicators["cci"].iloc[position], expected)

    def test_momentum_calculations_preserve_input_and_index(self):
        data = cci_impulse_bars()
        original = data.copy(deep=True)
        analyzer = TechnicalAnalyzer(data)
        for name in ("stoch", "williams_r", "cci"):
            analyzer.add_indicator(name)
        pd.testing.assert_frame_equal(data, original)
        pd.testing.assert_frame_equal(analyzer.data, original)
        for series in analyzer.indicators.values():
            pd.testing.assert_index_equal(series.index, data.index)

    def test_later_bars_do_not_rewrite_momentum_history(self):
        data = cci_impulse_bars()
        prefix = TechnicalAnalyzer(data.iloc[:22])
        full = TechnicalAnalyzer(data)
        for name in ("stoch", "williams_r", "cci"):
            prefix.add_indicator(name)
            full.add_indicator(name)
        for key, series in prefix.indicators.items():
            with self.subTest(key=key):
                pd.testing.assert_series_equal(series, full.indicators[key].iloc[:22])


if __name__ == "__main__":
    unittest.main()
