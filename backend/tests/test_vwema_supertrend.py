"""Hand-derived valid-subset calculations, not canonical strategy validation."""

import unittest
from unittest.mock import patch

import pandas as pd

from backend.tests.test_momentum_indicators import bars
from backend.tests.test_technical_analysis import ohlcv
from modules.alert_system import AlertSystem
from modules.config import INDICATORS_CONFIG
from modules.technical_analysis import TechnicalAnalyzer


def trend_bars(seed_count=10):
    """Asymmetric candles distinguish HL2 from Close; gaps dominate true range."""
    return bars([(101, 99, 100)] * seed_count
                + [(114, 108, 110), (94, 88, 90), (124, 118, 120), (104, 98, 100)])


class TestVwemaSupertrend(unittest.TestCase):
    def setUp(self):
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            self.enterContext(patch(target, side_effect=AssertionError("Network forbidden")))

    def test_vwema5_smooths_changes_after_a_constant_initial_window(self):
        analyzer = TechnicalAnalyzer(ohlcv([100] * 5 + [130, 80, 160], [1] * 8))
        analyzer.add_indicator("vwema_5")
        # alpha=1/3: (130+2*100)/3=110, then100, then120.
        for position, expected in ((5, 110), (6, 100), (7, 120)):
            with self.subTest(position=position):
                self.assertAlmostEqual(analyzer.indicators["vwema_5"].iloc[position], expected)
        self.assertEqual(analyzer.get_latest_indicators(), {"vwema_5": 120})

    def test_vwema20_uses_its_own_smoothing_period(self):
        analyzer = TechnicalAnalyzer(ohlcv([100] * 20 + [121, 81, 142], [1] * 23))
        analyzer.add_indicator("vwema_20")
        # alpha=2/21: (242+1900)/21=102, then100, then104.
        for position, expected in ((20, 102), (21, 100), (22, 104)):
            with self.subTest(position=position):
                self.assertAlmostEqual(analyzer.indicators["vwema_20"].iloc[position], expected)
        self.assertAlmostEqual(analyzer.get_latest_indicators()["vwema_20"], 104)

    def test_zero_volume_carries_the_last_value_instead_of_using_the_price_jump(self):
        for name, count, tail, expected in (
            ("vwema_5", 5, [130, 1000], [110, 110]),
            ("vwema_20", 20, [121, 1000], [102, 102]),
        ):
            with self.subTest(name=name):
                analyzer = TechnicalAnalyzer(ohlcv([100] * count + tail, [1] * count + [1, 0]))
                analyzer.add_indicator(name)
                for position, want in enumerate(expected, count):
                    self.assertAlmostEqual(analyzer.indicators[name].iloc[position], want)

    def test_vwema_consumes_the_configured_period(self):
        with patch.dict(INDICATORS_CONFIG["vwema_5"], {"period": 2}):
            analyzer = TechnicalAnalyzer(ohlcv([100, 100, 130, 90, 145], [1] * 5))
            analyzer.add_indicator("vwema_5")
        for position, expected in ((2, 120), (3, 100), (4, 130)):
            self.assertAlmostEqual(analyzer.indicators["vwema_5"].iloc[position], expected)

    def test_calculations_preserve_caller_data_and_datetime_index(self):
        data = trend_bars()
        original = data.copy(deep=True)
        analyzer = TechnicalAnalyzer(data)
        for name in ("vwema_5", "vwema_20", "supertrend"):
            analyzer.add_indicator(name)
        pd.testing.assert_frame_equal(data, original)
        pd.testing.assert_frame_equal(analyzer.data, original)
        for series in analyzer.indicators.values():
            pd.testing.assert_index_equal(series.index, original.index)

    def test_vwema_history_after_a_shared_constant_seed_ignores_later_bars(self):
        # This deliberately does not assert causality across the known seed bug.
        prefix = TechnicalAnalyzer(ohlcv([100] * 20 + [121, 81, 142], [1] * 23))
        full = TechnicalAnalyzer(ohlcv([100] * 20 + [121, 81, 142, 10000], [1] * 24))
        for name in ("vwema_5", "vwema_20"):
            with self.subTest(name=name):
                prefix.add_indicator(name)
                full.add_indicator(name)
                pd.testing.assert_series_equal(prefix.indicators[name], full.indicators[name].iloc[:23])

    def test_supertrend_uses_gap_atr_hl2_and_both_directions(self):
        analyzer = TechnicalAnalyzer(trend_bars())
        analyzer.add_indicator("supertrend")
        # TR tail14,22,34,22. Wilder ATR3.2,5.08,7.972,9.3748.
        # HL2 tail111,91,121,101; current basic lower/upper band is selected.
        for position, price, direction in (
            (10, 101.4, 1), (11, 106.24, -1), (12, 97.084, 1), (13, 72.8756, 1),
        ):
            with self.subTest(position=position):
                self.assertAlmostEqual(analyzer.indicators["supertrend"].iloc[position], price, places=9)
                self.assertEqual(analyzer.indicators["supertrend_trend"].iloc[position], direction)

    def test_supertrend_exposes_the_seed_only_after_normal_warmup(self):
        analyzer = TechnicalAnalyzer(trend_bars())
        analyzer.add_indicator("supertrend")
        for key in ("supertrend", "supertrend_trend"):
            self.assertTrue(analyzer.indicators[key].iloc[:10].isna().all())
            self.assertTrue(analyzer.indicators[key].iloc[10:].notna().all())
        latest = analyzer.get_latest_indicators()
        self.assertEqual(set(latest), {"supertrend", "supertrend_trend"})
        self.assertAlmostEqual(latest["supertrend"], 72.8756)
        self.assertEqual(latest["supertrend_trend"], 1)

    def test_supertrend_current_equality_takes_the_down_branch(self):
        # Eleven constant bars seed an exactly representable previous band94.
        for close, price, direction in ((93, 100.8, -1), (94, 101.5, -1), (95, 87.8, 1)):
            with self.subTest(close=close):
                data = bars([(101, 99, 100)] * 11 + [(close + 1, close - 1, close)])
                analyzer = TechnicalAnalyzer(data)
                analyzer.add_indicator("supertrend")
                latest = analyzer.get_latest_indicators()
                self.assertEqual(set(latest), {"supertrend", "supertrend_trend"})
                self.assertAlmostEqual(latest["supertrend"], price)
                self.assertEqual(latest["supertrend_trend"], direction)

    def test_supertrend_atr_failure_runs_the_real_manual_rolling_fallback(self):
        # Fault injection only: manual true-range/rolling/bands/directions stay real.
        with patch("modules.technical_analysis.ta.volatility.average_true_range",
                   side_effect=RuntimeError("fixture ATR failure")):
            analyzer = TechnicalAnalyzer(trend_bars())
            analyzer.add_indicator("supertrend")
        # Last ten TR means3.2,5.2,8.4,10.4 differ from Wilder after the first.
        for position, price, direction in (
            (10, 101.4, 1), (11, 106.6, -1), (12, 95.8, 1), (13, 69.8, 1),
        ):
            with self.subTest(position=position):
                self.assertAlmostEqual(analyzer.indicators["supertrend"].iloc[position], price)
                self.assertEqual(analyzer.indicators["supertrend_trend"].iloc[position], direction)

    def test_supertrend_config_controls_atr_window_and_band_multiplier(self):
        with patch.dict(INDICATORS_CONFIG["supertrend"], {"period": 2, "multiplier": 1}):
            analyzer = TechnicalAnalyzer(trend_bars(seed_count=2))
            analyzer.add_indicator("supertrend")
        # ATR8,15,24.5,23.25; seed/direction1,-1,1,1.
        for position, price in ((2, 103), (3, 106), (4, 96.5), (5, 77.75)):
            self.assertAlmostEqual(analyzer.indicators["supertrend"].iloc[position], price)
        self.assertEqual(analyzer.indicators["supertrend_trend"].iloc[2:].tolist(), [1, -1, 1, 1])

    def test_real_supertrend_direction_reaches_the_public_signal_reader(self):
        alerts = AlertSystem()
        for count, expected in ((11, "BEKLE"), (12, "SAT"), (13, "AL"), (14, "BEKLE")):
            with self.subTest(count=count):
                analyzer = TechnicalAnalyzer(trend_bars().iloc[:count])
                analyzer.add_indicator("supertrend")
                self.assertEqual(alerts.generate_signal(analyzer), expected)


if __name__ == "__main__":
    unittest.main()
