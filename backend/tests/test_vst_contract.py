"""Accepted ADR-0010 arithmetic/domain/readiness through real calculations."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backend.tests.test_technical_analysis import ohlcv
from modules.alert_system import AlertSystem
from modules.config import INDICATORS_CONFIG
from modules.technical_analysis import TechnicalAnalyzer


class TestVstContract(unittest.TestCase):
    def setUp(self):
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            self.enterContext(patch(target, side_effect=AssertionError("Network forbidden")))

    def calculated(self, data, name):
        analyzer = TechnicalAnalyzer(data)
        try:
            analyzer.add_indicator(name)
        except Exception as exc:
            self.fail(f"Valid {name} calculation raised {type(exc).__name__}: {exc}")
        return analyzer

    def rejected_atomically(self, data, name):
        caller = data.copy(deep=True)
        analyzer = TechnicalAnalyzer(data)
        keys = ("supertrend", "supertrend_trend") if name == "supertrend" else (name,)
        analyzer.indicators = {key: pd.Series(77.0, index=data.index) for key in (*keys, "unrelated")}
        objects = dict(analyzer.indicators)
        snapshots = {key: value.copy(deep=True) for key, value in objects.items()}
        try:
            analyzer.add_indicator(name)
        except Exception as exc:
            self.assertIsInstance(exc, ValueError)
        else:
            self.fail("Invalid input must raise ValueError")
        self.assertEqual(set(analyzer.indicators), set(snapshots))
        for key in snapshots:
            self.assertIs(analyzer.indicators[key], objects[key])
            pd.testing.assert_series_equal(analyzer.indicators[key], snapshots[key])
        pd.testing.assert_frame_equal(data, caller)
        pd.testing.assert_frame_equal(analyzer.data, caller)

    def test_vwema_weighted_recurrence_uses_volume_after_the_first_observation(self):
        with patch.dict(INDICATORS_CONFIG["vwema_5"], {"period": 2}):
            a = self.calculated(ohlcv([100, 200, 50], [1, 3, 2]), "vwema_5")
        # N/D:100/1, (1300/3)/(7/3), (1900/9)/(19/9).
        np.testing.assert_allclose(a.indicators["vwema_5"], [100, 1300 / 7, 100])

    def test_vwema_late_volume_changes_the_weight_not_just_the_seed(self):
        for volume, expected in ((1, 110), (1000, 21700 / 167)):
            with self.subTest(volume=volume):
                a = self.calculated(ohlcv([100] * 5 + [130], [1] * 5 + [volume]), "vwema_5")
                self.assertAlmostEqual(a.get_latest_indicators()["vwema_5"], expected)

    def test_vwema_all_prefixes_stay_unchanged_when_future_bars_are_added(self):
        data = ohlcv([1000, 999, 100, 200, 50, 130], [0, 0, 1, 3, 2, 1])
        for name in ("vwema_5", "vwema_20"):
            full = self.calculated(data, name)
            for count in range(len(data) + 1):
                with self.subTest(name=name, count=count):
                    prefix = self.calculated(data.iloc[:count], name)
                    pd.testing.assert_series_equal(prefix.indicators[name], full.indicators[name].iloc[:count])

    def test_vwema_leading_zero_volume_stays_missing_until_first_trade(self):
        a = self.calculated(ohlcv([1000, 999, 100, 130], [0, 0, 1, 1]), "vwema_5")
        self.assertTrue(a.indicators["vwema_5"].iloc[:2].isna().all())
        np.testing.assert_allclose(a.indicators["vwema_5"].iloc[2:], [100, 110])

    def test_vwema_all_zero_volume_does_not_substitute_close_or_zero(self):
        for name in ("vwema_5", "vwema_20"):
            with self.subTest(name=name):
                a = self.calculated(ohlcv([100, 130, 200], [0, 0, 0]), name)
                self.assertTrue(a.indicators[name].isna().all())
                self.assertEqual(a.get_latest_indicators(), {})

    def test_vwema_zero_gap_decays_mass_before_the_next_positive_bar(self):
        a = self.calculated(ohlcv([100] * 5 + [130, 999, 70], [1] * 6 + [0, 1]), "vwema_5")
        # Zero bar carries110, but mass2/3 makes next q650/7, not290/3.
        np.testing.assert_allclose(a.indicators["vwema_5"].iloc[5:], [110, 110, 650 / 7])

    def test_vwema_period_one_carries_only_zero_volume_bars(self):
        with patch.dict(INDICATORS_CONFIG["vwema_5"], {"period": 1}):
            a = self.calculated(ohlcv([90, 100, 999, 120], [0, 1, 0, 1]), "vwema_5")
        self.assertTrue(pd.isna(a.indicators["vwema_5"].iloc[0]))
        np.testing.assert_allclose(a.indicators["vwema_5"].iloc[1:], [100, 100, 120])

    def test_vwema_zero_mass_underflow_does_not_make_a_nan_recovery(self):
        tiny = np.nextafter(0.0, 1.0)
        with patch.dict(INDICATORS_CONFIG["vwema_5"], {"period": 2}):
            a = self.calculated(ohlcv([100, 999, 130], [tiny, 0, tiny]), "vwema_5")
        np.testing.assert_allclose(a.indicators["vwema_5"], [100, 100, 130])

    def test_vwema_can_weight_finite_prices_without_multiplying_huge_price_volume(self):
        a = self.calculated(ohlcv([1e308, 5e307], [1e308, 1e308]), "vwema_5")
        np.testing.assert_allclose(a.indicators["vwema_5"], [1e308, 8.333333333333333e307], rtol=1e-14)

    def test_empty_valid_frames_return_only_empty_float_series(self):
        for name, keys in (("vwema_5", {"vwema_5"}), ("vwema_20", {"vwema_20"}),
                           ("supertrend", {"supertrend", "supertrend_trend"})):
            with self.subTest(name=name):
                data = ohlcv([])
                a = self.calculated(data, name)
                self.assertEqual(set(a.indicators), keys)
                for value in a.indicators.values():
                    self.assertTrue(value.empty)
                    self.assertEqual(value.dtype, np.dtype(float))
                    pd.testing.assert_index_equal(value.index, data.index)

    def test_invalid_periods_are_rejected_before_empty_or_short_returns(self):
        for name in ("vwema_5", "supertrend"):
            for period in (0, -1, True, 1.5, "5", np.nan, np.inf):
                for count in (0, 1, 11):
                    with self.subTest(name=name, period=period, count=count):
                        with patch.dict(INDICATORS_CONFIG[name], {"period": period}):
                            self.rejected_atomically(ohlcv([100] * count), name)

    def test_invalid_supertrend_multipliers_are_atomic_even_on_short_data(self):
        for multiplier in (0, -1, True, "3", 1 + 0j, np.nan, np.inf):
            for count in (0, 1, 11):
                with self.subTest(multiplier=multiplier, count=count):
                    with patch.dict(INDICATORS_CONFIG["supertrend"], {"multiplier": multiplier}):
                        self.rejected_atomically(ohlcv([100] * count), "supertrend")

    def test_invalid_vwema_prices_or_volumes_are_rejected_atomically(self):
        for column, values in (("Close", (0, -1, True, "100", 100 + 0j, np.nan, np.inf, None)),
                               ("Volume", (-1, True, "1", 1 + 0j, np.nan, np.inf, None))):
            for value in values:
                with self.subTest(column=column, value=value):
                    data = ohlcv([100] * 11, [1] * 11)
                    data[column] = data[column].astype(object)
                    data.at[data.index[-1], column] = value
                    self.rejected_atomically(data, "vwema_5")

    def test_invalid_supertrend_prices_and_high_low_order_are_atomic(self):
        for column in ("High", "Low", "Close"):
            for value in (0, -1, True, "100", 100 + 0j, np.nan, np.inf, None):
                with self.subTest(column=column, value=value):
                    data = ohlcv([100] * 11)
                    data[column] = data[column].astype(object)
                    data.at[data.index[-1], column] = value
                    self.rejected_atomically(data, "supertrend")
        data = ohlcv([100] * 11)
        data["High"] = 98
        self.rejected_atomically(data, "supertrend")

    def test_supertrend_validation_is_not_hidden_by_atr_fallback(self):
        data = ohlcv([100] * 11)
        data["Close"] = np.nan
        with patch("modules.technical_analysis.ta.volatility.average_true_range",
                   side_effect=RuntimeError("fixture")) as atr:
            self.rejected_atomically(data, "supertrend")
        atr.assert_not_called()

    def test_supertrend_arithmetic_overflow_rejects_without_partial_records(self):
        with patch.dict(INDICATORS_CONFIG["supertrend"], {"multiplier": np.finfo(float).max}):
            with np.errstate(over="ignore", invalid="ignore"):
                self.rejected_atomically(ohlcv([100] * 11), "supertrend")

    def test_supertrend_short_and_full_prefixes_share_the_same_readiness(self):
        full_data = ohlcv([100] * 12)
        full = self.calculated(full_data, "supertrend")
        for count in (0, 1, 9, 10, 11, 12):
            with self.subTest(count=count):
                a = self.calculated(full_data.iloc[:count], "supertrend")
                for key in ("supertrend", "supertrend_trend"):
                    pd.testing.assert_series_equal(a.indicators[key], full.indicators[key].iloc[:count])
                    if count <= 10:
                        self.assertTrue(a.indicators[key].isna().all())
                if count <= 10:
                    self.assertEqual(a.get_latest_indicators(), {})
                else:
                    self.assertEqual(a.get_latest_indicators(), {"supertrend": 94.0, "supertrend_trend": 1.0})

    def test_real_readers_do_not_score_or_cross_unready_supertrend(self):
        alerts = AlertSystem()
        for count in (9, 10, 11):
            with self.subTest(count=count):
                a = self.calculated(ohlcv([100] * count), "supertrend")
                self.assertEqual(alerts.generate_signal(a), "BEKLE")
                result = alerts.generate_position_recommendation(a)
                self.assertEqual(result["bull_score"], 0 if count <= 10 else 1.5)
                self.assertEqual(result["bear_score"], 0)
                self.assertEqual("SuperTrend Pozitif" in result["bull_signals"], count == 11)

    def test_no_new_open_or_close_inside_range_validation_is_added(self):
        data = ohlcv([130] * 11)
        data["Open"] = "unused"
        data["High"], data["Low"] = 101, 99
        a = self.calculated(data, "supertrend")
        self.assertTrue(np.isfinite(a.indicators["supertrend"].iloc[10]))


if __name__ == "__main__":
    unittest.main()
