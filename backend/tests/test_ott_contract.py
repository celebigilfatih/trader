"""Accepted ADR-0006 numerical and input contract, with independent literals."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules.config import INDICATORS_CONFIG
from modules.technical_analysis import TechnicalAnalyzer


def candles(closes):
    return pd.DataFrame({"Open": closes, "High": [p + 1 for p in closes],
                         "Low": [p - 1 for p in closes], "Close": closes,
                         "Volume": 1000},
                        index=pd.date_range("2026-01-01", periods=len(closes)))


class TestOttContract(unittest.TestCase):
    def assert_outputs(self, analyzer, prices, signals, directions):
        self.assertEqual(set(analyzer.indicators), {"ott", "ott_signal", "ott_trend"})
        for name in analyzer.indicators:
            pd.testing.assert_index_equal(analyzer.indicators[name].index, analyzer.data.index)
        for actual, expected in zip(analyzer.indicators["ott"], prices):
            self.assertAlmostEqual(actual, expected, places=10)
        self.assertEqual(len(analyzer.indicators["ott"]), len(prices))
        for name, expected in (("ott_signal", signals), ("ott_trend", directions)):
            self.assertEqual(len(analyzer.indicators[name]), len(expected))
            for actual, value in zip(analyzer.indicators[name], expected):
                if value is None:
                    self.assertTrue(pd.isna(actual), name)
                else:
                    self.assertEqual(actual, value, name)
        self.assertTrue(pd.api.types.is_float_dtype(analyzer.indicators["ott_trend"]))

    def test_accepted_literal_table_seeds_before_loop_and_reverses_direction(self):
        analyzer = TechnicalAnalyzer(candles([100.0, 110.0, 90.0]))
        analyzer.add_indicator("ott")
        # Adjusted EMA weights 1,1/3,1/9 give VAR=100,107.5,1240/13.
        # Seed100; long stop107.5*.986=105.995; short stop(1240/13)*1.014=96.72.
        self.assert_outputs(analyzer, [100, 105.995, 96.72],
                            [None, "buy", "sell"], [None, 1, -1])

    def test_mirrored_literal_table_reverses_back_up(self):
        analyzer = TechnicalAnalyzer(candles([100.0, 90.0, 110.0]))
        analyzer.add_indicator("ott")
        # VAR=100,92.5,1360/13; seeded short stop93.795 then long67048/650.
        self.assert_outputs(analyzer, [100, 93.795, 103.15076923076923],
                            [None, "sell", "buy"], [None, -1, 1])

    def test_existing_else_equality_rule_and_one_sided_clamps_are_preserved(self):
        for close, price, signal, direction in ((100.0, 100.0, "sell", -1),
                                                 (100.1, 100.0, "buy", 1),
                                                 (99.9, 100.0, "sell", -1)):
            with self.subTest(close=close):
                analyzer = TechnicalAnalyzer(candles([100.0, close]))
                analyzer.add_indicator("ott")
                # Small moves keep the old stop via max/min; direction still updates.
                self.assert_outputs(analyzer, [100, price], [None, signal], [None, direction])

    def test_custom_existing_config_still_controls_adjusted_var_and_stop_percent(self):
        analyzer = TechnicalAnalyzer(candles([100.0, 110.0, 90.0]))
        with patch.dict(INDICATORS_CONFIG["ott"], {"period": 3, "percent": 2.0}):
            analyzer.add_indicator("ott")
        # Weights1,1/2,1/4 => VAR=100,320/3,680/7; stops .98/1.02.
        self.assert_outputs(analyzer, [100, 1568 / 15, 3468 / 35],
                            [None, "buy", "sell"], [None, 1, -1])

    def test_empty_input_returns_three_empty_series_with_original_index(self):
        data = candles([])
        analyzer = TechnicalAnalyzer(data)
        analyzer.add_indicator("ott")
        self.assert_outputs(analyzer, [], [], [])
        self.assertEqual(analyzer.get_latest_indicators(), {})
        pd.testing.assert_frame_equal(data, candles([]))

    def test_single_observation_has_price_but_no_calculated_direction(self):
        analyzer = TechnicalAnalyzer(candles([42.5]))
        analyzer.add_indicator("ott")
        self.assert_outputs(analyzer, [42.5], [None], [None])
        self.assertEqual(analyzer.get_latest_indicators(), {"ott": 42.5})

    def test_nonpositive_nonfinite_or_nonnumeric_close_is_rejected_before_mutation(self):
        for value in (0.0, -1.0, np.nan, np.inf, -np.inf, None, "invalid", 1 + 2j):
            for position in (0, 1, 2):
                with self.subTest(value=value, position=position):
                    data = candles([100.0, 110.0, 90.0])
                    if value is None or isinstance(value, (str, complex)):
                        data["Close"] = data["Close"].astype(object)
                    data.iloc[position, data.columns.get_loc("Close")] = value
                    original = data.copy(deep=True)
                    analyzer = TechnicalAnalyzer(data)
                    with self.assertRaisesRegex(ValueError, "Close.*positive.*finite|Close.*finite.*positive"):
                        analyzer.add_indicator("ott")
                    pd.testing.assert_frame_equal(data, original)
                    pd.testing.assert_frame_equal(analyzer.data, original)
                    self.assertEqual(analyzer.indicators, {})

    def test_invalid_recalculation_does_not_partially_overwrite_previous_result(self):
        analyzer = TechnicalAnalyzer(candles([100.0, 110.0, 90.0]))
        analyzer.add_indicator("ott")
        previous = {name: series.copy(deep=True) for name, series in analyzer.indicators.items()}
        analyzer.data.iloc[1, analyzer.data.columns.get_loc("Close")] = np.nan
        before = analyzer.data.copy(deep=True)
        with self.assertRaises(ValueError):
            analyzer.add_indicator("ott")
        pd.testing.assert_frame_equal(analyzer.data, before)
        for name, series in previous.items():
            pd.testing.assert_series_equal(analyzer.indicators[name], series)

    def test_temporal_close_is_rejected_as_non_price_without_mutation(self):
        for values in (pd.date_range("2026-01-01", periods=3),
                       pd.to_timedelta([1, 2, 3], unit="D")):
            with self.subTest(dtype=values.dtype):
                data = candles([100.0, 110.0, 90.0])
                data["Close"] = values
                analyzer = TechnicalAnalyzer(data)
                before = analyzer.data.copy(deep=True)
                with self.assertRaisesRegex(ValueError, "Close.*positive.*finite"):
                    analyzer.add_indicator("ott")
                self.assertEqual(analyzer.indicators, {})
                pd.testing.assert_frame_equal(analyzer.data, before)

    def test_timezone_gapped_index_caller_data_and_previous_series_are_preserved(self):
        data = candles([100.0, 110.0, 90.0])
        data.index = pd.DatetimeIndex(["2026-01-02", "2026-01-06", "2026-01-09"],
                                     tz="Europe/Istanbul", name="bar_time")
        original = data.copy(deep=True)
        analyzer = TechnicalAnalyzer(data)
        analyzer.add_indicator("ott")
        old_price = analyzer.indicators["ott"]
        old_trend = analyzer.indicators["ott_trend"]
        price_snapshot, trend_snapshot = old_price.copy(), old_trend.copy()
        analyzer.add_indicator("ott")
        self.assert_outputs(analyzer, [100, 105.995, 96.72],
                            [None, "buy", "sell"], [None, 1, -1])
        pd.testing.assert_frame_equal(data, original)
        pd.testing.assert_series_equal(old_price, price_snapshot)
        pd.testing.assert_series_equal(old_trend, trend_snapshot)


if __name__ == "__main__":
    unittest.main()
