"""Deterministic combo regressions; heuristic batch output, not trading evidence."""

import unittest
from unittest.mock import patch

import pandas as pd

from modules.config import INDICATORS_CONFIG
from modules.pattern_recognition_advanced import AdvancedPatternRecognition
from modules.technical_analysis import TechnicalAnalyzer


def candles(rows, start="2026-01-01"):
    data = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"],
                        index=pd.date_range(start, periods=len(rows)))
    data["Volume"] = 1000
    return data


def ob_chain(direction):
    rows = [(100, 101, 99, 100)] * 23
    if direction == "bullish":
        rows += [(101, 102, 100, 100.5), (101, 103, 100.5, 102),
                 (103, 105, 103, 104)]
    else:
        rows += [(99, 100, 98, 99.5), (99, 99.5, 97, 98),
                 (97, 97, 95, 96)]
    return candles(rows)


def bos_chain(direction):
    """One earlier swing and three closes breaking it; no later breaks.

    BOS deliberately uses its current batch scan, including future swing
    neighbors; the tenth bar from the end is included, newer nine excluded.
    All volumes are equal.
    """
    if direction == "bullish":
        rows = [(99, 100, 98, 99)] * 40
        rows[10] = (99, 105, 98, 100)
        rows[18:21] = [(105, 108, 104, 106), (107, 109, 107, 108),
                       (109, 111, 109, 110)]
    else:
        rows = [(101, 102, 100, 101)] * 40
        rows[10] = (101, 102, 96, 100)
        rows[18:21] = [(95, 96, 93, 94), (94, 94, 92, 93),
                       (91, 91, 90, 91)]
    return candles(rows)


def derived_gap(direction, low, high, date="2026-01-21"):
    """Real three-candle FVG for isolated matcher tests, not an end-to-end chain."""
    if direction == "bullish":
        rows = [(low - 1, low, low - 2, low - 1),
                (low, high + 2, low - 2, high),
                (high + 1, high + 2, high, high + 1)]
    else:
        rows = [(high + 1, high + 2, high, high + 1),
                (high, high + 2, low - 2, low),
                (low - 1, low, low - 2, low - 1)]
    start = pd.Timestamp(date) - pd.Timedelta(days=2)
    return AdvancedPatternRecognition(candles(rows, start)).detect_fair_value_gaps()


class TestFvgOrderBlockCombo(unittest.TestCase):
    def test_real_ohlcv_chains_have_literal_outputs_in_both_directions(self):
        # OB movement is 4% => strength 40; gap size 1 => combo strength 20.5.
        cases = [("bullish", (102, 103), (100, 102)),
                 ("bearish", (97, 98), (98, 100))]
        for direction, gap, block in cases:
            with self.subTest(direction=direction):
                result = AdvancedPatternRecognition(ob_chain(direction)).get_fvg_order_block_combo()
                self.assertEqual(result, [{"type": direction,
                                          "date": pd.Timestamp("2026-01-26"),
                                          "fvg_zone": gap, "order_block": block,
                                          "strength": 20.5}])

    def test_matcher_price_distance_is_strict_two_percent_on_either_side(self):
        # These caches come from separate real detectors. Only the matcher is
        # under test here; the end-to-end pipeline is exercised above.
        sources = {
            "bullish": [(100, 110, 90, 100), (100, 100, 98, 99),
                        (99, 101, 98, 100), (101, 103, 101, 102)],
            "bearish": [(100, 110, 90, 100), (100, 102, 100, 101),
                        (101, 102, 99, 100), (100, 101, 97, 98)],
        }
        # The reference OB edge is 100 in either direction, so ±2 is exactly 2%.
        for direction, rows in sources.items():
            for edge, count in [(97.99, 0), (98, 0), (98.01, 1),
                                (101.99, 1), (102, 0), (102.01, 0)]:
                with self.subTest(direction=direction, edge=edge):
                    analyzer = AdvancedPatternRecognition(candles(rows))
                    analyzer.detect_order_blocks(lookback=1)
                    low, high = ((edge, edge + 1) if direction == "bullish"
                                 else (edge - 1, edge))
                    analyzer.fvg_zones = derived_gap(direction, low, high)
                    self.assertEqual(len(analyzer.get_fvg_order_block_combo()), count)

    def test_matcher_does_not_pair_opposite_directions(self):
        for direction, opposite, low, high in [
            ("bullish", "bearish", 101, 102),
            ("bearish", "bullish", 98, 99),
        ]:
            with self.subTest(direction=direction):
                analyzer = AdvancedPatternRecognition(ob_chain(direction))
                blocks = analyzer.detect_order_blocks()
                self.assertEqual(len(blocks[direction]), 1)
                analyzer.fvg_zones = derived_gap(opposite, low, high)
                self.assertEqual(len(analyzer.fvg_zones[opposite]), 1)
                self.assertEqual(analyzer.get_fvg_order_block_combo(), [])


class TestFvgBosCombo(unittest.TestCase):
    def assert_bos_combo(self, result, direction):
        self.assertEqual(len(result), 1)
        actual = result[0]
        # Bullish swing strength=2*(105-100)/100*100=10. BOS strength
        # =15*(110-105)/105*100 + .3*10 = 521/7. Gap size=1:
        # combo=264/7; proximity=3/220; confidence=(264/7)*(217/220)=37.2.
        # Bearish swing=8, BOS=15*(96-91)/96*100 + .3*8=80.525;
        # gap size=2, combo=41.2625; proximity=1/91.
        if direction == "bullish":
            expected = {"type": direction, "date": pd.Timestamp("2026-01-21"),
                        "fvg_zone": (108, 109), "bos_price": 110, "prev_high": 105,
                        "strength": 37.714285714285715, "time_diff": 0,
                        "price_proximity": 0.013636363636363636, "confidence": 37.2}
        else:
            expected = {"type": direction, "date": pd.Timestamp("2026-01-21"),
                        "fvg_zone": (91, 93), "bos_price": 91, "prev_low": 96,
                        "strength": 41.2625, "time_diff": 0,
                        "price_proximity": 0.01098901098901099,
                        "confidence": 40.809065934065934}
        self.assertEqual(set(actual), set(expected))
        for key, value in expected.items():
            if isinstance(value, float):
                self.assertAlmostEqual(actual[key], value, places=10, msg=key)
            else:
                self.assertEqual(actual[key], value, key)

    def test_real_ohlcv_detection_chains_have_literal_outputs_in_both_directions(self):
        for direction in ("bullish", "bearish"):
            with self.subTest(direction=direction):
                analyzer = AdvancedPatternRecognition(bos_chain(direction))
                self.assert_bos_combo(analyzer.get_fvg_bos_combo(), direction)

    def test_real_weaker_breaks_are_filtered_by_default_strength(self):
        # Same gap is near the Jan 20 close too, but that break is too weak.
        # Lowering only the matcher cutoff demonstrates that candidate exists.
        analyzer = AdvancedPatternRecognition(bos_chain("bullish"))
        self.assert_bos_combo(analyzer.get_fvg_bos_combo(), "bullish")
        with patch.dict(INDICATORS_CONFIG["fvg_bos_combo"], {"strength_threshold": 20}):
            result = analyzer.get_fvg_bos_combo()
        self.assertEqual([row["bos_price"] for row in result], [110, 108])
        self.assertAlmostEqual(result[1]["strength"], 23.428571428571427, places=10)
        self.assertEqual(result[1]["time_diff"], 1)

    def test_matcher_time_window_includes_three_days_in_either_order(self):
        # FVG caches are separately detected real records at controlled dates;
        # BOS is always detected afresh from the real 40-bar data, never mocked.
        for direction, low, high in [("bullish", 108, 109), ("bearish", 91, 93)]:
            for offset, count in [(-4, 0), (-3, 1), (3, 1), (4, 0)]:
                with self.subTest(direction=direction, offset=offset):
                    analyzer = AdvancedPatternRecognition(bos_chain(direction))
                    date = pd.Timestamp("2026-01-21") + pd.Timedelta(days=offset)
                    analyzer.fvg_zones = derived_gap(direction, low, high, date)
                    result = analyzer.get_fvg_bos_combo()
                    self.assertEqual(len(result), count)
                    if count:
                        self.assertEqual(result[0]["date"], max(date, pd.Timestamp("2026-01-21")))
                        self.assertEqual(result[0]["time_diff"], 3)
                        self.assertEqual(result[0]["confidence"], 0)

    def test_matcher_price_threshold_is_inclusive(self):
        # Actual price distance is the independently derived 3/220. Adjacent
        # float thresholds exercise equality without decimal subtraction drift.
        for threshold, count in [(0.013636363636363634, 0),
                                 (0.013636363636363636, 1),
                                 (0.013636363636363638, 1)]:
            with self.subTest(threshold=threshold):
                with patch.dict(INDICATORS_CONFIG["fvg_bos_combo"],
                                {"proximity_threshold": threshold}):
                    result = AdvancedPatternRecognition(bos_chain("bullish")).get_fvg_bos_combo()
                self.assertEqual(len(result), count)

    def test_matcher_strength_threshold_is_inclusive(self):
        # Combo strength is 264/7; test both neighboring floats and equality.
        for threshold, count in [(37.71428571428571, 1),
                                 (37.714285714285715, 1),
                                 (37.71428571428572, 0)]:
            with self.subTest(threshold=threshold):
                with patch.dict(INDICATORS_CONFIG["fvg_bos_combo"],
                                {"strength_threshold": threshold}):
                    result = AdvancedPatternRecognition(bos_chain("bullish")).get_fvg_bos_combo()
                self.assertEqual(len(result), count)

    def test_matcher_rejects_distant_prices_and_opposite_directions(self):
        for direction, gap_direction, low, high in [
            ("bullish", "bullish", 120, 121),
            ("bearish", "bearish", 80, 82),
            ("bullish", "bearish", 108, 109),
            ("bearish", "bullish", 91, 93),
        ]:
            with self.subTest(direction=direction, gap_direction=gap_direction, low=low):
                analyzer = AdvancedPatternRecognition(bos_chain(direction))
                analyzer.fvg_zones = derived_gap(gap_direction, low, high)
                self.assertEqual(len(analyzer.fvg_zones[gap_direction]), 1)
                self.assertEqual(analyzer.get_fvg_bos_combo(), [])

    def test_matcher_sorts_by_confidence_with_time_and_price_penalties(self):
        analyzer = AdvancedPatternRecognition(bos_chain("bullish"))
        frames = [derived_gap("bullish", 108, 109, "2026-01-22")["bullish"],
                  derived_gap("bullish", 109, 110, "2026-01-20")["bullish"],
                  derived_gap("bullish", 108, 109, "2026-01-21")["bullish"]]
        analyzer.fvg_zones = {"bullish": pd.concat(frames, ignore_index=True),
                              "bearish": frames[0].iloc[:0].copy()}
        result = analyzer.get_fvg_bos_combo()
        self.assertEqual([row["fvg_zone"] for row in result],
                         [(108, 109), (109, 110), (108, 109)])
        self.assertEqual([row["date"] for row in result],
                         [pd.Timestamp("2026-01-21"), pd.Timestamp("2026-01-21"),
                          pd.Timestamp("2026-01-22")])
        self.assertEqual([row["time_diff"] for row in result], [0, 1, 1])
        for row, expected in zip(result, [37.2, 25.02857142857143, 24.8]):
            self.assertAlmostEqual(row["confidence"], expected, places=10)


class TestTechnicalAnalyzerCombos(unittest.TestCase):
    def test_real_order_block_combo_output_and_caller_data_preserved(self):
        for direction, gap, block in [("bullish", (102, 103), (100, 102)),
                                      ("bearish", (97, 98), (98, 100))]:
            with self.subTest(direction=direction):
                data = ob_chain(direction)
                original = data.copy(deep=True)
                analyzer = TechnicalAnalyzer(data)
                analyzer.add_indicator("fvg_ob_combo")
                self.assertEqual(analyzer.indicators["fvg_ob_combo"],
                                 [{"type": direction, "date": pd.Timestamp("2026-01-26"),
                                   "fvg_zone": gap, "order_block": block, "strength": 20.5}])
                pd.testing.assert_frame_equal(data, original)
                pd.testing.assert_frame_equal(analyzer.data, original)

    def test_real_bos_combo_output_and_caller_data_preserved(self):
        for direction in ("bullish", "bearish"):
            with self.subTest(direction=direction):
                data = bos_chain(direction)
                original = data.copy(deep=True)
                analyzer = TechnicalAnalyzer(data)
                analyzer.add_indicator("fvg_bos_combo")
                TestFvgBosCombo.assert_bos_combo(self, analyzer.indicators["fvg_bos_combo"], direction)
                pd.testing.assert_frame_equal(data, original)
                pd.testing.assert_frame_equal(analyzer.data, original)


if __name__ == "__main__":
    unittest.main()
