"""Characterize batch BOS on synthetic candles; no live/causal strategy claim."""

import unittest
from unittest.mock import patch

import pandas as pd

from modules.config import INDICATORS_CONFIG
from modules.pattern_recognition_advanced import AdvancedPatternRecognition
from modules.technical_analysis import TechnicalAnalyzer


def structure_break(direction, *, break_position=25, close=None, volume=1000):
    """40 flat bars, one isolated swing at 10 and one later break by default."""
    data = pd.DataFrame(
        {"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0,
         "Volume": 1000},
        index=pd.date_range("2026-01-01", periods=40),
    )
    if direction == "bullish":
        data.iloc[10, data.columns.get_loc("High")] = 125.0
        price = 126.25 if close is None else close
        data.iloc[break_position, data.columns.get_loc("High")] = max(130.0, price)
    else:
        data.iloc[10, data.columns.get_loc("Low")] = 80.0
        price = 79.2 if close is None else close
        data.iloc[break_position, data.columns.get_loc("Low")] = min(75.0, price)
    data.iloc[break_position, data.columns.get_loc("Close")] = price
    data.iloc[break_position, data.columns.get_loc("Volume")] = volume
    return data


class TestBreakOfStructure(unittest.TestCase):
    def test_real_swings_produce_both_break_directions_and_manual_strength(self):
        # Swing strengths: (125-100)/100*100*2=50; (100-80)/100*100*2=40.
        # A 1% close break gives 15+50*.3=30 or 15+40*.3=27, without volume boost.
        for direction, price, previous_key, level, strength in (
            ("bullish", 126.25, "prev_high", 125.0, 30.0),
            ("bearish", 79.2, "prev_low", 80.0, 27.0),
        ):
            with self.subTest(direction=direction):
                data = structure_break(direction)
                recognizer = AdvancedPatternRecognition(data)
                result = recognizer.detect_break_of_structure()
                self.assertEqual(len(result[direction]), 1)
                opposite = "bearish" if direction == "bullish" else "bullish"
                self.assertTrue(result[opposite].empty)
                row = result[direction].iloc[0]
                self.assertEqual(row["date"], pd.Timestamp("2026-01-26"))
                self.assertEqual(row["price"], price)
                self.assertEqual(row[previous_key], level)
                self.assertAlmostEqual(row["strength"], strength)
                self.assertFalse(row["volume_confirmation"])
                points = recognizer.structure_points["highs" if direction == "bullish" else "lows"]
                swing = next(point for point in points if point["idx"] == 10)
                self.assertEqual(swing["date"], pd.Timestamp("2026-01-11"))
                self.assertEqual(swing["price"], level)
                self.assertEqual(swing["strength"], 50.0 if direction == "bullish" else 40.0)

    def test_wick_beyond_level_and_equal_close_are_not_breaks(self):
        for direction, closes in (("bullish", (124.0, 125.0, 125.01)),
                                   ("bearish", (81.0, 80.0, 79.99))):
            for close, expected in zip(closes, (0, 0, 1)):
                with self.subTest(direction=direction, close=close):
                    # Fixture wick extends beyond the level even in negative cases.
                    result = AdvancedPatternRecognition(structure_break(direction, close=close)).detect_break_of_structure()
                    self.assertEqual(len(result[direction]), expected)

    def test_volume_boost_requires_strictly_more_than_twenty_percent(self):
        for direction, base_strength, boosted in (("bullish", 30.0, 39.0),
                                                   ("bearish", 27.0, 35.1)):
            for volume, confirmed, strength in ((1200, False, base_strength),
                                                 (1201, True, boosted)):
                with self.subTest(direction=direction, volume=volume):
                    result = AdvancedPatternRecognition(structure_break(direction, volume=volume)).detect_break_of_structure()
                    self.assertEqual(len(result[direction]), 1)
                    row = result[direction].iloc[0]
                    self.assertEqual(row["volume_confirmation"], confirmed)
                    self.assertAlmostEqual(row["strength"], strength)

    def test_swing_strength_threshold_includes_equality(self):
        for direction, boundary in (("bullish", 5.0), ("bearish", 4.0)):
            for threshold, expected in ((boundary, 1), (boundary + 0.0001, 0)):
                with self.subTest(direction=direction, threshold=threshold):
                    result = AdvancedPatternRecognition(structure_break(direction)).detect_break_of_structure(
                        swing_threshold=threshold)
                    self.assertEqual(len(result[direction]), expected)

    def test_swing_volume_factor_uses_local_average_and_caps_at_two(self):
        for direction, base in (("bullish", 50.0), ("bearish", 40.0)):
            for volume, factor, expected_break_strength in (
                (2000, 40 / 21, 305 / 7 if direction == "bullish" else 265 / 7),
                (10000, 2.0, 45.0 if direction == "bullish" else 39.0),
            ):
                with self.subTest(direction=direction, swing_volume=volume):
                    data = structure_break(direction)
                    data.iloc[10, data.columns.get_loc("Volume")] = volume
                    recognizer = AdvancedPatternRecognition(data)
                    result = recognizer.detect_break_of_structure()
                    points = recognizer.structure_points["highs" if direction == "bullish" else "lows"]
                    self.assertIn(10, [point["idx"] for point in points])
                    swing = next(point for point in points if point["idx"] == 10)
                    # For volume=2000: local mean=(19000+2000)/20=1050.
                    self.assertAlmostEqual(swing["volume_strength"], factor)
                    self.assertAlmostEqual(swing["strength"], base * factor)
                    self.assertEqual(len(result[direction]), 1)
                    self.assertAlmostEqual(result[direction].iloc[0]["strength"], expected_break_strength)

    def test_large_break_is_capped_before_existing_volume_boost(self):
        # Current heuristic may exceed 100 after the boost; it is not probability.
        for direction, close in (("bullish", 150.0), ("bearish", 40.0)):
            with self.subTest(direction=direction):
                result = AdvancedPatternRecognition(structure_break(
                    direction, close=close, volume=1300)).detect_break_of_structure()
                self.assertEqual(len(result[direction]), 1)
                self.assertTrue(result[direction].iloc[0]["volume_confirmation"])
                self.assertEqual(result[direction].iloc[0]["strength"], 130.0)

    def test_short_or_flat_data_does_not_invent_structure(self):
        for count in (0, 1, 9, 10, 11, 40):
            with self.subTest(count=count):
                data = structure_break("bullish").iloc[:count].copy()
                data.loc[:, ["Open", "High", "Low", "Close"]] = 100.0
                recognizer = AdvancedPatternRecognition(data)
                result = recognizer.detect_break_of_structure()
                self.assertTrue(result["bullish"].empty)
                self.assertTrue(result["bearish"].empty)
                self.assertEqual(recognizer.structure_points, {"highs": [], "lows": []})

    def test_lookback_excludes_swing_at_its_outer_boundary(self):
        # With 40 bars, swing at 10 is offset 30; range excludes offset=lookback.
        for direction in ("bullish", "bearish"):
            for lookback, expected in ((30, 0), (31, 1)):
                with self.subTest(direction=direction, lookback=lookback):
                    result = AdvancedPatternRecognition(structure_break(direction)).detect_break_of_structure(lookback=lookback)
                    self.assertEqual(len(result[direction]), expected)

    def test_batch_scan_includes_tenth_bar_from_end_but_not_newer_bars(self):
        for direction in ("bullish", "bearish"):
            for position, expected in ((30, 1), (31, 0)):
                with self.subTest(direction=direction, position=position):
                    data = structure_break(direction, break_position=position)
                    result = AdvancedPatternRecognition(data).detect_break_of_structure()
                    self.assertEqual(len(result[direction]), expected)
                    if expected:
                        self.assertEqual(result[direction].iloc[0]["date"], pd.Timestamp("2026-01-31"))

    def test_break_uses_prior_extreme_rather_than_weaker_recent_swing(self):
        for direction, column, recent, closes, key, extreme in (
            ("bullish", "High", 120.0, (123.0, 126.25), "prev_high", 125.0),
            ("bearish", "Low", 85.0, (82.0, 79.2), "prev_low", 80.0),
        ):
            for close, expected in zip(closes, (0, 1)):
                with self.subTest(direction=direction, close=close):
                    data = structure_break(direction, break_position=27, close=close)
                    data.iloc[18, data.columns.get_loc(column)] = recent
                    result = AdvancedPatternRecognition(data).detect_break_of_structure()
                    self.assertEqual(len(result[direction]), expected)
                    if expected:
                        self.assertEqual(result[direction].iloc[0][key], extreme)

    def test_a_later_swing_cannot_serve_as_previous_level(self):
        for direction in ("bullish", "bearish"):
            with self.subTest(direction=direction):
                data = structure_break(direction, break_position=6)
                recognizer = AdvancedPatternRecognition(data)
                result = recognizer.detect_break_of_structure()
                points = recognizer.structure_points["highs" if direction == "bullish" else "lows"]
                self.assertIn(10, [point["idx"] for point in points])
                self.assertTrue(result["bullish"].empty)
                self.assertTrue(result["bearish"].empty)

    def test_equal_neighbor_extremes_do_not_create_strict_swings(self):
        for direction, column, level in (("bullish", "High", 125.0),
                                          ("bearish", "Low", 80.0)):
            with self.subTest(direction=direction):
                data = structure_break(direction)
                data.iloc[11, data.columns.get_loc(column)] = level
                result = AdvancedPatternRecognition(data).detect_break_of_structure()
                self.assertTrue(result[direction].empty)

    def test_swings_five_bars_apart_survive_but_four_bars_apart_do_not(self):
        for direction, column, later_level, key, earlier_level in (
            ("bullish", "High", 120.0, "prev_high", 125.0),
            ("bearish", "Low", 85.0, "prev_low", 80.0),
        ):
            for second_position, selected_level in ((14, later_level), (15, earlier_level)):
                with self.subTest(direction=direction, second_position=second_position):
                    data = structure_break(direction, break_position=27)
                    data.iloc[second_position, data.columns.get_loc(column)] = later_level
                    recognizer = AdvancedPatternRecognition(data)
                    result = recognizer.detect_break_of_structure()
                    points = recognizer.structure_points["highs" if direction == "bullish" else "lows"]
                    previous_positions = {point["idx"] for point in points if point["idx"] < 27}
                    self.assertEqual(previous_positions, {14} if second_position == 14 else {10, 15})
                    self.assertEqual(len(result[direction]), 1)
                    self.assertEqual(result[direction].iloc[0][key], selected_level)

    def test_analyzer_publishes_real_bos_and_respects_threshold_config(self):
        for direction, rejected_threshold in (("bullish", 5.0001), ("bearish", 4.0001)):
            with self.subTest(direction=direction):
                data = structure_break(direction)
                original = data.copy(deep=True)
                analyzer = TechnicalAnalyzer(data)
                analyzer.add_indicator("bos")
                self.assertEqual(len(analyzer.indicators["bos_" + direction]), 1)
                row = analyzer.indicators["bos_" + direction].iloc[0]
                self.assertEqual(row["date"], pd.Timestamp("2026-01-26"))
                self.assertEqual(row["price"], 126.25 if direction == "bullish" else 79.2)
                with patch.dict(INDICATORS_CONFIG["bos"], {"swing_threshold": rejected_threshold}):
                    analyzer.add_indicator("bos")
                self.assertTrue(analyzer.indicators["bos_bullish"].empty)
                self.assertTrue(analyzer.indicators["bos_bearish"].empty)
                pd.testing.assert_frame_equal(data, original)

    def test_repeated_detection_preserves_caller_and_does_not_duplicate_events(self):
        data = structure_break("bullish")
        original = data.copy(deep=True)
        recognizer = AdvancedPatternRecognition(data)
        first = recognizer.detect_break_of_structure()
        second = recognizer.detect_break_of_structure()
        self.assertEqual(len(second["bullish"]), 1)
        pd.testing.assert_frame_equal(first["bullish"], second["bullish"])
        pd.testing.assert_frame_equal(first["bearish"], second["bearish"])
        pd.testing.assert_frame_equal(data, original)


if __name__ == "__main__":
    unittest.main()
