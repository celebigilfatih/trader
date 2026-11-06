"""Synthetic pattern regressions; no provider data or network access."""

import unittest

import pandas as pd

from modules.pattern_recognition import PatternRecognition
from modules.pattern_recognition_advanced import AdvancedPatternRecognition
from modules.technical_analysis import TechnicalAnalyzer


def candles(rows):
    """Rows are (open, high, low, close); dates and volume are fixed."""
    data = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"],
                        index=pd.date_range("2026-01-01", periods=len(rows)))
    data["Volume"] = 1000
    return data


class TestCandlestickPatterns(unittest.TestCase):
    def test_doji_includes_tolerance_boundary(self):
        data = candles([(100, 106, 96, 101), (100, 106, 96, 101.01),
                        (101, 106, 96, 100)])
        result = PatternRecognition(data).detect_doji()
        self.assertEqual(result.tolist(), [True, False, True])
        self.assertTrue(result.index.equals(data.index))

    def test_hammer_requires_long_lower_and_short_upper_shadow(self):
        result = PatternRecognition(candles([
            (100, 101.1, 98, 101),  # lower=2, upper=0.1, body=1
            (100, 101.1, 98.01, 101),
            (100, 101.11, 98, 101),
        ])).detect_hammer()
        self.assertEqual(result.tolist(), [True, False, False])

    def test_shooting_star_requires_long_upper_and_short_lower_shadow(self):
        result = PatternRecognition(candles([
            (100, 103, 99.9, 101),
            (100, 102.99, 99.9, 101),
            (100, 103, 99.89, 101),
        ])).detect_shooting_star()
        self.assertEqual(result.tolist(), [True, False, False])

    def test_engulfing_requires_strict_enclosure_and_opposite_direction(self):
        cases = [
            ("detect_engulfing_bullish", (104, 105, 99, 100),
             [(99, 106, 98, 105), (100, 106, 99, 105),
              (99, 105, 98, 104), (105, 106, 98, 99)]),
            ("detect_engulfing_bearish", (100, 105, 99, 104),
             [(105, 106, 98, 99), (104, 105, 98, 99),
              (105, 106, 99, 100), (99, 106, 98, 105)]),
        ]
        for method, first, seconds in cases:
            for second, expected in zip(seconds, [True, False, False, False]):
                with self.subTest(method=method, second=second):
                    recognizer = PatternRecognition(candles([first, second]))
                    self.assertEqual(getattr(recognizer, method)().tolist(),
                                     [False, expected])

    def test_star_patterns_require_small_middle_body_and_both_gaps(self):
        cases = [
            ("detect_morning_star", (110, 111, 99, 100),
             (96, 98, 95, 97), (100, 111, 99, 110),
             (95, 98, 94, 98), (100, 111, 98, 110)),
            ("detect_evening_star", (100, 111, 99, 110),
             (113, 115, 112, 114), (110, 111, 99, 100),
             (112, 116, 112, 115), (110, 112, 99, 100)),
        ]
        for method, first, middle, last, large_middle, touching_last in cases:
            for rows, expected in [([first, middle, last], True),
                                   ([first, large_middle, last], False),
                                   ([first, middle, touching_last], False)]:
                with self.subTest(method=method, expected=expected, rows=rows):
                    result = getattr(PatternRecognition(candles(rows)), method)()
                    self.assertEqual(result.tolist(), [False, False, expected])

    def test_latest_patterns_obey_lookback_and_return_last_detection(self):
        neutral = (100, 104, 99, 102)
        doji = (100, 106, 96, 100.5)
        data = candles([doji, neutral, doji, neutral, neutral, neutral])
        recognizer = PatternRecognition(data)
        self.assertIsNone(recognizer.get_latest_patterns(lookback=3)["doji"])
        self.assertEqual(recognizer.get_latest_patterns(lookback=4)["doji"], data.index[2])
        self.assertEqual(recognizer.get_pattern_signals(), {"doji": "BUY"})

    def test_patterns_preserve_caller_data(self):
        data = candles([(100, 104, 99, 102)] * 25)
        original = data.copy(deep=True)
        recognizer = PatternRecognition(data)
        recognizer.analyze_all_patterns()
        advanced = AdvancedPatternRecognition(data)
        advanced.detect_fair_value_gaps()
        advanced.detect_order_blocks()
        pd.testing.assert_frame_equal(data, original)


class TestChartPatterns(unittest.TestCase):
    def test_double_top_and_bottom_require_two_similar_extrema(self):
        for direction, name in [(1, "double_top"), (-1, "double_bottom")]:
            for second_height, expected in [(10, True), (20, False)]:
                with self.subTest(name=name, second_height=second_height):
                    prices = [100] * 35
                    prices[10] += direction * 10
                    prices[24] += direction * second_height
                    data = candles([(p, p + 1, p - 1, p) for p in prices])
                    result = TechnicalAnalyzer(data).detect_chart_patterns()
                    self.assertEqual(result[name], expected)
                    other = "double_bottom" if direction == 1 else "double_top"
                    self.assertFalse(result[other])

    def test_flat_and_single_extremum_do_not_form_double_patterns(self):
        for offset in (0, 10, -10):
            with self.subTest(offset=offset):
                prices = [100] * 25
                prices[12] += offset
                result = TechnicalAnalyzer(candles([
                    (p, p + 1, p - 1, p) for p in prices
                ])).detect_chart_patterns()
                self.assertFalse(result["double_top"])
                self.assertFalse(result["double_bottom"])


class TestAdvancedPatterns(unittest.TestCase):
    def test_fvg_requires_three_candles(self):
        for count in (0, 1, 2):
            with self.subTest(count=count):
                result = AdvancedPatternRecognition(candles([
                    (100, 101, 99, 100)
                ] * count)).detect_fair_value_gaps()
                self.assertTrue(result["bullish"].empty)
                self.assertTrue(result["bearish"].empty)

    def test_fvg_threshold_is_strict_in_both_directions(self):
        # Prior close=100, threshold=1% => size must exceed 1.
        for direction in ("bullish", "bearish"):
            for gap, expected_count in [(0.99, 0), (1.0, 0), (1.01, 1)]:
                with self.subTest(direction=direction, gap=gap):
                    last = ((102 + gap, 104, 101 + gap, 103)
                            if direction == "bullish"
                            else (98 - gap, 99 - gap, 96, 97))
                    data = candles([(100, 101, 99, 100), (100, 104, 96, 100), last])
                    result = AdvancedPatternRecognition(data).detect_fair_value_gaps(1.0)
                    self.assertEqual(len(result[direction]), expected_count)
                    opposite = "bearish" if direction == "bullish" else "bullish"
                    self.assertTrue(result[opposite].empty)
                    if expected_count:
                        row = result[direction].iloc[0]
                        self.assertEqual(row["date"], data.index[2])
                        self.assertAlmostEqual(row["size"], gap)
                        self.assertFalse(row["filled"])

    def test_fvg_fills_only_when_later_price_reaches_far_edge(self):
        for direction in ("bullish", "bearish"):
            for reaches_edge in (False, True):
                with self.subTest(direction=direction, reaches_edge=reaches_edge):
                    if direction == "bullish":
                        third = (104, 106, 103, 105)  # gap [101, 103]
                        fourth = (104, 106, 101 if reaches_edge else 102, 104)
                        low, high = 101, 103
                    else:
                        third = (96, 97, 94, 95)  # gap [97, 99]
                        fourth = (96, 99 if reaches_edge else 98, 94, 96)
                        low, high = 97, 99
                    data = candles([(100, 101, 99, 100), (100, 108, 92, 100),
                                    third, fourth])
                    result = AdvancedPatternRecognition(data).detect_fair_value_gaps()
                    self.assertEqual(len(result[direction]), 1)
                    row = result[direction].iloc[0]
                    self.assertEqual((row["low"], row["high"]), (low, high))
                    self.assertEqual(row["size"], 2)
                    self.assertEqual(row["filled"], reaches_edge)

    def test_order_blocks_direction_threshold_and_strength(self):
        for direction, first, second, third in [
            ("bullish", (102, 103, 100, 101), (101, 103, 100, 102), (102, 105, 101, 104)),
            ("bearish", (98, 100, 97, 99), (99, 100, 97, 98), (98, 99, 95, 96)),
        ]:
            data = candles([(100, 101, 99, 100), first, second, third])
            for threshold, count in [(3.99, 1), (4.0, 0), (4.01, 0)]:
                with self.subTest(direction=direction, threshold=threshold):
                    result = AdvancedPatternRecognition(data).detect_order_blocks(
                        lookback=1, threshold_percent=threshold)
                    self.assertEqual(len(result[direction]), count)
                    opposite = "bearish" if direction == "bullish" else "bullish"
                    self.assertTrue(result[opposite].empty)
                    if count:
                        row = result[direction].iloc[0]
                        self.assertEqual(row["date"], data.index[1])
                        self.assertEqual(row["strength"], 40)
                        self.assertEqual([row[k] for k in ("open", "high", "low", "close")],
                                         list(first))

    def test_order_blocks_require_continuation_and_enough_history(self):
        for rows in [
            [(102, 103, 100, 101), (101, 103, 100, 102), (102, 105, 101, 104)],
            [(100, 101, 99, 100), (102, 103, 100, 101),
             (101, 103, 100, 102), (102, 103, 100, 101)],
        ]:
            with self.subTest(rows=rows):
                result = AdvancedPatternRecognition(candles(rows)).detect_order_blocks(lookback=1)
                self.assertTrue(result["bullish"].empty)
                self.assertTrue(result["bearish"].empty)


if __name__ == "__main__":
    unittest.main()
