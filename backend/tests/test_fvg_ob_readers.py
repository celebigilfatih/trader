"""Real FVG/OB selectors and the actual legacy chart function, without app startup."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend.tests.test_pattern_combos import candles, ob_chain
from modules.config import INDICATORS_CONFIG
from modules.technical_analysis import TechnicalAnalyzer


def gap_frame(direction, count=3, filled=False):
    third = (104, 106, 103, 105) if direction == 'bullish' else (96, 97, 94, 95)
    later = ((104, 106, 101 if filled else 102, 104) if direction == 'bullish'
             else (96, 99 if filled else 98, 94, 96))
    return candles([(100, 101, 99, 100), (100, 108, 92, 100), third] + [later] * (count - 3))


def chart_function():
    source = (Path(__file__).resolve().parents[2] / 'app.py').read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'create_chart')
    namespace = dict(pd=pd, np=np, go=go, make_subplots=make_subplots, INDICATORS_CONFIG=INDICATORS_CONFIG)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-create-chart>', 'exec'), namespace)
    return namespace['create_chart']


class TestFvgObReaders(unittest.TestCase):
    def setUp(self):
        for target in ['socket.socket.connect', 'socket.socket.connect_ex', 'socket.create_connection']:
            self.enterContext(patch(target, side_effect=AssertionError('Network forbidden')))

    def calculate(self, frame, *names):
        before = frame.copy(deep=True)
        analyzer = TechnicalAnalyzer(frame)
        for name in names:
            analyzer.add_indicator(name)
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(analyzer.data, before)
        return analyzer

    def test_fvg_selector_exposes_literal_directional_dataframes(self):
        for direction, low, high in [('bullish', 101, 103), ('bearish', 97, 99)]:
            with self.subTest(direction=direction):
                frame = gap_frame(direction)
                analyzer = self.calculate(frame, 'fvg')
                self.assertEqual(set(analyzer.indicators), {'fvg_bullish', 'fvg_bearish'})
                result = analyzer.indicators['fvg_' + direction]
                self.assertEqual(list(result.columns), ['date', 'low', 'high', 'size', 'filled'])
                self.assertEqual(result.to_dict('records'), [dict(date=frame.index[2], low=low, high=high, size=2, filled=False)])
                opposite = 'bearish' if direction == 'bullish' else 'bullish'
                self.assertTrue(analyzer.indicators['fvg_' + opposite].empty)
                self.assertEqual(analyzer.get_latest_indicators(), {})  # Structured output is not a scalar.

    def test_fvg_selector_passes_configured_threshold_and_preserves_fill_state(self):
        for threshold, count in [(1.99, 1), (2., 0), (2.01, 0)]:
            for direction in ['bullish', 'bearish']:
                with self.subTest(threshold=threshold, direction=direction):
                    with patch.dict(INDICATORS_CONFIG['fvg'], threshold_percent=threshold):
                        analyzer = self.calculate(gap_frame(direction, 4, filled=True), 'fvg')
                    rows = analyzer.indicators['fvg_' + direction]
                    self.assertEqual(len(rows), count)
                    if count:
                        self.assertTrue(rows.iloc[0]['filled'])

    def test_order_block_selector_exposes_literal_ohlc_and_strength(self):
        for direction, values in [('bullish', (101, 102, 100, 100.5)), ('bearish', (99, 100, 98, 99.5))]:
            with self.subTest(direction=direction):
                frame = ob_chain(direction)
                analyzer = self.calculate(frame, 'order_block')
                self.assertEqual(set(analyzer.indicators), {'ob_bullish', 'ob_bearish'})
                result = analyzer.indicators['ob_' + direction]
                self.assertEqual(list(result.columns), ['date', 'open', 'high', 'low', 'close', 'strength'])
                self.assertEqual(result.to_dict('records'), [dict(date=frame.index[23], open=values[0], high=values[1], low=values[2], close=values[3], strength=40.)])
                self.assertTrue(analyzer.indicators['ob_' + ('bearish' if direction == 'bullish' else 'bullish')].empty)
                self.assertEqual(analyzer.get_latest_indicators(), {})

    def test_order_block_selector_passes_threshold_and_lookback(self):
        # The one block is three bars from the end, independent of extra history.
        for lookback, threshold, count in [(20, 3.99, 1), (20, 4., 0), (24, .5, 0), (23, .5, 1)]:
            with self.subTest(lookback=lookback, threshold=threshold):
                with patch.dict(INDICATORS_CONFIG['order_block'], lookback=lookback, threshold_percent=threshold):
                    analyzer = self.calculate(ob_chain('bullish'), 'order_block')
                self.assertEqual(len(analyzer.indicators['ob_bullish']), count)

    def test_empty_and_short_selectors_publish_empty_frames_with_expected_columns(self):
        for count in [0, 1, 2, 22]:
            for selector, prefix in [('fvg', 'fvg'), ('order_block', 'ob')]:
                with self.subTest(count=count, selector=selector):
                    analyzer = self.calculate(candles([(100, 101, 99, 100)] * count), selector)
                    for direction in ['bullish', 'bearish']:
                        self.assertIsInstance(analyzer.indicators[prefix + '_' + direction], pd.DataFrame)
                        self.assertTrue(analyzer.indicators[prefix + '_' + direction].empty)

    def shapes(self, frame, names, selected=None):
        analyzer = self.calculate(frame, *names)
        before = frame.copy(deep=True)
        snapshots = {key: value.copy(deep=True) for key, value in analyzer.indicators.items()}
        try:
            figure = chart_function()(frame, analyzer, selected if selected is not None else dict.fromkeys(names, True))
        except Exception as exc:
            self.fail(f'Actual chart raised {type(exc).__name__}: {exc}')
        for key, value in snapshots.items():
            pd.testing.assert_frame_equal(analyzer.indicators[key], value)
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(analyzer.data, before)
        self.assertEqual([trace.type for trace in figure.data], ['candlestick', 'bar'])
        return list(figure.layout.shapes)

    def test_actual_chart_renders_both_fvg_directions_from_real_frames(self):
        for direction, low, high, color in [('bullish', 101, 103, 'rgba(76, 175, 80, 0.2)'), ('bearish', 97, 99, 'rgba(244, 67, 54, 0.2)')]:
            with self.subTest(direction=direction):
                frame = gap_frame(direction, 10)
                shapes = self.shapes(frame, ['fvg'])
                self.assertEqual(len(shapes), 1)
                shape = shapes[0]
                self.assertEqual((shape.type, shape.x0, shape.x1, shape.y0, shape.y1), ('rect', frame.index[2], frame.index[7], low, high))
                self.assertEqual((shape.xref, shape.yref, shape.fillcolor), ('x', 'y', color))

    def test_actual_chart_renders_both_order_block_directions_and_clamps_end(self):
        for direction, low, high in [('bullish', 100, 102), ('bearish', 98, 100)]:
            with self.subTest(direction=direction):
                frame = ob_chain(direction)
                shapes = self.shapes(frame, ['order_block'])
                self.assertEqual(len(shapes), 1)
                shape = shapes[0]
                self.assertEqual((shape.x0, shape.x1, shape.y0, shape.y1), (frame.index[23], frame.index[25], low, high))
                self.assertEqual((shape.xref, shape.yref, shape.line.width), ('x', 'y', 2))

    def test_chart_mixed_selection_renders_both_zones_once(self):
        frame = ob_chain('bullish')
        shapes = self.shapes(frame, ['fvg', 'order_block'])
        self.assertEqual([(s.x0, s.y0, s.y1) for s in shapes], [(frame.index[25], 102, 103), (frame.index[23], 100, 102)])

    def test_order_block_chart_keeps_ten_bar_width_and_direction_colors(self):
        for direction, color in [('bullish', 'rgba(33, 150, 243, 0.15)'), ('bearish', 'rgba(255, 152, 0, 0.15)')]:
            with self.subTest(direction=direction):
                frame = ob_chain(direction)
                tail = candles([(100, 101, 99, 100)] * 10, start=frame.index[-1] + pd.Timedelta(days=1))
                frame = pd.concat([frame, tail])
                shapes = self.shapes(frame, ['order_block'])
                self.assertEqual(len(shapes), 1)
                self.assertEqual((shapes[0].x0, shapes[0].x1, shapes[0].fillcolor), (frame.index[23], frame.index[33], color))

    def test_disabled_or_empty_patterns_draw_no_zones(self):
        self.assertEqual(self.shapes(ob_chain('bullish'), ['fvg', 'order_block'], dict(fvg=False, order_block=False)), [])
        self.assertEqual(self.shapes(candles([(100, 101, 99, 100)] * 25), ['fvg', 'order_block']), [])

    def test_chart_does_not_invent_a_new_filled_zone_filter(self):
        frame = gap_frame('bullish', 4, filled=True)
        shapes = self.shapes(frame, ['fvg'])
        self.assertEqual(len(shapes), 1)
        self.assertEqual((shapes[0].x0, shapes[0].x1, shapes[0].y0, shapes[0].y1), (frame.index[2], frame.index[3], 101, 103))
