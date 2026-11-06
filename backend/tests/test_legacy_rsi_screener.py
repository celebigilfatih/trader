"""Legacy RSI with real constructor/calculator; only provider IO is injected."""
import ast
from pathlib import Path
import textwrap
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from backend.tests.test_technical_analysis import ohlcv
from modules.config import INDICATORS_CONFIG
from modules.data_fetcher import BISTDataFetcher
from modules.stock_screener import StockScreener


def rsi_button_source():
    source = (Path(__file__).resolve().parents[2] / 'app.py').read_text(encoding='utf-8')
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'show_stock_screener')
    button = next(n for n in ast.walk(function) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Call) and isinstance(n.test.func, ast.Attribute)
                  and n.test.func.attr == 'button' and any(k.arg == 'key' and isinstance(k.value, ast.Constant)
                                                          and k.value.value == 'rsi_scan_v2' for k in n.test.keywords))
    return ast.unparse(button)


def rsi_app(bounds=(65.625, 65.625), failing=False):
    setup = f'''
import streamlit as st
import pandas as pd
from unittest.mock import patch
from modules.data_fetcher import BISTDataFetcher
from modules.stock_screener import StockScreener
from backend.tests.test_technical_analysis import ohlcv
rsi_min, rsi_max = {bounds!r}
rsi_preset = 'Özel Aralık'
selected_interval = '15m'
def provider(self, symbol, **kwargs):
    st.session_state['last_provider_call'] = (symbol, kwargs)
    if {failing!r}:
        raise TimeoutError('Controlled source interruption')
    return ohlcv([100] * 21 + [102, 101])
screener = StockScreener({{'FIX.IS': 'Fixture'}})
'''
    return AppTest.from_string(setup + '''
try:
    with patch.object(BISTDataFetcher, 'get_stock_data', provider):
''' + textwrap.indent(rsi_button_source(), '        ') + '''
finally:
    screener.data_fetcher.session.close()
''').run(timeout=15)


class TestLegacyRsiScreener(unittest.TestCase):
    def setUp(self):
        for target in ['socket.socket.connect', 'socket.socket.connect_ex', 'socket.create_connection', 'time.sleep']:
            self.enterContext(patch(target, side_effect=AssertionError('Network/sleep forbidden')))

    def screener(self, frames):
        screener = StockScreener(dict((symbol, 'Name ' + symbol) for symbol in frames))
        self.addCleanup(screener.data_fetcher.session.close)
        def provider(symbol, **kwargs):
            frame = frames[symbol]
            if isinstance(frame, Exception):
                raise frame
            return frame
        boundary = self.enterContext(patch.object(screener.data_fetcher, 'get_stock_data', side_effect=provider))
        return screener, boundary

    def test_rational_rsi_bounds_are_inclusive_and_adjacent_bounds_exclude(self):
        # p22: after +2/-1, average-gain/loss ratio is21/11 => RSI65.625.
        frame = ohlcv([100] * 21 + [102, 101])
        before = frame.copy(deep=True)
        screener, _ = self.screener({'FIX.IS': frame})
        result = screener.screen_by_rsi(65.625, 65.625)
        self.assertEqual(result, [dict(symbol='FIX.IS', name='Name FIX.IS', rsi=65.625,
                                       current_price=101, signal='Overbought', interval='1d')])
        self.assertEqual(screener.screen_by_rsi(65.626, 100), [])
        self.assertEqual(screener.screen_by_rsi(0, 65.624), [])
        pd.testing.assert_frame_equal(frame, before)

    def test_results_sort_by_real_rsi_and_keep_zero_and_hundred(self):
        screener, _ = self.screener({'UP.IS': ohlcv(list(range(100, 140))),
                                    'MID.IS': ohlcv([100] * 21 + [102, 101]),
                                    'DOWN.IS': ohlcv(list(range(140, 100, -1)))})
        rows = screener.screen_by_rsi(0, 100)
        self.assertEqual([r['symbol'] for r in rows], ['DOWN.IS', 'MID.IS', 'UP.IS'])
        self.assertEqual([r['rsi'] for r in rows], [0., 65.625, 100.])
        self.assertEqual([r['signal'] for r in rows], ['Oversold', 'Overbought', 'Overbought'])

    def test_signal_labels_are_neutral_at_exact_35_and_65(self):
        # p2 after a flat seed: +14/-13 gives gain3.5/loss6.5 =>35;
        # +26/-7 gives gain6.5/loss3.5 =>65. Neither strict label cutoff fires.
        screener, _ = self.screener({'LOW.IS': ohlcv([100] * 21 + [114, 101]),
                                    'HIGH.IS': ohlcv([100] * 21 + [126, 119])})
        with patch.dict(INDICATORS_CONFIG['rsi'], period=2):
            rows = screener.screen_by_rsi(0, 100)
        self.assertEqual([(r['rsi'], r['signal']) for r in rows], [(35., 'Neutral'), (65., 'Neutral')])

    def test_interval_selects_provider_period_and_preserves_output_interval(self):
        screener, boundary = self.screener({'FIX.IS': ohlcv(list(range(100, 140)))})
        for interval, period in [('5m', '7d'), ('15m', '60d'), ('30m', '60d'), ('1h', '3mo'),
                                 ('4h', '6mo'), ('1d', '1y'), ('unmapped', '1y')]:
            with self.subTest(interval=interval):
                rows = screener.screen_by_rsi(100, 100, interval)
                boundary.assert_called_with('FIX.IS', period=period, interval=interval)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['interval'], interval)

    def test_none_empty_and_short_data_do_not_publish_unready_rsi(self):
        frames = {'NONE.IS': None, 'EMPTY.IS': ohlcv([])}
        frames.update({f'N{count}.IS': ohlcv(list(range(100, 100 + count))) for count in [20, 21, 22]})
        screener, boundary = self.screener(frames)
        rows = screener.screen_by_rsi(0, 100)
        self.assertEqual([r['symbol'] for r in rows], ['N22.IS'])
        self.assertEqual(rows[0]['rsi'], 100.)
        self.assertEqual(boundary.call_count, 5)

    def test_one_symbol_error_does_not_discard_a_later_real_match(self):
        screener, boundary = self.screener({'BROKEN.IS': TimeoutError('Controlled interruption'),
                                           'OK.IS': ohlcv([100] * 21 + [102, 101])})
        rows = screener.screen_by_rsi(60, 70)
        self.assertEqual([(r['symbol'], r['rsi']) for r in rows], [('OK.IS', 65.625)])
        self.assertEqual([call.args[0] for call in boundary.call_args_list], ['BROKEN.IS', 'OK.IS'])


class TestLegacyRsiButton(unittest.TestCase):
    def test_actual_button_displays_real_screener_result_and_forwards_interval(self):
        app = rsi_app()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 0)
        app.button[0].click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 1)
        row = app.dataframe[0].value.iloc[0]
        self.assertEqual((row['symbol'], row['rsi'], row['current_price'], row['interval']), ('FIX.IS', 65.625, 101, '15m'))
        self.assertEqual(app.session_state['last_provider_call'], ('FIX.IS', dict(period='60d', interval='15m')))
        self.assertTrue(app.success)

    def test_actual_button_no_match_displays_info_without_a_result_table(self):
        app = rsi_app((80, 90))
        app.button[0].click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertFalse(app.dataframe)
        self.assertFalse(app.success)
        self.assertTrue(app.info)
        self.assertIn('RSI aralığında hisse bulunamadı', app.info[0].value)
