"""Render the actual VWEMA assignment/card AST, not the full legacy app."""
import ast
from pathlib import Path
import unittest

import numpy  # Warm imports before Streamlit script timeout.
import pandas
from streamlit.testing.v1 import AppTest


def card_source():
    source = (Path(__file__).resolve().parents[2] / 'app.py').read_text(encoding='utf-8')
    dashboard = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name == 'show_modern_dashboard')
    assignment = next(node for node in ast.walk(dashboard)
                      if isinstance(node, ast.Assign) and any(
                          isinstance(target, ast.Name) and target.id == 'vwema_5' for target in node.targets))
    card = next(node for node in ast.walk(dashboard) if isinstance(node, ast.With) and any(
        isinstance(item.context_expr, ast.Subscript)
        and isinstance(item.context_expr.value, ast.Name) and item.context_expr.value.id == 'indicator_cols'
        and isinstance(item.context_expr.slice, ast.Constant) and item.context_expr.slice.value == 5
        for item in node.items))
    return ast.unparse(assignment) + '\n' + ast.unparse(card)


def render_card(indicators, current_price=110, real_zero_volume=False):
    setup = f'''
import streamlit as st
import numpy as np
import pandas as pd
import types
from modules.technical_analysis import TechnicalAnalyzer
current_price = {current_price!r}
indicator_cols = st.columns(6)
analyzer = types.SimpleNamespace(indicators={indicators})
'''
    if real_zero_volume:
        setup += '''
frame = pd.DataFrame({'Open': [100.] * 5, 'High': [101.] * 5,
                      'Low': [99.] * 5, 'Close': [100.] * 5, 'Volume': [0.] * 5})
analyzer = TechnicalAnalyzer(frame)
analyzer.add_indicator('vwema_5')
'''
    return AppTest.from_string(setup + card_source()).run(timeout=15)


class TestVstLegacyCard(unittest.TestCase):
    def assert_missing(self, app):
        self.assertFalse(app.exception)
        text = '\n'.join(element.value for element in app.markdown)
        self.assertIn('—', text)
        self.assertIn('Hesaplanamadı', text)
        for incorrect in ['₺0', '₺nan', '₺inf', 'Üzeri', 'Altı', '#00ff88', '#ff4757']:
            self.assertNotIn(incorrect, text)

    def test_none_nan_and_infinite_values_are_neutral(self):
        for value in ['None', 'np.nan', 'np.inf', '-np.inf']:
            with self.subTest(value=value):
                self.assert_missing(render_card("{'vwema_5': pd.Series([" + value + "])}"))

    def test_empty_or_missing_series_has_no_fabricated_zero(self):
        for indicators in ["{'vwema_5': pd.Series(dtype=float)}", '{}']:
            with self.subTest(indicators=indicators):
                self.assert_missing(render_card(indicators))

    def test_real_all_zero_volume_calculation_reaches_missing_card(self):
        self.assert_missing(render_card('{}', real_zero_volume=True))

    def test_finite_values_keep_currency_and_above_below_status(self):
        for price, status, color in [(110, 'Üzeri', '#00ff88'), (90, 'Altı', '#ff4757')]:
            with self.subTest(price=price):
                app = render_card("{'vwema_5': pd.Series([100.])}", current_price=price)
                self.assertFalse(app.exception)
                text = '\n'.join(element.value for element in app.markdown)
                for expected in ['₺100.00', status, color]:
                    self.assertIn(expected, text)
                self.assertNotIn('Hesaplanamadı', text)
