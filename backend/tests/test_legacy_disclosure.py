"""Render the actual legacy functions with Streamlit AppTest and isolated IO.

AST extraction avoids importing the monolithic app and its startup/network side
effects. Streamlit renders the unchanged function bodies; providers and ML are
controlled fixtures. This is not a full-app browser or live-provider test.
"""
import ast
import base64
import json
from pathlib import Path
import unittest

# Warm imports outside Streamlit's script timeout (first Plotly import can be slow).
import pandas
import numpy
import plotly.graph_objects

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[2] / "app.py"


def function_source(name):
    source = APP.read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


def news_item(title, score, time_status="valid"):
    return dict(title=title, description="Controlled RSS text", source="Fixture RSS",
                link="https://example.invalid/news", detected_symbols=["TEST.IS"],
                date="2026-10-01 10:00" if time_status == "valid" else "",
                published_at="2026-10-01T10:00:00+03:00" if time_status == "valid" else None,
                published_time_status=time_status, fetched_at="2026-10-01T07:01:00+00:00",
                sentiment_score=score, sentiment_label="Nötr" if score == 0 else "Pozitif",
                sentiment_status="failed" if score is None else "computed")


def source_diagnostic(source, status, error_code=None):
    return dict(source=source, status=status, attempts=1,
                elapsed_seconds=0.1, error_code=error_code)


def news_app(items, summary=None, list_diagnostics=None, summary_live_diagnostics=None):
    if summary is None:
        summary = dict(avg_sentiment=0.0, sentiment_label="Nötr", positive_news=0,
                       negative_news=0, neutral_news=1, total_news=len(items), scored_news=1,
                       failed_sentiment_count=1, top_symbols=[], last_updated="summary-clock")
    setup = f'''
import streamlit as st
import types
from unittest.mock import patch
ITEMS = {items!r}
SUMMARY = {summary!r}
LIST_DIAGNOSTICS = {list_diagnostics or []!r}
SUMMARY_LIVE_DIAGNOSTICS = {summary_live_diagnostics or []!r}
class FixtureSentiment:
    def get_market_news_summary(self):
        self.last_fetch_diagnostics = SUMMARY_LIVE_DIAGNOSTICS
        return SUMMARY
    def fetch_real_news(self, limit):
        self.last_fetch_diagnostics = LIST_DIAGNOSTICS
        return ITEMS
    def get_symbol_specific_news(self, symbol, limit):
        return self.fetch_real_news(limit)
sentiment_module = types.ModuleType("modules.sentiment_analyzer")
sentiment_module.SentimentAnalyzer = FixtureSentiment
config_module = types.ModuleType("modules.config")
config_module.BIST_SYMBOLS = {{"TEST.IS": "Fixture"}}
'''
    return AppTest.from_string(setup + function_source("show_news_feed") + '''
with patch.dict("sys.modules", {"modules.sentiment_analyzer": sentiment_module, "modules.config": config_module}):
    show_news_feed()
''').run(timeout=15)


def ml_app(spread=None):
    setup = f'''
import streamlit as st
import numpy as actual_np
import pandas as pd
import plotly.graph_objects as go
import types
from datetime import datetime, timedelta
from unittest.mock import patch
BIST_SYMBOLS = {{"TEST.IS": "Fixture"}}
class NumpyFixture:
    def __getattr__(self, name): return getattr(actual_np, name)
    def std(self, values):
        spread = {spread!r}
        return actual_np.std(values) if spread is None else float(spread)
np = NumpyFixture()
class BISTDataFetcher:
    def get_stock_data(self, *args, **kwargs):
        return pd.DataFrame({{key: [100.] * 120 for key in ["Open", "High", "Low", "Close", "Volume"]}}, index=pd.date_range("2026-01-01", periods=120))
class TechnicalAnalyzer:
    def __init__(self, data): self.indicators = {{}}
    def add_indicator(self, name): pass
class FixtureML:
    def prepare_features(self, *args): return pd.DataFrame({{"feature": [1.]}})
    def train_models(self, *args, **kwargs): return {{"successful_models": ["a", "b"]}}
    def predict_price(self, *args, **kwargs): return {{"a": 108., "b": 112.}}
    def get_feature_importance(self, *args): return pd.DataFrame()
ml_module = types.ModuleType("modules.ml_predictor")
ml_module.MLPredictor = FixtureML
'''
    return AppTest.from_string(setup + function_source("show_ai_predictions") + '''
with patch.dict("sys.modules", {"modules.ml_predictor": ml_module}):
    show_ai_predictions()
''').run(timeout=15)


def text_of(app):
    return "\n".join(str(e.value) for kind in ["markdown", "caption", "warning", "info", "error"] for e in app.get(kind))


def chart_values(trace):
    values = trace["y"]
    if isinstance(values, dict):
        return numpy.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
    return values


class LegacyNewsDisclosureTests(unittest.TestCase):
    def test_failed_score_remains_readable_and_unknown_dates_are_explicit(self):
        app = news_app([news_item("missing-story", None, "missing"), news_item("invalid-story", 0., "invalid")])
        text = text_of(app)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        for expected in ["RSS haberleri ve deneysel metin duygu analizi", "Duygu skoru, haber metninden hesaplanan deneysel", "missing-story", "invalid-story", "Yayın zamanı kaynakta belirtilmemiş.", "Yayın zamanı doğrulanamadı.", "Duygu skoru hesaplanamadı.", "Çekilme zamanı", "2026-10-01T07:01:00+00:00", "Özet oluşturulma zamanı", "summary-clock"]:
            self.assertIn(expected, text)
        self.assertNotIn("Gerçek zamanlı", text)

    def test_valid_publication_offset_is_visible(self):
        app = news_app([news_item("known-story", 0.)])
        self.assertIn("2026-10-01T10:00:00+03:00", text_of(app))

    def test_filters_exclude_failed_scores_but_keep_real_neutral(self):
        items = [news_item("failed-story", None), news_item("neutral-story", 0.), news_item("positive-story", .5), news_item("negative-story", -.5)]
        for choice, expected in [("Nötr", "neutral-story"), ("Pozitif", "positive-story"), ("Negatif", "negative-story")]:
            with self.subTest(choice=choice):
                app = news_app(items)
                app.sidebar.selectbox[1].select(choice).run()
                self.assertFalse(app.error)
                text = text_of(app)
                self.assertIn(expected, text)
                for item in items:
                    if item["title"] != expected:
                        self.assertNotIn(item["title"], text)

    def test_no_scored_summary_has_no_fake_neutral(self):
        summary = dict(avg_sentiment=None, sentiment_label=None, positive_news=0, negative_news=0, neutral_news=0, total_news=1, scored_news=0, failed_sentiment_count=1, top_symbols=[], last_updated="summary-clock")
        app = news_app([news_item("failed-story", None)], summary)
        self.assertFalse(app.error)
        self.assertIn("Özet için hesaplanmış duygu skoru yok.", text_of(app))
        self.assertIn("failed-story", text_of(app))
        self.assertIn("Özet oluşturulma zamanı", text_of(app))
        self.assertNotIn("Skor: 0.000", text_of(app))

    def test_no_news_does_not_invent_outage_reason(self):
        app = news_app([], {"error": "controlled-empty"})
        self.assertIn("Haber bulunamadı veya kaynaklara erişilemedi.", text_of(app))

    def test_summary_warning_uses_its_own_snapshot_with_successful_news_list(self):
        summary = dict(avg_sentiment=0., sentiment_label="Nötr", positive_news=0,
                       negative_news=0, neutral_news=1, total_news=1, scored_news=1,
                       failed_sentiment_count=0, top_symbols=[], last_updated="summary-clock",
                       source_diagnostics=[source_diagnostic("summary-source", "failed", "timeout")])
        app = news_app([news_item("available-story", 0.)], summary,
                       list_diagnostics=[source_diagnostic("list-source", "success")],
                       summary_live_diagnostics=[source_diagnostic("stale-source", "success")])
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        news_column, summary_column = app.get("column")[:2]
        warning = "Bazı haber kaynaklarına erişilemedi; sonuçlar eksik olabilir."
        self.assertEqual([e.value for e in app.warning], [warning])
        self.assertIn(warning, text_of(summary_column))
        self.assertIn("summary-source: Erişilemedi", text_of(summary_column))
        self.assertNotIn("stale-source", text_of(app))
        self.assertIn("list-source: Başarılı", text_of(news_column))
        self.assertIn("available-story", text_of(news_column))
        self.assertNotIn(warning, text_of(news_column))
        self.assertNotIn("list-source", text_of(summary_column))

    def test_list_warning_uses_its_own_fetch_including_symbol_filter(self):
        summary = {"error": "controlled-empty", "source_diagnostics": [source_diagnostic("summary-source", "empty")]}
        app = news_app([news_item("available-story", 0.)], summary,
                       list_diagnostics=[source_diagnostic("list-source", "failed", "connection_error")])
        for symbol in ["Tümü", "TEST.IS"]:
            with self.subTest(symbol=symbol):
                if symbol != "Tümü":
                    app.sidebar.selectbox[2].select(symbol).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                news_column, summary_column = app.get("column")[:2]
                warning = "Bazı haber kaynaklarına erişilemedi; sonuçlar eksik olabilir."
                self.assertEqual([e.value for e in app.warning], [warning])
                self.assertIn(warning, text_of(news_column))
                self.assertIn("list-source: Erişilemedi", text_of(news_column))
                self.assertIn("summary-source: Haber yok", text_of(summary_column))
                self.assertNotIn(warning, text_of(summary_column))
                self.assertNotIn("connection_error", text_of(app))

    def test_partial_sources_warn_without_hiding_usable_news_or_summary(self):
        summary = dict(avg_sentiment=0., sentiment_label="Nötr", positive_news=0,
                       negative_news=0, neutral_news=1, total_news=1, scored_news=1,
                       failed_sentiment_count=0, top_symbols=[], last_updated="summary-clock",
                       source_diagnostics=[source_diagnostic("summary-source", "partial", "parse_partial")])
        app = news_app([news_item("partial-story", 0.)], summary,
                       list_diagnostics=[source_diagnostic("list-source", "partial", "parse_partial")])
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        warning = "Bazı haber kaynaklarına erişilemedi; sonuçlar eksik olabilir."
        self.assertEqual([e.value for e in app.warning], [warning, warning])
        for expected in ["summary-source: Kısmi veri", "list-source: Kısmi veri", "partial-story", "Skor: 0.000"]:
            self.assertIn(expected, text_of(app))
        self.assertNotIn("parse_partial", text_of(app))

    def test_successful_or_empty_sources_do_not_warn_about_failure(self):
        for status in ["success", "empty"]:
            with self.subTest(status=status):
                summary = {"error": "controlled-empty", "source_diagnostics": [source_diagnostic("summary-source", status)]}
                app = news_app([], summary, list_diagnostics=[source_diagnostic("list-source", status)])
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertFalse(app.warning)
                label = "Başarılı" if status == "success" else "Haber yok"
                self.assertIn(f"summary-source: {label}", text_of(app))
                self.assertIn(f"list-source: {label}", text_of(app))

    def test_failed_sources_remain_visible_when_no_news_was_returned(self):
        summary = {"error": "controlled-empty", "source_diagnostics": [source_diagnostic("summary-source", "failed", "parse_failed")]}
        app = news_app([], summary, list_diagnostics=[source_diagnostic("list-source", "failed", "http_error")])
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.warning), 2)
        for expected in ["summary-source: Erişilemedi", "list-source: Erişilemedi", "Haber bulunamadı veya kaynaklara erişilemedi."]:
            self.assertIn(expected, text_of(app))


class LegacyMLDisclosureTests(unittest.TestCase):
    def test_warning_visible_before_model_run(self):
        app = ml_app()
        self.assertIn("Deneysel AI Tahminleri", text_of(app))
        self.assertIn("Deneysel model çıktısıdır. Yatırım tavsiyesi değildir; gelecekteki fiyatı veya getiriyi garanti etmez.", text_of(app))
        self.assertFalse(app.get("plotly_chart"))

    def test_valid_predictions_preserve_values_and_disclose_visual_scenarios(self):
        app = ml_app()
        app.button[0].click().run(timeout=15)
        self.assertFalse(app.error)
        self.assertFalse(app.exception)
        text = text_of(app)
        for expected in ["Son veri fiyatı", "Kaynak: Yahoo Finance", "Veri zamanı (son bar)", "Kullanılan veri aralığı", "2026-01-01", "2026-04-30", "Çekilme zamanı: Kaydedilmemiş", "Model uyumu (sezgisel)", "Model tahminlerinin birbirine yakınlığını özetler; doğruluk veya gerçekleşme olasılığı değildir.", "Üst yayılım senaryosu", "Model ortalaması", "Alt yayılım senaryosu", "₺112.00", "₺110.00", "₺108.00", "Güçlü Al", "95%", "Görsel senaryo — gerçekleşme olasılığı hesaplanmamıştır.", "Bant istatistiksel güven aralığı değildir."]:
            self.assertIn(expected, text)
        for forbidden in ["%30 Olasılık", "%40 Olasılık", "Canlı Piyasa", "AI Güveni"]:
            self.assertNotIn(forbidden, text)
        chart = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual([t["name"] for t in chart["data"]], ["Geçmiş Fiyat", "Şematik tahmin yolu", "Üst yayılım sınırı", "Model yayılımı (görsel)"])
        self.assertEqual(len(chart["data"][1]["x"]), 3)
        self.assertEqual(chart_values(chart["data"][1])[-1], 110.)
        self.assertEqual(chart_values(chart["data"][2])[-1], 112.)
        self.assertEqual(chart_values(chart["data"][3])[-1], 108.)

    def test_invalid_spread_hides_scenarios_and_band_without_five_percent_fallback(self):
        for spread in ["nan", "inf", "-1"]:
            with self.subTest(spread=spread):
                app = ml_app(spread)
                app.button[0].click().run(timeout=15)
                self.assertFalse(app.error)
                text = text_of(app)
                self.assertIn("Model yayılımı hesaplanamadı; senaryo ve bant gösterilemiyor.", text)
                self.assertIn("₺110.00", text)
                self.assertNotIn("Üst yayılım senaryosu", text)
                self.assertNotIn("Alt yayılım senaryosu", text)
                self.assertNotIn('class="metric-title">Model uyumu (sezgisel)', text)
                self.assertNotIn('class="metric-title">Risk Seviyesi', text)
                for chart in app.get("plotly_chart"):
                    self.assertNotIn("Model yayılımı (görsel)", chart.proto.spec)
                    self.assertEqual(len(json.loads(chart.proto.spec)["data"]), 2)


if __name__ == "__main__":
    unittest.main()
