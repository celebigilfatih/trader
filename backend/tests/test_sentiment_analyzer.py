from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from modules.sentiment_analyzer import SentimentAnalyzer


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        instant = cls(2026, 11, 5, 12, 30, 45, tzinfo=timezone.utc)
        return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)


def _entry(title="Synthetic report", **fields):
    return {"title": title, "description": "Synthetic description",
            "link": "https://example.invalid/news", **fields}


def _news(title, score, symbols=()):
    return {
        **_entry(title), "source": "Fixture Rss", "date": "",
        "published_at": None, "published_time_status": "missing",
        "fetched_at": "2026-11-05T12:30:45+00:00",
        "sentiment_status": "computed" if score is not None else "failed",
        "sentiment_score": score,
        "sentiment_label": "Nötr" if score is not None else None,
        "detected_symbols": list(symbols),
    }


class TestSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SentimentAnalyzer()
        self.analyzer.news_sources = {"fixture_rss": "https://example.invalid/rss"}
        # Fail fast if a test accidentally bypasses the feed response double.
        self.network = patch("socket.socket.connect", side_effect=AssertionError("Unexpected network"))
        self.network_mock = self.network.start()
        self.addCleanup(self.network.stop)
        transport = patch("modules.sentiment_analyzer.requests.get",
                          return_value=Mock(status_code=200, content=b"synthetic feed", headers={}))
        self.transport = transport.start()
        self.addCleanup(transport.stop)
        sleeper = patch("modules.sentiment_analyzer.time.sleep",
                        side_effect=AssertionError("Unexpected sleep"))
        self.sleep = sleeper.start()
        self.addCleanup(sleeper.stop)
        clock = patch("modules.sentiment_analyzer.datetime", _FixedDatetime)
        clock.start()
        self.addCleanup(clock.stop)

    def tearDown(self):
        self.network_mock.assert_not_called()
        self.sleep.assert_not_called()

    def fetch(self, entries, scores=None):
        with patch("modules.sentiment_analyzer.feedparser.parse",
                   return_value=SimpleNamespace(entries=entries)), patch.object(
                       self.analyzer, "_analyze_text_sentiment",
                       side_effect=scores or [0.0] * len(entries)):
            return self.analyzer.fetch_real_news(limit=30)

    def test_missing_publication_never_becomes_fetch_time(self):
        for fields in ({}, {"published": ""}, {"published": None}):
            with self.subTest(fields=fields):
                item = self.fetch([_entry(**fields)])[0]
                self.assertEqual(item["date"], "")
                self.assertIsNone(item["published_at"])
                self.assertEqual(item["published_time_status"], "missing")
                self.assertEqual(item["fetched_at"], "2026-11-05T12:30:45+00:00")

    def test_invalid_or_ambiguous_publication_never_becomes_fetch_time(self):
        for published in ("not a date", "Thu, 05 Nov 2026 09:00:00",
                          "Thu, 05 Nov 2026 09:00:00 XYZ",
                          "Thu, 05 Nov 2026 09:00:00 -0000",
                          "Thu, 05 Nov 2026 09:00:00 -00:00", 123):
            with self.subTest(published=published):
                item = self.fetch([_entry(published=published)])[0]
                self.assertEqual(item["date"], "")
                self.assertIsNone(item["published_at"])
                self.assertEqual(item["published_time_status"], "invalid")

    def test_aware_publication_preserves_source_offset(self):
        for published, iso, display in (
            ("Thu, 05 Nov 2026 09:00:00 +0300", "2026-11-05T09:00:00+03:00", "2026-11-05 09:00"),
            ("Thu, 05 Nov 2026 09:00:00 +03:00", "2026-11-05T09:00:00+03:00", "2026-11-05 09:00"),
            ("Thu, 05 Nov 2026 09:00:00 GMT", "2026-11-05T09:00:00+00:00", "2026-11-05 09:00"),
        ):
            with self.subTest(published=published):
                item = self.fetch([_entry(published=published)])[0]
                self.assertEqual(item.get("published_at"), iso)
                self.assertEqual(item.get("published_time_status"), "valid")
                self.assertEqual(item["date"], display)

    def test_cross_source_sort_uses_instants_and_places_unknown_times_last(self):
        self.analyzer.news_sources["second_rss"] = "https://example.invalid/second"
        feeds = [SimpleNamespace(entries=[
            _entry("unknown"),
            _entry("earlier local", published="Thu, 05 Nov 2026 12:00:00 +0300"),
        ]), SimpleNamespace(entries=[
            _entry("later instant", published="Thu, 05 Nov 2026 10:00:00 +0000"),
            _entry("invalid", published="bad date"),
        ])]
        with patch("modules.sentiment_analyzer.feedparser.parse", side_effect=feeds), patch.object(
                self.analyzer, "_analyze_text_sentiment", return_value=0.0):
            result = self.analyzer.fetch_real_news(limit=10)
        self.assertEqual([n["title"] for n in result],
                         ["later instant", "earlier local", "unknown", "invalid"])

    def test_real_zero_is_computed_and_failed_score_keeps_real_news(self):
        result = self.fetch([_entry("THYAO neutral"), _entry("GARAN unscored")], [0.0, None])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["sentiment_score"], 0.0)
        self.assertEqual(result[0].get("sentiment_status"), "computed")
        self.assertEqual(result[0]["sentiment_label"], "Nötr")
        self.assertIsNone(result[1]["sentiment_score"])
        self.assertIsNone(result[1]["sentiment_label"])
        self.assertEqual(result[1]["sentiment_status"], "failed")
        self.assertEqual(result[1]["detected_symbols"], ["GARAN"])
        self.assertEqual(result[1]["link"], "https://example.invalid/news")

    def test_sentiment_helper_preserves_average_and_rounding_formula(self):
        self.analyzer.vader_analyzer = Mock()
        self.analyzer.vader_analyzer.polarity_scores.return_value = {"compound": 0.4567}
        with patch("modules.sentiment_analyzer.TextBlob", return_value=SimpleNamespace(
                sentiment=SimpleNamespace(polarity=-0.123))):
            self.assertEqual(self.analyzer._analyze_text_sentiment("synthetic"), 0.167)

    def test_sentiment_helper_failure_returns_none_instead_of_neutral(self):
        self.analyzer.vader_analyzer = Mock()
        self.analyzer.vader_analyzer.polarity_scores.side_effect = RuntimeError("synthetic failure")
        with patch("builtins.print"):
            self.assertIsNone(self.analyzer._analyze_text_sentiment("synthetic"))

    def test_textblob_failure_also_returns_none(self):
        self.analyzer.vader_analyzer = Mock()
        self.analyzer.vader_analyzer.polarity_scores.return_value = {"compound": 0.0}
        with patch("modules.sentiment_analyzer.TextBlob", side_effect=RuntimeError("synthetic failure")), patch("builtins.print"):
            self.assertIsNone(self.analyzer._analyze_text_sentiment("synthetic"))

    def test_nonfinite_calculation_is_failed_instead_of_a_computed_score(self):
        self.analyzer.vader_analyzer = Mock()
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.analyzer.vader_analyzer.polarity_scores.return_value = {"compound": value}
                with patch("modules.sentiment_analyzer.TextBlob", return_value=SimpleNamespace(
                        sentiment=SimpleNamespace(polarity=0.0))):
                    self.assertIsNone(self.analyzer._analyze_text_sentiment("synthetic"))

    def test_mixed_summary_counts_only_computed_scores_but_all_real_news(self):
        news = [_news("positive", 0.6, ["THYAO"]), _news("negative", -0.3),
                _news("neutral", 0.0), _news("failed", None, ["THYAO", "GARAN"])]
        with patch.object(self.analyzer, "fetch_real_news", return_value=news):
            result = self.analyzer.get_market_news_summary()
        self.assertEqual(result["total_news"], 4)
        self.assertEqual(result.get("scored_news"), 3)
        self.assertEqual(result.get("failed_sentiment_count"), 1)
        self.assertEqual(result["avg_sentiment"], 0.1)
        self.assertEqual(result["sentiment_label"], "Nötr")
        self.assertEqual([result[k] for k in ("positive_news", "negative_news", "neutral_news")], [1, 1, 1])
        self.assertEqual(result["top_symbols"], [{"symbol": "THYAO", "mentions": 2}, {"symbol": "GARAN", "mentions": 1}])
        self.assertEqual(result["recent_news"], news)
        self.assertEqual(result["last_updated"], "2026-11-05 12:30:45")

    def test_summary_preserves_existing_thresholds_including_boundaries(self):
        news = [_news("positive counted", 0.15), _news("negative counted", -0.15),
                _news("upper boundary", 0.1), _news("lower boundary", -0.1)]
        with patch.object(self.analyzer, "fetch_real_news", return_value=news):
            result = self.analyzer.get_market_news_summary()
        self.assertEqual([result[k] for k in ("positive_news", "negative_news", "neutral_news")], [1, 1, 2])
        self.assertEqual(result["avg_sentiment"], 0.0)
        self.assertEqual(result["sentiment_label"], "Nötr")

    def test_no_scored_news_has_no_average_or_label(self):
        with patch.object(self.analyzer, "fetch_real_news", return_value=[_news("failed", None)]):
            result = self.analyzer.get_market_news_summary()
        self.assertEqual(result["total_news"], 1)
        self.assertEqual(result.get("scored_news"), 0)
        self.assertEqual(result.get("failed_sentiment_count"), 1)
        self.assertIsNone(result["avg_sentiment"])
        self.assertIsNone(result["sentiment_label"])
        self.assertEqual([result[k] for k in ("positive_news", "negative_news", "neutral_news")], [0, 0, 0])
        self.assertEqual(len(result["recent_news"]), 1)

    def test_empty_summary_has_complete_nullable_contract(self):
        with patch.object(self.analyzer, "fetch_real_news", return_value=[]):
            result = self.analyzer.get_market_news_summary()
        self.assertEqual(result["total_news"], 0)
        self.assertEqual(result.get("scored_news"), 0)
        self.assertEqual(result.get("failed_sentiment_count"), 0)
        self.assertIsNone(result["avg_sentiment"])
        self.assertIsNone(result["sentiment_label"])
        self.assertEqual(result["recent_news"], [])
        self.assertEqual(result["top_symbols"], [])

    def test_symbol_filter_keeps_failed_news_and_respects_limit(self):
        news = [_news("other", 0.5), _news("thyao failed", None),
                _news("symbol metadata", 0.0, ["THYAO"]), _news("THYAO beyond limit", 0.6)]
        with patch.object(self.analyzer, "fetch_real_news", return_value=news):
            result = self.analyzer.get_symbol_specific_news("THYAO", limit=2)
        self.assertEqual([n["title"] for n in result], ["thyao failed", "symbol metadata"])
        self.assertIsNone(result[0]["sentiment_score"])
        self.assertEqual(result[0]["sentiment_status"], "failed")


if __name__ == "__main__":
    unittest.main()
