"""Offline RSS failure isolation; article/ADR-0004 assertions survive ADR-0005."""

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import requests

from modules.sentiment_analyzer import SentimentAnalyzer


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        instant = cls(2026, 11, 5, 12, 30, 45, tzinfo=timezone.utc)
        return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)


def _entry(title="Synthetic retained report", **fields):
    return {
        "title": title,
        "description": "Synthetic fixture description",
        "link": "https://example.invalid/news/retained",
        "published": "Thu, 05 Nov 2026 09:00:00 +0300",
        **fields,
    }


def _feed(entries=(), **fields):
    return SimpleNamespace(entries=list(entries), status=200, bozo=False, **fields)


class TestRssFailurePaths(unittest.TestCase):
    def setUp(self):
        self.network = self._patch(
            "socket.socket.connect", side_effect=AssertionError("Unexpected network")
        )
        self.sleep = self._patch("modules.sentiment_analyzer.time.sleep")
        self._patch("modules.sentiment_analyzer.datetime", new=_FixedDatetime)
        self._patch("modules.sentiment_analyzer.SentimentIntensityAnalyzer")
        self.textblob = self._patch("modules.sentiment_analyzer.TextBlob")
        self._patch("builtins.print")
        self.parse = self._patch("modules.sentiment_analyzer.feedparser.parse")
        self.get = self._patch("modules.sentiment_analyzer.requests.get")
        self.get.side_effect = lambda url, **kwargs: self.response(url)
        self.responses = []
        self.analyzer = SentimentAnalyzer()
        self.analyzer.news_sources = {
            "first_fixture": "https://example.invalid/first/rss",
            "second_fixture": "https://example.invalid/second/rss",
        }
        self.sentiment = self._patch(
            "modules.sentiment_analyzer.SentimentAnalyzer._analyze_text_sentiment",
            return_value=0.0,
        )

    def _patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def tearDown(self):
        # The production loop catches exceptions, so assert that an accidental
        # network/sleep attempt did not merely get swallowed as a source error.
        self.network.assert_not_called()
        self.textblob.assert_not_called()
        for response in self.responses:
            response.close.assert_called_once_with()

    def response(self, url, status=200):
        response = Mock(status_code=status, content=url.encode(), headers={})
        self.responses.append(response)
        return response

    def fail_first_source(self, error):
        self.get.side_effect = [error] * 3 + [self.response("https://example.invalid/second/rss")]

    def assert_source_order(self):
        urls = [item.args[0] for item in self.get.call_args_list]
        ordered_sources = [url for i, url in enumerate(urls) if i == 0 or url != urls[i - 1]]
        self.assertEqual(ordered_sources, ["https://example.invalid/first/rss", "https://example.invalid/second/rss"])
        for item in self.get.call_args_list:
            self.assertEqual(item.kwargs, {"timeout": (8, 8)})
        for item in self.parse.call_args_list:
            self.assertIsInstance(item.args[0], bytes)

    def assert_retained_contract(self, item, entry, source="Second Fixture"):
        self.assertEqual(item["title"], entry["title"])
        self.assertEqual(item["description"], entry["description"])
        self.assertEqual(item["link"], entry["link"])
        self.assertEqual(item["source"], source)
        self.assertEqual(item["published_at"], "2026-11-05T09:00:00+03:00")
        self.assertEqual(item["published_time_status"], "valid")
        self.assertEqual(item["date"], "2026-11-05 09:00")
        self.assertEqual(item["fetched_at"], "2026-11-05T12:30:45+00:00")
        self.assertEqual(item["sentiment_status"], "computed")
        self.assertEqual(item["sentiment_score"], 0.0)
        self.assertEqual(item["sentiment_label"], "Nötr")
        self.assertEqual(item["detected_symbols"], [])

    def test_source_timeout_preserves_news_from_later_source(self):
        entry = _entry()
        self.fail_first_source(requests.Timeout("Synthetic timeout"))
        self.parse.return_value = _feed([entry])

        result = self.analyzer.fetch_real_news(limit=30)

        self.assert_source_order()
        self.assertEqual(len(result), 1)
        self.assert_retained_contract(result[0], entry)
        self.sentiment.assert_called_once_with(
            "Synthetic retained report Synthetic fixture description"
        )

    def test_source_connection_error_preserves_news_from_later_source(self):
        entry = _entry()
        self.fail_first_source(requests.ConnectionError("Synthetic disconnect"))
        self.parse.return_value = _feed([entry])

        result = self.analyzer.fetch_real_news(limit=30)

        self.assert_source_order()
        self.assertEqual(len(result), 1)
        self.assert_retained_contract(result[0], entry)
        self.assertEqual(self.sentiment.call_count, 1)

    def test_http_503_with_no_entries_does_not_abort_later_source(self):
        entry = _entry()
        self.get.side_effect = [self.response("first", 503) for _ in range(3)] + [
            self.response("https://example.invalid/second/rss")]
        self.parse.return_value = _feed([entry])

        result = self.analyzer.fetch_real_news(limit=30)

        self.assert_source_order()
        self.assertEqual(len(result), 1)
        self.assert_retained_contract(result[0], entry)
        self.assertEqual(self.sentiment.call_count, 1)

    def test_http_error_body_is_not_news_even_when_parser_finds_entries(self):
        retained = _entry()
        for status in (400, 403, 429, 500, 503):
            with self.subTest(status=status):
                self.parse.reset_mock()
                self.sentiment.reset_mock()
                self.get.reset_mock()
                count = 3 if status in (429, 500, 503) else 1
                error_body = b"Synthetic HTTP error body"
                failed = [self.response("https://example.invalid/first/rss", status) for _ in range(count)]
                for response in failed:
                    response.content = error_body
                self.get.side_effect = failed + [self.response("https://example.invalid/second/rss")]
                self.parse.side_effect = lambda body: _feed([
                    _entry("Synthetic HTTP error body") if body == error_body else retained])
                result = self.analyzer.fetch_real_news(limit=30)
                self.assert_source_order()
                self.assertEqual(len(result), 1)
                self.assert_retained_contract(result[0], retained)
                self.parse.assert_called_once_with(b"https://example.invalid/second/rss")
                self.sentiment.assert_called_once_with(
                    "Synthetic retained report Synthetic fixture description")

    def test_redirect_metadata_with_usable_entries_remains_compatible(self):
        redirected = _entry("Synthetic redirected report")
        retained = _entry()
        self.parse.side_effect = [
            SimpleNamespace(status=301, bozo=False, entries=[redirected]),
            _feed([retained]),
        ]
        result = self.analyzer.fetch_real_news(limit=30)
        self.assert_source_order()
        self.assertEqual(len(result), 2)
        self.assert_retained_contract(result[0], redirected, source="First Fixture")
        self.assert_retained_contract(result[1], retained)

    def test_bozo_parse_keeps_usable_entries_and_later_source_news(self):
        partial_entry = _entry("Synthetic partial report", published="invalid fixture time")
        later_entry = _entry()
        self.parse.side_effect = [
            SimpleNamespace(status=200, bozo=True,
                            bozo_exception=ValueError("Synthetic malformed feed"),
                            entries=[partial_entry]),
            _feed([later_entry]),
        ]

        result = self.analyzer.fetch_real_news(limit=30)

        self.assert_source_order()
        self.assertEqual([item["title"] for item in result],
                         [later_entry["title"], partial_entry["title"]])
        self.assert_retained_contract(result[0], later_entry)
        partial = result[1]
        self.assertEqual(partial["source"], "First Fixture")
        self.assertEqual(partial["link"], partial_entry["link"])
        self.assertIsNone(partial["published_at"])
        self.assertEqual(partial["published_time_status"], "invalid")
        self.assertEqual(partial["date"], "")
        self.assertEqual(partial["fetched_at"], "2026-11-05T12:30:45+00:00")
        self.assertEqual(partial["sentiment_status"], "computed")
        self.assertEqual(partial["sentiment_score"], 0.0)
        self.assertEqual(self.sentiment.call_count, 2)

    def test_bozo_parse_with_no_entries_does_not_abort_later_source(self):
        entry = _entry()
        self.parse.side_effect = [
            SimpleNamespace(status=200, bozo=True,
                            bozo_exception=ValueError("Synthetic malformed feed"), entries=[]),
            _feed([entry]),
        ]

        result = self.analyzer.fetch_real_news(limit=30)

        self.assert_source_order()
        self.assertEqual(len(result), 1)
        self.assert_retained_contract(result[0], entry)
        self.assertEqual(self.sentiment.call_count, 1)

    def test_all_sources_fail_returns_no_fabricated_news(self):
        self.get.side_effect = [requests.Timeout("Synthetic timeout")] * 3 + [
            requests.ConnectionError("Synthetic disconnect")] * 3

        result = self.analyzer.fetch_real_news(limit=30)

        self.assert_source_order()
        self.assertEqual(result, [])
        self.sentiment.assert_not_called()

    def test_all_sources_fail_summary_has_no_neutral_or_scored_observation(self):
        self.get.side_effect = [requests.Timeout("Synthetic timeout")] * 3 + [
            requests.ConnectionError("Synthetic disconnect")] * 3

        result = self.analyzer.get_market_news_summary()

        self.assert_source_order()
        self.assertNotIn("error", result)
        for field in ("total_news", "scored_news", "failed_sentiment_count",
                      "positive_news", "negative_news", "neutral_news"):
            self.assertEqual(result[field], 0, field)
        self.assertIsNone(result["avg_sentiment"])
        self.assertIsNone(result["sentiment_label"])
        self.assertEqual(result["recent_news"], [])
        self.assertEqual(result["top_symbols"], [])
        self.assertEqual(result["last_updated"], "2026-11-05 12:30:45")
        self.sentiment.assert_not_called()

    def test_partial_outage_with_nlp_failure_keeps_unscored_real_news(self):
        entry = _entry(published=None)
        self.fail_first_source(requests.Timeout("Synthetic timeout"))
        self.parse.return_value = _feed([entry])
        self.sentiment.return_value = None

        result = self.analyzer.get_market_news_summary()

        self.assert_source_order()
        self.assertEqual(result["total_news"], 1)
        self.assertEqual(result["failed_sentiment_count"], 1)
        for field in ("scored_news", "positive_news", "negative_news", "neutral_news"):
            self.assertEqual(result[field], 0, field)
        self.assertIsNone(result["avg_sentiment"])
        self.assertIsNone(result["sentiment_label"])
        self.assertEqual(len(result["recent_news"]), 1)
        retained = result["recent_news"][0]
        self.assertEqual(retained["title"], entry["title"])
        self.assertEqual(retained["description"], entry["description"])
        self.assertEqual(retained["link"], entry["link"])
        self.assertEqual(retained["source"], "Second Fixture")
        self.assertIsNone(retained["published_at"])
        self.assertEqual(retained["published_time_status"], "missing")
        self.assertEqual(retained["date"], "")
        self.assertEqual(retained["fetched_at"], "2026-11-05T12:30:45+00:00")
        self.assertEqual(retained["sentiment_status"], "failed")
        self.assertIsNone(retained["sentiment_score"])
        self.assertIsNone(retained["sentiment_label"])
        self.assertEqual(result["last_updated"], "2026-11-05 12:30:45")
        self.assertEqual(self.sentiment.call_count, 1)


if __name__ == "__main__":
    unittest.main()
