"""ADR-0005 transport and diagnostic contract, with no sockets or real sleeps."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import requests
from requests.structures import CaseInsensitiveDict

from modules.sentiment_analyzer import SentimentAnalyzer


def _response(status=200, body=b"synthetic feed", headers=None):
    return Mock(status_code=status, content=body,
                headers=CaseInsensitiveDict(headers or {}))


class TestRssTransport(unittest.TestCase):
    def setUp(self):
        self.network = self._patch("socket.socket.connect", side_effect=AssertionError("Unexpected network"))
        self.get = self._patch("modules.sentiment_analyzer.requests.get")
        self.sleep = self._patch("modules.sentiment_analyzer.time.sleep")
        self.clock = self._patch("modules.sentiment_analyzer.time.monotonic", return_value=100.0)
        self.parse = self._patch("modules.sentiment_analyzer.feedparser.parse",
                                 return_value=SimpleNamespace(entries=[], bozo=False))
        self._patch("modules.sentiment_analyzer.SentimentIntensityAnalyzer")
        self._patch("modules.sentiment_analyzer.SentimentAnalyzer._analyze_text_sentiment", return_value=0.0)
        self._patch("builtins.print")
        self.analyzer = SentimentAnalyzer()
        self.analyzer.news_sources = {"fixture_rss": "https://example.invalid/rss"}
        self.response = _response()
        self.get.return_value = self.response

    def _patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def tearDown(self):
        self.network.assert_not_called()

    def diagnostic(self, status, attempts, error_code):
        self.assertEqual(len(self.analyzer.last_fetch_diagnostics), 1)
        value = self.analyzer.last_fetch_diagnostics[0]
        self.assertEqual(set(value), {"source", "status", "attempts", "elapsed_seconds", "error_code"})
        self.assertEqual(value["source"], "fixture_rss")
        self.assertEqual(value["status"], status)
        self.assertEqual(value["attempts"], attempts)
        self.assertEqual(value["error_code"], error_code)
        self.assertGreaterEqual(value["elapsed_seconds"], 0)
        return value

    def test_success_fetches_bytes_once_with_explicit_connect_read_timeout(self):
        self.parse.return_value = SimpleNamespace(entries=[{"title": "THYAO synthetic"}], bozo=False)
        result = self.analyzer.fetch_real_news()
        self.assertEqual(len(result), 1)
        self.get.assert_called_once_with("https://example.invalid/rss", timeout=(8, 8))
        self.parse.assert_called_once_with(b"synthetic feed")
        self.response.close.assert_called_once_with()
        self.sleep.assert_not_called()
        self.diagnostic("success", 1, None)

    def test_normal_empty_parse_is_empty_without_error_code(self):
        self.assertEqual(self.analyzer.fetch_real_news(), [])
        self.diagnostic("empty", 1, None)
        self.response.close.assert_called_once_with()

    def test_successful_empty_body_is_parsed_as_bytes_and_can_be_empty(self):
        self.response.content = b""
        self.assertEqual(self.analyzer.fetch_real_news(), [])
        self.parse.assert_called_once_with(b"")
        self.diagnostic("empty", 1, None)

    def test_connect_read_and_generic_timeout_exhaust_exactly_three_attempts(self):
        for exception in (requests.ConnectTimeout, requests.ReadTimeout, requests.Timeout):
            with self.subTest(exception=exception):
                self.get.reset_mock()
                self.sleep.reset_mock()
                self.get.side_effect = exception("synthetic timeout")
                self.assertEqual(self.analyzer.fetch_real_news(), [])
                self.assertEqual(self.get.call_count, 3)
                self.assertEqual(self.sleep.call_args_list, [call(1.5), call(3.0)])
                self.parse.assert_not_called()
                self.diagnostic("failed", 3, "timeout")

    def test_connection_error_retries_then_success_clears_final_error(self):
        self.get.side_effect = [requests.ConnectionError("synthetic"), self.response]
        self.assertEqual(self.analyzer.fetch_real_news(), [])
        self.assertEqual(self.get.call_count, 2)
        self.sleep.assert_called_once_with(1.5)
        self.parse.assert_called_once_with(self.response.content)
        self.diagnostic("empty", 2, None)

    def test_connection_exhaustion_is_classified(self):
        self.get.side_effect = requests.ConnectionError("synthetic")
        self.analyzer.fetch_real_news()
        self.diagnostic("failed", 3, "connection_error")
        self.parse.assert_not_called()

    def test_all_retryable_statuses_retry_and_close_each_response(self):
        for status in (408, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                self.get.reset_mock()
                self.sleep.reset_mock()
                responses = [_response(status) for _ in range(3)]
                self.get.side_effect = responses
                self.assertEqual(self.analyzer.fetch_real_news(), [])
                self.assertEqual(self.get.call_count, 3)
                self.assertEqual(self.sleep.call_args_list, [call(1.5), call(3.0)])
                self.parse.assert_not_called()
                for response in responses:
                    response.close.assert_called_once_with()
                self.diagnostic("failed", 3, "http_error")

    def test_retryable_http_failure_then_success_parses_only_success_body(self):
        failed = _response(502, b"error body")
        self.get.side_effect = [failed, self.response]
        self.analyzer.fetch_real_news()
        self.parse.assert_called_once_with(b"synthetic feed")
        failed.close.assert_called_once_with()
        self.response.close.assert_called_once_with()
        self.sleep.assert_called_once_with(1.5)
        self.diagnostic("empty", 2, None)

    def test_retry_after_on_429_or_503_defers_even_empty_or_invalid_header(self):
        for status in (429, 503):
            for header in ("5", "Thu, 05 Nov 2026 12:30:45 GMT", "invalid", ""):
                with self.subTest(status=status, header=header):
                    self.get.reset_mock()
                    response = _response(status, headers={"retry-after": header})
                    self.get.return_value = response
                    self.assertEqual(self.analyzer.fetch_real_news(), [])
                    self.get.assert_called_once()
                    response.close.assert_called_once_with()
                    self.sleep.assert_not_called()
                    self.parse.assert_not_called()
                    self.diagnostic("failed", 1, "retry_after_deferred")

    def test_retry_after_on_other_retryable_status_does_not_change_policy(self):
        response = _response(500, headers={"Retry-After": "5"})
        self.get.side_effect = [response, self.response]
        self.analyzer.fetch_real_news()
        self.assertEqual(self.get.call_count, 2)
        self.sleep.assert_called_once_with(1.5)
        self.diagnostic("empty", 2, None)

    def test_nonretryable_http_failures_and_unfollowed_redirect_never_parse(self):
        for status in (301, 304, 400, 401, 403, 404, 410, 422, 501, 505):
            with self.subTest(status=status):
                self.get.reset_mock()
                response = _response(status, b"not news")
                self.get.return_value = response
                self.assertEqual(self.analyzer.fetch_real_news(), [])
                self.get.assert_called_once()
                self.parse.assert_not_called()
                self.sleep.assert_not_called()
                response.close.assert_called_once_with()
                self.diagnostic("failed", 1, "http_error")

    def test_redirect_history_uses_final_response_status(self):
        self.response.history = [_response(301)]
        self.response.status_code = 403
        self.analyzer.fetch_real_news()
        self.parse.assert_not_called()
        self.response.close.assert_called_once_with()
        self.diagnostic("failed", 1, "http_error")

    def test_followed_redirect_success_parses_final_body(self):
        self.response.history = [_response(301)]
        self.analyzer.fetch_real_news()
        self.parse.assert_called_once_with(b"synthetic feed")
        self.diagnostic("empty", 1, None)

    def test_unclassified_fetch_errors_do_not_retry_or_parse(self):
        for error in (requests.TooManyRedirects("synthetic"), requests.RequestException("synthetic"), ValueError("synthetic")):
            with self.subTest(error=type(error)):
                self.get.reset_mock()
                self.get.side_effect = error
                self.analyzer.fetch_real_news()
                self.get.assert_called_once()
                self.parse.assert_not_called()
                self.sleep.assert_not_called()
                self.diagnostic("failed", 1, "fetch_failed")

    def test_response_body_error_closes_response_without_parser_or_retry(self):
        type(self.response).content = property(lambda response: (_ for _ in ()).throw(ValueError("synthetic")))
        self.analyzer.fetch_real_news()
        self.response.close.assert_called_once_with()
        self.get.assert_called_once()
        self.parse.assert_not_called()
        self.sleep.assert_not_called()
        self.diagnostic("failed", 1, "fetch_failed")

    def test_parser_exception_does_not_retry_transport(self):
        self.parse.side_effect = ValueError("synthetic parser exception")
        self.assertEqual(self.analyzer.fetch_real_news(), [])
        self.get.assert_called_once()
        self.response.close.assert_called_once_with()
        self.sleep.assert_not_called()
        self.diagnostic("failed", 1, "parse_failed")

    def test_response_close_error_keeps_failed_diagnostic_and_later_source(self):
        self.analyzer.news_sources["later_rss"] = "https://example.invalid/later"
        failed = _response()
        failed.close.side_effect = OSError("synthetic close failure")
        self.get.side_effect = [failed, self.response]
        self.analyzer.fetch_real_news()
        self.assertEqual([(d["source"], d["status"], d["attempts"], d["error_code"])
                          for d in self.analyzer.last_fetch_diagnostics],
                         [("fixture_rss", "failed", 1, "fetch_failed"), ("later_rss", "empty", 1, None)])
        self.parse.assert_called_once_with(b"synthetic feed")
        self.sleep.assert_not_called()

    def test_bozo_usable_entries_are_partial_and_nlp_failure_is_independent(self):
        self.parse.return_value = SimpleNamespace(entries=[{"title": "THYAO synthetic"}], bozo=True)
        with patch.object(self.analyzer, "_analyze_text_sentiment", return_value=None):
            result = self.analyzer.fetch_real_news()
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0]["sentiment_score"])
        self.assertEqual(result[0]["sentiment_status"], "failed")
        self.diagnostic("partial", 1, "parse_partial")
        self.get.assert_called_once()
        self.sleep.assert_not_called()

    def test_bozo_empty_parse_is_failed_and_not_retried(self):
        self.parse.return_value = SimpleNamespace(entries=[], bozo=True)
        self.assertEqual(self.analyzer.fetch_real_news(), [])
        self.diagnostic("failed", 1, "parse_failed")
        self.get.assert_called_once()
        self.sleep.assert_not_called()

    def test_elapsed_uses_monotonic_and_includes_backoff(self):
        current = [100.0]
        def get(*args, **kwargs):
            current[0] += 2
            if self.get.call_count < 3:
                raise requests.Timeout("synthetic")
            return self.response
        self.get.side_effect = get
        self.clock.side_effect = lambda: current[0]
        self.sleep.side_effect = lambda delay: current.__setitem__(0, current[0] + delay)
        self.analyzer.fetch_real_news()
        self.assertEqual(self.diagnostic("empty", 3, None)["elapsed_seconds"], 10.5)

    def test_diagnostics_reset_and_summary_copies_each_record_from_same_fetch(self):
        summary = self.analyzer.get_market_news_summary()
        self.get.assert_called_once()
        self.assertEqual(summary.get("source_diagnostics"), self.analyzer.last_fetch_diagnostics)
        self.assertIsNot(summary["source_diagnostics"], self.analyzer.last_fetch_diagnostics)
        self.assertIsNot(summary["source_diagnostics"][0], self.analyzer.last_fetch_diagnostics[0])
        self.analyzer.last_fetch_diagnostics[0]["status"] = "mutated"
        self.assertEqual(summary["source_diagnostics"][0]["status"], "empty")
        self.get.side_effect = requests.Timeout("synthetic")
        self.analyzer.fetch_real_news()
        self.diagnostic("failed", 3, "timeout")
        self.assertEqual(summary["source_diagnostics"][0]["status"], "empty")
        self.analyzer.news_sources = {}
        self.assertEqual(self.analyzer.fetch_real_news(), [])
        self.assertEqual(self.analyzer.last_fetch_diagnostics, [])

    def test_failed_source_does_not_block_next_source_and_records_both(self):
        self.analyzer.news_sources["later_rss"] = "https://example.invalid/later"
        self.get.side_effect = [_response(403), self.response]
        self.parse.return_value = SimpleNamespace(entries=[{"title": "retained synthetic"}], bozo=False)
        result = self.analyzer.fetch_real_news()
        self.assertEqual([item["title"] for item in result], ["retained synthetic"])
        self.assertEqual([(d["source"], d["status"], d["attempts"], d["error_code"])
                          for d in self.analyzer.last_fetch_diagnostics],
                         [("fixture_rss", "failed", 1, "http_error"), ("later_rss", "success", 1, None)])
        self.parse.assert_called_once_with(b"synthetic feed")


if __name__ == "__main__":
    unittest.main()
