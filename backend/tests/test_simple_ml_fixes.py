"""ML-04–06 regressions: real RF lifecycle/names, explicit numeric boundary.

Boundary returns isolate output validity/cache handling; they do not test RF
fitting. Genuine training/importance and constant-return prediction run separately.
"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules.simple_ml_predictor import SimpleMLPredictor
from backend.tests.test_simple_ml_predictor import BASE_COLUMNS, ReturnEstimatorBoundary, prices


class TestSimpleMLFixes(unittest.TestCase):
    def setUp(self):
        for name in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection",
                     "smtplib.SMTP", "smtplib.SMTP_SSL", "joblib.load", "joblib.dump"):
            guard = patch(name, side_effect=AssertionError(name))
            guard.start()
            self.addCleanup(guard.stop)
        self.p = SimpleMLPredictor()
        self.data = prices(range(100, 141))

    def fitted_with_cached_prediction(self):
        prediction = self.p.predict_next_day(self.data, {})
        self.assertNotIn("error", prediction)
        self.assertTrue(self.p.is_trained)
        self.assertIsNotNone(self.p.last_prediction)

    def assert_training_invalidated(self, result):
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertFalse(self.p.is_trained)
        self.assertIsNone(self.p.last_prediction)
        self.assertTrue(self.p.get_feature_importance().empty)

    def boundary(self, value=.025):
        self.p.model = ReturnEstimatorBoundary(value)
        self.p.is_trained = True

    def test_short_retrain_clears_old_trained_state_cache_and_importance(self):
        self.fitted_with_cached_prediction()
        self.assert_training_invalidated(self.p.train_quick_model(prices([100, 101]), {}))
        with patch.object(self.p.model, "predict", side_effect=AssertionError("old model used")):
            self.assertEqual(self.p.predict_next_day(prices([100, 101]), {}), {"error": "Yetersiz veri"})

    def test_preparation_error_during_retrain_invalidates_old_training(self):
        self.fitted_with_cached_prediction()
        self.assert_training_invalidated(self.p.train_quick_model(self.data.drop(columns="Close"), {}))

    def test_fit_error_during_retrain_invalidates_old_training(self):
        self.fitted_with_cached_prediction()
        with patch.object(self.p.model, "fit", side_effect=ValueError("controlled fit failure")):
            result = self.p.train_quick_model(self.data, {})
        self.assert_training_invalidated(result)
        self.assertIn("controlled fit failure", result["error"])

    def test_metric_prediction_error_during_retrain_invalidates_old_training(self):
        self.fitted_with_cached_prediction()
        with patch.object(self.p.model, "predict", side_effect=ValueError("controlled metric failure")):
            result = self.p.train_quick_model(self.data, {})
        self.assert_training_invalidated(result)

    def test_nonfinite_training_metrics_cannot_publish_success(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.fitted_with_cached_prediction()
                # RF fit remains real; only metric prediction boundary is invalid.
                with patch.object(self.p.model, "predict", side_effect=lambda x: np.full(len(x), value)):
                    result = self.p.train_quick_model(self.data, {})
                self.assert_training_invalidated(result)

    def test_bare_feature_importance_keeps_all_ten_actual_feature_names_and_weights(self):
        self.assertTrue(self.p.train_quick_model(self.data, {})["success"])
        result = self.p.get_feature_importance()
        self.assertEqual(list(result.columns), ["feature", "importance"])
        self.assertEqual(set(result["feature"]), set(BASE_COLUMNS))
        self.assertEqual(len(result), 10)
        # RF numeric weights are consumed in the independent literal schema order.
        np.testing.assert_array_equal(result.set_index("feature").loc[BASE_COLUMNS, "importance"],
                                      self.p.model.feature_importances_)
        self.assertTrue(result["importance"].is_monotonic_decreasing)

    def test_partial_optional_feature_importance_uses_exact_trained_order(self):
        for indicators, optional in (({"rsi": 60.}, ["rsi"]),
                                     ({"ema_8": 120.}, ["ema_8_ratio"]),
                                     ({"vwap": 120.}, ["vwap_ratio", "vwap_distance"]),
                                     ({"rsi": 60., "vwap": 120.}, ["rsi", "vwap_ratio", "vwap_distance"])):
            with self.subTest(optional=optional):
                values = {k: pd.Series(v, index=self.data.index) for k, v in indicators.items()}
                self.assertTrue(self.p.train_quick_model(self.data, values)["success"])
                result = self.p.get_feature_importance()
                expected = BASE_COLUMNS[:6] + optional + BASE_COLUMNS[6:]
                self.assertEqual(set(result["feature"]), set(expected))
                self.assertEqual(len(result), len(expected))
                np.testing.assert_array_equal(result.set_index("feature").loc[expected, "importance"],
                                              self.p.model.feature_importances_)

    def test_successful_retrain_clears_cache_and_replaces_optional_feature_names(self):
        rsi = {"rsi": pd.Series(60., index=self.data.index)}
        self.assertNotIn("error", self.p.predict_next_day(self.data, rsi))
        self.assertTrue(self.p.train_quick_model(self.data, {})["success"])
        self.assertTrue(self.p.is_trained)
        self.assertIsNone(self.p.last_prediction)
        self.assertEqual(set(self.p.get_feature_importance()["feature"]), set(BASE_COLUMNS))
        self.assertNotIn("error", self.p.predict_next_day(self.data, {}))
        self.assertIsNotNone(self.p.last_prediction)

    def test_invalid_importance_metadata_does_not_return_truncated_or_invented_names(self):
        self.assertTrue(self.p.train_quick_model(self.data, {})["success"])
        for invalid in ([], ["wrong"], ["same"] * 10):
            with self.subTest(invalid=invalid), patch.object(self.p.model, "feature_names_in_", np.array(invalid)):
                self.assertTrue(self.p.get_feature_importance().empty)

    def test_nonfinite_return_is_error_and_clears_previous_prediction(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.boundary()
                self.assertNotIn("error", self.p.predict_next_day(prices([20, 25]), {}))
                self.p.model.value = value
                result = self.p.predict_next_day(prices([20, 25]), {})
                self.assertIn("error", result)
                self.assertNotIn("signal", result)
                self.assertIsNone(self.p.last_prediction)
                self.assertTrue(self.p.is_trained)

    def test_invalid_current_price_is_error_and_clears_previous_prediction(self):
        for value in (np.nan, np.inf, -np.inf, 0., -1.):
            with self.subTest(value=value):
                self.boundary()
                self.assertNotIn("error", self.p.predict_next_day(prices([20, 25]), {}))
                self.assertIn("error", self.p.predict_next_day(prices([20, value]), {}))
                self.assertIsNone(self.p.last_prediction)

    def test_nonfinite_derived_values_and_nonpositive_predicted_price_are_errors(self):
        for current, value in ((1e308, 1.), (1e-310, 1e308), (25., -1.), (25., -2.)):
            with self.subTest(current=current, value=value):
                self.boundary(value)
                result = self.p.predict_next_day(prices([20., current]), {})
                self.assertIn("error", result)
                self.assertIsNone(self.p.last_prediction)

    def test_estimator_exception_after_success_clears_cache_and_allows_retry(self):
        self.boundary()
        data = prices([20, 25])
        self.assertNotIn("error", self.p.predict_next_day(data, {}))
        with patch.object(self.p.model, "predict", side_effect=ValueError("controlled inference failure")):
            self.assertEqual(self.p.predict_next_day(data, {}), {"error": "controlled inference failure"})
        self.assertIsNone(self.p.last_prediction)
        self.assertTrue(self.p.is_trained)
        self.assertNotIn("error", self.p.predict_next_day(data, {}))


if __name__ == "__main__":
    unittest.main()
