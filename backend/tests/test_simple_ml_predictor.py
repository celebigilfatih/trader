"""Offline numerical/lifecycle tests for SimpleMLPredictor.

Real forests train on synthetic prices. A separately named estimator boundary
double selects exact returns only for inference arithmetic and thresholds;
those cases do not claim to test sklearn's fitting or financial accuracy.
"""

from contextlib import ExitStack
from math import sqrt
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules.simple_ml_predictor import SimpleMLPredictor


BASE_COLUMNS = [
    "price_change", "volume_change", "high_low_ratio", "sma_5", "sma_20",
    "price_sma_ratio", "close_lag_1", "close_lag_3", "return_lag_1",
    "volatility",
]


def prices(closes, volumes=None):
    """Gapped, timezone-aware index also exercises label preservation."""
    values = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"Close": values, "High": values + 2, "Low": values - 2,
         "Volume": volumes if volumes is not None else np.arange(len(values)) + 100},
        index=pd.date_range("2026-01-05", periods=len(values), freq="2D", tz="UTC"),
    )


class ReturnEstimatorBoundary:
    """Inference-only double; exposes the actual frame received by predict."""

    def __init__(self, value):
        self.value = value
        self.received = None

    def predict(self, frame):
        self.received = frame.copy(deep=True)
        return np.array([self.value])


class TestSimpleMLPredictor(unittest.TestCase):
    def setUp(self):
        self.guards = ExitStack()
        self.addCleanup(self.guards.close)
        # Any accidental provider/email/model persistence call fails locally.
        for name in ("socket.create_connection", "socket.socket.connect",
                     "smtplib.SMTP", "smtplib.SMTP_SSL", "joblib.dump", "joblib.load"):
            self.guards.enter_context(patch(name, side_effect=AssertionError(name)))
        self.predictor = SimpleMLPredictor()

    def test_feature_schema_ratios_and_one_period_returns(self):
        # Reversing a ratio or using a two-period return must fail here.
        data = prices([10, 12, 9, 18], [100, 150, 120, 240])
        features = self.predictor.prepare_features(data, {})
        self.assertEqual(list(features), BASE_COLUMNS)
        np.testing.assert_allclose(features["price_change"], [0, .2, -.25, 1])
        np.testing.assert_allclose(features["volume_change"], [0, .5, -.2, 1])
        np.testing.assert_allclose(features["high_low_ratio"], [1.5, 1.4, 11 / 7, 1.25])
        np.testing.assert_allclose(features["return_lag_1"], [0, .2, -.25, 1])

    def test_lags_are_past_prices_and_warmup_is_zero(self):
        features = self.predictor.prepare_features(prices([10, 12, 9, 18, 15]), {})
        np.testing.assert_array_equal(features["close_lag_1"], [0, 10, 12, 9, 18])
        np.testing.assert_array_equal(features["close_lag_3"], [0, 0, 0, 10, 12])

    def test_rolling_windows_and_sample_standard_deviation(self):
        # End at 32 to distinguish rolling from expanding means and std(ddof=0).
        data = prices(list(range(10, 33)))
        features = self.predictor.prepare_features(data, {})
        self.assertEqual(features["sma_5"].iloc[3], 0)
        self.assertEqual(features["sma_5"].iloc[4], 12)
        self.assertEqual(features["sma_5"].iloc[-1], 30)
        self.assertEqual(features["sma_20"].iloc[18], 0)
        self.assertEqual(features["sma_20"].iloc[19], 19.5)
        self.assertEqual(features["sma_20"].iloc[-1], 22.5)
        self.assertEqual(features["price_sma_ratio"].iloc[18], 1)
        self.assertAlmostEqual(features["price_sma_ratio"].iloc[-1], 32 / 22.5)
        self.assertEqual(features["volatility"].iloc[8], 0)
        self.assertAlmostEqual(features["volatility"].iloc[9], sqrt(55 / 6))
        self.assertAlmostEqual(features["volatility"].iloc[-1], sqrt(55 / 6))

    def test_optional_indicators_have_independent_normalized_values_and_order(self):
        data = prices([10, 12, 15])
        indicators = {"rsi": pd.Series([20, 50, 80], index=data.index)}
        for name, denominator in (("ema_5", 5), ("ema_8", 6), ("ema_13", 10), ("ema_21", 15)):
            indicators[name] = pd.Series(denominator, index=data.index)
        indicators["vwap"] = pd.Series([8, 15, 10], index=data.index)
        features = self.predictor.prepare_features(data, indicators)
        self.assertEqual(list(features), BASE_COLUMNS[:6] + [
            "rsi", "ema_5_ratio", "ema_8_ratio", "ema_13_ratio", "ema_21_ratio",
            "vwap_ratio", "vwap_distance", *BASE_COLUMNS[6:],
        ])
        np.testing.assert_allclose(features["rsi"], [.2, .5, .8])
        for name, expected in (("ema_5_ratio", [2, 2.4, 3]),
                               ("ema_8_ratio", [10 / 6, 2, 2.5]),
                               ("ema_13_ratio", [1, 1.2, 1.5]),
                               ("ema_21_ratio", [2 / 3, .8, 1])):
            np.testing.assert_allclose(features[name], expected)
        np.testing.assert_allclose(features["vwap_ratio"], [1.25, .8, 1.5])
        np.testing.assert_allclose(features["vwap_distance"], [.2, -.25, 1 / 3])

    def test_nonfinite_features_forward_fill_and_safe_ratio_fallbacks(self):
        data = prices([10, 12, 15], [0, 100, 0])
        data.loc[data.index[1], "Low"] = 0
        indicators = {"rsi": pd.Series([np.nan, 50, np.inf], index=data.index),
                      "vwap": pd.Series([0, np.inf, 10], index=data.index)}
        features = self.predictor.prepare_features(data, indicators)
        np.testing.assert_allclose(features["volume_change"], [0, 0, -1])
        np.testing.assert_allclose(features["high_low_ratio"], [1.5, 1, 17 / 13])
        np.testing.assert_allclose(features["rsi"], [0, .5, .5])
        np.testing.assert_allclose(features["vwap_ratio"], [1, 0, 1.5])
        np.testing.assert_allclose(features["vwap_distance"], [1, 0, 1 / 3])
        self.assertTrue(np.isfinite(features.to_numpy()).all())

    def test_input_frames_indicators_and_index_are_preserved(self):
        data = prices([10, 12, 15])
        # RSI is label-aligned even when caller order differs.
        rsi = pd.Series([80, 20, 50], index=data.index[[2, 0, 1]])
        original, original_rsi = data.copy(deep=True), rsi.copy(deep=True)
        features = self.predictor.prepare_features(data, {"rsi": rsi})
        pd.testing.assert_frame_equal(data, original)
        pd.testing.assert_series_equal(rsi, original_rsi)
        pd.testing.assert_index_equal(features.index, original.index)
        np.testing.assert_allclose(features["rsi"], [.2, .5, .8])
        features.iloc[0, 0] = 999
        pd.testing.assert_frame_equal(data, original)

    def test_empty_feature_matrix_preserves_schema_without_inventing_rows(self):
        features = self.predictor.prepare_features(prices([]), {})
        self.assertTrue(features.empty)
        self.assertEqual(list(features), BASE_COLUMNS)
        pd.testing.assert_index_equal(features.index, prices([]).index)

    def test_training_requires_thirty_valid_next_day_targets(self):
        # 30 observations supply only 29 future returns, so no estimator fits.
        data = prices(range(10, 40))
        with patch.object(self.predictor.model, "fit", wraps=self.predictor.model.fit) as fit:
            result = self.predictor.train_quick_model(data, {})
        self.assertEqual(result, {"success": False, "error": "Yetersiz veri"})
        self.assertFalse(self.predictor.is_trained)
        fit.assert_not_called()

    def test_real_training_at_boundary_uses_chronological_split_and_next_day_target(self):
        data = prices(range(10, 41))
        original = data.copy(deep=True)
        with patch.object(self.predictor.model, "fit", wraps=self.predictor.model.fit) as fit, \
             patch.object(self.predictor.model, "predict", wraps=self.predictor.model.predict) as predict:
            result = self.predictor.train_quick_model(data, {})
        self.assertTrue(result["success"], result)
        self.assertTrue(self.predictor.is_trained)
        self.assertEqual((result["train_samples"], result["test_samples"]), (21, 9))
        train_frame, train_targets = fit.call_args.args
        test_frame = predict.call_args.args[0]
        pd.testing.assert_index_equal(train_frame.index, data.index[:21])
        pd.testing.assert_index_equal(train_targets.index, data.index[:21])
        pd.testing.assert_index_equal(test_frame.index, data.index[21:30])
        self.assertEqual(list(train_frame), BASE_COLUMNS)
        self.assertEqual(list(test_frame), BASE_COLUMNS)
        self.assertEqual(train_frame["close_lag_1"].iloc[-1], 29)
        self.assertEqual(test_frame["close_lag_1"].iloc[0], 30)
        np.testing.assert_allclose(train_targets, [1 / value for value in range(10, 31)])
        # Real estimator predictions are not a golden float sequence. Metrics
        # must agree with an independent target and explicit squared/absolute errors.
        held_out = self.predictor.model.predict(test_frame)
        expected_targets = [1 / value for value in range(31, 40)]
        squared = sum((a - b) ** 2 for a, b in zip(expected_targets, held_out)) / 9
        absolute = sum(abs(a - b) for a, b in zip(expected_targets, held_out)) / 9
        self.assertAlmostEqual(result["mse"], squared, places=14)
        self.assertAlmostEqual(result["mae"], absolute, places=14)
        self.assertEqual(len(self.predictor.model.estimators_), 50)
        pd.testing.assert_frame_equal(data, original)

    def test_real_automatic_training_and_prediction_on_constant_return_prices(self):
        # Every genuine training target is 2%; no sklearn output array is pinned.
        data = prices(100 * 1.02 ** np.arange(61))
        original = data.copy(deep=True)
        result = self.predictor.predict_next_day(data, {})
        self.assertNotIn("error", result)
        self.assertTrue(self.predictor.is_trained)
        self.assertAlmostEqual(result["current_price"], 100 * 1.02 ** 60)
        self.assertAlmostEqual(result["predicted_return"], 2)
        self.assertAlmostEqual(result["predicted_price"], 100 * 1.02 ** 61)
        self.assertAlmostEqual(result["price_change"], 2 * 1.02 ** 60)
        self.assertAlmostEqual(result["confidence"], .4)
        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(self.predictor.last_prediction, result)
        pd.testing.assert_frame_equal(data, original)

    def test_estimator_boundary_return_arithmetic_signals_and_confidence(self):
        data = prices([20, 25])
        # Exact boundary inputs: this tests the consumer, not forest fitting.
        cases = [(-.1, "SELL", .2), (-.010001, "SELL", .49999),
                 (-.01, "HOLD", .5), (0, "HOLD", .6), (.01, "HOLD", .5),
                 (.010001, "BUY", .49999), (.025, "BUY", .35), (.1, "BUY", .2)]
        for value, signal, confidence in cases:
            with self.subTest(value=value):
                model = ReturnEstimatorBoundary(value)
                self.predictor.model, self.predictor.is_trained = model, True
                result = self.predictor.predict_next_day(data, {})
                self.assertNotIn("error", result)
                self.assertEqual(result["signal"], signal)
                self.assertAlmostEqual(result["current_price"], 25)
                self.assertAlmostEqual(result["predicted_price"], 25 + 25 * value)
                self.assertAlmostEqual(result["price_change"], 25 * value)
                self.assertAlmostEqual(result["predicted_return"], value * 100)
                self.assertAlmostEqual(result["confidence"], confidence)
                pd.testing.assert_index_equal(model.received.index, data.index[-1:])
                self.assertEqual(list(model.received), BASE_COLUMNS)
                np.testing.assert_allclose(model.received.iloc[0], [
                    .25, .01, 27 / 23, 0, 0, 1, 20, 0, .25, 0,
                ])
                self.assertEqual(self.predictor.last_prediction, result)

    def test_automatic_training_failure_is_returned_without_prediction(self):
        with patch.object(self.predictor.model, "predict", side_effect=AssertionError("must not predict")):
            result = self.predictor.predict_next_day(prices([10, 11]), {})
        self.assertEqual(result, {"error": "Yetersiz veri"})
        self.assertIsNone(self.predictor.last_prediction)
        self.assertFalse(self.predictor.is_trained)

    def test_training_and_inference_exception_paths_are_isolated(self):
        missing_close = prices(range(10, 41)).drop(columns="Close")
        result = self.predictor.train_quick_model(missing_close, {})
        self.assertFalse(result["success"])
        self.assertIn("Close", result["error"])
        self.assertFalse(self.predictor.is_trained)
        self.predictor.is_trained = True
        with patch.object(self.predictor.model, "predict", side_effect=ValueError("synthetic estimator failure")):
            result = self.predictor.predict_next_day(prices([10, 11]), {})
        self.assertEqual(result, {"error": "synthetic estimator failure"})
        self.assertIsNone(self.predictor.last_prediction)

    def test_real_estimator_rejects_feature_schema_drift_on_inference(self):
        data = prices(range(10, 51))
        self.assertTrue(self.predictor.train_quick_model(data, {})["success"])
        result = self.predictor.predict_next_day(data, {"rsi": pd.Series(50, index=data.index)})
        self.assertIn("error", result)
        self.assertIn("feature", result["error"].lower())
        self.assertIsNone(self.predictor.last_prediction)

    def test_untrained_feature_importance_is_empty(self):
        self.assertTrue(self.predictor.get_feature_importance().empty)


if __name__ == "__main__":
    unittest.main()
