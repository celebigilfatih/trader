"""Offline numeric feature and training contracts for the real ML helper.

Training uses the existing LinearRegression model as an explicit fast subset
and a real sklearn constant baseline for nonzero metric errors; this is not an
integration test of the four default estimators. The target boundary is
supplied independently to isolate training metrics. Native target and four-model
integration are tested separately in test_ml_contract under Accepted ADR-0007.
Feature preparation, cleaning, both scalers, fitting, prediction and metrics
remain real. Fit-failure doubles are used only in the isolation tests.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from modules.ml_predictor import MLPredictor


def prices(count=101):
    close = np.arange(100.0, 100.0 + count)
    return pd.DataFrame(
        {"Open": close - 2, "High": close + 5, "Low": close - 5,
         "Close": close, "Volume": 1000.0 + 10 * np.arange(count)},
        index=pd.date_range("2026-01-01", periods=count, name="session"),
    )


class RecordingLinearRegression(LinearRegression):
    """A real estimator retaining only its numerical consumer inputs."""

    def fit(self, x, y):
        self.fit_x = np.array(x, copy=True)
        self.fit_y = np.array(y, copy=True)
        self.predict_inputs = []
        return super().fit(x, y)

    def predict(self, x):
        self.predict_inputs.append(np.array(x, copy=True))
        return super().predict(x)


class FailingFit:
    """Explicit estimator-boundary double, not a numerical model oracle."""

    def fit(self, x, y):
        raise ValueError("controlled fit failure")


class TestMLFeaturesTraining(unittest.TestCase):
    def setUp(self):
        for target in ("socket.create_connection", "socket.socket.connect",
                       "socket.socket.connect_ex", "modules.ml_predictor.joblib.dump",
                       "modules.ml_predictor.joblib.load"):
            guard = patch(target, side_effect=AssertionError("unexpected network/model I/O"))
            guard.start()
            self.addCleanup(guard.stop)
        self.predictor = MLPredictor()

    def training_case(self, valid_count=60, constant_target=None):
        """Independent known-label boundary; last future observation is absent."""
        data = prices(valid_count + 1)
        values = ([constant_target] * valid_count if constant_target is not None
                  else [0.01 + i / 1000 for i in range(valid_count)])
        target = pd.Series(values + [np.nan], index=data.index)
        model = RecordingLinearRegression()
        self.predictor.models = {"linear_regression": model}
        boundary = patch.object(self.predictor, "create_target_variable", return_value=target)
        boundary.start()
        self.addCleanup(boundary.stop)
        return data, target, model

    def test_price_and_volume_ratios_use_the_current_and_previous_rows(self):
        features = self.predictor.prepare_features(prices(), {})
        row = features.iloc[50]
        for name, expected in {"price_change": 1 / 149, "volume_change": 10 / 1490,
                               "high_low_ratio": 155 / 145,
                               "open_close_ratio": 148 / 150}.items():
            with self.subTest(feature=name):
                self.assertAlmostEqual(row[name], expected)

    def test_rolling_price_volume_and_sample_volatility_use_their_windows(self):
        row = self.predictor.prepare_features(prices(), {}).iloc[50]
        expected = {"sma_5": 148, "sma_20": 140.5, "sma_ratio": 148 / 140.5,
                    "trend_5": 148 / 150, "trend_20": 140.5 / 150,
                    "volatility_5": np.sqrt(2.5), "volatility_20": np.sqrt(35),
                    "volume_sma_5": 1480, "volume_sma_20": 1405,
                    "volume_ratio": 1500 / 1405}
        for name, value in expected.items():
            with self.subTest(feature=name):
                self.assertAlmostEqual(row[name], value)

    def test_all_lags_refer_to_past_prices_and_volumes(self):
        row = self.predictor.prepare_features(prices(), {}).iloc[50]
        for lag, close, volume in ((1, 149, 1490), (2, 148, 1480),
                                   (3, 147, 1470), (5, 145, 1450), (10, 140, 1400)):
            with self.subTest(lag=lag):
                self.assertEqual(row[f"close_lag_{lag}"], close)
                self.assertEqual(row[f"volume_lag_{lag}"], volume)
                self.assertAlmostEqual(row[f"return_lag_{lag}"], 150 / close - 1)

    def test_price_position_uses_twenty_row_high_low_range_and_flat_fallback(self):
        row = self.predictor.prepare_features(prices(), {}).iloc[50]
        # Rows 31..50: low=126, high=155, close=150.
        self.assertAlmostEqual(row["price_position"], 24 / 29)
        spiked = prices()
        spiked.loc[spiked.index[35], "High"] = 300.0
        # The older high remains in the 20-row window, outside the last 5 rows.
        self.assertAlmostEqual(self.predictor.prepare_features(spiked, {}).iloc[50]["price_position"],
                               24 / 174)
        flat = prices()
        flat[["Open", "High", "Low", "Close"]] = 100.0
        self.assertEqual(self.predictor.prepare_features(flat, {}).iloc[50]["price_position"], 0.5)

    def test_optional_indicators_use_independent_normalization_and_distances(self):
        data = prices()
        values = {name: pd.Series(value, index=data.index) for name, value in
                  {"rsi": 60, "macd": 3, "macd_signal": 1, "ema_20": 125,
                   "supertrend": 120, "supertrend_trend": -1,
                   "bb_middle": 140, "bb_upper": 180, "bb_lower": 100}.items()}
        row = self.predictor.prepare_features(data, values).iloc[50]
        expected = {"rsi_normalized": 0.6, "macd": 3, "macd_signal": 1,
                    "macd_signal_diff": 2, "ema_20_ratio": 1.2,
                    "ema_20_distance": 1 / 6, "supertrend_distance": 0.2,
                    "supertrend_signal": -1, "bb_position": 1 / 14,
                    "bb_middle": 140, "bb_upper": 180, "bb_lower": 100}
        for name, value in expected.items():
            with self.subTest(feature=name):
                self.assertIn(name, row.index)
                self.assertAlmostEqual(row[name], value)

    def test_missing_macd_signal_defaults_to_zero_and_unsupported_inputs_are_ignored(self):
        data = prices()
        features = self.predictor.prepare_features(data, {
            "macd": pd.Series(3.0, index=data.index), "rsi": [60] * len(data),
            "ema_short": pd.Series([120.0]), "unrecognized": pd.Series(8, index=data.index),
        })
        self.assertIn("macd_signal_diff", features)
        self.assertEqual(features.iloc[50]["macd_signal_diff"], 3)
        self.assertEqual(len(features.columns), 32)
        for absent in ("rsi_normalized", "ema_short_ratio", "unrecognized", "macd_signal"):
            self.assertNotIn(absent, features)

    def test_feature_preparation_clips_percentages_but_does_not_fit_quantiles(self):
        data = prices()
        data[["Open", "High", "Low", "Close"]] = 100.0
        data.loc[data.index[30:60], "Close"] = 2000.0
        data.loc[data.index[99:], "High"] = [300.0, 500.0]
        features = self.predictor.prepare_features(data, {})
        self.assertEqual(features.iloc[30]["price_change"], 10)
        self.assertAlmostEqual(features.iloc[60]["price_change"], -0.95)
        # Raw ratio survives; only fitted training bounds may clip it later.
        self.assertEqual(features.iloc[100]["high_low_ratio"], 5)
        self.assertTrue(np.isfinite(features.to_numpy()).all())

    def test_feature_generation_preserves_caller_and_aligns_series_by_index(self):
        data = prices()
        # Reversed storage order must still match each session label.
        rsi = pd.Series(np.arange(100.0, 201.0), index=data.index).iloc[::-1]
        indicators = {"rsi": rsi}
        original_data, original_rsi = data.copy(deep=True), rsi.copy(deep=True)
        features = self.predictor.prepare_features(data, indicators)
        self.assertAlmostEqual(features.iloc[40]["rsi_normalized"], 1.4)
        pd.testing.assert_index_equal(features.index, data.index)
        pd.testing.assert_frame_equal(data, original_data)
        pd.testing.assert_series_equal(rsi, original_rsi)
        self.assertIs(indicators["rsi"], rsi)

    def test_training_rejects_forty_nine_known_labels_before_scaler_or_fit(self):
        data, _, model = self.training_case(valid_count=49)
        result = self.predictor.train_models(data, {})
        self.assertIn("error", result)
        self.assertIn("En az 50", result["error"])
        self.assertEqual(result["available_data"], 49)
        self.assertFalse(self.predictor.is_trained)
        self.assertEqual(self.predictor.trained_models, {})
        self.assertFalse(hasattr(self.predictor.scaler, "mean_"))
        self.assertFalse(hasattr(model, "fit_x"))

    def test_training_accepts_fifty_known_labels_and_stores_feature_order(self):
        data, _, model = self.training_case(valid_count=50)
        result = self.predictor.train_models(data, {})
        self.assertNotIn("error", result)
        self.assertEqual((result["train_samples"], result["test_samples"]), (39, 10))
        self.assertEqual(result["feature_count"], 30)
        self.assertEqual(self.predictor.feature_names, [
            "price_change", "high_low_ratio", "open_close_ratio", "volume_change",
            "volatility_5", "volatility_20", "trend_5", "trend_20",
            "close_lag_1", "volume_lag_1", "return_lag_1",
            "close_lag_2", "volume_lag_2", "return_lag_2",
            "close_lag_3", "volume_lag_3", "return_lag_3",
            "close_lag_5", "volume_lag_5", "return_lag_5",
            "close_lag_10", "volume_lag_10", "return_lag_10",
            "sma_5", "sma_20", "sma_ratio", "volume_sma_5", "volume_sma_20",
            "volume_ratio", "price_position",
        ])
        self.assertIs(self.predictor.trained_models["linear_regression"], model)
        self.assertTrue(self.predictor.is_trained)

    def test_chronological_split_and_both_scalers_fit_only_training_prefix(self):
        data, _, model = self.training_case(valid_count=100)
        rsi = pd.Series([20.0] * 75 + [80.0] * 26, index=data.index)
        result = self.predictor.train_models(data, {"rsi": rsi}, test_size=0.25)
        self.assertNotIn("error", result)
        self.assertEqual((result["train_samples"], result["test_samples"]), (74, 25))
        self.assertEqual(self.predictor.scaler.n_samples_seen_, 74)
        self.assertEqual(self.predictor.target_scaler.n_samples_seen_, 74)
        rsi_column = self.predictor.feature_names.index("rsi_normalized")
        self.assertAlmostEqual(self.predictor.scaler.mean_[rsi_column], 0.2)
        np.testing.assert_allclose(model.fit_x[:, rsi_column], 0, atol=1e-14)
        np.testing.assert_allclose(model.predict_inputs[1][:, rsi_column], 0.6, atol=1e-14)
        np.testing.assert_allclose(model.fit_y, np.arange(74) / 73, atol=1e-14)
        np.testing.assert_allclose(self.predictor.target_scaler.data_min_, [0.01])
        np.testing.assert_allclose(self.predictor.target_scaler.data_max_, [0.083])
        self.assertEqual(model.fit_x.shape, (74, 31))

    def test_real_fit_prediction_inverse_scaling_and_metrics_on_constant_known_target(self):
        data, _, model = self.training_case(constant_target=0.025)
        result = self.predictor.train_models(data, {})
        self.assertNotIn("error", result)
        self.assertEqual(result["successful_models"], ["linear_regression"])
        np.testing.assert_allclose(model.fit_y, 0, atol=1e-14)
        for key, expected in {"train_score": 1, "test_score": 1, "rmse": 0, "mae": 0}.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(result[key], expected, places=12)
                self.assertAlmostEqual(result["models"]["linear_regression"][key], expected, places=12)

        # A real constant estimator predicts normalized zero, hence return .01.
        # Held-out targets are .058..069: errors .048..059, sum squares .034490,
        # mean absolute error .0535, target centered sum squares .000143.
        self.predictor = MLPredictor()
        data, _, _ = self.training_case()
        self.predictor.models = {"constant_baseline": DummyRegressor(strategy="constant", constant=0)}
        result = self.predictor.train_models(data, {})
        self.assertNotIn("error", result)
        for key, expected in {"test_score": 1 - 34490 / 143,
                              "rmse": np.sqrt(0.034490 / 12), "mae": 0.0535}.items():
            with self.subTest(nonzero_metric=key):
                self.assertAlmostEqual(result[key], expected, places=10)

    def test_failed_fit_is_reported_and_remaining_real_model_still_trains(self):
        data, _, model = self.training_case(constant_target=0.025)
        self.predictor.models = {"broken": FailingFit(), "linear_regression": model}
        result = self.predictor.train_models(data, {})
        self.assertNotIn("error", result)
        self.assertEqual(result["models"]["broken"], {"error": "controlled fit failure"})
        self.assertEqual(result["successful_models"], ["linear_regression"])
        self.assertEqual(list(self.predictor.trained_models), ["linear_regression"])
        self.assertTrue(self.predictor.is_trained)
        for metric in ("train_score", "test_score", "rmse", "mae"):
            self.assertEqual(result[metric], result["models"]["linear_regression"][metric])

    def test_all_fit_failures_leave_a_fresh_instance_untrained(self):
        data, _, _ = self.training_case()
        self.predictor.models = {"broken": FailingFit()}
        result = self.predictor.train_models(data, {})
        self.assertIn("Hiçbir model", result["error"])
        self.assertEqual(result["model_errors"], {"broken": {"error": "controlled fit failure"}})
        self.assertFalse(self.predictor.is_trained)
        self.assertEqual(self.predictor.trained_models, {})

    def test_training_preserves_prices_indicators_and_target(self):
        data, target, _ = self.training_case()
        rsi = pd.Series(60.0, index=data.index)
        originals = (data.copy(deep=True), target.copy(deep=True), rsi.copy(deep=True))
        self.assertNotIn("error", self.predictor.train_models(data, {"rsi": rsi}))
        pd.testing.assert_frame_equal(data, originals[0])
        pd.testing.assert_series_equal(target, originals[1])
        pd.testing.assert_series_equal(rsi, originals[2])

    def test_training_preparation_error_is_structured(self):
        result = self.predictor.train_models(prices().drop(columns="Close"), {})
        self.assertIn("Model eğitim hatası", result["error"])
        self.assertIn("Close", result["error"])
        self.assertFalse(self.predictor.is_trained)


if __name__ == "__main__":
    unittest.main()
