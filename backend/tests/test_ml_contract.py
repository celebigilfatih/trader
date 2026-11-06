"""Accepted ADR-0007 numerical contracts; no forecast accuracy claim.

Native targets/four estimators run separately from the explicit constant-label
boundary used to isolate inverse scaling. Network and persistence are blocked.
"""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from modules.ml_predictor import MLPredictor
from backend.tests.test_ml_features_training import prices, RecordingLinearRegression, FailingFit


class PredictFailure(LinearRegression):
    def predict(self, x):
        raise ValueError("controlled metric prediction failure")


class TestMLContract(unittest.TestCase):
    def setUp(self):
        for name in ("socket.socket.connect", "socket.socket.connect_ex",
                     "socket.create_connection", "joblib.dump", "joblib.load"):
            guard = patch(name, side_effect=AssertionError(name))
            guard.start()
            self.addCleanup(guard.stop)
        self.p = MLPredictor()
        self.p.models = {"linear_regression": RecordingLinearRegression()}

    def train_constant_boundary(self, all_models=False):
        # Explicit labels isolate transformations; native labels tested below.
        if all_models:
            self.p = MLPredictor()
        data = prices(61)
        target = pd.Series([.025] * 60 + [np.nan], index=data.index)
        with patch.object(self.p, "create_target_variable", return_value=target):
            result = self.p.train_models(data, {})
        self.assertNotIn("error", result)
        return data

    def test_target_unknown_tail_stays_nan_for_each_horizon(self):
        data = pd.DataFrame({"Close": [100., 110., 121., 133.1]},
                            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-07", "2026-01-08"]))
        original = data.copy(deep=True)
        for h, expected in ((1, [.1, .1, .1, np.nan]),
                            (2, [.21, .21, np.nan, np.nan]),
                            (3, [.331, np.nan, np.nan, np.nan])):
            with self.subTest(h=h):
                actual = self.p.create_target_variable(data, h)
                np.testing.assert_allclose(actual, expected, atol=1e-12)
                pd.testing.assert_index_equal(actual.index, data.index)
        pd.testing.assert_frame_equal(data, original)

    def test_target_distinguishes_real_zero_and_invalid_prices(self):
        for invalid in (0., -1., np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                result = self.p.create_target_variable(pd.DataFrame({"Close": [100., 100., invalid, 110.]}))
                np.testing.assert_allclose(result, [0., np.nan, np.nan, np.nan])

    def test_target_empty_and_beyond_history_have_no_labels(self):
        for empty in (pd.DataFrame({"Close": pd.Series(dtype=float)}), pd.DataFrame(columns=["Close"])):
            with self.subTest(empty_dtype=empty["Close"].dtype):
                target = self.p.create_target_variable(empty)
                self.assertTrue(target.empty)
                self.assertEqual(target.dtype, np.dtype(float))
        for h in (3, 10, np.int64(4), 10**20):
            self.assertTrue(self.p.create_target_variable(pd.DataFrame({"Close": [100., 110., 120.]}), h).isna().all())

    def test_invalid_horizons_and_nonnumeric_prices_are_rejected(self):
        for h in (True, False, 0, -1, 1.5, "2", np.nan):
            with self.subTest(h=h), self.assertRaises(ValueError):
                self.p.create_target_variable(prices(), h)
        for values in (["100", "110"], [100+1j, 110+1j], [True, False],
                       pd.to_datetime(["2026-01-01", "2026-01-02"])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.p.create_target_variable(pd.DataFrame({"Close": values}))

    def test_native_fifty_labels_required_and_horizon_purge_counts(self):
        result = self.p.train_models(prices(52), {}, prediction_horizon=3)
        self.assertIn("error", result)
        self.assertEqual(result["available_data"], 49)
        result = self.p.train_models(prices(53), {}, prediction_horizon=3)
        self.assertNotIn("error", result)
        self.assertEqual((result["train_samples"], result["test_samples"]), (37, 10))
        self.assertEqual(self.p.scaler.n_samples_seen_, 37)
        # Last retained train target uses rows36 and39, both before test row40.
        self.assertAlmostEqual(self.p.target_scaler.data_min_[0], 3/136)
        self.assertAlmostEqual(self.p.target_scaler.data_max_[0], .03)

    def test_purge_uses_source_positions_across_missing_labels(self):
        data = prices(56)
        data.loc[data.index[38], "Close"] = np.nan
        # h3 invalid source rows35,38 and terminal53..55 ->51 known labels.
        # split40/11: first test source42, train only source<=38, excluding35/38 ->37.
        result = self.p.train_models(data, {}, prediction_horizon=3)
        self.assertNotIn("error", result)
        self.assertEqual((result["train_samples"], result["test_samples"]), (37, 11))

    def test_purge_or_test_with_fewer_than_two_samples_fails(self):
        for count, h, test_size in ((100, 50, .2), (51, 1, .01)):
            with self.subTest(count=count, h=h, test_size=test_size):
                result = self.p.train_models(prices(count), {}, prediction_horizon=h, test_size=test_size)
                self.assertIn("error", result)
                self.assertFalse(self.p.is_trained)
                self.assertEqual(self.p.trained_models, {})

    def test_training_rejects_unsorted_and_duplicate_index_without_sorting(self):
        for data in (prices().iloc[::-1], prices().set_axis([0] * 101)):
            before = data.copy(deep=True)
            self.assertIn("error", self.p.train_models(data, {}))
            self.assertFalse(self.p.is_trained)
            pd.testing.assert_frame_equal(data, before)

    def test_cleaning_and_feature_prefix_do_not_learn_from_appended_future(self):
        raw = pd.DataFrame({"raw": [0., 10., 20.]})
        np.testing.assert_array_equal(self.p.clean_features(raw)["raw"], [0., 10., 20.])
        extended = pd.DataFrame({"raw": [0., 10., 20., 10000.]})
        pd.testing.assert_frame_equal(self.p.clean_features(raw), self.p.clean_features(extended).iloc[:3])
        data = prices(65)
        pd.testing.assert_frame_equal(self.p.prepare_features(data.iloc[:50], {}),
                                      self.p.prepare_features(data, {}).iloc[:50])

    def test_train_bounds_are_single_pass_and_never_clip_percentage_or_constant(self):
        train = pd.DataFrame({"raw": [0., 10., 20.], "price_change": [-1., 0., 10.], "constant": [7.] * 3})
        before = train.copy(deep=True)
        bounds = self.p._fit_clip_bounds(train)
        self.assertEqual(set(bounds), {"raw"})
        np.testing.assert_allclose(bounds["raw"], [.2, 19.8])
        result = self.p._apply_clip_bounds(pd.DataFrame({"raw": [-100., 10000.],
                                                       "price_change": [-1., 10.], "constant": [7., 7.]}), bounds)
        np.testing.assert_allclose(result["raw"], [.2, 19.8])
        np.testing.assert_array_equal(result["price_change"], [-1., 10.])
        pd.testing.assert_frame_equal(train, before)

    def test_holdout_does_not_change_fitted_bounds_or_scalers(self):
        data = prices(61)
        first = self.p.train_models(data, {})
        self.assertNotIn("error", first)
        bounds = dict(self.p.clip_bounds)
        mean, scale = self.p.scaler.mean_.copy(), self.p.scaler.scale_.copy()
        np.testing.assert_allclose(bounds["close_lag_1"], [46., 144.54])
        col = self.p.feature_names.index("close_lag_1")
        self.assertAlmostEqual(mean[col], (46 + sum(range(100, 145)) + 144.54) / 47)
        data.loc[data.index[48:], ["Open", "High", "Low", "Close", "Volume"]] *= 10
        self.assertNotIn("error", self.p.train_models(data, {}))
        self.assertEqual(self.p.clip_bounds, bounds)
        np.testing.assert_array_equal(self.p.scaler.mean_, mean)
        np.testing.assert_array_equal(self.p.scaler.scale_, scale)

    def test_all_real_estimators_inverse_constant_target_to_literal_price(self):
        data = self.train_constant_boundary(all_models=True)
        expected_names = {"random_forest", "gradient_boosting", "linear_regression", "svr", "ensemble"}
        with contextlib.redirect_stdout(io.StringIO()):
            predictions = self.p.predict_price(data, {})
        self.assertEqual(set(predictions), expected_names)
        for name in expected_names:
            with self.subTest(model=name):
                self.assertAlmostEqual(predictions[name], 164., places=8)

    def test_inference_receives_training_bounds_scaled_input_without_refitting(self):
        data = self.train_constant_boundary()
        model = self.p.trained_models["linear_regression"]
        # Independent numeric check on lag1: train ends row46,
        # q99=144.54; training mean uses .01-clipped46 and100..144,144.54.
        mean = (46 + sum(range(100, 145)) + 144.54) / 47
        values = np.array([46., *range(100, 145), 144.54])
        expected = (144.54 - mean) / np.sqrt(np.mean((values - mean)**2))
        with patch.object(self.p.scaler, "fit", side_effect=AssertionError("refit")), \
             patch.object(self.p.target_scaler, "fit", side_effect=AssertionError("refit")):
            self.assertNotIn("error", self.p.predict_price(data, {}))
        col = self.p.feature_names.index("close_lag_1")
        self.assertAlmostEqual(model.predict_inputs[-1][0, col], expected)

    def test_native_four_model_smoke_returns_finite_prices_and_ensemble(self):
        self.p = MLPredictor()
        data = prices(90)
        result = self.p.train_models(data, {}, prediction_horizon=3)
        self.assertEqual(len(result.get("successful_models", [])), 4)
        with contextlib.redirect_stdout(io.StringIO()):
            predictions = self.p.predict_price(data, {}, prediction_horizon=3)
        self.assertNotIn("error", predictions)
        self.assertEqual(len(predictions), 5)
        values = [predictions[name] for name in self.p.models]
        self.assertTrue(all(np.isfinite(value) and value > 0 for value in values))
        self.assertAlmostEqual(predictions["ensemble"], sum(values) / 4)

    def test_missing_features_horizon_and_bad_index_are_prediction_errors(self):
        data = self.train_constant_boundary()
        for h in (3, 0, True):
            self.assertIn("error", self.p.predict_price(data, {}, prediction_horizon=h))
        for bad in (data.iloc[::-1], data.set_axis([0] * len(data))):
            self.assertIn("error", self.p.predict_price(bad, {}))
        self.p.feature_names.append("missing_feature")
        self.assertIn("error", self.p.predict_price(data, {}))

    def test_extra_feature_is_ignored_but_stored_order_preserved(self):
        data = self.train_constant_boundary()
        plain = self.p.predict_price(data, {})
        extra = self.p.predict_price(data, {"rsi": pd.Series(60., index=data.index)})
        self.assertEqual(plain, extra)

    def test_failed_retrain_discards_old_models_and_fitted_metadata(self):
        data = self.train_constant_boundary()
        self.p.models = {"broken": FailingFit()}
        self.assertIn("error", self.p.train_models(data, {}))
        self.assertFalse(self.p.is_trained)
        self.assertEqual(self.p.trained_models, {})
        self.assertEqual(self.p.feature_names, [])
        self.assertEqual(self.p.clip_bounds, {})
        self.assertIsNone(self.p.prediction_horizon)
        self.assertIn("error", self.p.predict_price(data, {}))

    def test_insufficient_retrain_also_invalidates_old_training(self):
        self.train_constant_boundary()
        self.assertIn("error", self.p.train_models(prices(10), {}))
        self.assertFalse(self.p.is_trained)
        self.assertEqual(self.p.trained_models, {})

    def test_model_that_fails_after_fit_is_not_published(self):
        self.p.models = {"broken": PredictFailure(), "linear_regression": RecordingLinearRegression()}
        result = self.p.train_models(prices(61), {})
        self.assertNotIn("error", result)
        self.assertEqual(result["successful_models"], ["linear_regression"])
        self.assertEqual(list(self.p.trained_models), ["linear_regression"])
        self.assertIn("error", result["models"]["broken"])


if __name__ == "__main__":
    unittest.main()
