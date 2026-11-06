"""Trusted temporary joblib roundtrips and fail-before-mutation validation."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from modules.ml_predictor import MLPredictor
from backend.tests.test_ml_features_training import prices


class TestMLArtifacts(unittest.TestCase):
    def setUp(self):
        for name in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            guard = patch(name, side_effect=AssertionError(name))
            guard.start()
            self.addCleanup(guard.stop)
        directory = tempfile.TemporaryDirectory(prefix="trader-ml-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "model.joblib"
        self.p = MLPredictor()
        self.p.models = {"linear_regression": LinearRegression()}
        self.data = prices(83)
        result = self.p.train_models(self.data, {}, prediction_horizon=3)
        self.assertNotIn("error", result)

    def package(self):
        self.p.save_models(str(self.path))
        return joblib.load(self.path)

    def assert_rejected_preserving_instance(self, package):
        previous = self.p.predict_price(self.data, {}, prediction_horizon=3)
        identities = (self.p.trained_models, self.p.scaler, self.p.target_scaler,
                      self.p.feature_names)
        bounds = copy.deepcopy(self.p.clip_bounds)
        joblib.dump(package, self.path)
        with self.assertRaises(ValueError):
            self.p.load_models(str(self.path))
        for actual, original in zip((self.p.trained_models, self.p.scaler, self.p.target_scaler,
                                     self.p.feature_names), identities):
            self.assertIs(actual, original)
        self.assertTrue(self.p.is_trained)
        self.assertEqual(self.p.prediction_horizon, 3)
        self.assertEqual(self.p.clip_bounds, bounds)
        self.assertEqual(self.p.predict_price(self.data, {}, prediction_horizon=3), previous)

    def test_roundtrip_preserves_bounds_horizon_feature_order_and_real_prediction(self):
        expected = self.p.predict_price(self.data, {}, prediction_horizon=3)
        package = self.package()
        self.assertEqual(package["format_version"], 1)
        self.assertEqual(package["prediction_horizon"], 3)
        self.assertEqual(package["preprocessing"]["clip_bounds"], self.p.clip_bounds)
        loaded = MLPredictor()
        loaded.load_models(str(self.path))
        self.assertEqual(loaded.feature_names, self.p.feature_names)
        self.assertEqual(loaded.clip_bounds, self.p.clip_bounds)
        np.testing.assert_array_equal(loaded.scaler.mean_, self.p.scaler.mean_)
        np.testing.assert_array_equal(loaded.target_scaler.data_min_, self.p.target_scaler.data_min_)
        self.assertEqual(loaded.predict_price(self.data, {}, prediction_horizon=3), expected)
        self.assertIn("error", loaded.predict_price(self.data, {}, prediction_horizon=7))

    def test_untrained_or_failed_retrain_cannot_write_a_model_file(self):
        with self.assertRaises(ValueError):
            MLPredictor().save_models(str(self.path))
        self.assertFalse(self.path.exists())
        self.p.train_models(prices(10), {})
        with self.assertRaises(ValueError):
            self.p.save_models(str(self.path))
        self.assertFalse(self.path.exists())

    def test_legacy_unknown_and_noninteger_versions_are_rejected_before_mutation(self):
        package = self.package()
        for version in (None, 2, True, 1.0, "1"):
            candidate = copy.deepcopy(package)
            if version is None:
                candidate.pop("format_version", None)
            else:
                candidate["format_version"] = version
            with self.subTest(version=version):
                self.assert_rejected_preserving_instance(candidate)

    def test_missing_required_fields_and_nonmapping_package_are_rejected(self):
        package = self.package()
        for field in ("models", "scaler", "target_scaler", "feature_names", "is_trained",
                      "preprocessing", "prediction_horizon"):
            candidate = copy.deepcopy(package)
            candidate.pop(field)
            with self.subTest(field=field):
                self.assert_rejected_preserving_instance(candidate)
        self.assert_rejected_preserving_instance([package])

    def test_bad_horizon_state_and_feature_schema_are_rejected(self):
        package = self.package()
        cases = [("prediction_horizon", x) for x in (0, -1, True, 1.5, None)]
        cases += [("is_trained", x) for x in (False, 1, "yes")]
        cases += [("feature_names", x) for x in ([], "Close", ["a", "a"], [1] * 30)]
        for field, value in cases:
            candidate = copy.deepcopy(package)
            candidate[field] = value
            with self.subTest(field=field, value=value):
                self.assert_rejected_preserving_instance(candidate)

    def test_invalid_bounds_are_rejected_and_valid_empty_bounds_roundtrip(self):
        package = self.package()
        for bounds in (None, [], {"absent": [0., 1.]}, {"price_change": [-1., 10.]},
                       {"sma_5": [1., 0.]}, {"sma_5": [1., 1.]}, {"sma_5": [0., np.inf]},
                       {"sma_5": [0.]}, {"sma_5": ["0", "1"]}):
            candidate = copy.deepcopy(package)
            candidate["preprocessing"] = {"clip_bounds": bounds}
            with self.subTest(bounds=bounds):
                self.assert_rejected_preserving_instance(candidate)
        candidate = copy.deepcopy(package)
        candidate["preprocessing"] = {}
        self.assert_rejected_preserving_instance(candidate)
        self.p.clip_bounds = {}
        self.p.save_models(str(self.path))
        loaded = MLPredictor()
        loaded.load_models(str(self.path))
        self.assertEqual(loaded.clip_bounds, {})
        self.assertNotIn("error", loaded.predict_price(self.data, {}, prediction_horizon=3))

    def test_unfitted_wrong_type_and_wrong_dimension_scalers_are_rejected(self):
        package = self.package()
        for field, value in (("scaler", StandardScaler()), ("scaler", object()),
                              ("scaler", StandardScaler().fit([[1.], [2.]])),
                              ("target_scaler", StandardScaler().fit([[1.], [2.]]))):
            candidate = copy.deepcopy(package)
            candidate[field] = value
            with self.subTest(field=field, value=type(value).__name__):
                self.assert_rejected_preserving_instance(candidate)

    def test_nonfinite_malformed_or_inconsistent_scaler_parameters_are_rejected(self):
        package = self.package()
        for field, attribute, value in (("scaler", "mean_", [np.nan] * 30),
                                         ("scaler", "scale_", [1.] * 29),
                                         ("scaler", "scale_", [0.] * 30),
                                         ("scaler", "var_", [-1.] * 30),
                                         ("scaler", "feature_names_in_", ["wrong"] * 30),
                                         ("target_scaler", "scale_", [np.inf]),
                                         ("target_scaler", "data_range_", [-1.])):
            candidate = copy.deepcopy(package)
            setattr(candidate[field], attribute, np.asarray(value))
            with self.subTest(field=field, attribute=attribute):
                self.assert_rejected_preserving_instance(candidate)

    def test_missing_unfitted_or_feature_mismatched_models_are_rejected(self):
        package = self.package()
        for models in ({}, [], {"linear_regression": LinearRegression()},
                       {"linear_regression": LinearRegression().fit([[1.], [2.]], [0., 1.])}):
            candidate = copy.deepcopy(package)
            candidate["models"] = models
            with self.subTest(models=models):
                self.assert_rejected_preserving_instance(candidate)

    def test_inconsistent_derived_scaler_parameters_preserve_working_model(self):
        package = self.package()
        for field, attribute, value in (
                ("target_scaler", "min_", np.array([0.])),
                ("target_scaler", "scale_", package["target_scaler"].scale_ * 2),
                ("scaler", "scale_", package["scaler"].scale_ * 2)):
            candidate = copy.deepcopy(package)
            setattr(candidate[field], attribute, value)
            with self.subTest(field=field, attribute=attribute):
                self.assert_rejected_preserving_instance(candidate)

    def test_invalid_or_mismatched_scaler_sample_counts_preserve_working_model(self):
        package = self.package()
        for field, count in (("scaler", np.nan), ("target_scaler", np.inf),
                             ("scaler", 2.5), ("scaler", True),
                             ("target_scaler", package["target_scaler"].n_samples_seen_ + 1)):
            candidate = copy.deepcopy(package)
            candidate[field].n_samples_seen_ = count
            with self.subTest(field=field, count=count):
                self.assert_rejected_preserving_instance(candidate)


if __name__ == "__main__":
    unittest.main()
