"""ADR-0008 literal bounds and helper contracts.

Cap tests use an explicit estimator boundary and real identity target scaler;
native model integration remains separately covered by test_ml_contract.
"""
import contextlib
import copy
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules.ml_predictor import MLPredictor


class ScaledReturnBoundary:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        if isinstance(self.value, Exception):
            raise self.value
        return np.full(len(frame), self.value)


class TestMLHorizonConfidence(unittest.TestCase):
    def setUp(self):
        for name in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection",
                     "joblib.dump", "joblib.load"):
            guard = patch(name, side_effect=AssertionError(name))
            guard.start()
            self.addCleanup(guard.stop)
        self.p = MLPredictor()

    def boundary(self, horizon, values):
        # Controlled model outputs, actual prepare/scaler/inverse/predict consumer.
        data = pd.DataFrame({"Open": [100., 100.], "High": [105., 105.],
                             "Low": [95., 95.], "Close": [100., 100.], "Volume": [1000., 1000.]})
        features = self.p.prepare_features(data, {})
        self.p.feature_names = list(features)
        self.p.scaler.fit(features)
        self.p.target_scaler.fit([[0.], [1.]])  # Exact identity target scale.
        self.p.trained_models = {name: ScaledReturnBoundary(value) for name, value in values.items()}
        self.p.prediction_horizon = horizon
        self.p.is_trained = True
        return data

    def prediction(self, data, horizon):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.p.predict_price(data, {}, prediction_horizon=horizon)

    def test_return_bounds_cover_every_interval_edge_and_large_integer(self):
        for h, expected in ((1, (-.1, .1)), (2, (-.25, .25)), (3, (-.25, .25)),
                            (7, (-.25, .25)), (8, (-.5, .5)), (14, (-.5, .5)),
                            (30, (-.5, .5)), (31, (-.7, 1.)), (10**100, (-.7, 1.)),
                            (np.int64(7), (-.25, .25))):
            with self.subTest(h=h):
                self.assertEqual(self.p.get_prediction_return_bounds(h), expected)

    def test_return_bounds_reject_invalid_horizons(self):
        for h in (True, False, np.bool_(True), 0, -1, 1.5, 3., "3", None, np.nan, np.inf):
            with self.subTest(h=h), self.assertRaises(ValueError):
                self.p.get_prediction_return_bounds(h)

    def test_predict_uses_one_interval_cap_both_directions_and_keeps_equalities(self):
        for h, raw, expected in ((1, .4, 110.), (1, -.4, 90.),
                                  (2, .4, 125.), (3, -.4, 75.), (7, .25, 125.),
                                  (8, .8, 150.), (14, -.8, 50.), (30, -.5, 50.),
                                  (31, 2., 200.), (31, -2., 30.),
                                  (31, -.7, 30.), (31, 1., 200.), (3, 0., 100.)):
            with self.subTest(h=h, raw=raw):
                data = self.boundary(h, {"controlled": raw})
                result = self.prediction(data, h)
                self.assertEqual(set(result), {"controlled", "ensemble"})
                self.assertAlmostEqual(result["controlled"], expected)
                self.assertAlmostEqual(result["ensemble"], expected)

    def test_predict_consumes_shared_bounds_helper(self):
        data = self.boundary(3, {"controlled": .8})
        # Explicit dependency-boundary values prove the consumer uses the helper.
        with patch.object(self.p, "get_prediction_return_bounds", return_value=(-.02, .02)):
            self.assertAlmostEqual(self.prediction(data, 3)["ensemble"], 102.)

    def test_invalid_models_are_skipped_and_ensemble_uses_valid_prices(self):
        data = self.boundary(3, {"up": .4, "down": -.4, "nan": np.nan,
                                 "inf": np.inf, "broken": ValueError("controlled failure")})
        result = self.prediction(data, 3)
        self.assertEqual(set(result), {"up", "down", "ensemble"})
        self.assertAlmostEqual(result["up"], 125.)
        self.assertAlmostEqual(result["down"], 75.)
        self.assertAlmostEqual(result["ensemble"], 100.)
        self.p.trained_models = {"bad": ScaledReturnBoundary(np.nan)}
        self.assertIn("error", self.prediction(data, 3))

    def test_predict_keeps_finite_positive_current_price_validation(self):
        for current in (0., -1., np.inf, np.nan):
            with self.subTest(current=current):
                data = self.boundary(3, {"controlled": .1})
                data.loc[1, "Close"] = current
                self.assertIn("error", self.prediction(data, 3))

    def test_price_helper_reads_actual_prediction_mapping(self):
        data = self.boundary(3, {"a": .025, "b": -.05})
        predictions = self.prediction(data, 3)
        result = self.p.get_model_confidence(predictions, current_price=100.)
        self.assertEqual(set(result), {"a", "b"})
        self.assertAlmostEqual(result["a"]["confidence"], .75)
        self.assertEqual(result["a"]["prediction_strength"], "High")
        self.assertAlmostEqual(result["b"]["confidence"], .5)
        self.assertEqual(result["b"]["prediction_strength"], "Medium")

    def test_price_helper_literal_zero_and_strict_thresholds(self):
        for price, score, strength in ((100., 1., "High"), (102.9, .71, "High"),
                                        (103., .7, "Medium"), (106., .4, "Low"),
                                        (95., .5, "Medium"), (120., 0., "Low")):
            with self.subTest(price=price):
                result = self.p.get_model_confidence({"model": price}, 100.)["model"]
                self.assertAlmostEqual(result["confidence"], score)
                self.assertEqual(result["prediction_strength"], strength)

    def test_legacy_records_keep_units_extra_fields_and_large_finite_returns(self):
        records = {"zero": {"predicted_return": 0., "extra": "kept"},
                   "small": {"predicted_return": -.025},
                   "large": {"predicted_return": 1e308}, "ensemble": object()}
        before = copy.deepcopy({k: v for k, v in records.items() if k != "ensemble"})
        result = self.p.get_model_confidence(records)
        self.assertEqual(result["zero"], {"confidence": 1., "prediction_strength": "High"})
        self.assertAlmostEqual(result["small"]["confidence"], .75)
        self.assertEqual(result["large"], {"confidence": 0., "prediction_strength": "Low"})
        self.assertEqual({k: v for k, v in records.items() if k != "ensemble"}, before)
        self.assertEqual(self.p.get_model_confidence(records, 100.), result)

    def test_price_context_is_required_and_invalid_context_fails_for_every_payload(self):
        with self.assertRaises(ValueError):
            self.p.get_model_confidence({"a": 100.})
        for context in (0., -1., True, np.bool_(False), "100", 1+0j, np.nan, np.inf, [100.]):
            for payload in ({}, {"ensemble": 100.}, {"a": 100.}, {"a": {"predicted_return": 0.}}):
                with self.subTest(context=context, payload=payload), self.assertRaises(ValueError):
                    self.p.get_model_confidence(payload, current_price=context)

    def test_invalid_price_or_legacy_return_never_produces_a_score(self):
        for value in (True, np.bool_(True), "103", 1+0j, np.nan, np.inf, -np.inf, [103.], None):
            for payload in ({"a": value}, {"a": {"predicted_return": value}}):
                with self.subTest(value=value, payload=payload), self.assertRaises(ValueError):
                    self.p.get_model_confidence(payload, 100.)
        for value in (0., -1.):
            with self.subTest(price=value), self.assertRaises(ValueError):
                self.p.get_model_confidence({"a": value}, 100.)

    def test_mixed_error_nonmapping_payload_and_invalid_model_names_are_rejected(self):
        for payload in ([100.], None, "bad", {"error": "failure"}, {"error": 103.}, {"a": {}},
                        {"a": 100., "b": {"predicted_return": 0.}},
                        {1: 100.}, {"": {"predicted_return": .01}}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.p.get_model_confidence(payload, 100.)

    def test_empty_and_ensemble_only_have_no_independent_score(self):
        for payload in ({}, {"ensemble": object()}):
            self.assertEqual(self.p.get_model_confidence(payload), {})
            self.assertEqual(self.p.get_model_confidence(payload, 100.), {})

    def test_helper_rejects_ratio_overflow_and_preserves_prediction_mapping(self):
        with self.assertRaises(ValueError):
            self.p.get_model_confidence({"a": 1e308}, 1e-308)
        predictions = {"a": np.float64(102.5), "ensemble": np.float64(102.5)}
        before = predictions.copy()
        self.p.get_model_confidence(predictions, np.int64(100))
        self.assertEqual(predictions, before)


if __name__ == "__main__":
    unittest.main()
