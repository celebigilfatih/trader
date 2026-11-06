"""Independent numeric helper oracles; no claim of forecasting effectiveness."""

import unittest

import numpy as np
import pandas as pd

from modules.ml_predictor import MLPredictor
from modules.simple_ml_predictor import SimpleMLPredictor


class TestMlHelpers(unittest.TestCase):
    def test_safe_divide_replaces_only_nonfinite_results_with_each_default(self):
        numerator = np.array([8., 0., 9., -9., np.inf, np.nan])
        denominator = np.array([2., 0., 0., 3., 2., 1.])
        before_num, before_den = numerator.copy(), denominator.copy()
        for predictor, default in ((MLPredictor(), 0.), (SimpleMLPredictor(), 1.)):
            with self.subTest(predictor=type(predictor).__name__):
                actual = predictor.safe_divide(numerator, denominator)
                np.testing.assert_array_equal(actual, [4., default, default, -3., default, default])
                custom = predictor.safe_divide(numerator, denominator, default_value=-7.)
                np.testing.assert_array_equal(custom, [4., -7., -7., -3., -7., -7.])
        np.testing.assert_array_equal(numerator, before_num)
        np.testing.assert_array_equal(denominator, before_den)

    def test_safe_divide_preserves_numpy_broadcast_and_scalar_arithmetic(self):
        for predictor in (MLPredictor(), SimpleMLPredictor()):
            with self.subTest(predictor=type(predictor).__name__):
                np.testing.assert_array_equal(predictor.safe_divide(
                    np.array([[6., 8.], [10., 12.]]), np.array([2., 4.])),
                    [[3., 2.], [5., 3.]])
                self.assertEqual(predictor.safe_divide(np.float64(9), np.float64(3)).item(), 3.)

    def test_stateless_cleaner_clips_percentages_without_fitting_quantiles(self):
        # ADR-0007: learned bounds belong to training, never stateless cleaning.
        data = pd.DataFrame({"price_change": [-2., -1., 0., 10., 12.],
                             "return_lag_1": [-5., -.5, 1., 20., 9.],
                             "raw_price": [0., 10., 20., 30., 100.],
                             "constant": [7.] * 5},
                            index=pd.date_range("2026-01-01", periods=5))
        original = data.copy(deep=True)
        actual = MLPredictor().clean_features(data)
        # Raw prices are unchanged; percentages keep their fixed bounds.
        expected = pd.DataFrame({"price_change": [-1., -1., 0., 10., 10.],
                                 "return_lag_1": [-1., -.5, 1., 10., 9.],
                                 "raw_price": [0., 10., 20., 30., 100.],
                                 "constant": [7.] * 5}, index=data.index)
        pd.testing.assert_frame_equal(actual, expected, atol=1e-12, rtol=1e-12)
        pd.testing.assert_frame_equal(data, original)

    def test_batch_cleaner_forward_fills_then_zeros_leading_missing_values(self):
        data = pd.DataFrame({"price_change": [np.nan, .5, np.inf, -np.inf, -2., 12.],
                             "raw": [np.nan, np.inf, 5., -np.inf, 9., np.nan]})
        original = data.copy(deep=True)
        result = MLPredictor().clean_features(data)
        np.testing.assert_array_equal(result["price_change"], [0., .5, .5, .5, -1., 10.])
        np.testing.assert_array_equal(result["raw"], [0., 0., 5., 5., 9., 9.])
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        pd.testing.assert_frame_equal(data, original)

    def test_batch_cleaner_keeps_empty_numeric_matrix_and_constant_columns(self):
        empty = pd.DataFrame({"price_change": pd.Series(dtype=float),
                              "raw": pd.Series(dtype=float)})
        pd.testing.assert_frame_equal(MLPredictor().clean_features(empty), empty)
        constants = pd.DataFrame({"raw": [3., 3., 3.], "return_lag_1": [0., 0., 0.]})
        pd.testing.assert_frame_equal(MLPredictor().clean_features(constants), constants)

    def test_numeric_diagnostics_distinguish_nan_infinity_and_strict_large_threshold(self):
        data = pd.DataFrame({"boundary": [-1e10, 1e10, 0.],
                             "overflow": [-1e10 - 1, 1e10 + 1, np.inf],
                             "missing": [np.nan, 1., 0.],
                             "negative_inf": [-np.inf, 0., 0.]})
        original = data.copy(deep=True)
        result = MLPredictor().diagnostic_features(data)
        self.assertEqual(result, {
            "infinity_columns": ["overflow", "negative_inf"],
            "nan_columns": ["missing"],
            "large_value_columns": ["overflow", "negative_inf"],
            "total_features": 4, "total_rows": 3,
        })
        pd.testing.assert_frame_equal(data, original)

    def test_target_valid_prefix_uses_future_bar_ratio_and_keeps_index(self):
        data = pd.DataFrame({"Close": [100., 110., 121., 133.1]},
                            index=pd.to_datetime(["2026-01-02", "2026-01-05",
                                                  "2026-01-06", "2026-01-08"]))
        before = data.copy(deep=True)
        for horizon, values in ((1, [.1, .1, .1]), (2, [.21, .21]), (3, [.331])):
            with self.subTest(horizon=horizon):
                target = MLPredictor().create_target_variable(data, horizon)
                np.testing.assert_allclose(target.iloc[:len(values)], values, atol=1e-12, rtol=0)
                pd.testing.assert_index_equal(target.index, data.index)
        # Terminal NaN and native target integration are pinned in test_ml_contract.
        pd.testing.assert_frame_equal(data, before)

    def test_confidence_helper_valid_return_records_use_absolute_change_and_edges(self):
        # This helper accepts return records, not predict_price's price mapping;
        # that interface mismatch is separately reported, not normalized here.
        for change, expected, strength in ((0., 1., "High"), (.029, .71, "High"),
                                            (.03, .7, "Medium"), (.059, .41, "Medium"),
                                            (.06, .4, "Low"), (-.05, .5, "Medium"),
                                            (.1, 0., "Low"), (.2, 0., "Low")):
            with self.subTest(change=change):
                result = MLPredictor().get_model_confidence({
                    "synthetic": {"predicted_return": change}, "ensemble": 100.,
                })
                self.assertEqual(set(result), {"synthetic"})
                self.assertAlmostEqual(result["synthetic"]["confidence"], expected)
                self.assertEqual(result["synthetic"]["prediction_strength"], strength)

    def test_untrained_prediction_reports_error_without_changing_training_state(self):
        predictor = MLPredictor()
        result = predictor.predict_price(pd.DataFrame(), {})
        self.assertIn("error", result)
        self.assertIn("eğitilmemiş", result["error"])
        self.assertFalse(predictor.is_trained)
        self.assertEqual(predictor.trained_models, {})


if __name__ == "__main__":
    unittest.main()
