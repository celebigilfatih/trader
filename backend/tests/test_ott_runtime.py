"""Narrow pandas runtime regressions; numerical contracts have separate tests."""

import unittest

import pandas as pd

from modules.technical_analysis import TechnicalAnalyzer


def prices(closes):
    return pd.DataFrame({"Open": closes, "High": [p + 1 for p in closes],
                         "Low": [p - 1 for p in closes], "Close": closes,
                         "Volume": 1000},
                        index=pd.date_range("2026-01-01", periods=len(closes)))


class TestOttRuntime(unittest.TestCase):
    def test_real_ott_calculation_runs_with_current_pandas(self):
        for closes in ([100.0, 110.0, 90.0], [90.0, 95.0, 100.0], [100.0] * 4):
            with self.subTest(closes=closes):
                data = prices(closes)
                analyzer = TechnicalAnalyzer(data)
                analyzer.add_indicator("ott")
                # Existing plotting outputs must remain available. No trend
                # math is asserted here; ADR-0006 has separate contract tests.
                for name in ("ott", "ott_signal"):
                    self.assertIsInstance(analyzer.indicators[name], pd.Series)
                    pd.testing.assert_index_equal(analyzer.indicators[name].index, data.index)

    def test_ott_runtime_preserves_caller_prices_and_index(self):
        data = prices([100.0, 110.0, 90.0])
        original = data.copy(deep=True)
        analyzer = TechnicalAnalyzer(data)
        analyzer.add_indicator("ott")
        pd.testing.assert_frame_equal(data, original)


if __name__ == "__main__":
    unittest.main()
