"""Actual extracted Streamlit panel with controlled IO and real bound helper."""
import json
import unittest

import numpy as np

from backend.tests.test_legacy_disclosure import ml_app, text_of, chart_values


class LegacyMLHorizonTests(unittest.TestCase):
    def rendered(self, horizon=3):
        app = ml_app(start="2021-01-01")
        if horizon != 3:
            app.selectbox[1].select(horizon).run(timeout=15)
        app.button[0].click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        return app

    def test_selector_preserves_numeric_options_and_explains_observation_units(self):
        app = ml_app()
        selector = app.selectbox[1]
        self.assertEqual(selector.label, "Tahmin ufku (gözlem)")
        self.assertEqual(selector.options, ["3 gözlem ilerisi", "7 gözlem ilerisi",
                                            "14 gözlem ilerisi", "30 gözlem ilerisi"])
        self.assertEqual(selector.value, 3)
        self.assertIn("takvim günü veya kesin tarih değildir", selector.proto.help)

    def test_stale_data_card_uses_actual_last_index_and_has_no_calendar_target(self):
        app = self.rendered()
        text = text_of(app)
        self.assertIn("Son veri gözleminden 3 gözlem ilerisi.", text)
        self.assertIn("Son veri gözlemi: 2021-04-30", text)
        self.assertIn("3 gözlem ilerisi tahmini", text)
        for forbidden in ("tarihindeki fiyat tahmini", "gün sonra", "Gün Tahmini", "Gün AI Fiyat"):
            self.assertNotIn(forbidden, text)

    def test_caption_uses_matching_shared_bounds_for_each_ui_horizon(self):
        for h, expected in ((3, "-25% ile +25%"), (7, "-25% ile +25%"),
                            (14, "-50% ile +50%"), (30, "-50% ile +50%")):
            with self.subTest(h=h):
                text = text_of(self.rendered(h))
                self.assertIn(f"Deneysel model çıktısına uygulanan sınır: {expected}.", text)
                self.assertIn("Bu sınır, modelin doğruluk ölçüsü değildir.", text)
                self.assertIn(f"Son veri gözleminden {h} gözlem ilerisi.", text)
                self.assertIn("Son veri gözlemi: 2021-04-30", text)
                self.assertIn("Model uyumu (sezgisel)", text)
                self.assertIn("95%", text)

    def test_chart_uses_zero_anchor_and_schematic_steps_with_unchanged_spread_formula(self):
        chart = json.loads(self.rendered().get("plotly_chart")[0].proto.spec)
        self.assertEqual(chart["layout"]["xaxis"]["title"]["text"], "Şematik gözlem adımı")
        self.assertEqual(chart["layout"]["xaxis"]["range"], [0, 3])
        anchor, line, upper, lower = chart["data"]
        self.assertEqual(anchor["name"], "Son veri fiyatı")
        self.assertEqual(anchor["x"], [0])
        self.assertEqual(list(chart_values(anchor)), [100.])
        self.assertEqual(line["x"], [0, 1, 2, 3])
        np.testing.assert_allclose(chart_values(line), [100., 310/3, 320/3, 110.])
        self.assertEqual(upper["x"], [1, 2, 3])
        self.assertEqual(lower["x"], [1, 2, 3])
        np.testing.assert_allclose(chart_values(upper), [310/3+(8/3)**.5, 320/3+(8/3)**.5, 110+(8/3)**.5])
        np.testing.assert_allclose(chart_values(lower), [310/3-(8/3)**.5, 320/3-(8/3)**.5, 110-(8/3)**.5])


if __name__ == "__main__":
    unittest.main()
