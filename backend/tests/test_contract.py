from __future__ import annotations

from datetime import datetime, timezone
import unittest

from backend.app.core.contract import DataContract, error_response, ok_response


class TestContract(unittest.TestCase):
    def test_ok_response_includes_contract_and_correlation(self) -> None:
        contract = DataContract(
            symbol="THYAO.IS",
            source="unit",
            fetched_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
            as_of=datetime(2026, 8, 19, tzinfo=timezone.utc),
            staleness_seconds=12.34,
        )
        payload = ok_response({"ok": True}, contract=contract, correlation_id="abc")

        self.assertEqual(payload["data"]["ok"], True)
        self.assertEqual(payload["meta"]["correlation_id"], "abc")
        self.assertEqual(payload["meta"]["symbol"], "THYAO.IS")
        self.assertEqual(payload["meta"]["source"], "unit")

    def test_error_response_has_shape(self) -> None:
        payload = error_response("invalid_input", "Bad input", correlation_id="corr")
        self.assertEqual(
            payload,
            {
                "error_code": "invalid_input",
                "message": "Bad input",
                "correlation_id": "corr",
            },
        )


if __name__ == "__main__":
    unittest.main()
