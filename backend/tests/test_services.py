from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from backend.app.services.portfolio import PortfolioService
from backend.app.services.security import AuthService
from backend.app.services.watchlist import WatchlistService


class TestServices(unittest.TestCase):
    def test_auth_invite_magic_and_session_round_trip(self) -> None:
        auth = AuthService(invite_ttl_minutes=5, session_ttl_minutes=30)
        token = auth.create_invite("Test@Example.Com ")

        session = auth.consume_magic_link(token)
        self.assertIsNotNone(session)
        self.assertEqual(session.email, "test@example.com")

        # Reuse is not allowed:
        self.assertIsNone(auth.consume_magic_link(token))
        self.assertEqual(auth.validate_session(session.token), "test@example.com")

    def test_watchlist_defaults_and_operations(self) -> None:
        service = WatchlistService(defaults=["THYAO.IS", "GARAN.IS"])
        self.assertEqual(service.list_for_user("u1"), ["THYAO.IS", "GARAN.IS"])

        service.set_for_user("u1", ["THYAO.IS", "tupRS.is", ""])
        self.assertEqual(service.list_for_user("u1"), ["THYAO.IS", "tupRS.is"])

        service.add_symbol("u1", "SASA.IS")
        self.assertEqual(service.list_for_user("u1")[-1], "SASA.IS")

        service.remove_symbol("u1", "SASA.IS")
        self.assertNotIn("SASA.IS", service.list_for_user("u1"))

    def test_portfolio_weighted_average_and_remove(self) -> None:
        service = PortfolioService()
        h1 = service.upsert("u1", "THYAO.IS", quantity=2, average_cost=100.0)
        self.assertEqual(h1.quantity, 2)
        self.assertEqual(h1.average_cost, 100.0)

        h2 = service.upsert("u1", "THYAO.IS", quantity=3, average_cost=150.0)
        self.assertEqual(h2.quantity, 5)
        self.assertEqual(h2.average_cost, 130.0)

        self.assertTrue(service.remove("u1", "THYAO.IS"))
        self.assertFalse(service.remove("u1", "THYAO.IS"))

    def test_session_expires_after_ttl(self) -> None:
        auth = AuthService(invite_ttl_minutes=5, session_ttl_minutes=1)
        invite = auth.create_invite("user@example.com")
        session = auth.consume_magic_link(invite)
        self.assertIsNotNone(session)

        # Force expiry by mutating internal store (test-only path)
        key = auth._hash_token(session.token)
        auth.sessions[key].expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        self.assertIsNone(auth.validate_session(session.token))


if __name__ == "__main__":
    unittest.main()
