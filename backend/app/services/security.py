from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import hashlib
import secrets


@dataclass
class MagicInvite:
    email: str
    expires_at: datetime
    consumed: bool = False


@dataclass
class SessionToken:
    email: str
    token: str
    expires_at: datetime


class AuthService:
    def __init__(self, invite_ttl_minutes: int = 15, session_ttl_minutes: int = 120) -> None:
        self.invite_ttl = timedelta(minutes=invite_ttl_minutes)
        self.session_ttl = timedelta(minutes=session_ttl_minutes)
        self.invites: Dict[str, MagicInvite] = {}
        self.sessions: Dict[str, SessionToken] = {}

    @staticmethod
    def _hash_token(raw: str) -> str:
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_email(email: str) -> str:
        return email.strip().lower()

    def create_invite(self, email: str) -> str:
        raw = secrets.token_urlsafe(32)
        token = self._hash_token(raw)
        now = datetime.now(timezone.utc)
        self.invites[token] = MagicInvite(
            email=self._normalize_email(email),
            expires_at=now + self.invite_ttl,
            consumed=False,
        )
        return raw

    def consume_magic_link(self, raw_code: str) -> Optional[SessionToken]:
        invite = self.invites.get(self._hash_token(raw_code))
        if not invite or invite.consumed or invite.expires_at < datetime.now(timezone.utc):
            return None

        invite.consumed = True
        raw_session = secrets.token_urlsafe(40)
        hashed_session = self._hash_token(raw_session)
        now = datetime.now(timezone.utc)
        session_token = SessionToken(
            email=invite.email,
            token=hashed_session,
            expires_at=now + self.session_ttl,
        )
        self.sessions[hashed_session] = session_token
        return SessionToken(email=session_token.email, token=raw_session, expires_at=session_token.expires_at)

    def validate_session(self, raw_token: str) -> Optional[str]:
        token_hash = self._hash_token(raw_token)
        session = self.sessions.get(token_hash)
        if not session:
            return None
        if session.expires_at < datetime.now(timezone.utc):
            self.sessions.pop(token_hash, None)
            return None
        return session.email
