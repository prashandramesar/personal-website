"""One-password login for the recipe tool, using a signed cookie."""

import hashlib
import hmac
import os
import threading
import time
from collections import deque
from urllib.parse import urlsplit

from fastapi import Request

COOKIE_NAME = "recipes_session"
SESSION_SECONDS = 30 * 24 * 60 * 60
MAX_FAILURES = 5
FAILURE_WINDOW_SECONDS = 15 * 60


class LoginLimiter:
    """Slows password guessing. There's one account, so the limit is global."""

    def __init__(self) -> None:
        self._failures: deque[float] = deque()
        self._lock = threading.Lock()

    def _trim(self, now: float) -> None:
        while self._failures and now - self._failures[0] > FAILURE_WINDOW_SECONDS:
            self._failures.popleft()

    def allowed(self) -> bool:
        with self._lock:
            self._trim(time.monotonic())
            return len(self._failures) < MAX_FAILURES

    def record_failure(self) -> None:
        with self._lock:
            self._failures.append(time.monotonic())

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()


limiter = LoginLimiter()


def password_matches(given: str, expected: str) -> bool:
    return hmac.compare_digest(given.encode(), expected.encode())


def _signature(secret: str, expires: str) -> str:
    return hmac.new(secret.encode(), expires.encode(), hashlib.sha256).hexdigest()


def make_session(secret: str, now: float | None = None) -> str:
    expires = str(int((now if now is not None else time.time()) + SESSION_SECONDS))
    return f"{expires}.{_signature(secret, expires)}"


def session_is_valid(token: str | None, secret: str, now: float | None = None) -> bool:
    if not token or "." not in token:
        return False
    expires, signature = token.split(".", 1)
    if not hmac.compare_digest(signature, _signature(secret, expires)):
        return False
    return expires.isdigit() and int(expires) > (
        now if now is not None else time.time()
    )


def is_same_origin(request: Request) -> bool:
    """Reject a form posted from another site, which would carry the cookie."""
    source = request.headers.get("origin") or request.headers.get("referer")
    if not source:
        return False
    return urlsplit(source).netloc.lower() == request.headers.get("host", "").lower()


def secure_cookies() -> bool:
    return os.environ.get("PRODUCTION", "").lower() == "true"
