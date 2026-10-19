"""Kept-alive HTTPS connections to the two fixed voice services (Groq and Fish Audio).

Opening a new secure connection costs over a second from this laptop, so each service gets one
connection that is reused. Servers drop idle connections after a while, so warm() reconnects in
the background the moment J.A.R.V.I.S wakes up, while you are still speaking.

Rules: HTTPS only, one fixed host each, never follows redirects (a key can never be forwarded),
and the warm-up request sends no key and no data.
"""
from __future__ import annotations

import threading
from typing import Any


class KeepAlive:
    def __init__(self, base_url: str):
        if not base_url.startswith("https://"):
            raise ValueError("only HTTPS hosts are allowed")
        self.base_url = base_url
        self._client: Any = None
        self._lock = threading.Lock()

    def client(self) -> Any:
        """The shared client (created on first use). Safe to use from several threads."""
        with self._lock:
            if self._client is None:
                import httpx

                self._client = httpx.Client(follow_redirects=False,
                                            limits=httpx.Limits(max_connections=4, keepalive_expiry=120.0))
            return self._client

    def warm(self) -> None:
        """Open (or refresh) the connection in the background. Sends no key and no data."""
        def work() -> None:
            try:
                self.client().head(self.base_url, timeout=5.0)
            except Exception:
                pass                       # warming is only a speed-up; failures don't matter

        threading.Thread(target=work, name="jarvis-warm", daemon=True).start()

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None


GROQ = KeepAlive("https://api.groq.com")
FISH = KeepAlive("https://api.fish.audio")


def warm_all() -> None:
    GROQ.warm()
    FISH.warm()
