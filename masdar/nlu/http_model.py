"""A model reached over plain HTTPS with `requests`, so no vendor library is needed.

`gemini.py` and `groq.py` differ in their addresses, headers and reply
shapes; what they share is here: one session, a per-call timeout, and
network failures named in Arabic and marked as passing (`transient`), so
the chain rests that provider and asks the next one.
"""

from __future__ import annotations

import requests

from masdar.nlu.lexicon import Lexicon
from masdar.nlu.llm import TIMEOUT_SECONDS, LlmUnavailable, LlmUnderstanding

# Statuses that pass on their own: the request was fine, the server was not.
TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class HttpUnderstanding(LlmUnderstanding):
    """The reading contract of `LlmUnderstanding`, over a `requests` session."""

    def __init__(self, session, lexicon: Lexicon, key: str, model: str, today=None):
        super().__init__(None, lexicon, model=model, today=today)
        self._session = session
        self._key = key

    def _headers(self) -> dict[str, str]:
        """The headers of every call; the key travels here, never in the URL."""
        raise NotImplementedError

    def _request(self, method: str, url: str, timeout: float | None = None, **kwargs):
        try:
            return self._session.request(
                method, url, headers=self._headers(), timeout=timeout or TIMEOUT_SECONDS,
                **kwargs,
            )
        except requests.Timeout as exc:
            raise LlmUnavailable(f"انتهت مهلة الاتصال بـ {self.VENDOR}", transient=True) from exc
        except requests.RequestException as exc:
            raise LlmUnavailable(f"تعذّر الاتصال بـ {self.VENDOR}", transient=True) from exc
