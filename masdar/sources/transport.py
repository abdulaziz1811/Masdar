"""How bytes are actually fetched.

`HttpClient` owns caching, retries and provenance; a transport owns the one
question of how a request reaches the host. They are separate because the
Saudi portals forced the issue: every one of them refuses connections from
outside the Kingdom, so a direct `requests.get()` can only ever work from a
Saudi address. Running Masdar elsewhere needs a different route, and that
must not mean rewriting the adapters.

Two transports ship:

* `DirectTransport` -- ordinary HTTPS. Correct inside Saudi Arabia.
* `FirecrawlTransport` -- routes through Firecrawl with a Saudi exit and a
  real browser, which was verified to reach open.data.gov.sa (HTTP 200) when
  direct and non-Saudi routes all failed. It also renders single-page apps,
  which plain HTTP cannot.

Selected with `MASDAR_HTTP_BACKEND=direct|firecrawl`.
"""

from __future__ import annotations

import abc
import contextlib
import os
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

from masdar.sources.base import (
    RetryableSourceError,
    SourceError,
    SourceRejected,
    SourceUnreachable,
)

DEFAULT_TIMEOUT = 20.0
CONNECT_TIMEOUT = 8.0

# Identifies the agent while presenting the header set a normal browser sends.
# The portals are public and unauthenticated; the extra headers are there
# because their edge rejects requests that negotiate content like a script,
# not to defeat any access control.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36 Masdar/0.1 (+open-data research agent)"
)

BROWSER_HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "application/json;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "ar,ar-SA;q=0.9,en;q=0.8",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}

# The F5/Imperva rejection page open.data.gov.sa serves. It arrives as HTTP
# 200 with an HTML body, so status code alone cannot detect it.
_WAF_MARKERS = (
    "Request Rejected",
    "The requested URL was rejected",
    "Your support ID is",
)


def looks_rejected(body: bytes, media_type: str) -> bool:
    """Whether a 200 response is really a WAF refusal.

    Worth a dedicated check: treated as a normal body, the rejection page
    parses as a dataset list with zero entries, which would turn "the edge
    blocked me" into "this source has no data" -- the one confusion this
    project exists to prevent.
    """
    if "html" not in (media_type or "") and media_type:
        return False
    head = body[:2048].decode("utf-8", errors="replace")
    return sum(marker in head for marker in _WAF_MARKERS) >= 2


# GASTAT's gateway answers a keyless request to a protected route with
# "Failed to resolve API Key variable request.header.apikey".
_MISSING_KEY_MARKERS = ("api key", "apikey", "api_key", "unauthorized", "credentials")


def describe_refusal(response, url: str) -> str:
    """Say which kind of refusal this was, since the remedies differ.

    A missing API key, a geo-block and a network policy all surface as 401 or
    403, and telling them apart is the difference between "set this variable"
    and "you cannot reach this host at all".
    """
    snippet = ""
    with contextlib.suppress(Exception):
        snippet = response.content[:512].decode("utf-8", errors="replace").lower()
    status = response.status_code
    if status == 401 or any(marker in snippet for marker in _MISSING_KEY_MARKERS):
        return (
            f"هذا المسار يتطلب مفتاح API (HTTP {status}). عيّن المفتاح في متغيّر "
            "البيئة المذكور في إعداد المصدر (auth.key_env) وأعد المحاولة. "
            "هذا لا يعني عدم وجود البيانات."
        )
    if status == 407:
        return f"رُفض من البروكسي (HTTP {status})"
    return (
        f"رُفض الوصول (HTTP {status}) — قد يكون المضيف محجوباً جغرافياً أو "
        "بسياسة الشبكة. هذا لا يعني عدم وجود البيانات."
    )


@dataclass
class Response:
    """A transport's raw result, before provenance is attached."""

    url: str
    final_url: str
    status: int
    content: bytes
    media_type: str
    headers: dict[str, str]


class Transport(abc.ABC):
    name = "transport"

    @abc.abstractmethod
    def get(
        self,
        url: str,
        source_id: str,
        params: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        """Fetch `url`, or raise `SourceUnreachable`/`SourceError`."""

    def describe(self) -> str:
        return self.name


class DirectTransport(Transport):
    """Plain HTTPS via `requests`."""

    name = "direct"

    def __init__(self, timeout: float = DEFAULT_TIMEOUT, max_bytes: int = 80 * 1024 * 1024):
        self.timeout = timeout
        self.max_bytes = max_bytes
        self._session = requests.Session()
        self._session.headers.update(BROWSER_HEADERS)
        self._session.headers["User-Agent"] = os.environ.get(
            "MASDAR_USER_AGENT", DEFAULT_USER_AGENT
        )

    def get(
        self,
        url: str,
        source_id: str,
        params: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        try:
            response = self._session.get(
                url,
                params=params,
                headers=headers or None,
                # A host that does not accept the connection within a few
                # seconds is not going to; a slow download is another matter.
                timeout=(min(CONNECT_TIMEOUT, self.timeout), self.timeout),
                stream=True,
                allow_redirects=True,
            )
        except requests.exceptions.SSLError as exc:
            raise SourceUnreachable(
                source_id, f"{urlparse(url).netloc}: فشل التحقق من شهادة TLS"
            ) from exc
        except requests.exceptions.RequestException as exc:
            from masdar.sources.http import concise_reason

            raise SourceUnreachable(source_id, concise_reason(exc, urlparse(url).netloc)) from exc

        if response.status_code >= 400:
            # Read a little of the body: it says which refusal this is.
            response.content  # noqa: B018 - forces the stream to buffer
        if response.status_code == 404:
            raise SourceError(source_id, f"not found: {url}")
        if response.status_code in (401, 403, 407):
            raise SourceRejected(source_id, describe_refusal(response, url))
        if response.status_code in (429, 500, 502, 503, 504):
            raise RetryableSourceError(source_id, f"HTTP {response.status_code} for {url}")
        if response.status_code >= 400:
            raise SourceError(source_id, f"HTTP {response.status_code} for {url}")

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            chunks.append(chunk)
            total += len(chunk)
            if total > self.max_bytes:
                raise SourceError(source_id, f"response exceeds {self.max_bytes} bytes: {url}")

        return Response(
            url=url,
            final_url=response.url,
            status=response.status_code,
            content=b"".join(chunks),
            media_type=(response.headers.get("Content-Type", "") or "").split(";")[0].strip(),
            headers=dict(response.headers),
        )


class FirecrawlTransport(Transport):
    """Fetches through Firecrawl with a Saudi exit and a real browser.

    Needed because the portals are geo-restricted and one of them is a
    single-page app: this route was the only one that reached
    open.data.gov.sa from outside the Kingdom, and it executes the page's
    JavaScript so an SPA yields rendered content rather than an empty shell.

    Requires `FIRECRAWL_API_KEY`. Costs a credit per fetch, so it is opt-in
    rather than the default.
    """

    name = "firecrawl"
    endpoint = "https://api.firecrawl.dev/v2/scrape"

    def __init__(
        self,
        api_key: str | None = None,
        country: str = "SA",
        timeout: float = 90.0,
        wait_for_ms: int = 5000,
    ):
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY", "")
        self.country = country
        self.timeout = timeout
        self.wait_for_ms = wait_for_ms
        self._session = requests.Session()

    def describe(self) -> str:
        return f"firecrawl (exit: {self.country})"

    def get(
        self,
        url: str,
        source_id: str,
        params: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        if headers:
            # Forwarding a publisher's API key through a scraping service
            # would hand the credential to a third party the key's owner never
            # agreed to share it with. Refused outright, not silently dropped:
            # dropping it would turn into a 401 that reads like a data problem.
            raise SourceError(
                source_id,
                "هذا الطلب يحمل مفتاح API، ولن يُرسَل المفتاح عبر وسيط خارجي "
                "(Firecrawl). شغّل الأداة من داخل السعودية بالمنفذ المباشر "
                "(MASDAR_HTTP_BACKEND=direct) لاستخدام المسارات التي تتطلب مفتاحاً.",
            )
        if not self.api_key:
            raise SourceUnreachable(
                source_id,
                "FIRECRAWL_API_KEY غير مُعيَّن، ولا يمكن استخدام منفذ Firecrawl",
            )
        if params:
            from urllib.parse import urlencode

            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{urlencode(params)}"

        payload = {
            "url": url,
            "formats": ["rawHtml"],
            "location": {"country": self.country},
            "onlyMainContent": False,
            "maxAge": 0,
        }
        if self.wait_for_ms:
            payload["waitFor"] = self.wait_for_ms
        try:
            response = self._session.post(
                self.endpoint,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as exc:
            raise SourceUnreachable(source_id, f"تعذّر الوصول إلى Firecrawl: {exc}") from exc

        if response.status_code >= 400:
            raise SourceError(
                source_id, f"Firecrawl أعاد HTTP {response.status_code}: {response.text[:200]}"
            )

        body = response.json()
        if not body.get("success", True):
            raise SourceError(source_id, f"Firecrawl فشل: {str(body.get('error'))[:200]}")

        data = body.get("data") or {}
        content = data.get("rawHtml") or data.get("html") or data.get("markdown") or ""
        metadata = data.get("metadata") or {}
        return Response(
            url=url,
            final_url=metadata.get("url") or url,
            status=int(metadata.get("statusCode") or 200),
            content=content.encode("utf-8"),
            media_type=(metadata.get("contentType") or "text/html").split(";")[0].strip(),
            headers={},
        )


def saudi_exit_transport() -> FirecrawlTransport:
    """The route for geo-restricted sources when this server is abroad.

    Their APIs answer JSON, so there is no page to wait for; 30 seconds covers
    the exit's slowest answers seen (the largest publisher list, ~275 KB).
    """
    return FirecrawlTransport(timeout=30.0, wait_for_ms=0)


def build_transport(name: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> Transport:
    name = (name or os.environ.get("MASDAR_HTTP_BACKEND") or "direct").strip().lower()
    if name in ("direct", "", "requests"):
        return DirectTransport(timeout=timeout)
    if name == "firecrawl":
        return FirecrawlTransport()
    raise ValueError(f"unknown HTTP backend '{name}' (expected 'direct' or 'firecrawl')")
