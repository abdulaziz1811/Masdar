"""HTTP access with caching, retries and honest failure reporting.

Three things here matter to the rest of the system:

1. Every fetch returns a `Fetched` carrying a sha256 and a retrieval
   timestamp, so provenance is a by-product of downloading rather than
   something a later layer has to remember to attach.
2. Being blocked is reported as `SourceUnreachable`, never as an empty
   result. A geo-block, a WAF rejection and a proxy denial all mean
   "unknown", not "absent".
3. How bytes arrive is a transport's concern (see `transport.py`), because
   the Saudi portals require a Saudi exit and one of them requires a real
   browser. Swapping routes must not touch any adapter.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import requests

from masdar.sources.base import (
    RetryableSourceError,
    SourceError,
    SourceRejected,
    SourceUnreachable,
)
from masdar.sources.transport import Transport, build_transport, looks_rejected

DEFAULT_TIMEOUT = 20.0
DEFAULT_RETRIES = 3
# How long a cached response may stand in for the source. Sources are asked
# live precisely so that a newly published year is seen; a copy kept forever
# would answer "not published" about a year published since. Six hours keeps
# a conversation's repeated questions cheap without outliving a publication
# day. Override with MASDAR_CACHE_MAX_AGE_HOURS (0 = always ask the source).
DEFAULT_CACHE_MAX_AGE = timedelta(hours=6)


# When a source cannot be reached, an older saved copy (up to this age) is
# better than nothing -- a presentation should not fail because one ministry
# is down -- provided the answer says it is a saved copy, from when, and that
# newer data may exist. Override with MASDAR_STALE_MAX_DAYS (0 = never).
DEFAULT_STALE_MAX_AGE = timedelta(days=90)


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return None


def cache_max_age() -> timedelta:
    hours = _env_float("MASDAR_CACHE_MAX_AGE_HOURS")
    return DEFAULT_CACHE_MAX_AGE if hours is None else timedelta(hours=hours)


def stale_max_age() -> timedelta:
    days = _env_float("MASDAR_STALE_MAX_DAYS")
    return DEFAULT_STALE_MAX_AGE if days is None else timedelta(days=days)


# A host that could not be reached is not asked again for this long. Every
# attempt at a dead host waits out its timeouts and retries -- a minute or
# more -- and a live demonstration cannot spend that on each question. A
# saved copy still answers meanwhile. Override with MASDAR_HOST_COOLDOWN_SECONDS
# (0 = always try).
DEFAULT_HOST_COOLDOWN = timedelta(minutes=2)


def host_cooldown() -> timedelta:
    seconds = _env_float("MASDAR_HOST_COOLDOWN_SECONDS")
    return DEFAULT_HOST_COOLDOWN if seconds is None else timedelta(seconds=seconds)


def concise_reason(exc: Exception, host: str) -> str:
    """A one-line cause a user can act on.

    `requests` nests the real problem several exceptions deep, and printing
    the whole chain buries the one fact that matters: whether the host was
    blocked, was slow, or does not resolve.
    """
    text = str(exc)
    if "403" in text and "Tunnel" in text:
        return f"{host}: محجوب بسياسة الشبكة أو البروكسي (403 على CONNECT)"
    if isinstance(exc, requests.exceptions.ProxyError):
        return f"{host}: تعذّر الاتصال عبر البروكسي"
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return f"{host}: انتهت مهلة الاتصال"
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return f"{host}: انتهت مهلة القراءة"
    if "NameResolutionError" in text or "getaddrinfo" in text:
        return f"{host}: فشل تحويل اسم النطاق (DNS)"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return f"{host}: تعذّر الاتصال بالمضيف — قد يكون مقيَّداً جغرافياً"
    return f"{host}: {type(exc).__name__}"


def cache_dir() -> Path:
    configured = os.environ.get("MASDAR_CACHE_DIR")
    return Path(configured) if configured else Path.home() / ".cache" / "masdar"


@dataclass(frozen=True)
class Fetched:
    """One successful fetch, with everything provenance needs."""

    url: str
    final_url: str
    status: int
    content: bytes
    media_type: str
    retrieved_at: datetime
    sha256: str
    from_cache: bool = False
    headers: dict[str, str] | None = None
    # A saved copy used because the source could not be reached just now.
    stale: bool = False

    @property
    def byte_size(self) -> int:
        return len(self.content)

    def text(self, encoding: str | None = None) -> str:
        return self.content.decode(encoding or "utf-8", errors="replace")

    def json(self):
        try:
            return json.loads(self.text())
        except json.JSONDecodeError as exc:
            raise SourceError(urlparse(self.url).netloc, f"invalid JSON: {exc}") from exc

    def last_modified(self) -> datetime | None:
        """The server's Last-Modified header, when it sends one."""
        raw = (self.headers or {}).get("Last-Modified")
        if not raw:
            return None
        for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
            try:
                parsed = datetime.strptime(raw, fmt)
            except ValueError:
                continue
            return parsed.replace(tzinfo=parsed.tzinfo or UTC)
        return None


class HttpClient:
    """Caching, retrying wrapper around a transport. One instance per run."""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        use_cache: bool = True,
        offline: bool = False,
        transport: Transport | None = None,
        max_age: timedelta | None = None,
        stale_age: timedelta | None = None,
        cooldown: timedelta | None = None,
        down_hosts: dict | None = None,
    ):
        self.timeout = timeout
        self.retries = retries
        self.use_cache = use_cache
        self.max_age = cache_max_age() if max_age is None else max_age
        self.stale_age = stale_max_age() if stale_age is None else stale_age
        self.cooldown = host_cooldown() if cooldown is None else cooldown
        # host -> (until, reason): hosts that failed to connect just now.
        # Clients in one server may share it (`down_hosts`), so what the
        # start-up warm-up learnt spares the presenter's first question.
        self._down: dict[str, tuple[datetime, str]] = {} if down_hosts is None else down_hosts
        # `offline` makes any un-cached request fail loudly instead of
        # silently degrading, which keeps test runs honest.
        self.offline = offline
        self.transport = transport or build_transport(timeout=timeout)

    def describe_transport(self) -> str:
        return self.transport.describe()

    # -- cache ---------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        host = urlparse(url).netloc.replace(":", "_") or "unknown"
        return cache_dir() / host / f"{digest}.bin"

    def _read_cache(self, url: str, max_age: timedelta | None = None) -> Fetched | None:
        path = self._cache_path(url)
        meta_path = path.with_suffix(".json")
        if not (path.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            retrieved_at = datetime.fromisoformat(meta["retrieved_at"])
        except (OSError, ValueError, KeyError):
            return None
        if retrieved_at.tzinfo is None:
            retrieved_at = retrieved_at.replace(tzinfo=UTC)
        # Offline runs have no source to ask, so any copy is the best
        # available -- and it carries its own retrieval time into the
        # workbook. Online, an old copy must not stand in for the source.
        limit = self.max_age if max_age is None else max_age
        if not self.offline and datetime.now(UTC) - retrieved_at > limit:
            return None
        try:
            content = path.read_bytes()
        except OSError:
            return None
        return Fetched(
            url=url,
            final_url=meta.get("final_url", url),
            status=meta.get("status", 200),
            content=content,
            media_type=meta.get("media_type", "application/octet-stream"),
            retrieved_at=retrieved_at,
            sha256=meta.get("sha256", hashlib.sha256(content).hexdigest()),
            from_cache=True,
            headers=meta.get("headers", {}),
        )

    def _write_cache(self, fetched: Fetched) -> None:
        path = self._cache_path(fetched.url)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(fetched.content)
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "final_url": fetched.final_url,
                        "status": fetched.status,
                        "media_type": fetched.media_type,
                        "retrieved_at": fetched.retrieved_at.isoformat(),
                        "sha256": fetched.sha256,
                        "headers": fetched.headers or {},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass  # a cache that cannot be written must not fail the request

    # -- fetch ---------------------------------------------------------
    def get(
        self,
        url: str,
        source_id: str = "",
        params: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> Fetched:
        """Fetch a URL.

        `headers` are request headers, typically an API key. They are passed
        to the transport and never stored: the cache records the response,
        keyed by URL, and nothing about how it was authorised.
        """
        source_id = source_id or urlparse(url).netloc or "local"
        if url.startswith("file://"):
            return self._get_file(url, source_id)

        if self.use_cache and params is None:
            cached = self._read_cache(url)
            if cached is not None:
                return cached

        if self.offline:
            raise SourceUnreachable(source_id, f"offline mode and no cached copy of {url}")

        try:
            self._check_host(url, source_id)
            return self._get_live(url, source_id, params, headers)
        except SourceUnreachable:
            saved = self._stale_copy(url, params)
            if saved is None:
                raise
            return saved

    def _check_host(self, url: str, source_id: str) -> None:
        down = self._down.get(urlparse(url).netloc)
        if down is None:
            return
        until, reason = down
        if datetime.now(UTC) < until:
            raise SourceUnreachable(source_id, f"{reason} (تعذّر قبل قليل؛ لن يُعاد قبل مهلة قصيرة)")
        self._down.pop(urlparse(url).netloc, None)

    def _mark_down(self, url: str, reason: str) -> None:
        if self.cooldown > timedelta(0):
            self._down[urlparse(url).netloc] = (datetime.now(UTC) + self.cooldown, reason)

    def _stale_copy(self, url: str, params: dict | None) -> Fetched | None:
        """An older saved copy, for when the source cannot be reached."""
        if not self.use_cache or params is not None or self.stale_age <= timedelta(0):
            return None
        cached = self._read_cache(url, max_age=self.stale_age)
        return replace(cached, stale=True) if cached is not None else None

    def _get_live(
        self,
        url: str,
        source_id: str,
        params: dict | None,
        headers: dict[str, str] | None,
    ) -> Fetched:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.transport.get(url, source_id, params, headers)
            except SourceRejected:
                # A policy or WAF refusal is deterministic; repeating it only
                # wastes time and looks like abuse.
                raise
            except (SourceUnreachable, RetryableSourceError) as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                if isinstance(exc, SourceUnreachable):
                    # The host itself could not be reached (network, DNS,
                    # timeout). A refusal (401/403/WAF) or a 5xx concerns one
                    # path, and the rest of the host may answer: not marked.
                    self._mark_down(url, exc.reason)
                    raise
                raise SourceUnreachable(
                    source_id, f"{exc.reason} (بعد {self.retries} محاولات)"
                ) from exc

            if looks_rejected(response.content, response.media_type):
                raise SourceRejected(
                    source_id,
                    "جدار الحماية رفض الطلب (صفحة Request Rejected) — الطلب لم يبدُ "
                    "كطلب متصفح، أو المسار محمي. هذا ليس دليلاً على عدم وجود البيانات.",
                )

            fetched = Fetched(
                url=url,
                final_url=response.final_url,
                status=response.status,
                content=response.content,
                media_type=response.media_type,
                retrieved_at=datetime.now(UTC),
                sha256=hashlib.sha256(response.content).hexdigest(),
                headers=response.headers,
            )
            if self.use_cache and params is None:
                self._write_cache(fetched)
            return fetched

        raise SourceUnreachable(source_id, f"exhausted retries for {url}: {last_error}")

    def _get_file(self, url: str, source_id: str) -> Fetched:
        """Read a local file through the same interface as a remote one.

        Fixtures are served this way so offline runs exercise exactly the
        code path a live fetch takes, rather than a test-only shortcut.
        """
        path = Path(url2pathname(urlparse(url).path))
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise SourceError(source_id, f"cannot read {path}: {exc}") from exc
        suffix = path.suffix.lower().lstrip(".")
        media_types = {
            "csv": "text/csv",
            "json": "application/json",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "html": "text/html",
        }
        stat = path.stat()
        return Fetched(
            url=url,
            final_url=url,
            status=200,
            content=content,
            media_type=media_types.get(suffix, "application/octet-stream"),
            retrieved_at=datetime.now(UTC),
            sha256=hashlib.sha256(content).hexdigest(),
            headers={
                "Last-Modified": datetime.fromtimestamp(stat.st_mtime, UTC).strftime(
                    "%a, %d %b %Y %H:%M:%S GMT"
                )
            },
        )
