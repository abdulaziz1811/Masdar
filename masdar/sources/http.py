"""HTTP access with caching, retries and honest failure reporting.

Two things here matter to the rest of the system:

1. Every fetch returns a `Fetched` carrying a sha256 and a retrieval
   timestamp, so provenance is a by-product of downloading rather than
   something a later layer has to remember to attach.
2. Being blocked is reported as `SourceUnreachable`, never as an empty
   result. A corporate proxy, a geo-block and a policy denial all look the
   same from here, and all of them mean "unknown", not "absent".
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import requests

from masdar.sources.base import SourceError, SourceUnreachable

USER_AGENT = "Masdar/0.1 (open-data research agent; +https://github.com/abdulaziz1811/masdar)"

DEFAULT_TIMEOUT = 20.0
DEFAULT_RETRIES = 3
MAX_BYTES = 80 * 1024 * 1024


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
        return f"{host}: تعذّر الاتصال بالمضيف"
    return f"{host}: {type(exc).__name__}"


def cache_dir() -> Path:
    configured = os.environ.get("MASDAR_CACHE_DIR")
    base = Path(configured) if configured else Path.home() / ".cache" / "masdar"
    return base


@dataclass(frozen=True)
class Fetched:
    """One successful HTTP response, with everything provenance needs."""

    url: str
    final_url: str
    status: int
    content: bytes
    media_type: str
    retrieved_at: datetime
    sha256: str
    from_cache: bool = False
    headers: dict[str, str] = None  # type: ignore[assignment]

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
    """Thin `requests` wrapper. One instance per run."""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        use_cache: bool = True,
        offline: bool = False,
    ):
        self.timeout = timeout
        self.retries = retries
        self.use_cache = use_cache
        # `offline` makes any un-cached request fail loudly instead of
        # silently degrading, which keeps test runs honest.
        self.offline = offline
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

    # -- cache ---------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        host = urlparse(url).netloc.replace(":", "_") or "unknown"
        return cache_dir() / host / f"{digest}.bin"

    def _read_cache(self, url: str) -> Fetched | None:
        path = self._cache_path(url)
        meta_path = path.with_suffix(".json")
        if not (path.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            content = path.read_bytes()
        except OSError:
            return None
        return Fetched(
            url=url,
            final_url=meta.get("final_url", url),
            status=meta.get("status", 200),
            content=content,
            media_type=meta.get("media_type", "application/octet-stream"),
            retrieved_at=datetime.fromisoformat(meta["retrieved_at"]),
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

    # -- fetch ---------------------------------------------------------
    def get(self, url: str, source_id: str = "", params: dict | None = None) -> Fetched:
        source_id = source_id or urlparse(url).netloc or "local"
        if url.startswith("file://"):
            return self._get_file(url, source_id)
        if self.use_cache and params is None:
            cached = self._read_cache(url)
            if cached is not None:
                return cached

        if self.offline:
            raise SourceUnreachable(source_id, f"offline mode and no cached copy of {url}")

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self._session.get(
                    url, params=params, timeout=self.timeout, stream=True, allow_redirects=True
                )
            except requests.exceptions.SSLError as exc:
                raise SourceUnreachable(
                    source_id, f"{urlparse(url).netloc}: فشل التحقق من شهادة TLS"
                ) from exc
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                host = urlparse(url).netloc
                raise SourceUnreachable(
                    source_id,
                    f"{concise_reason(exc, host)} (بعد {self.retries} محاولات)",
                ) from exc

            if response.status_code in (429, 500, 502, 503, 504) and attempt < self.retries - 1:
                time.sleep(2 ** attempt)
                continue

            if response.status_code == 404:
                raise SourceError(source_id, f"not found: {url}")
            if response.status_code in (401, 403, 407):
                raise SourceUnreachable(
                    source_id,
                    f"رُفض الوصول (HTTP {response.status_code}) — قد يكون المضيف "
                    "محجوباً جغرافياً أو بسياسة الشبكة",
                )
            if response.status_code >= 400:
                raise SourceError(source_id, f"HTTP {response.status_code} for {url}")

            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_BYTES:
                    raise SourceError(source_id, f"response exceeds {MAX_BYTES} bytes: {url}")
            content = b"".join(chunks)

            fetched = Fetched(
                url=url,
                final_url=response.url,
                status=response.status_code,
                content=content,
                media_type=(response.headers.get("Content-Type", "") or "").split(";")[0].strip(),
                retrieved_at=datetime.now(UTC),
                sha256=hashlib.sha256(content).hexdigest(),
                headers=dict(response.headers),
            )
            if self.use_cache and params is None:
                self._write_cache(fetched)
            return fetched

        raise SourceUnreachable(source_id, f"exhausted retries for {url}: {last_error}")
