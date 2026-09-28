"""Adapter contract shared by every data source.

Portals differ wildly -- a JSON API here, an HTML publications list there --
so each one gets an adapter whose only job is to turn its own quirks into
`DatasetCandidate` objects. Nothing downstream of this module knows what a
CKAN response or an ASPX page looks like.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from masdar.domain.models import DataRequest, DatasetCandidate


class SourceError(Exception):
    """A source could not answer. Never means "the data does not exist"."""

    def __init__(self, source_id: str, reason: str):
        self.source_id = source_id
        self.reason = reason
        super().__init__(f"{source_id}: {reason}")


class SourceUnreachable(SourceError):
    """Network, DNS, TLS or policy blocked us before we saw any data."""


class RetryableSourceError(SourceError):
    """A transient failure (429 or 5xx) that is worth trying again."""


class SourceRejected(SourceUnreachable):
    """The host answered, but a WAF refused the request.

    Kept apart from a plain network failure because the remedy is different:
    a rejection means the request did not look legitimate to the edge (wrong
    headers, a path the WAF guards, an out-of-region address), not that the
    host is down. It is still a form of "unknown", never of "absent".
    """


@dataclass(frozen=True)
class SourceDescriptor:
    """One entry from sources.yaml."""

    id: str
    name_ar: str
    name_en: str
    base_url: str
    adapter: str
    authority: int = 50
    operator_ar: str = ""
    operator_en: str = ""
    api_verified: bool = False
    api: dict = field(default_factory=dict)
    notes_ar: str = ""
    topics: tuple[str, ...] = ()
    # How the site serves its content. An SPA cannot be read by scraping
    # HTML -- its data arrives over XHR -- so this decides whether an
    # HTML adapter can work at all.
    rendering: str = "server"          # "server" | "spa"
    # Verified access facts, so the next person does not re-derive them.
    geo_restricted: bool = False       # refuses connections outside Saudi Arabia
    waf: bool = False                  # edge rejects non-browser-looking requests
    access_checked: str = ""           # ISO date of the last verification
    access_notes_ar: str = ""
    # False for a source that is documented but not yet usable -- a known API
    # whose base URL or request shape is still missing. It stays in the
    # registry so the knowledge is not lost, but is never queried.
    enabled: bool = True
    # Credential *locations*, never credentials. Secrets are read from the
    # environment at call time and never written to this repository.
    auth: dict = field(default_factory=dict)
    # True for fixture/demo sources. Anything derived from one is labelled
    # as sample data so it can never be mistaken for an official figure.
    synthetic: bool = False

    def covers_topic(self, topic_id: str | None) -> bool:
        if topic_id is None:
            return True
        return topic_id in self.topics

    def url(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


class SourceAdapter(abc.ABC):
    """Searches one source and reports what it holds."""

    def __init__(self, descriptor: SourceDescriptor, http=None):
        self.descriptor = descriptor
        self.http = http

    @property
    def id(self) -> str:
        return self.descriptor.id

    @abc.abstractmethod
    def search(self, request: DataRequest, limit: int = 10) -> list[DatasetCandidate]:
        """Return candidates this source offers for the request.

        Raise `SourceUnreachable` if the source could not be contacted, and
        `SourceError` if it answered but unusably. Returning `[]` is a
        positive statement: "I looked and I have nothing."
        """

    def request_headers(self) -> dict[str, str]:
        """Credentials for this source, read from the environment at call time.

        The descriptor's `auth` block names where a key lives (`key_env`) and
        which header carries it; the value itself is never in the repository.
        No variable set means no header -- public endpoints need none.
        """
        import os

        auth = self.descriptor.auth or {}
        key_env = auth.get("key_env")
        if not key_env:
            return {}
        key = os.environ.get(key_env, "").strip()
        if not key:
            return {}
        return {auth.get("header", "apikey"): key}

    def fetch(self, url: str):
        """Download one of this source's resources, with its credentials.

        The orchestrator calls this rather than the HTTP client directly, so
        knowing how a source authenticates stays inside the source layer and
        credentials never travel on domain objects such as `Resource`, which
        are printed, serialised to JSON and written into workbooks.
        """
        if self.http is None:
            raise SourceError(self.id, "no HTTP client configured")
        return self.http.get(url, source_id=self.id, headers=self.request_headers())

    def probe(self) -> str:
        """Cheap liveness check used by `masdar doctor`."""
        raise NotImplementedError
