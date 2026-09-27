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

    def probe(self) -> str:
        """Cheap liveness check used by `masdar doctor`."""
        raise NotImplementedError
