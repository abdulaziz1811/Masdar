"""Loads sources.yaml and builds the adapters named in it."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import yaml

from masdar.sources.base import SourceAdapter, SourceDescriptor

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
SOURCES_FILE = CONFIG_DIR / "sources.yaml"


def outside_ksa() -> bool:
    """Whether this server runs outside the Kingdom (MASDAR_OUTSIDE_KSA=1).

    Set on a host abroad. Sources verified to refuse non-Saudi connections
    (`geo_restricted`) are then not asked at all: each attempt would only
    wait for a timeout, and the answer would read "could not reach" on every
    question instead of saying once, on the landing page, that this server
    cannot see them.
    """
    return os.environ.get("MASDAR_OUTSIDE_KSA", "").strip().lower() in ("1", "true", "yes", "on")


def saudi_exit() -> bool:
    """Whether sources that refuse non-Saudi connections get a Saudi exit.

    Outside the Kingdom with FIRECRAWL_API_KEY set, their requests go through
    Firecrawl's Saudi exit (sources/transport.py) and every other source's
    stay direct. Verified for the national platform's JSON API; its files are
    refused on that route, so they are offered as links (see
    SaudiOpenDataAdapter).
    """
    return outside_ksa() and bool(os.environ.get("FIRECRAWL_API_KEY", "").strip())


def via_saudi_exit(descriptor: SourceDescriptor) -> bool:
    return descriptor.geo_restricted and saudi_exit()


def reachable_here(descriptor: SourceDescriptor) -> bool:
    return not (descriptor.geo_restricted and outside_ksa()) or saudi_exit()


def load_descriptors(path: Path | None = None) -> tuple[SourceDescriptor, ...]:
    raw = yaml.safe_load((path or SOURCES_FILE).read_text(encoding="utf-8"))
    return tuple(
        SourceDescriptor(
            id=entry["id"],
            name_ar=entry["name_ar"],
            name_en=entry["name_en"],
            base_url=entry["base_url"],
            adapter=entry["adapter"],
            authority=int(entry.get("authority", 50)),
            operator_ar=entry.get("operator_ar", ""),
            operator_en=entry.get("operator_en", ""),
            api_verified=bool(entry.get("api_verified", False)),
            api=entry.get("api") or {},
            notes_ar=(entry.get("notes_ar") or "").strip(),
            topics=tuple(entry.get("topics", ())),
            synthetic=bool(entry.get("synthetic", False)),
            international=bool(entry.get("international", False)),
            site_url=str(entry.get("site_url", "") or ""),
            rendering=entry.get("rendering", "server"),
            geo_restricted=bool(entry.get("geo_restricted", False)),
            waf=bool(entry.get("waf", False)),
            access_checked=str(entry.get("access_checked", "") or ""),
            access_notes_ar=(entry.get("access_notes_ar") or "").strip(),
            enabled=bool(entry.get("enabled", True)),
            auth=entry.get("auth") or {},
        )
        for entry in raw.get("sources", [])
    )


@functools.lru_cache(maxsize=1)
def _cached_descriptors() -> tuple[SourceDescriptor, ...]:
    return load_descriptors()


def build_adapter(descriptor: SourceDescriptor, http=None) -> SourceAdapter:
    """Instantiate the adapter class named by `descriptor.adapter`."""
    # Imported here so the adapter modules can import from this package.
    from masdar.sources.adapters.fixture import FixtureAdapter
    from masdar.sources.adapters.gastat_api import GastatApiAdapter
    from masdar.sources.adapters.gastat_cdata import GastatCdataAdapter
    from masdar.sources.adapters.gastat_site import GastatSiteAdapter
    from masdar.sources.adapters.html_index import HtmlIndexAdapter
    from masdar.sources.adapters.live_data import LiveDataAdapter
    from masdar.sources.adapters.opendatasoft import OpendatasoftAdapter
    from masdar.sources.adapters.saudi_open_data import SaudiOpenDataAdapter
    from masdar.sources.adapters.worldbank import WorldBankAdapter

    registry: dict[str, type[SourceAdapter]] = {
        "saudi_open_data": SaudiOpenDataAdapter,
        "gastat_api": GastatApiAdapter,
        "gastat_cdata": GastatCdataAdapter,
        "gastat_site": GastatSiteAdapter,
        "opendatasoft": OpendatasoftAdapter,
        "html_index": HtmlIndexAdapter,
        "live_data": LiveDataAdapter,
        "worldbank": WorldBankAdapter,
        "fixture": FixtureAdapter,
    }
    try:
        cls = registry[descriptor.adapter]
    except KeyError:
        raise ValueError(
            f"source '{descriptor.id}' names unknown adapter '{descriptor.adapter}'"
        ) from None
    return cls(descriptor, http)


class Registry:
    """The set of sources available to a run."""

    def __init__(
        self, descriptors: tuple[SourceDescriptor, ...], http=None, use_saudi_exit: bool = True
    ):
        self.descriptors = descriptors
        self._http = http
        self._saudi_http = None
        # False: geo-restricted sources are left out rather than asked through
        # the Saudi exit -- for background work that must not spend its
        # credits or its rate (see chat/warmup.py).
        self._use_saudi_exit = use_saudi_exit
        self._by_id = {d.id: d for d in descriptors}

    def _reachable(self, descriptor: SourceDescriptor) -> bool:
        if self._use_saudi_exit:
            return reachable_here(descriptor)
        return not (descriptor.geo_restricted and outside_ksa())

    @classmethod
    def load(cls, path: Path | None = None, http=None) -> Registry:
        descriptors = load_descriptors(path) if path else _cached_descriptors()
        return cls(descriptors, http)

    def get(self, source_id: str) -> SourceDescriptor | None:
        return self._by_id.get(source_id)

    def adapter(self, source_id: str) -> SourceAdapter:
        descriptor = self._by_id[source_id]
        if self._http is not None and self._use_saudi_exit and via_saudi_exit(descriptor):
            if self._saudi_http is None:
                from masdar.sources.transport import saudi_exit_transport

                self._saudi_http = self._http.routed(saudi_exit_transport())
            return build_adapter(descriptor, self._saudi_http)
        return build_adapter(descriptor, self._http)

    def not_reachable(self, topic_id: str | None) -> tuple[SourceDescriptor, ...]:
        """Sources on the topic that this server cannot ask at all."""
        return tuple(
            d for d in self.descriptors
            if d.enabled and not self._reachable(d) and d.covers_topic(topic_id)
        )

    def for_topic(self, topic_id: str | None) -> tuple[SourceDescriptor, ...]:
        """Sources that claim the topic, most authoritative first."""
        matches = [
            d for d in self.descriptors
            if d.enabled and self._reachable(d) and d.covers_topic(topic_id)
        ]
        return tuple(sorted(matches, key=lambda d: -d.authority))

    def plan(
        self, topic_id: str | None, preferred: tuple[str, ...] = ()
    ) -> tuple[SourceDescriptor, ...]:
        """Search order: verified before unverified, then by preference.

        Preference comes from the topic lexicon (who originates this kind of
        statistic), authority only breaks the remaining ties.
        """
        candidates = self.for_topic(topic_id)
        by_id = {d.id: d for d in candidates}
        ordered: list[SourceDescriptor] = []
        for source_id in preferred:
            descriptor = by_id.pop(source_id, None)
            if descriptor is not None:
                ordered.append(descriptor)
        ordered.extend(d for d in candidates if d.id in by_id)
        # Verified sources first, preference order kept within each group.
        # The run consults only a few sources, and a slot spent on one whose
        # endpoint was never confirmed is a slot a working source did not get.
        # A source moves up by being verified, not by edits to this ordering.
        ordered.sort(key=lambda d: not d.api_verified)
        return tuple(ordered)


DEMO_SOURCES_FILE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "sources.demo.yaml"
)


def load_registry(demo: bool = False, http=None, use_saudi_exit: bool = True) -> Registry:
    """The registry for a run.

    Demo sources live in a separate file and are only ever added when asked
    for, so synthetic data cannot leak into a normal search.
    """
    descriptors = load_descriptors()
    if demo:
        descriptors = descriptors + load_descriptors(DEMO_SOURCES_FILE)
    return Registry(descriptors, http, use_saudi_exit=use_saudi_exit)
