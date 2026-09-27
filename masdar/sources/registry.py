"""Loads sources.yaml and builds the adapters named in it."""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

from masdar.sources.base import SourceAdapter, SourceDescriptor

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
SOURCES_FILE = CONFIG_DIR / "sources.yaml"


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
    from masdar.sources.adapters.html_index import HtmlIndexAdapter
    from masdar.sources.adapters.saudi_open_data import SaudiOpenDataAdapter

    registry: dict[str, type[SourceAdapter]] = {
        "saudi_open_data": SaudiOpenDataAdapter,
        "gastat_api": GastatApiAdapter,
        "html_index": HtmlIndexAdapter,
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

    def __init__(self, descriptors: tuple[SourceDescriptor, ...], http=None):
        self.descriptors = descriptors
        self._http = http
        self._by_id = {d.id: d for d in descriptors}

    @classmethod
    def load(cls, path: Path | None = None, http=None) -> Registry:
        descriptors = load_descriptors(path) if path else _cached_descriptors()
        return cls(descriptors, http)

    def get(self, source_id: str) -> SourceDescriptor | None:
        return self._by_id.get(source_id)

    def adapter(self, source_id: str) -> SourceAdapter:
        descriptor = self._by_id[source_id]
        return build_adapter(descriptor, self._http)

    def for_topic(self, topic_id: str | None) -> tuple[SourceDescriptor, ...]:
        """Sources that claim the topic, most authoritative first."""
        matches = [d for d in self.descriptors if d.enabled and d.covers_topic(topic_id)]
        return tuple(sorted(matches, key=lambda d: -d.authority))

    def plan(
        self, topic_id: str | None, preferred: tuple[str, ...] = ()
    ) -> tuple[SourceDescriptor, ...]:
        """Search order: the topic's preferred sources, then the rest.

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
        return tuple(ordered)


DEMO_SOURCES_FILE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "sources.demo.yaml"
)


def load_registry(demo: bool = False, http=None) -> Registry:
    """The registry for a run.

    Demo sources live in a separate file and are only ever added when asked
    for, so synthetic data cannot leak into a normal search.
    """
    descriptors = load_descriptors()
    if demo:
        descriptors = descriptors + load_descriptors(DEMO_SOURCES_FILE)
    return Registry(descriptors, http)
