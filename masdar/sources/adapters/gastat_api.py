"""Adapter for GASTAT's statistical database API.

Verified live on 2026-09-27 against
`https://database.stats.gov.sa/gastatapi/portal/api/v1/`:

* `indicators/cardsinfo` -> `{"info": [...]}`, headline indicators. Confirmed
  by direct call: HTTP 200, `application/json`, bilingual `ar_title`/
  `en_title` and a `frequency` holding the period.
* `indicators/public` -> the indicator catalog, roughly 2,600 entries and
  about 4.5 MB. Confirmed HTTP 200 and `application/json`.
* `indicators/getDataForChart?api=<token>` -> the observations for one
  indicator, OData-shaped, with per-indicator column names ending in `_TIME`
  and `_OBSV`.

No key, token or registration: these endpoints answer anonymous requests.

Two consequences shape this adapter. The catalog has no server-side search --
GASTAT's own WAF rejects a query string on it -- so filtering happens here
after fetching the whole catalog once. And an entry's `frequency_*` fields
give periodicity ("Annual"), never which years exist, so coverage is left
unknown and is established by reading the observations themselves.
"""

from __future__ import annotations

from masdar.domain.models import (
    Coverage,
    DataRequest,
    DatasetCandidate,
    Resource,
)
from masdar.nlu.normalize import contains_phrase
from masdar.sources.base import SourceAdapter, SourceError

# Field names taken from the live responses and from GASTAT's own client.
_CATALOG_LIST_KEYS = ("data", "value", "items", "results")
_TITLE_AR = ("title_ar", "ar_title", "titleAr")
_TITLE_EN = ("title_en", "en_title", "titleEn")
_FREQUENCY = ("frequency_en", "frequency_ar", "frequency")
_INDEX_NAME = ("index_name_en", "index_name_ar", "index_name")

MAX_CATALOG_SCAN = 20_000


def _first(entry: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value not in (None, "", [], {}):
            return str(value)
    return ""


class GastatApiAdapter(SourceAdapter):
    """Searches the indicator catalog and points at each indicator's data."""

    def _url(self, key: str, default: str) -> str:
        return self.descriptor.url(self.descriptor.api.get(key, default))

    # -- catalog -------------------------------------------------------
    def _catalog(self) -> list[dict]:
        """The full indicator catalog.

        Fetched whole because the endpoint has no search; the HTTP layer
        caches it on disk so the cost is paid once per run, not per query.
        """
        fetched = self.http.get(
            self._url("catalog", "/gastatapi/portal/api/v1/indicators/public"),
            source_id=self.id,
        )
        payload = fetched.json()

        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = next(
                (payload[key] for key in _CATALOG_LIST_KEYS if isinstance(payload.get(key), list)),
                None,
            )
            if rows is None:
                raise SourceError(
                    self.id,
                    "لم يُعرَف شكل فهرس المؤشرات؛ المفاتيح الموجودة: "
                    f"{sorted(payload)[:12]}",
                )
        else:
            raise SourceError(self.id, f"شكل غير متوقع للفهرس: {type(payload).__name__}")

        return [r for r in rows[:MAX_CATALOG_SCAN] if isinstance(r, dict)]

    # -- search --------------------------------------------------------
    def search(self, request: DataRequest, limit: int = 10) -> list[DatasetCandidate]:
        if self.http is None:
            raise SourceError(self.id, "no HTTP client configured")

        terms = request.search_terms()
        chart_path = self.descriptor.api.get(
            "observations", "/gastatapi/portal/api/v1/indicators/getDataForChart"
        )

        scored: list[tuple[float, DatasetCandidate]] = []
        for entry in self._catalog():
            token = entry.get("api_url")
            # Only leaf indicators carry a data token; branch nodes are
            # navigation, not data.
            if not isinstance(token, str) or not token.strip():
                continue

            title_ar = _first(entry, _TITLE_AR)
            title_en = _first(entry, _TITLE_EN)
            haystack = " ".join(filter(None, (title_ar, title_en, _first(entry, _INDEX_NAME))))
            if not haystack:
                continue

            hits = sum(1 for term in terms if term and contains_phrase(haystack, term))
            if not hits:
                continue

            score = float(hits)
            for dimension in request.dimensions:
                words = self.lexicon_words(dimension)
                if any(contains_phrase(haystack, word) for word in words):
                    score += 1.5

            separator = "&" if "?" in chart_path else "?"
            resource_url = self.descriptor.url(
                f"{chart_path}{separator}api={token.strip()}"
            )
            periodicity = _first(entry, _FREQUENCY)

            scored.append((
                score,
                DatasetCandidate(
                    source_id=self.id,
                    dataset_id=str(entry.get("id") or title_en or title_ar)[:120],
                    title_ar=title_ar or title_en,
                    title_en=title_en,
                    description=(
                        "مؤشر من قاعدة بيانات الهيئة العامة للإحصاء"
                        + (f" — التكرار: {periodicity}" if periodicity else "")
                    ),
                    landing_url=self.descriptor.base_url,
                    publisher_ar=self.descriptor.name_ar,
                    publisher_en=self.descriptor.name_en,
                    resources=(
                        Resource(url=resource_url, format="JSON", title=title_ar or title_en),
                    ),
                    # `frequency_*` is periodicity, not coverage: it says
                    # "Annual", never which years exist. The years come from
                    # the observations, so nothing is claimed here.
                    claimed_coverage=Coverage.unknown(),
                ),
            ))

        scored.sort(key=lambda pair: -pair[0])
        return [candidate for _, candidate in scored[:limit]]

    def lexicon_words(self, dimension) -> tuple[str, ...]:
        from masdar.nlu.lexicon import load_lexicon

        return load_lexicon().dimension_words.get(dimension, ())

    # -- diagnostics ---------------------------------------------------
    def probe(self) -> str:
        """Checks the cheap headline endpoint, not the 4.5 MB catalog."""
        fetched = self.http.get(
            self._url("headlines", "/gastatapi/portal/api/v1/indicators/cardsinfo"),
            source_id=self.id,
        )
        payload = fetched.json()
        cards = payload.get("info") if isinstance(payload, dict) else None
        if not isinstance(cards, list):
            keys = sorted(payload)[:10] if isinstance(payload, dict) else type(payload).__name__
            raise SourceError(self.id, f"شكل غير متوقع لـ cardsinfo؛ المفاتيح: {keys}")
        sample = cards[0].get("en_title") if cards and isinstance(cards[0], dict) else ""
        return (
            f"HTTP {fetched.status}, {len(cards)} مؤشراً رئيسياً (مثال: {sample}), "
            "بلا مفتاح"
        )
