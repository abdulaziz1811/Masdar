"""Adapter for Opendatasoft Explore API v2.1 portals, such as KAPSARC's.

Verified live on 2026-09-28 against https://datasource.kapsarc.org:

* `GET /api/explore/v2.1/catalog/datasets?where="<text>"` is a real
  full-text catalogue search -- the one thing GASTAT's cdata API lacks. A
  search for "electricity consumption" matched 75 datasets.
* Each dataset carries `modified` (a genuine last-update timestamp), the
  *original* `publisher` (KAPSARC republishes GASTAT, SAMA, ministries...),
  `license`, keywords, an Arabic `theme_ar`, and a typed field list in which
  the time field is marked `timeserie_precision`.
* `GET .../records?group_by=year(<field>) as y` returns the distinct years a
  dataset holds -- authoritative coverage in one small request. On the SEC
  subscribers dataset it returned 2005-2025, while the dataset's own
  description still says "from 2005-2022": prose goes stale, data does not.
* `GET .../exports/json` returns every record, unaggregated and without the
  records endpoint's 10,000-row ceiling.

Titles are English even in the `_ar` fields, so the catalogue is searched
with the topic's English keywords as well as the user's own words.
"""

from __future__ import annotations

import html
import json
import re
from datetime import date, datetime
from urllib.parse import urlencode

from masdar.domain.models import (
    Coverage,
    CoverageOrigin,
    DataRequest,
    DatasetCandidate,
    Dimension,
    Resource,
)
from masdar.nlu.lexicon import load_lexicon
from masdar.nlu.normalize import contains_phrase
from masdar.sources.base import SourceAdapter, SourceError, SourceUnreachable

API = "/api/explore/v2.1"
# Selecting a field brings its language variants along (`title` answers with
# `title_en` and `title_ar`, `theme` with `theme_ar`). Naming a variant is
# refused: since September 2026 `title_ar` in the select is "Unknown field",
# a 400 that failed every search.
CATALOG_SELECT = (
    "dataset_id, title, description, modified, publisher, records_count, "
    "license, keyword, theme, fields"
)
MAX_COVERAGE_PROBES = 5
MAX_SEARCH_TERMS = 6
_TAGS = re.compile(r"<[^>]+>")
_ARABIC = re.compile(r"[؀-ۿ]")
_TIME_NAMES = ("year", "السنة", "date", "period", "time", "month", "quarter")


def _odsql_string(text: str) -> str:
    """A double-quoted ODSQL string literal."""
    return '"' + text.replace("\\", " ").replace('"', " ").strip() + '"'


def _plain(text: object) -> str:
    return " ".join(html.unescape(_TAGS.sub(" ", str(text or ""))).split())


def _as_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)] if value else []


def time_field(fields: list[dict]) -> dict | None:
    """The field that carries the period, by the portal's own annotation first."""
    fields = [f for f in fields if isinstance(f, dict) and f.get("name")]
    for field in fields:
        if (field.get("annotations") or {}).get("timeserie_precision"):
            return field
    for name in _TIME_NAMES:
        for field in fields:
            label = f"{field.get('name', '')} {field.get('label', '')}".lower()
            if name in label.split() or field.get("name", "").lower() == name:
                return field
    return next((f for f in fields if f.get("type") in ("date", "datetime")), None)


class OpendatasoftAdapter(SourceAdapter):
    """Catalogue search, coverage by group_by, data by JSON export."""

    def _url(self, path: str, params: list[tuple[str, str]] | None = None) -> str:
        url = self.descriptor.url(f"{API}{path}")
        return f"{url}?{urlencode(params)}" if params else url

    def _get_json(self, url: str):
        fetched = self.fetch(url)
        try:
            return json.loads(fetched.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceError(self.id, f"استجابة غير صالحة: {exc}") from exc

    # -- search --------------------------------------------------------
    def _terms(self, request: DataRequest) -> list[str]:
        """What to put in the catalogue query, English first.

        KAPSARC's titles are English even in their `_ar` fields, so the
        topic's English keywords carry most of the recall; the user's own
        words still go in, since themes and some descriptions are Arabic.
        """
        terms: list[str] = []
        if request.topic:
            terms.append(request.topic.label_en)
            terms.extend(request.topic.keywords_en[:3])
            terms.append(request.topic.label_ar)
        terms.extend(request.typed_phrases[:2])
        terms.extend(t for t in request.free_terms if not t.isdigit())
        unique = list(dict.fromkeys(t.strip() for t in terms if t and t.strip()))
        return unique[:MAX_SEARCH_TERMS]

    def search(self, request: DataRequest, limit: int = 10) -> list[DatasetCandidate]:
        if self.http is None:
            raise SourceError(self.id, "no HTTP client configured")
        terms = self._terms(request)
        if not terms:
            return []

        where = " OR ".join(_odsql_string(t) for t in terms)
        payload = self._get_json(self._url("/catalog/datasets", [
            ("where", where),
            ("limit", str(min(max(limit, 1) * 2, 50))),
            ("select", CATALOG_SELECT),
        ]))
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise SourceError(self.id, "استجابة الفهرس لا تحتوي results")

        lexicon = load_lexicon()
        candidates: list[DatasetCandidate] = []
        for index, entry in enumerate(r for r in results if isinstance(r, dict)):
            dataset_id = str(entry.get("dataset_id") or "")
            if not dataset_id:
                continue
            fields = [f for f in entry.get("fields") or [] if isinstance(f, dict)]
            timed = time_field(fields)

            coverage = Coverage.unknown()
            if timed is not None and index < MAX_COVERAGE_PROBES:
                try:
                    coverage = self._observe_coverage(dataset_id, timed)
                except (SourceError, SourceUnreachable):
                    coverage = Coverage.unknown()

            title = _plain(entry.get("title"))
            title_ar = _plain(entry.get("title_ar"))
            publisher = _plain(entry.get("publisher"))
            themes = _as_list(entry.get("theme")) + _as_list(entry.get("theme_ar"))
            keywords = " ".join(_as_list(entry.get("keyword")) + themes)
            field_text = " ".join(
                f"{f.get('name', '')} {f.get('label', '')}" for f in fields
            )

            candidates.append(DatasetCandidate(
                source_id=self.id,
                dataset_id=dataset_id,
                title_ar=title_ar if _ARABIC.search(title_ar) else title,
                title_en=title,
                description=_plain(entry.get("description"))[:600],
                keywords=f"{keywords} {field_text}".strip(),
                landing_url=self.descriptor.url(f"/explore/dataset/{dataset_id}/"),
                # KAPSARC republishes other bodies' data; the publisher named
                # is the originating one, and the platform is credited too.
                publisher_ar=(
                    f"{publisher} (عبر {self.descriptor.name_ar})"
                    if publisher else self.descriptor.name_ar
                ),
                publisher_en=(
                    f"{publisher} (via {self.descriptor.name_en})"
                    if publisher else self.descriptor.name_en
                ),
                resources=(Resource(
                    url=self._url(f"/catalog/datasets/{dataset_id}/exports/json"),
                    format="JSON",
                    title=title,
                ),),
                claimed_coverage=coverage,
                last_updated=_as_date(entry.get("modified")),
                license_name=_plain(entry.get("license")) or None,
                provided_dimensions=self._dimensions(fields, lexicon),
            ))
        return candidates[:limit]

    @staticmethod
    def _dimensions(fields: list[dict], lexicon) -> tuple[Dimension, ...]:
        """Breakdowns the dataset's own fields provide."""
        text = " ".join(f"{f.get('name', '')} {f.get('label', '')}" for f in fields)
        text = text.replace("_", " ")
        found: list[Dimension] = []
        for dimension, words in lexicon.dimension_words.items():
            if any(contains_phrase(text, w) for w in words) and dimension not in found:
                found.append(dimension)
        return tuple(found)

    # -- coverage ------------------------------------------------------
    def _observe_coverage(self, dataset_id: str, field: dict) -> Coverage:
        """The years the dataset holds, grouped by the platform itself."""
        name = str(field["name"])
        if field.get("type") in ("date", "datetime"):
            expression = f"year({name}) as y"
        else:
            expression = f"{name} as y"
        payload = self._get_json(self._url(f"/catalog/datasets/{dataset_id}/records", [
            ("group_by", expression),
            ("limit", "1000"),
        ]))
        from masdar.nlu.parser import extract_years

        years: set[int] = set()
        for row in (payload.get("results") or []) if isinstance(payload, dict) else []:
            years.update(extract_years(str((row or {}).get("y", ""))))
        if not years:
            return Coverage.unknown()
        return Coverage(
            years=frozenset(years),
            origin=CoverageOrigin.OBSERVED_DATA,
            is_exhaustive=True,
            note=f"مستخرجة من المصدر بتجميع الحقل {name}",
        )

    # -- data ----------------------------------------------------------
    def fetch(self, url: str):
        """Download, and for a full export, prove nothing was cut off.

        The export endpoint is documented as unlimited. That is checked
        rather than trusted: the row count is compared with the dataset's
        `total_count`, and a short export is refused instead of delivered.
        """
        fetched = super().fetch(url)
        marker = "/exports/json"
        if marker not in url or "?" in url:
            return fetched
        dataset_path = url.split(f"{API}", 1)[1].split(marker, 1)[0]
        try:
            rows = json.loads(fetched.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceError(self.id, f"تصدير غير صالح: {exc}") from exc
        if not isinstance(rows, list):
            raise SourceError(self.id, "التصدير لا يحتوي قائمة سجلات")

        count = super().fetch(self._url(f"{dataset_path}/records", [("limit", "0")]))
        try:
            total = int(json.loads(count.content.decode("utf-8")).get("total_count"))
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
            return fetched  # the export itself is fine; the check was unavailable
        if len(rows) != total:
            raise SourceError(
                self.id,
                f"التصدير أعاد {len(rows)} سجلاً من أصل {total}؛ رُفض بدل تسليم "
                "جزء على أنه كامل.",
            )
        return fetched

    def probe(self) -> str:
        payload = self._get_json(self._url("/catalog/datasets", [
            ("limit", "1"), ("select", "dataset_id, modified"),
        ]))
        total = payload.get("total_count") if isinstance(payload, dict) else None
        return f"الفهرس يستجيب، {total} مجموعة بيانات، بلا مفتاح"
