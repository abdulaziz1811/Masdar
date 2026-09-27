"""Adapter for GASTAT's cdata statistics API (api.stats.gov.sa).

Verified live on 2026-09-27 against
`https://api.stats.gov.sa/v1/stats/{dataset_id}`. The "Public" APIs answer
anonymous requests -- the developer-portal key and secret are not required
for them, though a key is sent if one is configured, for any API that does
need it.

The API is OData-flavoured and unusually well suited to this agent:

* `dimensions[]=REGION&dimensions[]=YEAR` groups server-side, which answers
  a request for "حسب المناطق" directly instead of by post-processing.
* `format=JSON` returns `{"value": [...]}`; `CSV` and `HTML` are also offered.
* Columns are suffixed by role: `YEAR_TIME` carries the period, `*_OBSV` the
  measures, and `*_ARAB`/`*_ENGL`/`*_CODE` the dimension labels.
* `$top`, `$skip` and `$orderby` page and sort.

The important consequence is coverage. Asking for `dimensions[]=YEAR` alone
returns one row per year the dataset holds -- a handful of rows rather than
the whole table -- so the years can be established authoritatively and
cheaply, from the publisher's own data, before anything is downloaded. On
the electricity-connection dataset that query returns 2017-2019 and
2021-2022: no 2020, and nothing after 2022. A request for 2026 is therefore
answered as a verified absence, not a guess.
"""

from __future__ import annotations

from urllib.parse import urlencode

from masdar.domain.models import (
    Coverage,
    CoverageOrigin,
    DataRequest,
    DatasetCandidate,
    Dimension,
    Resource,
)
from masdar.export.tabular import read_json
from masdar.nlu.normalize import contains_phrase
from masdar.sources.base import SourceAdapter, SourceError, SourceUnreachable

# Our dimension vocabulary mapped onto the API's dimension names. Only the
# ones a dataset actually declares are ever requested.
DIMENSION_NAMES: dict[Dimension, tuple[str, ...]] = {
    Dimension.REGION: ("REGION",),
    Dimension.CITY: ("CITY", "GOVERNORATE"),
    Dimension.SECTOR: ("SECTOR", "ECONOMIC_ACTIVITY"),
    Dimension.GENDER: ("GENDER", "SEX"),
    Dimension.NATIONALITY: ("NATIONALITY",),
    Dimension.AGE: ("AGE_GROUP", "AGE"),
    Dimension.MONTH: ("MONTH",),
    Dimension.QUARTER: ("QUARTER",),
    Dimension.ACTIVITY: ("ACTIVITY", "ECONOMIC_ACTIVITY"),
}

TIME_DIMENSION = "YEAR"
DEFAULT_FORMAT = "JSON"
MAX_COVERAGE_PROBES = 5


class GastatCdataAdapter(SourceAdapter):
    """Serves the datasets declared for this source in sources.yaml.

    The API has no catalogue endpoint, so the datasets it exposes are listed
    in configuration. Each entry is a dataset id from the developer portal
    plus its titles, dimensions and topics; adding another is a config edit,
    not a code change.
    """

    # -- configuration -------------------------------------------------
    def _datasets(self) -> list[dict]:
        declared = self.descriptor.api.get("datasets") or []
        if not isinstance(declared, list):
            raise SourceError(self.id, "`datasets` في الإعدادات يجب أن تكون قائمة")
        return [d for d in declared if isinstance(d, dict) and d.get("id")]

    def _path(self, dataset_id: str) -> str:
        template = self.descriptor.api.get("dataset_path", "/v1/stats/{id}")
        return self.descriptor.url(template.replace("{id}", dataset_id))

    def _auth_headers(self) -> dict[str, str]:
        """An API key only if one is configured; Public APIs need none.

        Read from the environment at call time. Credentials are never stored
        in this repository.
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

    def _query_url(self, dataset_id: str, params: list[tuple[str, str]]) -> str:
        # Brackets kept literal: this is the form verified against the API.
        return f"{self._path(dataset_id)}?{urlencode(params, safe='[]')}"

    def _requested_dimensions(self, request: DataRequest, available: list[str]) -> list[str]:
        """Dimension names to group by: the ones asked for that exist here."""
        chosen: list[str] = []
        for dimension in request.dimensions:
            for name in DIMENSION_NAMES.get(dimension, ()):
                if name in available and name not in chosen:
                    chosen.append(name)
                    break
        return chosen

    # -- coverage ------------------------------------------------------
    def _observe_coverage(self, dataset_id: str) -> Coverage:
        """Ask the dataset which years it holds.

        `dimensions[]=YEAR` returns one row per year, so this is authoritative
        evidence at the cost of a few rows -- the publisher enumerating its
        own coverage rather than us inferring it.
        """
        url = self._query_url(
            dataset_id, [("dimensions[]", TIME_DIMENSION), ("format", DEFAULT_FORMAT)]
        )
        fetched = self.http.get(url, source_id=self.id)
        table = read_json(fetched.content)
        years = table.observed_years()
        if not years:
            return Coverage.unknown()
        return Coverage(
            years=years,
            origin=CoverageOrigin.OBSERVED_DATA,
            is_exhaustive=True,
            note="مستخرجة من المصدر باستعلام dimensions[]=YEAR",
        )

    # -- search --------------------------------------------------------
    def search(self, request: DataRequest, limit: int = 10) -> list[DatasetCandidate]:
        if self.http is None:
            raise SourceError(self.id, "no HTTP client configured")

        terms = request.search_terms()
        topic_id = request.topic.id if request.topic else None

        matched: list[tuple[float, dict]] = []
        for entry in self._datasets():
            score = 0.0
            if topic_id and topic_id in (entry.get("topics") or []):
                score += 3.0
            haystack = " ".join(
                str(entry.get(key, ""))
                for key in ("title_ar", "title_en", "description", "keywords")
            )
            score += sum(1 for term in terms if term and contains_phrase(haystack, term))
            if score <= 0:
                continue
            available = [str(d) for d in (entry.get("dimensions") or [])]
            if self._requested_dimensions(request, available):
                score += 2.0
            matched.append((score, entry))

        matched.sort(key=lambda pair: -pair[0])

        candidates: list[DatasetCandidate] = []
        for index, (_, entry) in enumerate(matched[:limit]):
            dataset_id = str(entry["id"])
            available = [str(d) for d in (entry.get("dimensions") or [])]

            params: list[tuple[str, str]] = []
            for name in self._requested_dimensions(request, available):
                params.append(("dimensions[]", name))
            if TIME_DIMENSION in available:
                params.append(("dimensions[]", TIME_DIMENSION))
            params.append(("format", DEFAULT_FORMAT))

            # Coverage is worth a request only for the few best matches.
            coverage = Coverage.unknown()
            if index < MAX_COVERAGE_PROBES and TIME_DIMENSION in available:
                try:
                    coverage = self._observe_coverage(dataset_id)
                except (SourceError, SourceUnreachable):
                    # Not fatal: the orchestrator will still open the data.
                    coverage = Coverage.unknown()

            candidates.append(
                DatasetCandidate(
                    source_id=self.id,
                    dataset_id=dataset_id,
                    title_ar=str(entry.get("title_ar") or entry.get("title_en") or dataset_id),
                    title_en=str(entry.get("title_en") or ""),
                    description=str(entry.get("description") or ""),
                    landing_url=self._path(dataset_id),
                    publisher_ar=self.descriptor.name_ar,
                    publisher_en=self.descriptor.name_en,
                    resources=(
                        Resource(
                            url=self._query_url(dataset_id, params),
                            format="JSON",
                            title=str(entry.get("title_ar") or dataset_id),
                        ),
                    ),
                    claimed_coverage=coverage,
                    license_name=str(entry.get("license") or "") or None,
                )
            )
        return candidates

    # -- diagnostics ---------------------------------------------------
    def probe(self) -> str:
        datasets = self._datasets()
        if not datasets:
            raise SourceError(self.id, "لا توجد مجموعات بيانات معلنة في الإعدادات")
        dataset_id = str(datasets[0]["id"])
        coverage = self._observe_coverage(dataset_id)
        years = sorted(coverage.years)
        span = f"{years[0]}–{years[-1]}" if years else "غير معروفة"
        keyed = "مع مفتاح" if self._auth_headers() else "بلا مفتاح"
        return (
            f"{len(datasets)} مجموعة معلنة، {dataset_id}: {len(years)} سنة "
            f"({span}) — {keyed}"
        )
