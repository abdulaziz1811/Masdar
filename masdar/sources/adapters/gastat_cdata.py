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

from pathlib import Path
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
from masdar.nlu.lexicon import CONFIG_DIR
from masdar.nlu.normalize import contains_phrase
from masdar.sources.base import SourceAdapter, SourceError, SourceUnreachable
from masdar.sources.openapi import datasets_from_dir

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
    # GASTAT splits household energy by winter vs the rest of the year.
    Dimension.SEASON: ("CONSUMP_OPERATION_PERIOD", "SEASON"),
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
    def _spec_dir(self) -> Path | None:
        configured = self.descriptor.api.get("spec_dir")
        if not configured:
            return None
        path = Path(str(configured))
        return path if path.is_absolute() else CONFIG_DIR / path

    def spec_problems(self) -> list[str]:
        """Spec files that could not be read, for `masdar specs`/doctor."""
        directory = self._spec_dir()
        if directory is None:
            return []
        return datasets_from_dir(directory, self.descriptor.base_url)[1]

    def _datasets(self) -> list[dict]:
        """Hand-declared datasets, then every dataset in the spec directory.

        A declared entry wins over a spec entry with the same id: it was
        written by someone who looked at the data, and may carry a better
        title or keywords than the portal's.
        """
        declared = self.descriptor.api.get("datasets") or []
        if not isinstance(declared, list):
            raise SourceError(self.id, "`datasets` في الإعدادات يجب أن تكون قائمة")
        entries = [dict(d) for d in declared if isinstance(d, dict) and d.get("id")]
        seen = {str(e["id"]) for e in entries}

        directory = self._spec_dir()
        if directory is not None:
            from_specs, _ = datasets_from_dir(directory, self.descriptor.base_url)
            for entry in from_specs:
                if str(entry["id"]) not in seen:
                    seen.add(str(entry["id"]))
                    entries.append(entry)
        return entries

    def _path(self, entry: dict) -> str:
        dataset_id = str(entry["id"])
        path = entry.get("path")
        if not path:
            template = self.descriptor.api.get("dataset_path", "/v1/stats/{id}")
            path = template.replace("{id}", dataset_id)
        # A spec may name a server; openapi.py only keeps one on this host.
        server = entry.get("server")
        if server:
            return f"{str(server).rstrip('/')}/{str(path).lstrip('/')}"
        return self.descriptor.url(str(path))

    @staticmethod
    def _time_dimension(entry: dict) -> str:
        explicit = entry.get("time_dimension")
        if explicit:
            return str(explicit)
        dims = [str(d) for d in entry.get("dimensions") or []]
        return TIME_DIMENSION if TIME_DIMENSION in dims else ""

    def _query_url(self, entry: dict, params: list[tuple[str, str]]) -> str:
        # Brackets kept literal: this is the form verified against the API.
        return f"{self._path(entry)}?{urlencode(params, safe='[]')}"

    def _requested_dimensions(
        self, request: DataRequest, available: list[str]
    ) -> list[tuple[Dimension, str]]:
        """The asked-for breakdowns this dataset can actually group by.

        Returns both our dimension and the API's name for it, so the
        candidate can report what it provides without the rest of the system
        having to recognise names like CONSUMP_OPERATION_PERIOD.
        """
        chosen: list[tuple[Dimension, str]] = []
        taken: set[str] = set()
        for dimension in request.dimensions:
            for name in DIMENSION_NAMES.get(dimension, ()):
                if name in available and name not in taken:
                    chosen.append((dimension, name))
                    taken.add(name)
                    break
        return chosen

    # -- coverage ------------------------------------------------------
    def _observe_coverage(self, entry: dict) -> Coverage:
        """Ask the dataset which periods it holds.

        Grouping by the time dimension alone returns one row per period, so
        this is authoritative evidence at the cost of a few rows -- the
        publisher enumerating its own coverage rather than us inferring it.
        """
        time_dimension = self._time_dimension(entry)
        if not time_dimension:
            return Coverage.unknown()
        url = self._query_url(
            entry, [("dimensions[]", time_dimension), ("format", DEFAULT_FORMAT)]
        )
        fetched = self.fetch(url)
        table = read_json(fetched.content)
        years = table.observed_years()
        if not years:
            return Coverage.unknown()
        return Coverage(
            years=years,
            origin=CoverageOrigin.OBSERVED_DATA,
            is_exhaustive=True,
            note=f"مستخرجة من المصدر باستعلام dimensions[]={time_dimension}",
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
            time_dimension = self._time_dimension(entry)

            grouped = self._requested_dimensions(request, available)
            params: list[tuple[str, str]] = []
            for _, name in grouped:
                params.append(("dimensions[]", name))
            if time_dimension:
                params.append(("dimensions[]", time_dimension))
            params.append(("format", DEFAULT_FORMAT))

            # Coverage is worth a request only for the few best matches.
            coverage = Coverage.unknown()
            if index < MAX_COVERAGE_PROBES and time_dimension:
                try:
                    coverage = self._observe_coverage(entry)
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
                    landing_url=self._path(entry),
                    publisher_ar=self.descriptor.name_ar,
                    publisher_en=self.descriptor.name_en,
                    resources=(
                        Resource(
                            url=self._query_url(entry, params),
                            format="JSON",
                            title=str(entry.get("title_ar") or dataset_id),
                        ),
                    ),
                    claimed_coverage=coverage,
                    license_name=str(entry.get("license") or "") or None,
                    # The server grouped by these, so they are present by
                    # construction -- no need to guess from column names.
                    provided_dimensions=tuple(d for d, _ in grouped),
                )
            )
        return candidates

    # -- diagnostics ---------------------------------------------------
    def probe(self) -> str:
        datasets = self._datasets()
        if not datasets:
            raise SourceError(self.id, "لا توجد مجموعات بيانات معلنة في الإعدادات")
        dataset_id = str(datasets[0]["id"])
        coverage = self._observe_coverage(datasets[0])
        years = sorted(coverage.years)
        span = f"{years[0]}–{years[-1]}" if years else "غير معروفة"
        keyed = "مع مفتاح" if self.request_headers() else "بلا مفتاح"
        return (
            f"{len(datasets)} مجموعة معلنة، {dataset_id}: {len(years)} سنة "
            f"({span}) — {keyed}"
        )
