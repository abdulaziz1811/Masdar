"""Core domain model for Masdar.

Every type here exists to serve one rule: the agent may only tell the user
something it can point at. A claim without a `Provenance` cannot be built,
and an availability answer without `Coverage` evidence degrades to
`Verdict.UNVERIFIED` rather than guessing.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from datetime import date, datetime

# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------

class Calendar(enum.Enum):
    GREGORIAN = "gregorian"
    HIJRI = "hijri"


class PeriodKind(enum.Enum):
    SINGLE = "single"          # "2026"
    RANGE = "range"            # "from 2019 to 2024"
    LATEST = "latest"          # "the newest available"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True)
class Period:
    """A requested span of time, always normalised to Gregorian years.

    A Hijri year straddles two Gregorian years, so `years` may hold both
    candidates while `calendar` remembers what the user actually typed.
    """

    years: tuple[int, ...] = ()
    kind: PeriodKind = PeriodKind.UNSPECIFIED
    calendar: Calendar = Calendar.GREGORIAN
    raw: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "years", tuple(sorted(set(self.years))))

    @property
    def is_open_ended(self) -> bool:
        return self.kind in (PeriodKind.LATEST, PeriodKind.UNSPECIFIED)

    def label(self) -> str:
        if not self.years:
            # Without a period the agent answers with the newest data it finds.
            return "أحدث سنة متاحة" if self.kind is PeriodKind.LATEST else "الأحدث المتاح"
        if len(self.years) == 1:
            return str(self.years[0])
        return f"{self.years[0]}–{self.years[-1]}"


# --------------------------------------------------------------------------
# What the user asked for
# --------------------------------------------------------------------------

class Dimension(enum.Enum):
    """A breakdown axis the user asked the numbers to be split by."""

    REGION = "region"           # حسب المناطق
    CITY = "city"
    SECTOR = "sector"           # حسب القطاع
    GENDER = "gender"
    NATIONALITY = "nationality"
    AGE = "age"
    MONTH = "month"
    QUARTER = "quarter"
    ACTIVITY = "activity"
    SEASON = "season"

    @property
    def label_ar(self) -> str:
        return _DIMENSION_LABELS_AR.get(self, self.value)


_DIMENSION_LABELS_AR = {
    Dimension.REGION: "المناطق",
    Dimension.CITY: "المدن",
    Dimension.SECTOR: "القطاع",
    Dimension.GENDER: "الجنس",
    Dimension.NATIONALITY: "الجنسية",
    Dimension.AGE: "الفئة العمرية",
    Dimension.MONTH: "الشهر",
    Dimension.QUARTER: "الربع",
    Dimension.ACTIVITY: "النشاط",
    Dimension.SEASON: "الفصل",
}


@dataclass(frozen=True)
class Topic:
    """A canonical subject, decoupled from the words the user used for it."""

    id: str
    label_ar: str
    label_en: str
    keywords_ar: tuple[str, ...] = ()
    keywords_en: tuple[str, ...] = ()
    # Registry source ids expected to publish this topic, best first.
    preferred_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataRequest:
    """A parsed, machine-actionable version of the user's question."""

    raw_query: str
    period: Period = field(default_factory=Period)
    topic: Topic | None = None
    dimensions: tuple[Dimension, ...] = ()
    free_terms: tuple[str, ...] = ()
    # Lexicon phrases that actually occur in the question. They say far more
    # than the topic's full keyword list: "متوسط العمر المتوقع" picks out one
    # health dataset, whereas the health keywords match them all.
    typed_phrases: tuple[str, ...] = ()
    language: str = "ar"
    # Regions named in the question («في الرياض»): the answer is narrowed to
    # them where the data has a column of regions.
    places: tuple[str, ...] = ()
    # Parts of the question the parser could not resolve; surfaced to the
    # user instead of being silently dropped.
    unresolved: tuple[str, ...] = ()

    @property
    def is_answerable(self) -> bool:
        """Enough signal to attempt a search at all."""
        return bool(self.topic or self.free_terms)

    def search_terms(self) -> tuple[str, ...]:
        terms: list[str] = []
        if self.topic:
            terms.extend([self.topic.label_ar, self.topic.label_en])
            terms.extend(self.topic.keywords_ar)
        terms.extend(self.free_terms)
        seen: dict[str, None] = {}
        for t in terms:
            if t:
                seen.setdefault(t, None)
        return tuple(seen)


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------

class CoverageOrigin(enum.Enum):
    """Where a coverage statement came from, ordered by how much we trust it."""

    OBSERVED_DATA = "observed_data"      # parsed out of the actual file
    METADATA_CLAIM = "metadata_claim"    # the portal's declared coverage
    INFERRED_TITLE = "inferred_title"    # guessed from a title/filename

    @property
    def is_authoritative(self) -> bool:
        return self is CoverageOrigin.OBSERVED_DATA

    @property
    def can_support_availability(self) -> bool:
        """Title-guessing may never be used to promise data exists."""
        return self in (CoverageOrigin.OBSERVED_DATA, CoverageOrigin.METADATA_CLAIM)


@dataclass(frozen=True)
class Coverage:
    """Which years a dataset actually holds, and how we know."""

    years: frozenset[int]
    origin: CoverageOrigin
    # True when `years` is known to enumerate everything the dataset holds.
    is_exhaustive: bool = False
    note: str = ""

    @classmethod
    def unknown(cls) -> Coverage:
        return cls(
            years=frozenset(),
            origin=CoverageOrigin.INFERRED_TITLE,
            note="لا توجد بيانات عن التغطية الزمنية",
        )

    @property
    def is_empty(self) -> bool:
        return not self.years

    def latest(self) -> int | None:
        return max(self.years) if self.years else None

    def describe(self) -> str:
        """Human-readable coverage that does not imply continuity.

        Printing "2017–2022" for a source missing 2020 reads as an unbroken
        run, so gaps are named. Real series have them: GASTAT's
        electricity-connection data skips 2020.
        """
        if not self.years:
            return "غير معروفة"
        ordered = sorted(self.years)
        if len(ordered) == 1:
            return str(ordered[0])
        missing = [y for y in range(ordered[0], ordered[-1] + 1) if y not in self.years]
        span = f"{ordered[0]}\u2013{ordered[-1]}"
        if not missing:
            return span
        if len(missing) <= 5:
            return f"{span} (ناقصة: {'، '.join(str(y) for y in missing)})"
        return f"{span} (متقطعة، المتاح {len(ordered)} سنة)"


@dataclass(frozen=True)
class Provenance:
    """The audit trail for one retrieved artefact. Never synthesised."""

    source_id: str
    publisher_ar: str
    publisher_en: str
    landing_url: str
    retrieved_at: datetime
    resource_url: str | None = None
    # The publisher's own "last updated" stamp, not our fetch time.
    last_updated: date | None = None
    last_updated_note: str = ""
    sha256: str | None = None
    byte_size: int | None = None
    media_type: str | None = None
    license_name: str | None = None
    # The source could not be reached, and this is an older saved copy.
    stale: bool = False

    def citation_ar(self) -> str:
        parts = [self.publisher_ar]
        if self.last_updated:
            parts.append(f"آخر تحديث: {self.last_updated.isoformat()}")
        else:
            parts.append("آخر تحديث: غير معلن")
        parts.append(f"تاريخ الاستخراج: {self.retrieved_at.date().isoformat()}")
        parts.append(self.resource_url or self.landing_url)
        return " — ".join(parts)


@dataclass(frozen=True)
class Resource:
    """A downloadable file attached to a dataset."""

    url: str
    format: str                      # normalised upper-case: XLSX, CSV, PDF...
    title: str = ""
    byte_size: int | None = None
    media_type: str | None = None

    @property
    def is_tabular(self) -> bool:
        return self.format in {"XLSX", "XLS", "CSV", "TSV", "JSON"}


@dataclass
class DatasetCandidate:
    """A dataset a source offered us, before we decide whether it fits."""

    source_id: str
    dataset_id: str
    title_ar: str
    title_en: str = ""
    description: str = ""
    # Search words a source attaches to a dataset beyond its title. Used for
    # ranking only, never shown: a title says "الطاقة الكهربائية" where a user
    # types "الكهرباء", and curated keywords are what bridge the two.
    keywords: str = ""
    landing_url: str = ""
    publisher_ar: str = ""
    publisher_en: str = ""
    resources: tuple[Resource, ...] = ()
    claimed_coverage: Coverage = field(default_factory=Coverage.unknown)
    last_updated: date | None = None
    license_name: str | None = None
    # Breakdowns this result is known to provide, when the source can say so
    # -- an API asked to group by a dimension provides it by construction.
    # Sources that cannot declare leave this empty and are judged by their
    # column names instead.
    provided_dimensions: tuple[Dimension, ...] = ()
    # What a reader must know about this kind of data, stated by the source
    # adapter and carried into the answer's notes (e.g. "a live snapshot,
    # not an annual statistic").
    caveats: tuple[str, ...] = ()
    # Set when this server cannot fetch the file (the national platform's
    # files from outside the Kingdom): nothing is downloaded, the verdict
    # stands on the source's declared coverage, and this says why the reader
    # gets a link instead of a workbook.
    download_note: str = ""
    # How closely the source's own search matched the question, from 0 to 1,
    # for a source that can tell apart what the generic ranker sees as ties
    # (the World Bank's many near-identical indicator names). 0 means the
    # source gave no opinion.
    relevance: float = 0.0
    # Filled in by the ranker.
    score: float = 0.0
    match_reasons: tuple[str, ...] = ()

    def tabular_resources(self) -> list[Resource]:
        """Every file that can become a table, most convenient first."""
        order = {"XLSX": 0, "XLS": 1, "CSV": 2, "TSV": 3, "JSON": 4}
        return sorted(
            (r for r in self.resources if r.is_tabular),
            key=lambda r: order.get(r.format, 99),
        )

    def best_tabular_resource(self) -> Resource | None:
        ranked = self.tabular_resources()
        return ranked[0] if ranked else None


# --------------------------------------------------------------------------
# What we hand back
# --------------------------------------------------------------------------

class Verdict(enum.Enum):
    """The only four outcomes the agent is allowed to report."""

    AVAILABLE = "available"              # verified present for the whole period
    PARTIAL = "partial"                 # verified present for part of it
    NOT_AVAILABLE = "not_available"     # verified absent
    UNVERIFIED = "unverified"           # found something, cannot confirm the period
    NO_SOURCE = "no_source"             # searched successfully, found nothing
    # Distinct from NO_SOURCE on purpose: "I could not reach the sources" is
    # not the same claim as "the data does not exist", and conflating the two
    # is the most misleading thing this system could do.
    SOURCE_UNREACHABLE = "source_unreachable"

    @property
    def is_positive(self) -> bool:
        return self in (Verdict.AVAILABLE, Verdict.PARTIAL)


@dataclass(frozen=True)
class YearSuggestion:
    """A fallback offered when the asked-for year is not published."""

    year: int
    reason_ar: str
    # Set when the year comes from a different dataset than the answer's:
    # the offer is then "this other publisher has newer data", and saying
    # whose it is keeps the offer from reading as a claim about the first.
    source_ar: str = ""
    title_ar: str = ""


@dataclass
class Finding:
    """One source's verified answer for the request."""

    candidate: DatasetCandidate
    provenance: Provenance
    verdict: Verdict
    coverage: Coverage
    matched_years: tuple[int, ...] = ()
    missing_years: tuple[int, ...] = ()
    export_path: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Consulted:
    """One source asked during a search, and what it returned."""

    source_id: str
    name_ar: str
    # Results offered, or None when the source could not answer.
    found: int | None
    international: bool = False


@dataclass
class Answer:
    """The complete, self-describing reply to one DataRequest."""

    request: DataRequest
    verdict: Verdict
    findings: tuple[Finding, ...] = ()
    suggestions: tuple[YearSuggestion, ...] = ()
    # Human-readable trace of what was searched and why it was rejected.
    audit: tuple[str, ...] = ()
    # Sources that failed to answer, as (source_id, reason). Reported
    # alongside any verdict so partial outages are never invisible.
    source_errors: tuple[tuple[str, str], ...] = ()
    # Every source asked, in the order asked, so an answer can show where it
    # looked -- including the sources that had nothing.
    consulted: tuple[Consulted, ...] = ()
    # Sources that publish on the topic but this server cannot reach at all
    # (outside the Kingdom), by Arabic name: "nothing found" must not read as
    # "nothing exists" when they were never asked.
    not_searched: tuple[str, ...] = ()
    # Where a reader can look on, as (label, url): GASTAT's own search for the
    # same words when nothing was found here.
    elsewhere: tuple[tuple[str, str], ...] = ()
    message_ar: str = ""

    @property
    def primary(self) -> Finding | None:
        return self.findings[0] if self.findings else None

    def with_message(self, message: str) -> Answer:
        return replace(self, message_ar=message)
