"""The agent loop: understand, search, verify, export, explain.

The order matters. Verification happens before export and before any wording
is chosen, so the sentence the user reads is derived from evidence already in
hand rather than from an intention formed earlier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from masdar.domain.models import (
    Answer,
    Coverage,
    CoverageOrigin,
    DataRequest,
    DatasetCandidate,
    Finding,
    Provenance,
    Verdict,
)
from masdar.export import excel
from masdar.export.tabular import Table, read_table
from masdar.nlu.lexicon import Lexicon, load_lexicon
from masdar.nlu.parser import parse
from masdar.pipeline import resolve
from masdar.pipeline import verify as verify_module
from masdar.pipeline.reply import compose_message
from masdar.sources.base import SourceDescriptor, SourceError, SourceUnreachable
from masdar.sources.http import HttpClient
from masdar.sources.registry import Registry


@dataclass
class AgentConfig:
    max_sources: int = 4
    per_source_limit: int = 8
    # How many ranked candidates we are willing to open files for. Opening a
    # file is what upgrades a claim to OBSERVED_DATA, so this is the main
    # trade-off between cost and certainty.
    max_downloads: int = 3
    download: bool = True
    out_dir: Path = field(default_factory=lambda: Path("out"))
    export: bool = True
    today: date | None = None


@dataclass
class SearchOutcome:
    """What the search phase found, and how many sources were asked."""

    candidates: list[DatasetCandidate]
    errors: list[tuple[str, str]]
    audit: list[str]
    attempted: int


class Agent:
    def __init__(
        self,
        registry: Registry | None = None,
        http: HttpClient | None = None,
        lexicon: Lexicon | None = None,
        config: AgentConfig | None = None,
    ):
        self.config = config or AgentConfig()
        self.http = http or HttpClient()
        self.registry = registry or Registry.load(http=self.http)
        self.lexicon = lexicon or load_lexicon()

    # -- public --------------------------------------------------------
    def answer(self, query: str) -> Answer:
        request = parse(query, self.lexicon)
        today = self.config.today or date.today()

        if not request.is_answerable:
            return Answer(
                request=request,
                verdict=Verdict.NO_SOURCE,
                audit=("لم يتضمن الطلب موضوعاً يمكن البحث عنه.",),
            ).with_message(compose_message(
                Answer(request=request, verdict=Verdict.NO_SOURCE)
            ))

        search = self._search(request)
        candidates, errors, audit = search.candidates, search.errors, search.audit
        descriptors = {d.id: d for d in self.registry.descriptors}
        ranked = resolve.rank(request, candidates, descriptors, self.lexicon, today)
        audit.append(f"عدد النتائج المرشحة بعد الترتيب: {len(ranked)}")

        if not ranked:
            # "every source I tried failed" is a different statement from
            # "the sources answered and had nothing".
            verdict = (
                Verdict.SOURCE_UNREACHABLE
                if search.attempted and len(errors) == search.attempted
                else Verdict.NO_SOURCE
            )
            answer = Answer(
                request=request,
                verdict=verdict,
                audit=tuple(audit),
                source_errors=tuple(errors),
            )
            return answer.with_message(compose_message(answer))

        findings = self._evaluate(request, ranked, descriptors, today, audit)
        findings.sort(key=_finding_sort_key)

        best = findings[0] if findings else None
        verdict = best.verdict if best else Verdict.NO_SOURCE
        suggestions = (
            verify_module.suggest_years(request, best.coverage, best.matched_years)
            if best
            else ()
        )

        answer = Answer(
            request=request,
            verdict=verdict,
            findings=tuple(findings),
            suggestions=suggestions,
            audit=tuple(audit),
            source_errors=tuple(errors),
        )
        return answer.with_message(compose_message(answer))

    # -- search --------------------------------------------------------
    def _search(self, request: DataRequest) -> SearchOutcome:
        topic_id = request.topic.id if request.topic else None
        preferred = request.topic.preferred_sources if request.topic else ()
        full_plan = self.registry.plan(topic_id, preferred)
        # Real sources first, capped; demo sources are additive so that
        # --demo always exercises the pipeline even when the cap is small.
        real = [d for d in full_plan if not d.synthetic][: self.config.max_sources]
        plan = real + [d for d in full_plan if d.synthetic]

        audit = [
            "الموضوع المستخرج: " + (request.topic.label_ar if request.topic else "غير محدد"),
            "الفترة المطلوبة: " + request.period.label(),
            "ترتيب المصادر: " + "، ".join(d.name_ar for d in plan),
        ]
        candidates: list[DatasetCandidate] = []
        errors: list[tuple[str, str]] = []

        for descriptor in plan:
            try:
                adapter = self.registry.adapter(descriptor.id)
                found = adapter.search(request, limit=self.config.per_source_limit)
            except SourceUnreachable as exc:
                errors.append((descriptor.id, f"تعذّر الوصول: {exc.reason}"))
                audit.append(f"{descriptor.name_ar}: تعذّر الوصول ({exc.reason})")
                continue
            except SourceError as exc:
                errors.append((descriptor.id, exc.reason))
                audit.append(f"{descriptor.name_ar}: خطأ ({exc.reason})")
                continue
            except Exception as exc:  # an adapter bug must not sink the run
                errors.append((descriptor.id, f"خطأ غير متوقع: {exc}"))
                audit.append(f"{descriptor.name_ar}: خطأ غير متوقع ({exc})")
                continue

            candidates.extend(found)
            audit.append(f"{descriptor.name_ar}: {len(found)} نتيجة")

        return SearchOutcome(
            candidates=candidates, errors=errors, audit=audit, attempted=len(plan)
        )

    # -- verify and export ---------------------------------------------
    def _evaluate(
        self,
        request: DataRequest,
        ranked: list[DatasetCandidate],
        descriptors: dict[str, SourceDescriptor],
        today: date,
        audit: list[str],
    ) -> list[Finding]:
        findings: list[Finding] = []
        downloads = 0

        for candidate in ranked:
            descriptor = descriptors.get(candidate.source_id)
            coverage = candidate.claimed_coverage
            table: Table | None = None
            fetched = None
            notes: list[str] = []

            may_download = (
                self.config.download
                and downloads < self.config.max_downloads
                and candidate.best_tabular_resource() is not None
            )
            if may_download:
                downloads += 1
                observed, table, fetched, problem = self._observe(candidate)
                if problem:
                    notes.append(problem)
                    audit.append(f"{candidate.title_ar}: {problem}")
                if observed is not None:
                    coverage = _merge_coverage(candidate.claimed_coverage, observed)
                    audit.append(
                        f"{candidate.title_ar}: تم فتح الملف والتحقق من "
                        f"{len(observed.years)} سنة"
                    )
            elif candidate.best_tabular_resource() is None:
                notes.append("لا يتوفر ملف جدولي لهذه النتيجة، لذا لم يُنشأ ملف إكسل.")

            verification = verify_module.verify(request, coverage, today=today)
            notes.extend(verification.notes)

            if table is not None:
                present = table.observed_dimensions(self.lexicon.dimension_words)
                # A source that grouped by a dimension provides it, whatever
                # its column happens to be called.
                present |= set(candidate.provided_dimensions)
                for dimension in resolve.missing_dimensions(request, present):
                    notes.append(
                        "تنبيه: الملف لا يحتوي عموداً للتفصيل المطلوب "
                        f"({dimension.label_ar})."
                    )

            provenance = self._provenance(candidate, descriptor, fetched)
            export_path: str | None = None
            if (
                self.config.export
                and table is not None
                and verification.verdict.is_positive
            ):
                sliced = table.filter_years(frozenset(verification.matched_years))
                if sliced.rows:
                    path = excel.default_path(
                        self.config.out_dir,
                        request,
                        candidate,
                        years=verification.matched_years,
                    )
                    excel.write_workbook(
                        path=path,
                        table=sliced,
                        request=request,
                        candidate=candidate,
                        provenance=provenance,
                        verdict=verification.verdict,
                        coverage=verification.coverage,
                        notes=tuple(notes),
                        synthetic=bool(descriptor and descriptor.synthetic),
                    )
                    export_path = str(path)

            findings.append(
                Finding(
                    candidate=candidate,
                    provenance=provenance,
                    verdict=verification.verdict,
                    coverage=verification.coverage,
                    matched_years=verification.matched_years,
                    missing_years=verification.missing_years,
                    export_path=export_path,
                    notes=tuple(notes),
                )
            )

            # A confirmed, exported answer ends the search; anything weaker
            # keeps looking in case a better source exists.
            if verification.verdict is Verdict.AVAILABLE and export_path:
                break

        return findings

    def _observe(
        self, candidate: DatasetCandidate
    ) -> tuple[Coverage | None, Table | None, object | None, str | None]:
        """Open the actual file to see which years it holds."""
        resource = candidate.best_tabular_resource()
        if resource is None:
            return None, None, None, None
        try:
            fetched = self.http.get(resource.url, source_id=candidate.source_id)
        except SourceUnreachable as exc:
            return None, None, None, f"تعذّر تنزيل الملف للتحقق: {exc.reason}"
        except SourceError as exc:
            return None, None, None, f"تعذّر قراءة الملف: {exc.reason}"

        try:
            table = read_table(fetched.content, resource.format)
        except Exception as exc:
            return None, None, fetched, f"تعذّر تحليل الملف ({resource.format}): {exc}"

        years = table.observed_years()
        if not years:
            # The file opened but has no readable year axis, so it cannot
            # upgrade the claim; the declared coverage stands.
            return (
                None,
                table,
                fetched,
                "تم فتح الملف لكن لم يتم التعرف على عمود أو عنوان يحدد السنة.",
            )

        return (
            Coverage(
                years=years,
                origin=CoverageOrigin.OBSERVED_DATA,
                is_exhaustive=True,
                note="تم استخراج السنوات من محتوى الملف نفسه",
            ),
            table,
            fetched,
            None,
        )

    def _provenance(
        self,
        candidate: DatasetCandidate,
        descriptor: SourceDescriptor | None,
        fetched=None,
    ) -> Provenance:
        resource = candidate.best_tabular_resource() or (
            candidate.resources[0] if candidate.resources else None
        )
        last_updated = candidate.last_updated
        note = "" if last_updated else "لم يعلن المصدر تاريخ تحديث لهذه البيانات"
        if last_updated is None and fetched is not None:
            server_stamp = fetched.last_modified()
            if server_stamp is not None:
                last_updated = server_stamp.date()
                note = "مأخوذ من ترويسة Last-Modified للملف، وليس من بيانات المصدر الوصفية"

        return Provenance(
            source_id=candidate.source_id,
            publisher_ar=candidate.publisher_ar or (descriptor.name_ar if descriptor else ""),
            publisher_en=candidate.publisher_en or (descriptor.name_en if descriptor else ""),
            landing_url=candidate.landing_url,
            retrieved_at=fetched.retrieved_at if fetched else datetime.now(UTC),
            resource_url=resource.url if resource else None,
            last_updated=last_updated,
            last_updated_note=note,
            sha256=fetched.sha256 if fetched else None,
            byte_size=fetched.byte_size if fetched else None,
            media_type=fetched.media_type if fetched else None,
            license_name=candidate.license_name,
        )


def _merge_coverage(enumerated: Coverage, observed: Coverage) -> Coverage:
    """Combine coverage a source enumerated with coverage seen in a download.

    A downloaded response can be one page of a larger result -- APIs cap rows
    by default -- so the years visible in it may be fewer than the years that
    exist. Where a source has already enumerated its coverage authoritatively
    (GASTAT answers a `dimensions[]=YEAR` query with one row per year), that
    enumeration must not be narrowed by a partial page: doing so would
    manufacture a false absence, which is the one error this system must never
    make. The two are therefore unioned, never replaced.
    """
    if enumerated.origin is CoverageOrigin.OBSERVED_DATA and not enumerated.is_empty:
        return Coverage(
            years=enumerated.years | observed.years,
            origin=CoverageOrigin.OBSERVED_DATA,
            is_exhaustive=enumerated.is_exhaustive,
            note=enumerated.note,
        )
    return observed


_VERDICT_RANK = {
    Verdict.AVAILABLE: 0,
    Verdict.PARTIAL: 1,
    Verdict.UNVERIFIED: 2,
    Verdict.NOT_AVAILABLE: 3,
    Verdict.SOURCE_UNREACHABLE: 4,
    Verdict.NO_SOURCE: 5,
}


def _finding_sort_key(finding: Finding) -> tuple:
    return (
        _VERDICT_RANK.get(finding.verdict, 9),
        0 if finding.export_path else 1,
        -finding.candidate.score,
    )
