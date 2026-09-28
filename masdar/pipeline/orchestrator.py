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
        return self.answer_request(parse(query, self.lexicon))

    def answer_request(self, request: DataRequest) -> Answer:
        """Answer an already-understood request.

        A conversation builds requests from more than one message ("and for
        2021?" keeps the previous topic), so understanding and answering are
        separate steps.
        """
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

        # A search can succeed from a local catalogue while the data behind it
        # cannot be fetched; such sources are listed with the ones whose search
        # failed, so no outage is invisible.
        reported = {source_id for source_id, _ in errors}
        for finding in findings:
            source_id = finding.candidate.source_id
            if finding.verdict is Verdict.SOURCE_UNREACHABLE and source_id not in reported:
                reported.add(source_id)
                errors.append((source_id, "تعذّر تنزيل البيانات من المصدر"))
        suggestions = (
            verify_module.suggest_years(request, best.coverage, best.matched_years)
            if best
            else ()
        )
        newer = verify_module.newer_elsewhere(best, findings) if best else None
        if newer is not None:
            suggestions = (newer, *suggestions)

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
        # Sources that could not deliver a file for this question. Their other
        # datasets are not tried again: each attempt would spend a download on
        # a source already known to be down, starving the ones that answer.
        down: dict[str, str] = {}

        for candidate in ranked:
            descriptor = descriptors.get(candidate.source_id)
            coverage = candidate.claimed_coverage
            table: Table | None = None
            fetched = None
            notes: list[str] = []

            if candidate.source_id in down and (
                coverage.is_empty or not coverage.origin.can_support_availability
            ):
                findings.append(
                    Finding(
                        candidate=candidate,
                        provenance=self._provenance(candidate, descriptor),
                        verdict=Verdict.SOURCE_UNREACHABLE,
                        coverage=coverage,
                        notes=(
                            "لم يُحاوَل التنزيل: المصدر لم يُجب قبل قليل "
                            f"({down[candidate.source_id]})",
                        ),
                    )
                )
                continue

            may_download = (
                self.config.download
                and downloads < self.config.max_downloads
                and candidate.best_tabular_resource() is not None
            )
            if may_download:
                downloads += 1
                try:
                    observed, table, fetched, problem = self._observe(candidate)
                except _Unreachable as exc:
                    down.setdefault(candidate.source_id, exc.reason)
                    known = candidate.claimed_coverage
                    if known.is_empty or not known.origin.can_support_availability:
                        # No file could be fetched and nothing trustworthy is
                        # known about the years: "unverified" would read as a
                        # finding. It is an outage, and sorts as one.
                        audit.append(f"{candidate.title_ar}: {exc.reason}")
                        findings.append(
                            Finding(
                                candidate=candidate,
                                provenance=self._provenance(candidate, descriptor),
                                verdict=Verdict.SOURCE_UNREACHABLE,
                                coverage=known,
                                notes=(exc.reason,),
                            )
                        )
                        continue
                    # The source already told us its years; only the file is
                    # missing, so the verdict stands on that and no workbook
                    # is written.
                    observed, table, fetched, problem = None, None, None, exc.reason
                if problem:
                    notes.append(problem)
                    audit.append(f"{candidate.title_ar}: {problem}")
                if fetched is not None and fetched.from_cache:
                    notes.append(
                        "الملف من نسخة محفوظة جُلبت من المصدر في "
                        f"{fetched.retrieved_at:%Y-%m-%d %H:%M} (UTC)."
                    )
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
                sliced, dropped = sliced.without_inapplicable_columns()
                if dropped:
                    notes.append(
                        f"أُزيلت من الملف أعمدة كل قيمها «لا ينطبق» ({len(dropped)}): "
                        + "، ".join(dropped[:6]) + ("…" if len(dropped) > 6 else "")
                    )
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

            # A confirmed, exported answer for the years asked ends the search;
            # anything weaker keeps looking in case a better source exists. A
            # request for "the latest" keeps looking too: the first dataset
            # with any year answers it, but not necessarily with the newest.
            if (
                verification.verdict is Verdict.AVAILABLE
                and export_path
                and request.period.years
            ):
                break

        return findings

    def _observe(
        self, candidate: DatasetCandidate
    ) -> tuple[Coverage | None, Table | None, object | None, str | None]:
        """Open the actual file to see which years it holds.

        Every tabular file of the dataset is tried in order until one opens
        with a readable year axis: publishers often attach the same table as
        XLSX and CSV, and one damaged copy should not cost the answer.
        """
        resources = candidate.tabular_resources()
        if not resources:
            return None, None, None, None

        problems: list[str] = []
        fallback: tuple[Table, object] | None = None
        unreachable = 0
        for resource in resources:
            try:
                fetched = self._fetch(candidate.source_id, resource.url)
            except SourceUnreachable as exc:
                unreachable += 1
                problems.append(f"تعذّر تنزيل ملف {resource.format}: {exc.reason}")
                continue
            except SourceError as exc:
                problems.append(f"تعذّر قراءة ملف {resource.format}: {exc.reason}")
                continue
            try:
                table = read_table(fetched.content, resource.format)
            except Exception as exc:
                problems.append(f"تعذّر تحليل ملف {resource.format}: {exc}")
                continue

            years = table.observed_years()
            if not years:
                fallback = fallback or (table, fetched)
                problems.append(f"ملف {resource.format} لا يحتوي عموداً يحدد السنة")
                continue

            note = "؛ ".join(problems) if problems else None
            if note:
                note = f"استُخدم ملف {resource.format} بعد تعذّر غيره: {note}"
            return (
                Coverage(
                    years=years,
                    origin=CoverageOrigin.OBSERVED_DATA,
                    is_exhaustive=True,
                    note="تم استخراج السنوات من محتوى الملف نفسه",
                ),
                table,
                fetched,
                note,
            )

        # Nothing had a year axis. A file that opened can still be exported,
        # but it cannot upgrade the claim; the declared coverage stands.
        joined = "؛ ".join(problems)
        if unreachable == len(resources):
            raise _Unreachable(joined)
        if fallback is not None:
            return None, fallback[0], fallback[1], joined
        return None, None, None, joined

    def _fetch(self, source_id: str, url: str):
        """Let the owning adapter download its resource, credentials and all."""
        if self.registry.get(source_id) is not None:
            return self.registry.adapter(source_id).fetch(url)
        return self.http.get(url, source_id=source_id)

    def _provenance(
        self,
        candidate: DatasetCandidate,
        descriptor: SourceDescriptor | None,
        fetched=None,
    ) -> Provenance:
        resource = candidate.best_tabular_resource() or (
            candidate.resources[0] if candidate.resources else None
        )
        # Cite the file that was actually read -- the sha256 is of that file,
        # which may not be the one ranked first if that one failed to open.
        cited_url = fetched.url if fetched is not None else (
            resource.url if resource else None
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
            resource_url=cited_url,
            last_updated=last_updated,
            last_updated_note=note,
            sha256=fetched.sha256 if fetched else None,
            byte_size=fetched.byte_size if fetched else None,
            media_type=fetched.media_type if fetched else None,
            license_name=candidate.license_name,
        )


class _Unreachable(Exception):
    """Every file of a dataset failed for lack of access, none for content."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


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
    rank: float = _VERDICT_RANK.get(finding.verdict, 9)
    # "Unverified" with nothing known about the years -- a dataset never
    # opened -- carries no evidence at all, so it does not outrank a source
    # that was opened and showed the year is absent. The answer then says
    # what was established, and the unopened result is still listed.
    if finding.verdict is Verdict.UNVERIFIED and finding.coverage.is_empty:
        rank = _VERDICT_RANK[Verdict.NOT_AVAILABLE] + 0.5
    return (
        rank,
        0 if finding.export_path else 1,
        -finding.candidate.score,
    )
