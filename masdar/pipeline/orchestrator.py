"""The agent loop: understand, search, verify, export, explain.

The order matters. Verification happens before export and before any wording
is chosen, so the sentence the user reads is derived from evidence already in
hand rather than from an intention formed earlier.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path

from masdar.domain.models import (
    Answer,
    Consulted,
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
from masdar.nlu.understand import LLM, Understanding, understand
from masdar.pipeline import resolve
from masdar.pipeline import verify as verify_module
from masdar.pipeline.reply import compose_message
from masdar.sources.base import SourceDescriptor, SourceError, SourceUnreachable
from masdar.sources.http import HttpClient
from masdar.sources.registry import Registry


@dataclass
class AgentConfig:
    # Enough for every verified source of a topic: several search a local
    # catalogue (specs, configured indicators) and cost no request to ask.
    max_sources: int = 6
    per_source_limit: int = 8
    # How many ranked candidates we are willing to open files for. Opening a
    # file is what upgrades a claim to OBSERVED_DATA, so this is the main
    # trade-off between cost and certainty.
    max_downloads: int = 3
    download: bool = True
    out_dir: Path = field(default_factory=lambda: Path("out"))
    export: bool = True
    today: date | None = None
    # A time limit for one answer, in seconds, for interactive use. Sources
    # are then asked in parallel; one still silent when its share of the
    # time is up is reported as not answering, and keeps working in the
    # background so its reply is cached for the next question. None: no
    # limit, sources asked one after another (the command line, the tests).
    answer_seconds: float | None = None


# Of an answer's time limit, the share the search may use; the rest is for
# opening files.
SEARCH_SHARE = 0.6
# However late it is, a file gets at least this long to arrive.
MIN_FETCH_SECONDS = 8.0
# A source that ran out of time is not asked again for this long, unless its
# late reply arrives first: the next question should not wait on it again.
SLOW_COOLDOWN_SECONDS = 120.0


@dataclass
class SearchOutcome:
    """What the search phase found, and how many sources were asked."""

    candidates: list[DatasetCandidate]
    errors: list[tuple[str, str]]
    audit: list[str]
    attempted: int
    consulted: list[Consulted] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        registry: Registry | None = None,
        http: HttpClient | None = None,
        lexicon: Lexicon | None = None,
        config: AgentConfig | None = None,
        llm=None,
        slow: dict[str, float] | None = None,
    ):
        self.config = config or AgentConfig()
        self.http = http or HttpClient()
        self.registry = registry or Registry.load(http=self.http)
        self.lexicon = lexicon or load_lexicon()
        # Optional: a language model reads questions the rules cannot (see
        # nlu/llm.py).
        self.llm = llm
        # Sources that ran out of time recently: id -> when to ask again.
        # May be shared between agents serving the same page.
        self._slow: dict[str, float] = {} if slow is None else slow

    # -- public --------------------------------------------------------
    def answer(self, query: str) -> Answer:
        understood = understand(query, self.lexicon, llm=self.llm)
        answer = self.answer_request(understood.request)
        return replace(answer, audit=(*understanding_audit(understood), *answer.audit))

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

        deadline = (
            time.monotonic() + self.config.answer_seconds
            if self.config.answer_seconds
            else None
        )
        search = self._search(request, deadline)
        candidates, errors, audit = search.candidates, search.errors, search.audit
        descriptors = {d.id: d for d in self.registry.descriptors}
        ranked = resolve.rank(request, candidates, descriptors, self.lexicon, today)
        off_subject = [c for c in ranked if not resolve.on_subject(request, c)]
        if off_subject:
            ranked = [c for c in ranked if resolve.on_subject(request, c)]
            audit.append(
                f"استُبعدت {len(off_subject)} نتيجة لا تذكر أياً من كلمات السؤال"
            )
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
                consulted=tuple(search.consulted),
            )
            return answer.with_message(compose_message(answer))

        findings = self._evaluate(request, ranked, descriptors, today, audit, deadline)
        international = frozenset(d.id for d in descriptors.values() if d.international)
        findings.sort(key=lambda f: _finding_sort_key(f, international))

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
            consulted=tuple(search.consulted),
        )
        return answer.with_message(compose_message(answer))

    # -- search --------------------------------------------------------
    def _search(self, request: DataRequest, deadline: float | None = None) -> SearchOutcome:
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
        consulted: list[Consulted] = []

        for descriptor, (found, error, note) in zip(
            plan, self._run_searches(plan, request, deadline), strict=True
        ):
            if error is not None:
                errors.append((descriptor.id, error))
                audit.append(f"{descriptor.name_ar}: {note}")
            consulted.append(Consulted(
                descriptor.id, descriptor.name_ar,
                None if found is None else len(found), descriptor.international,
            ))
            if found is None:
                continue

            candidates.extend(found)
            audit.append(f"{descriptor.name_ar}: {len(found)} نتيجة")

        return SearchOutcome(
            candidates=candidates, errors=errors, audit=audit, attempted=len(plan),
            consulted=consulted,
        )

    def _search_one(
        self, descriptor: SourceDescriptor, request: DataRequest
    ) -> tuple[list[DatasetCandidate] | None, str | None, str | None]:
        """(results, error, audit note) for one source. Never raises."""
        try:
            adapter = self.registry.adapter(descriptor.id)
            found = adapter.search(request, limit=self.config.per_source_limit)
        except SourceUnreachable as exc:
            return None, f"تعذّر الوصول: {exc.reason}", f"تعذّر الوصول ({exc.reason})"
        except SourceError as exc:
            return None, exc.reason, f"خطأ ({exc.reason})"
        except Exception as exc:  # an adapter bug must not sink the run
            return None, f"خطأ غير متوقع: {exc}", f"خطأ غير متوقع ({exc})"
        # A late reply means the source is back: ask it again at once.
        self._slow.pop(descriptor.id, None)
        return found, None, None

    def _run_searches(
        self, plan: list[SourceDescriptor], request: DataRequest, deadline: float | None
    ) -> list[tuple[list[DatasetCandidate] | None, str | None, str | None]]:
        """Every source's search, in plan order.

        Without a deadline one after another. With one, in parallel, waiting
        at most the search's share of the time; a source still busy then is
        reported as not answering and left to finish in the background.
        """
        if deadline is None:
            return [self._search_one(d, request) for d in plan]

        now = time.monotonic()
        resting = {
            d.id for d in plan if self._slow.get(d.id, 0.0) > now
        }
        asked = [d for d in plan if d.id not in resting]
        pool = ThreadPoolExecutor(max_workers=max(1, len(asked)), thread_name_prefix="search")
        futures = {d.id: pool.submit(self._search_one, d, request) for d in asked}
        budget = max(1.0, (deadline - now) * SEARCH_SHARE)
        wait(futures.values(), timeout=budget)
        pool.shutdown(wait=False)

        results = []
        for descriptor in plan:
            future = futures.get(descriptor.id)
            if future is None:
                results.append((
                    None, "لم يُجب في سؤال سابق قبل قليل؛ يُسأل مجدداً بعد دقيقتين",
                    "مستبعد مؤقتاً لبطئه في سؤال سابق",
                ))
            elif future.done():
                results.append(future.result())
            else:
                self._slow[descriptor.id] = time.monotonic() + SLOW_COOLDOWN_SECONDS
                seconds = round(budget)
                results.append((
                    None, f"لم يُجب خلال {seconds} ثانية", f"لم يُجب خلال {seconds} ثانية",
                ))
        return results

    # -- verify and export ---------------------------------------------
    def _evaluate(
        self,
        request: DataRequest,
        ranked: list[DatasetCandidate],
        descriptors: dict[str, SourceDescriptor],
        today: date,
        audit: list[str],
        deadline: float | None = None,
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
            notes: list[str] = list(candidate.caveats)

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
                    observed, table, fetched, problem = self._observe_within(
                        candidate, deadline
                    )
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
                if fetched is not None and fetched.stale:
                    notes.append(
                        "تعذّر الوصول إلى المصدر الآن، فاستُخدمت آخر نسخة محفوظة منه "
                        f"(جُلبت في {fetched.retrieved_at:%Y-%m-%d %H:%M} UTC). "
                        "قد تكون نُشرت بيانات أحدث بعد ذلك التاريخ."
                    )
                    audit.append(f"{candidate.title_ar}: نسخة محفوظة لتعذّر الوصول")
                elif fetched is not None and fetched.from_cache:
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

    def _observe_within(
        self, candidate: DatasetCandidate, deadline: float | None
    ) -> tuple[Coverage | None, Table | None, object | None, str | None]:
        """`_observe`, bounded by what is left of the answer's time.

        A file that has not arrived in time is treated like one that could
        not be reached; the download carries on in the background and is
        cached, so asking again shortly after finds it at once.
        """
        if deadline is None:
            return self._observe(candidate)
        seconds = max(MIN_FETCH_SECONDS, deadline - time.monotonic())
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fetch")
        future = pool.submit(self._observe, candidate)
        try:
            return future.result(timeout=seconds)
        except FutureTimeout:
            raise _Unreachable(f"لم يصل الملف خلال {round(seconds)} ثانية") from None
        finally:
            pool.shutdown(wait=False)

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
            stale=bool(fetched is not None and fetched.stale),
        )


def understanding_audit(understood: Understanding) -> tuple[str, ...]:
    """How the question was read, first in the audit trail."""
    if understood.method == LLM:
        line = "فُهم السؤال بمساعدة نموذج لغوي (لم تحدد القواعد موضوعاً)"
        return (f"{line}: {understood.note}",) if understood.note else (line,)
    return (understood.note,) if understood.note else ()


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


def _finding_sort_key(finding: Finding, international: frozenset[str] = frozenset()) -> tuple:
    rank: float = _VERDICT_RANK.get(finding.verdict, 9)
    # "Unverified" with nothing known about the years -- a dataset never
    # opened -- carries no evidence at all, so it does not outrank a source
    # that was opened and showed the year is absent. The answer then says
    # what was established, and the unopened result is still listed.
    if finding.verdict is Verdict.UNVERIFIED and finding.coverage.is_empty:
        rank = _VERDICT_RANK[Verdict.NOT_AVAILABLE] + 0.5
    return (
        rank,
        # At the same verdict, the Saudi publisher's own file comes before an
        # international compiler's copy; a better verdict still wins.
        finding.candidate.source_id in international,
        0 if finding.export_path else 1,
        -finding.candidate.score,
    )
