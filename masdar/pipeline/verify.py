"""Deciding what may be claimed about a requested period.

This module is the reason the agent can be trusted. It converts evidence
into one of the `Verdict` values under a single rule: the strength of the
claim may never exceed the strength of the evidence behind it.

* Data read out of the file itself (`OBSERVED_DATA`) can prove presence and
  absence.
* A portal's declared coverage (`METADATA_CLAIM`) can support presence, and
  absence only when the declaration is exhaustive.
* A year guessed from a title (`INFERRED_TITLE`) proves nothing. It may
  direct the search, never the answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from masdar.domain.models import (
    Coverage,
    CoverageOrigin,
    DataRequest,
    Finding,
    PeriodKind,
    Verdict,
    YearSuggestion,
)

# Annual statistics are typically released well into the following year, so a
# year that has only just ended is normally not published yet. Used for
# explanations and suggestions only -- never to override evidence.
TYPICAL_ANNUAL_LAG_MONTHS = 6


@dataclass(frozen=True)
class Verification:
    verdict: Verdict
    coverage: Coverage
    matched_years: tuple[int, ...] = ()
    missing_years: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()


def _plausibility_notes(request: DataRequest, today: date) -> tuple[str, ...]:
    """Explain, in advance, why a very recent year is unlikely to exist."""
    notes: list[str] = []
    for year in request.period.years:
        if year > today.year:
            notes.append(f"سنة {year} لم تبدأ بعد، لذا لا توجد إحصاءات سنوية لها.")
        elif year == today.year:
            notes.append(
                f"سنة {year} لم تنتهِ بعد؛ الإحصاءات السنوية الكاملة لها عادةً "
                "تُنشر بعد نهاية السنة."
            )
        elif year == today.year - 1 and today.month <= TYPICAL_ANNUAL_LAG_MONTHS:
            notes.append(
                f"سنة {year} انتهت حديثاً، وقد لا تكون الإحصاءات السنوية لها "
                "قد نُشرت بعد."
            )
    return tuple(notes)


def verify(request: DataRequest, coverage: Coverage, today: date | None = None) -> Verification:
    """Compare what was asked for against what we can actually show."""
    today = today or date.today()
    requested = frozenset(request.period.years)

    # "latest available" / no period given: any coverage answers it.
    if not requested:
        if coverage.is_empty:
            return Verification(
                verdict=Verdict.UNVERIFIED,
                coverage=coverage,
                notes=("لا توجد معلومات عن السنوات التي يغطيها المصدر.",),
            )
        latest = coverage.latest()
        assert latest is not None
        if not coverage.origin.can_support_availability:
            return Verification(
                verdict=Verdict.UNVERIFIED,
                coverage=coverage,
                notes=(
                    f"السنة {latest} مستنتجة من العنوان فقط ولم يتم التحقق منها "
                    "داخل الملف.",
                ),
            )
        kind_note = (
            "لم تُحدَّد سنة في الطلب، فتم اختيار أحدث سنة متاحة."
            if request.period.kind is PeriodKind.UNSPECIFIED
            else "تم اختيار أحدث سنة متاحة."
        )
        return Verification(
            verdict=Verdict.AVAILABLE,
            coverage=coverage,
            matched_years=(latest,),
            notes=(kind_note,),
        )

    matched = tuple(sorted(requested & coverage.years))
    missing = tuple(sorted(requested - coverage.years))
    notes = list(_plausibility_notes(request, today))

    if coverage.is_empty:
        return Verification(
            verdict=Verdict.UNVERIFIED,
            coverage=coverage,
            missing_years=tuple(sorted(requested)),
            notes=tuple([*notes, "تعذّر تحديد السنوات التي يغطيها هذا المصدر."]),
        )

    # A title hint can point the search at a dataset but can never answer it.
    if coverage.origin is CoverageOrigin.INFERRED_TITLE:
        hint = (
            f"العنوان يشير إلى السنوات {sorted(coverage.years)} لكن لم يتم فتح "
            "الملف للتأكد، فلا يمكن تأكيد التوفر."
        )
        return Verification(
            verdict=Verdict.UNVERIFIED,
            coverage=coverage,
            matched_years=matched,
            missing_years=missing,
            notes=tuple([*notes, hint]),
        )

    if coverage.origin is CoverageOrigin.METADATA_CLAIM:
        notes.append("التوفر مبني على البيانات الوصفية المعلنة من المصدر.")
        if not matched and not coverage.is_exhaustive:
            # The declaration may simply be incomplete; that is not proof of
            # absence.
            return Verification(
                verdict=Verdict.UNVERIFIED,
                coverage=coverage,
                missing_years=missing,
                notes=tuple([
                    *notes,
                    "التغطية المعلنة لا تشمل السنة المطلوبة، لكنها غير معلنة "
                    "كقائمة كاملة، فلا يمكن الجزم بعدم التوفر.",
                ]),
            )

    if not matched:
        return Verification(
            verdict=Verdict.NOT_AVAILABLE,
            coverage=coverage,
            missing_years=missing,
            notes=tuple(notes),
        )
    if missing:
        return Verification(
            verdict=Verdict.PARTIAL,
            coverage=coverage,
            matched_years=matched,
            missing_years=missing,
            notes=tuple(notes),
        )
    return Verification(
        verdict=Verdict.AVAILABLE,
        coverage=coverage,
        matched_years=matched,
        notes=tuple(notes),
    )


def suggest_years(
    request: DataRequest,
    coverage: Coverage,
    matched_years: tuple[int, ...] = (),
    limit: int = 3,
) -> tuple[YearSuggestion, ...]:
    """Nearest usable years when the requested one is not there.

    Drawn only from coverage strong enough to stand behind, so a user is
    never redirected to a year that is itself a guess. Nothing is suggested
    when the request was already satisfied, and "the latest available year"
    always means the true maximum -- never the maximum of what is left after
    the requested years are removed.
    """
    if coverage.is_empty or not coverage.origin.can_support_availability:
        return ()

    requested = frozenset(request.period.years)
    matched = frozenset(matched_years)

    # Already answered: an open-ended request that got a year, or an explicit
    # request whose every year was found.
    if not requested and matched:
        return ()
    if requested and requested <= matched:
        return ()

    def is_useful(year: int) -> bool:
        return year not in matched and year not in requested

    suggestions: list[YearSuggestion] = []
    seen: set[int] = set()

    def add(year: int, reason: str) -> None:
        if is_useful(year) and year not in seen:
            seen.add(year)
            suggestions.append(YearSuggestion(year=year, reason_ar=reason))

    add(max(coverage.years), "أحدث سنة متاحة في هذا المصدر")

    # Neighbours of the years we could not deliver, not of the request as a
    # whole: if 2023 was found and 2025 was not, 2022 is not a useful offer.
    unmet = sorted(requested - matched) or sorted(requested)
    for target in unmet[:2]:
        below = [y for y in coverage.years if y < target]
        above = [y for y in coverage.years if y > target]
        if below:
            add(max(below), f"أقرب سنة متاحة قبل {target}")
        if above:
            add(min(above), f"أقرب سنة متاحة بعد {target}")

    return tuple(suggestions[:limit])


def newer_elsewhere(best: Finding, others: Sequence[Finding]) -> YearSuggestion | None:
    """A strictly newer year held by another dataset whose file was opened.

    The answer is chosen for relevance first, so a dataset that matches the
    question less closely is not promoted over it merely for being newer.
    But "what is the latest?" is the question, and silence about a newer
    year the agent itself has seen would be a quiet omission. Only files
    actually opened count: they were among the most relevant results, and
    their years are observed, not declared.
    """
    baseline = best.coverage.latest() if best.coverage.origin.can_support_availability else None
    newest: Finding | None = None
    for finding in others:
        if finding is best or finding.provenance.sha256 is None:
            continue
        if finding.coverage.origin is not CoverageOrigin.OBSERVED_DATA:
            continue
        latest = finding.coverage.latest()
        if latest is None or (baseline is not None and latest <= baseline):
            continue
        if newest is None or latest > (newest.coverage.latest() or 0):
            newest = finding
    if newest is None:
        return None
    return YearSuggestion(
        year=newest.coverage.latest() or 0,
        reason_ar="أحدث سنة وجدتها في مصدر آخر",
        source_ar=newest.provenance.publisher_ar,
        title_ar=newest.candidate.title_ar or newest.candidate.title_en,
    )
