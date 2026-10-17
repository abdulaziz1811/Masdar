"""Whether an opened result is what was asked, beyond its years.

`verify` answers one question: are the years asked in the evidence? This
module answers the others, which can only be asked once the file is open --
the table read from a workbook, its columns -- and which may weaken what
`verify` concluded:

* **A release, not a series.** One bulletin lacking a past year says nothing
  of the other bulletins (`RELEASE_NOTE`): «غير متوفرة» becomes unverified.
* **The measure.** A count asked, and the table read is shares.
* **The subject.** Named by the title, the keywords or the table read -- not
  only by a description that lists everything a dataset touches.
* **The month.** A monthly release for another month holds the year, not
  necessarily the month asked: unverified.
* **The breakdown.** «حسب المناطق» answered by a national total: partial.

Every rule here has a test named after the answer that made it necessary
(tests/test_subject_fit.py, tests/test_logic.py, tests/test_gastat_site.py).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from masdar.domain.models import DataRequest, DatasetCandidate, Dimension, Verdict
from masdar.export.tabular import Table
from masdar.nlu.lexicon import Lexicon
from masdar.nlu.normalize import months_named
from masdar.pipeline import resolve
from masdar.pipeline import verify as verify_module
from masdar.pipeline.verify import Verification

RELEASE_NOTE = (
    "هذا إصدار واحد من إصدارات الجهة (يغطي {held})، لا سلسلة كاملة؛ عدم ظهور "
    "{asked} فيه لا يعني أنها لم تُنشر في إصدار آخر."
)
MEASURE_NOTE = "طُلب عدد، والجدول المقروء من الملف نسب أو معدلات."
_MONTH_NAMES_AR = ("يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو",
                   "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر")


@dataclass(frozen=True)
class Fit:
    """`verify`'s conclusion, adjusted for how well the result fits the question."""

    verification: Verification
    # Said before anything else in the answer's notes, in this order.
    leading_notes: tuple[str, ...] = ()
    table_title: str = ""
    measure_mismatch: bool = False
    names_subject: bool = True
    missing_dimensions: tuple[Dimension, ...] = ()


def judge(
    request: DataRequest,
    candidate: DatasetCandidate,
    table: Table | None,
    verification: Verification,
    lexicon: Lexicon,
    today: date,
) -> Fit:
    """Weigh an opened (or unopened: `table` None) result against the question."""
    verification = _as_one_release(candidate, verification, today)

    # In a workbook of tables, which one was read, and is it what was asked:
    # «عدد المشتغلين» once got «التوزيع النسبي للمشتغلين (%)».
    table_title = table.title if table is not None and table.sheet_name else ""
    mismatch = bool(table_title) and resolve.text_measure_mismatch(request, table_title)
    named = resolve.names_subject(request, candidate, table_title)
    absent = _missing_breakdown(request, candidate, table, table_title, lexicon)

    leading: list[str] = []
    if absent:
        labels = "، ".join(d.label_ar for d in absent)
        leading.append(f"الملف لا يحتوي التفصيل المطلوب ({labels})؛ الأرقام فيه على مستوى أعم.")
        if verification.verdict is Verdict.AVAILABLE:
            # «حسب المناطق» answered with a national total is not the data
            # asked for: the years are there, the breakdown is not.
            verification = replace(verification, verdict=Verdict.PARTIAL)

    months = resolve.other_month(request, candidate)
    if months and verification.verdict.is_positive:
        # The year is in the file; the month asked may not be.
        leading.append(
            f"هذا الإصدار لشهر {_month_names(months)}، والمطلوب "
            f"{_month_names(months_named(request.raw_query))}؛ لم يُتحقق من وجود "
            "الشهر المطلوب في الملف."
        )
        verification = replace(verification, verdict=Verdict.UNVERIFIED)

    if mismatch:
        leading.append(MEASURE_NOTE)

    return Fit(
        verification=verification,
        leading_notes=tuple(leading),
        table_title=table_title,
        measure_mismatch=mismatch,
        names_subject=named,
        missing_dimensions=absent,
    )


def _as_one_release(
    candidate: DatasetCandidate, verification: Verification, today: date
) -> Verification:
    """A past year missing from one bulletin is not missing from all of them.

    The 2026 Hajj bulletin holds 2026; that is no evidence 2023 was never
    published. A year not yet over is another matter: that it is missing is
    the expected answer (as in orchestrator `_absence_needs_every_lead`).
    """
    if not (
        candidate.is_release
        and verification.verdict in (Verdict.NOT_AVAILABLE, Verdict.UNVERIFIED)
        and any(year < today.year for year in verification.missing_years)
        and verification.coverage.years
    ):
        return verification
    return replace(
        verification,
        verdict=Verdict.UNVERIFIED,
        notes=(
            *(n for n in verification.notes if n != verify_module.TITLE_YEAR_NOTE),
            RELEASE_NOTE.format(
                held=years_text(verification.coverage.years),
                asked=years_text(verification.missing_years),
            ),
        ),
    )


def _missing_breakdown(
    request: DataRequest,
    candidate: DatasetCandidate,
    table: Table | None,
    table_title: str,
    lexicon: Lexicon,
) -> tuple[Dimension, ...]:
    """Breakdowns asked for that the opened table does not provide."""
    if table is None or not request.dimensions:
        return ()
    present = table.observed_dimensions(
        lexicon.dimension_words,
        lexicon.places if Dimension.REGION in request.dimensions else None,
    )
    # A source that grouped by a dimension provides it, whatever its column
    # happens to be called; a title that already says it («للسعوديين») is
    # that breakdown's one value.
    present |= set(candidate.provided_dimensions)
    present |= resolve.dimensions_named(
        request, f"{candidate.title_ar} {candidate.title_en} {table_title}",
        lexicon.dimension_words,
    )
    return resolve.missing_dimensions(request, present)


def years_text(years) -> str:
    """«2023» or «2019–2023»."""
    years = sorted(years)
    if not years:
        return "—"
    return str(years[0]) if len(years) == 1 else f"{years[0]}–{years[-1]}"


def _month_names(months: frozenset[int]) -> str:
    return "، ".join(_MONTH_NAMES_AR[m - 1] for m in sorted(months))
