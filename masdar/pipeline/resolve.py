"""Ranking candidates against a request.

Scoring is additive and every component records a human-readable reason, so
the ordering can always be explained to the user. An opaque relevance score
would undermine the point of an agent that justifies its answers.
"""

from __future__ import annotations

from datetime import date

from masdar.domain.models import DataRequest, DatasetCandidate, Dimension
from masdar.nlu.lexicon import Lexicon, load_lexicon
from masdar.nlu.normalize import contains_phrase
from masdar.sources.base import SourceDescriptor

W_TERM = 2.0
W_TOPIC_LABEL = 3.0
W_YEAR_CLAIMED = 6.0
W_TABULAR = 5.0
W_DIMENSION = 3.0
W_AUTHORITY = 0.06      # 0-100 authority contributes up to 6 points
W_FRESHNESS = 2.0


def _freshness(last_updated: date | None, today: date) -> float:
    if last_updated is None:
        return 0.0
    age_years = max(0.0, (today - last_updated).days / 365.25)
    # Full marks within a year, decaying to zero at five.
    return W_FRESHNESS * max(0.0, 1.0 - age_years / 5.0)


def score_candidate(
    request: DataRequest,
    candidate: DatasetCandidate,
    descriptor: SourceDescriptor | None = None,
    lexicon: Lexicon | None = None,
    today: date | None = None,
) -> DatasetCandidate:
    """Return the candidate with `score` and `match_reasons` filled in."""
    lexicon = lexicon or load_lexicon()
    today = today or date.today()
    haystack = " ".join(
        filter(None, (candidate.title_ar, candidate.title_en, candidate.description))
    )

    score = 0.0
    reasons: list[str] = []

    if request.topic:
        for label in (request.topic.label_ar, request.topic.label_en):
            if label and contains_phrase(haystack, label):
                score += W_TOPIC_LABEL
                reasons.append(f"العنوان يذكر الموضوع: {label}")
                break

    hits = [t for t in request.free_terms if contains_phrase(haystack, t)]
    if hits:
        score += W_TERM * len(hits)
        reasons.append(f"تطابق مصطلحات: {', '.join(hits)}")

    requested = frozenset(request.period.years)
    if requested and candidate.claimed_coverage.years:
        overlap = requested & candidate.claimed_coverage.years
        if overlap:
            score += W_YEAR_CLAIMED
            reasons.append(f"التغطية المعلنة تشمل: {sorted(overlap)}")

    if candidate.best_tabular_resource() is not None:
        score += W_TABULAR
        reasons.append("يتضمن ملفاً جدولياً قابلاً للتحويل إلى إكسل")
    elif candidate.resources:
        reasons.append("الملفات المتاحة غير جدولية (PDF مثلاً)")

    for dimension in request.dimensions:
        words = lexicon.dimension_words.get(dimension, ())
        if any(contains_phrase(haystack, word) for word in words):
            score += W_DIMENSION
            reasons.append(f"العنوان يشير إلى التفصيل المطلوب: {dimension.value}")

    if descriptor is not None:
        score += W_AUTHORITY * descriptor.authority
        if descriptor.authority >= 90:
            reasons.append(f"مصدر مرجعي: {descriptor.name_ar}")

    fresh = _freshness(candidate.last_updated, today)
    if fresh > 0:
        score += fresh
        reasons.append(f"محدَّث في {candidate.last_updated}")

    candidate.score = round(score, 3)
    candidate.match_reasons = tuple(reasons)
    return candidate


def rank(
    request: DataRequest,
    candidates: list[DatasetCandidate],
    descriptors: dict[str, SourceDescriptor] | None = None,
    lexicon: Lexicon | None = None,
    today: date | None = None,
) -> list[DatasetCandidate]:
    descriptors = descriptors or {}
    lexicon = lexicon or load_lexicon()
    scored = [
        score_candidate(request, c, descriptors.get(c.source_id), lexicon, today)
        for c in candidates
    ]
    # Deterministic: score, then source authority, then title.
    scored.sort(
        key=lambda c: (
            -c.score,
            -(descriptors[c.source_id].authority if c.source_id in descriptors else 0),
            c.title_ar,
        )
    )
    return scored


def missing_dimensions(
    request: DataRequest, present: set[Dimension]
) -> tuple[Dimension, ...]:
    """Requested breakdowns the data does not actually provide."""
    return tuple(d for d in request.dimensions if d not in present)
