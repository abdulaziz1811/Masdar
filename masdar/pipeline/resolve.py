"""Ranking candidates against a request.

Scoring is additive and every component records a human-readable reason, so
the ordering can always be explained to the user. An opaque relevance score
would undermine the point of an agent that justifies its answers.
"""

from __future__ import annotations

from datetime import date

from masdar.domain.models import DataRequest, DatasetCandidate, Dimension
from masdar.nlu.lexicon import Lexicon, load_lexicon
from masdar.nlu.normalize import (
    contains_phrase,
    mentions,
    normalize,
    period_grain,
    stems,
    strip_article,
    strip_phrase_articles,
    title_period,
)
from masdar.nlu.parser import MEASURES, shared_measures, typed_phrase_score
from masdar.sources.base import SourceDescriptor

W_TERM = 2.0
W_TOPIC_LABEL = 3.0
W_YEAR_CLAIMED = 6.0
W_TABULAR = 5.0
W_DIMENSION = 3.0
W_AUTHORITY = 0.06      # 0-100 authority contributes up to 6 points
W_FRESHNESS = 2.0
# Per year of how recent a period a title names, for "the latest", over the
# last LATEST_SPAN years: a month is a tenth of a point, above the noise of a
# site's own ordering -- July's CPI once came before August's.
W_LATEST_PERIOD = 1.2
LATEST_SPAN = 3
# A whole year asked, a release for the whole year found: GASTAT's annual
# industrial production index before its December bulletin.
W_WHOLE_YEAR = 2.0
# Per pair of the question's words found side by side in the title, as asked:
# «لأسعار المستهلك» is the consumer price index, while «مؤشر ثقة المستهلك»
# shares two words with «مؤشر لأسعار المستهلك» but none of its pairs.
W_WORD_PAIR = 1.5
# An international compiler's copy of a figure ranks below the Saudi
# publisher's own, even when both match the question equally.
W_INTERNATIONAL = -3.0
# A source's own reading of how well a result fits (see
# DatasetCandidate.relevance), worth up to this much.
W_SOURCE_RELEVANCE = 4.0
# A source's own relevance at or above this counts as being on the subject
# (see `on_subject`); the World Bank adapter already drops anything below it.
SUBJECT_RELEVANCE = 0.5
# The quantity asked for (see `shared_measures`).
W_MEASURE = 2.0
# A count asked, a rate found: «عدد المستشفيات» is not «أسرّة المستشفيات لكل
# 1000 شخص». Worth the measure bonus in reverse, and said in the answer.
W_MEASURE_MISMATCH = -2.0
_RATE_MARKERS = ("%", "لكل", "per", "نسبه", "معدل")
# Words the publishers use for the same thing, normalised.
SYNONYMS = {
    "نفطيه": ("بتروليه",),
    "نفط": ("بترول",),
    "بتروليه": ("نفطيه",),
    # A publisher known by its short name: the platform says the full one.
    "سدايا": ("الهيئة السعودية للبيانات والذكاء الاصطناعي", "sdaia"),
}
# Free terms too plain to say what a result must be about.
_PLAIN = frozenset({"غير", "حسب", "لعدد", "بعدد"})
# A publisher's name says whose data, not what data: «عدد الخيول في سدايا»
# is not answered by any SDAIA table.
_PUBLISHER_NAMES = frozenset({"سدايا", "sdaia"})


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
        filter(
            None,
            (candidate.title_ar, candidate.title_en, candidate.description, candidate.keywords),
        )
    )

    score = 0.0
    reasons: list[str] = []

    if request.topic:
        for label in (request.topic.label_ar, request.topic.label_en):
            if label and contains_phrase(haystack, label):
                score += W_TOPIC_LABEL
                reasons.append(f"العنوان يذكر الموضوع: {label}")
                break

    shared = shared_measures(request.raw_query, candidate.title_ar or candidate.title_en)
    if shared:
        score += W_MEASURE * len(shared)
        reasons.append(f"يطابق ما يُقاس في السؤال: {', '.join(shared)}")
    elif measure_mismatch(request, candidate):
        score += W_MEASURE_MISMATCH
        reasons.append("السؤال يطلب عدداً، والنتيجة نسبة أو معدل")

    typed, typed_hits = typed_phrase_score(request, haystack)
    if typed:
        score += typed
        reasons.append(f"يطابق عبارة السؤال: {', '.join(typed_hits)}")

    pairs = [p for p in _word_pairs(request.raw_query, lexicon.stopwords)
             if contains_phrase(candidate.title_ar or candidate.title_en, p)]
    if pairs:
        score += W_WORD_PAIR * len(pairs)
        reasons.append(f"كلمات السؤال متجاورة في العنوان: {'، '.join(pairs)}")

    # Naming the publisher («... من وزارة الموارد البشرية») asks for that
    # publisher's data, so its name counts as matching words; it only adds
    # to the ranking, never lets an off-subject result through (`on_subject`).
    publisher = " ".join(filter(None, (candidate.publisher_ar, candidate.publisher_en)))
    hits = [
        t for t in request.free_terms
        if contains_phrase(haystack, t) or (publisher and contains_phrase(publisher, t))
    ]
    if hits:
        score += W_TERM * len(hits)
        reasons.append(f"تطابق مصطلحات: {', '.join(hits)}")

    requested = frozenset(request.period.years)
    if requested and candidate.claimed_coverage.years:
        overlap = requested & candidate.claimed_coverage.years
        if overlap:
            score += W_YEAR_CLAIMED
            reasons.append(f"التغطية المعلنة تشمل: {sorted(overlap)}")

    if (
        requested
        and period_grain(request.raw_query) == "year"
        and period_grain(candidate.title_ar or candidate.title_en) == "year"
    ):
        score += W_WHOLE_YEAR
        reasons.append("إصدار للسنة كاملة، كما سُئل")

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

    if candidate.relevance > 0:
        score += W_SOURCE_RELEVANCE * min(candidate.relevance, 1.0)
        if candidate.relevance >= 0.5:
            reasons.append("الأقرب إلى صياغة السؤال بين نتائج المصدر")

    if descriptor is not None:
        score += W_AUTHORITY * descriptor.authority
        if descriptor.authority >= 90:
            reasons.append(f"مصدر مرجعي: {descriptor.name_ar}")
        if descriptor.international:
            score += W_INTERNATIONAL
            reasons.append(f"مصدر دولي، يأتي بعد الجهة السعودية: {descriptor.name_ar}")

    fresh = _freshness(candidate.last_updated, today)
    if fresh > 0:
        score += fresh
        reasons.append(f"محدَّث في {candidate.last_updated}")

    # "The latest" with no year asked: of a series published per month or
    # quarter, the newest release -- «للربع الثاني 2026» before «للربع الأول».
    if not request.period.years:
        period = title_period(candidate.title_ar or candidate.title_en)
        if period is None and candidate.claimed_coverage.latest():
            period = candidate.claimed_coverage.latest() + 1.0
        if period is not None:
            since = today.year - LATEST_SPAN + 1
            score += W_LATEST_PERIOD * max(0.0, min(float(LATEST_SPAN), period - since))

    candidate.score = round(score, 3)
    candidate.match_reasons = tuple(reasons)
    return candidate


def _word_pairs(question: str, stopwords: frozenset[str]) -> list[str]:
    """Neighbouring words of the question, filler and periods left out."""
    words = [
        w for w in normalize(question).split()
        if not w.isdigit() and w not in stopwords and strip_article(w) not in stopwords
        and period_grain(w + " 2000") != "month"
    ]
    return [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]


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


# What GASTAT publishes a concept under, when the title does not name it:
# non-oil exports are in «التجارة الدولية السلعية غير البترولية», the
# unemployment rate in «إحصاءات سوق العمل», inflation in the consumer price
# index, pilgrims in «إحصاءات الحج». A result naming the publication is on the
# subject of the concept. Written as GASTAT spells them: they are also asked
# of its website's search, which matches words, not their families --
# «الحجاج» finds nothing there, «الحج» finds the bulletin.
SUBJECT_ALIASES = {
    "صادرات": ("التجارة الدولية", "التجارة الخارجية"),
    "واردات": ("التجارة الدولية", "التجارة الخارجية"),
    "ميزان تجاري": ("التجارة الدولية", "التجارة الخارجية"),
    # Not «القوى العاملة»: «القوى العاملة في خدمة الحجاج» is not unemployment.
    "بطاله": ("سوق العمل",),
    "توظيف": ("سوق العمل",),
    "مشتغلين": ("سوق العمل",),
    "عاطلين": ("سوق العمل",),
    "تضخم": ("أسعار المستهلك",),
    "حجاج": ("الحج",),
    "حاج": ("الحج",),
    "معتمرين": ("العمرة",),
    "معتمرون": ("العمرة",),
    "سياح": ("السياحة",),
    "ناتج محلي": ("الحسابات القومية",),
}


def _aliases(phrase: str) -> tuple[str, ...]:
    return SUBJECT_ALIASES.get(strip_phrase_articles(normalize(phrase)), ())


def subject_aliases(request: DataRequest) -> tuple[str, ...]:
    """GASTAT's names for what the question asks about, as GASTAT spells them."""
    phrases = (*request.typed_phrases, *request.free_terms, *request.raw_query.split())
    found: list[str] = []
    for phrase in phrases:
        for name in _aliases(phrase):
            if name not in found:
                found.append(name)
    return tuple(found)


def on_subject(request: DataRequest, candidate: DatasetCandidate) -> bool:
    """Whether a candidate is about what was asked, not merely returned by a search.

    A search returns anything that shares a word or a topic with the
    question, and a topic is broad: «أحدث بيانات البطالة» is a labour-market
    question, and so is a table of the workforce serving pilgrims. A result
    must name what the user actually typed: the topic phrases found in the
    question («البطالة»), or, when no topic was recognised, one of its other
    words («عدد الخيول بالسعودية» once came back as a pension table because
    both mention Saudi Arabia). A source's own confident reading of the
    question also counts.
    """
    if candidate.relevance >= SUBJECT_RELEVANCE:
        return True
    haystack = " ".join(
        filter(
            None,
            (candidate.title_ar, candidate.title_en, candidate.description, candidate.keywords),
        )
    )
    if request.topic is not None:
        if not request.typed_phrases:
            # The topic came from elsewhere (a language model's reading):
            # the sources were searched for it, and that is all there is.
            return True
        return any(
            mentions(haystack, name)
            for phrase in request.typed_phrases
            for name in (phrase, *_aliases(phrase))
        )
    # No topic recognised: the question's own words are all there is, and
    # one shared word is not a subject. «عدد المستفيدين من معسكرات سدايا»
    # once returned «نسبة السكان المستفيدين من خدمات الكهرباء» on
    # «المستفيدين» alone. At least half of them must be named -- the
    # publisher's name counting, since «سدايا» asks for SDAIA's data -- and
    # at least one word of the subject itself.
    terms = [t for t in request.free_terms if t not in _PLAIN]
    if not terms:
        return False
    named = {t for t in terms if mentions_term(candidate, t)}
    subject = [t for t in terms if strip_article(t) not in _PUBLISHER_NAMES]
    if subject and not named & set(subject):
        return False
    return len(named) >= (len(terms) + 1) // 2


def specific_terms(request: DataRequest) -> tuple[str, ...]:
    """The question's own words beyond the recognised topic («تملك», «النفطية»)."""
    return tuple(t for t in request.free_terms if t not in _PLAIN and len(t) >= 3)


def mentions_term(candidate: DatasetCandidate, term: str) -> bool:
    haystack = " ".join(filter(None, (
        candidate.title_ar, candidate.title_en, candidate.description, candidate.keywords,
        candidate.publisher_ar, candidate.publisher_en,
    )))
    words = (term, *SYNONYMS.get(strip_article(term), ()), *load_lexicon().alternatives(term))
    return any(mentions(haystack, w) for w in words)


def prefer_specific(request: DataRequest, ranked: list[DatasetCandidate]) -> list[DatasetCandidate]:
    """Those naming the most of the question's specific words, if any do; else all.

    A soft rule: «نسبة تملك المساكن» should prefer a table about ownership to
    one about electricity in dwellings. When no result names the word, all
    are kept and the answer says what it could not match (`unmatched_terms`).
    """
    terms = specific_terms(request)
    if not terms:
        return ranked
    # Those naming the most of them: «لشهر مايو» is named by the May bulletin,
    # and every monthly one says «لشهر».
    named = [(sum(1 for t in terms if mentions_term(c, t)), c) for c in ranked]
    most = max((n for n, _ in named), default=0)
    return [c for n, c in named if n == most] if most else ranked


def unmatched_terms(request: DataRequest, candidate: DatasetCandidate) -> tuple[str, ...]:
    return tuple(t for t in specific_terms(request) if not mentions_term(candidate, t))


def measure_mismatch(request: DataRequest, candidate: DatasetCandidate) -> bool:
    """The question asks for a count, and the result is a rate or a share."""
    asked = set(stems(request.raw_query))
    if not asked & set(MEASURES[0]):
        return False
    title = normalize(candidate.title_ar or candidate.title_en)
    return any(marker in title for marker in _RATE_MARKERS)





def missing_dimensions(
    request: DataRequest, present: set[Dimension]
) -> tuple[Dimension, ...]:
    """Requested breakdowns the data does not actually provide."""
    return tuple(d for d in request.dimensions if d not in present)
