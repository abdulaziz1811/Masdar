"""Turns a natural-language question into a `DataRequest`.

The parser is deliberately conservative: anything it cannot place is kept in
`free_terms` (so search still sees it) or `unresolved` (so the user is told
it was not understood). Silently dropping part of a question is the one
failure mode that would make the whole agent untrustworthy.
"""

from __future__ import annotations

import re

from masdar.domain.models import Calendar, DataRequest, Dimension, Period, PeriodKind, Topic
from masdar.nlu import hijri
from masdar.nlu.lexicon import Lexicon, load_lexicon
from masdar.nlu.normalize import (
    contains_phrase,
    normalize,
    normalize_light,
    stems,
    strip_article,
    strip_phrase_articles,
    tokens,
)

_YEAR = re.compile(r"(?<!\d)(\d{4})(?!\d)")

# Connectors as they look AFTER normalisation: alef-maksura folds to ya, so
# "إلى" arrives as "الي" and "حتى" as "حتي". Longest first.
_CONNECTORS = ("الي", "حتي", "لغايه", "ل", "to", "until", "through", "-", "–", "—")
_CONNECTOR_RE = "|".join(re.escape(c) for c in _CONNECTORS)

# "من 2019 إلى 2024" / "بين 2019 و 2024" / "2019-2024" / "from 2019 to 2024"
_RANGE_PATTERNS = (
    re.compile(rf"\bمن\s*(\d{{4}})\s*(?:{_CONNECTOR_RE})\s*(\d{{4}})"),
    re.compile(r"\bبين\s*(\d{4})\s*(?:و|الي)\s*(\d{4})"),
    re.compile(r"(\d{4})\s*[-–—/]\s*(\d{4})"),
    re.compile(rf"\bfrom\s*(\d{{4}})\s*(?:{_CONNECTOR_RE})\s*(\d{{4}})", re.IGNORECASE),
    re.compile(r"(\d{4})\s*(?:to|through)\s*(\d{4})", re.IGNORECASE),
)

_HIJRI_MARKERS = ("هجري", "هجريه", "هـ", "ه.", "ah")
_GREGORIAN_MARKERS = ("ميلادي", "ميلاديه", "م.", "ce", "ad", "gregorian")

_LATEST_MARKERS = (
    "احدث", "اخر", "الاخيره", "الاخير", "اخر تحديث", "المتاح", "المتوفر",
    "latest", "newest", "most recent", "last",
)

# Public name for other modules: words about the period, not the subject.
LATEST_MARKERS = _LATEST_MARKERS


# Phrases that mean "give me the file" rather than naming a subject.
_FORMAT_HINTS = ("اكسل", "excel", "xlsx", "csv", "ملف", "جدول")


def _marked(context: str, markers: tuple[str, ...]) -> bool:
    for marker in markers:
        if marker.isascii():
            if re.search(rf"(?<![a-z]){re.escape(marker)}(?![a-z])", context):
                return True
        elif marker in context:
            return True
    return False


def _classify_year(year: int, light_text: str, position: int) -> Calendar | None:
    """Decide which calendar a bare 4-digit number belongs to."""
    window = light_text[position: position + 24].lower()
    before = light_text[max(0, position - 12): position].lower()
    context = window + " " + before

    # A marker settles the calendar only for a number that could be in it,
    # and a Latin marker only as a word of its own: «umrah 2018» is not the
    # Hijri year 2018 (it once read as 2579), nor «price 1445» a Gregorian one.
    if hijri.looks_hijri(year) and _marked(context, _HIJRI_MARKERS):
        return Calendar.HIJRI
    if hijri.looks_gregorian(year) and _marked(context, _GREGORIAN_MARKERS):
        return Calendar.GREGORIAN
    # The plausible Hijri and Gregorian ranges do not overlap, so an
    # unmarked number is unambiguous on its own.
    if hijri.looks_hijri(year):
        return Calendar.HIJRI
    if hijri.looks_gregorian(year):
        return Calendar.GREGORIAN
    return None


def _expand(year: int, calendar: Calendar) -> tuple[int, ...]:
    """Map a year in its own calendar onto Gregorian year(s)."""
    if calendar is Calendar.HIJRI:
        return hijri.hijri_to_gregorian_years(year)
    return (year,)


def parse_period(text: str) -> Period:
    light = normalize_light(text)
    flat = normalize(text)

    for pattern in _RANGE_PATTERNS:
        match = pattern.search(light)
        if not match:
            continue
        start_raw, end_raw = int(match.group(1)), int(match.group(2))
        cal_start = _classify_year(start_raw, light, match.start(1))
        cal_end = _classify_year(end_raw, light, match.start(2))
        calendar = cal_start or cal_end or Calendar.GREGORIAN
        if cal_start is None or cal_end is None:
            continue
        lo, hi = sorted((start_raw, end_raw))
        years: list[int] = []
        for y in range(lo, hi + 1):
            years.extend(_expand(y, calendar))
        return Period(
            years=tuple(years),
            kind=PeriodKind.RANGE,
            calendar=calendar,
            raw=match.group(0).strip(),
        )

    found: list[tuple[int, Calendar]] = []
    for match in _YEAR.finditer(light):
        year = int(match.group(1))
        calendar = _classify_year(year, light, match.start(1))
        if calendar is not None:
            found.append((year, calendar))

    if found:
        calendar = found[0][1]
        years: list[int] = []
        for year, cal in found:
            years.extend(_expand(year, cal))
        kind = PeriodKind.SINGLE if len(found) == 1 else PeriodKind.RANGE
        raw = " ".join(str(y) for y, _ in found)
        return Period(years=tuple(years), kind=kind, calendar=calendar, raw=raw)

    if any(contains_phrase(flat, marker) for marker in _LATEST_MARKERS):
        return Period(kind=PeriodKind.LATEST, raw="latest")

    return Period(kind=PeriodKind.UNSPECIFIED)


def _score_topic(text: str, topic: Topic) -> tuple[float, tuple[str, ...]]:
    """Weight keyword hits by phrase length so specific beats generic."""
    score = 0.0
    hits: list[str] = []
    phrases = (topic.label_ar, topic.label_en, *topic.keywords_ar, *topic.keywords_en)
    for phrase in phrases:
        if not phrase:
            continue
        if contains_phrase(text, phrase):
            weight = len(normalize(phrase).split())
            score += weight * weight  # quadratic: "الطاقة الكهربائية" >> "الطاقة"
            hits.append(phrase)
    return score, tuple(hits)


def detect_topic(text: str, lexicon: Lexicon) -> tuple[Topic | None, tuple[str, ...]]:
    best: tuple[float, Topic | None, tuple[str, ...]] = (0.0, None, ())
    for topic in lexicon.topics:
        score, hits = _score_topic(text, topic)
        if score > best[0]:
            best = (score, topic, hits)
    return best[1], best[2]


def topics_in_text(text: str, lexicon: Lexicon | None = None) -> list[str]:
    """Every topic a text supports, strongest first.

    Unlike `detect_topic`, which picks one reading of a user's question, this
    labels a dataset for recall: a dataset about household electricity is
    both an electricity and a housing dataset, and should be found as either.
    """
    lexicon = lexicon or load_lexicon()
    # Qualifiers such as "بالأسعار الجارية" hold a topic word without being
    # about that topic; take them out before labelling.
    text = _without_phrases(text, lexicon.qualifiers)
    scored = []
    for topic in lexicon.topics:
        score, _ = _score_topic(text, topic)
        if score > 0:
            scored.append((score, topic.id))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [topic_id for _, topic_id in scored]


# Per token, squared: a three-word phrase the user typed is decisive.
W_TYPED = 3.0
def typed_phrase_score(request: DataRequest, text: str) -> tuple[float, tuple[str, ...]]:
    """Reward text containing the phrases the user actually typed.

    Weighted by length squared, like topic detection, so a specific phrase
    ("متوسط العمر المتوقع") outweighs any number of generic topic words.
    """
    score = 0.0
    hits: list[str] = []
    for phrase in request.typed_phrases:
        if phrase and contains_phrase(text, phrase):
            tokens = len(phrase.split())
            score += W_TYPED * tokens * tokens
            hits.append(phrase)
    return score, tuple(hits)


# The quantity asked for: «عدد الحجاج» wants a count of pilgrims, not the
# workforce serving them, though both titles name the pilgrims. Normalised,
# grouped where they mean the same quantity.
MEASURES = (
    ("عدد", "اعداد", "تعداد"),
    ("نسبه",),
    ("معدل",),
    ("متوسط",),
)


def shared_measures(question: str, title: str) -> list[str]:
    """Measure words («عدد», «نسبة», «معدل», «متوسط») in both texts."""
    asked = set(stems(question))
    named = set(stems(title))
    return [group[0] for group in MEASURES if asked & set(group) and named & set(group)]


def detect_dimensions(text: str, lexicon: Lexicon) -> tuple[tuple[Dimension, ...], tuple[str, ...]]:
    found: list[Dimension] = []
    matched_words: list[str] = []
    for dimension, words in lexicon.dimension_words.items():
        # Longest first, and every word of it spent: «وفئات العمر» must not
        # leave «فئات» behind as a subject word.
        for word in sorted(words, key=len, reverse=True):
            if contains_phrase(text, word):
                found.append(dimension)
                matched_words.extend(normalize(word).split())
                break
    return tuple(found), tuple(matched_words)


_YEARISH = re.compile(r"^\d{3,4}\D{0,3}$")


def detect_places(text: str, lexicon: Lexicon) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(regions named in the text, the words that named them)."""
    found: list[str] = []
    words: list[str] = []
    for name, variants in lexicon.places.items():
        for variant in sorted(variants, key=len, reverse=True):
            if contains_phrase(text, variant):
                found.append(name)
                words.extend(normalize(variant).split())
                break
    return tuple(found), tuple(words)


def _free_terms(text: str, consumed: set[str], stopwords: frozenset[str]) -> tuple[str, ...]:
    result: list[str] = []
    for token in tokens(text):
        stem = strip_article(token)
        if token.isdigit() or _YEARISH.match(token):
            continue
        if token in stopwords or stem in stopwords:
            continue
        # «وفي», «وكم», «وعن»: a joined «و» before a word that carries nothing.
        if token.startswith("و") and token[1:] in stopwords:
            continue
        if token in consumed or stem in consumed:
            continue
        if token in _FORMAT_HINTS or len(token) < 3:
            continue
        if token not in result:
            result.append(token)
    return tuple(result)


def _without_phrases(text: str, phrases: tuple[str, ...]) -> str:
    """The text with the given phrases removed, compared article-insensitively."""
    remaining = " " + " ".join(stems(text)) + " "
    for phrase in sorted(phrases, key=len, reverse=True):
        needle = strip_phrase_articles(phrase)
        if needle:
            remaining = remaining.replace(f" {needle} ", " ")
    return remaining.strip()


def detect_language(text: str) -> str:
    arabic = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return "ar" if arabic >= latin else "en"


def parse(query: str, lexicon: Lexicon | None = None) -> DataRequest:
    """Parse a user question into a `DataRequest`."""
    lexicon = lexicon or load_lexicon()

    period = parse_period(query)
    topic, topic_hits = detect_topic(query, lexicon)
    # Words that named the topic are spent: "متوسط العمر المتوقع" is life
    # expectancy, and its "العمر" must not also request an age breakdown.
    dimensions, dimension_words = detect_dimensions(
        _without_phrases(query, topic_hits), lexicon
    )

    places, place_words = detect_places(query, lexicon)
    if places and Dimension.REGION not in dimensions:
        # A named region is answered from the regional breakdown.
        dimensions = (*dimensions, Dimension.REGION)

    consumed: set[str] = set(dimension_words) | set(place_words)
    # «للسعوديين» named the breakdown as surely as «السعوديين».
    consumed.update(strip_article(w) for w in (*dimension_words, *place_words))
    for hit in topic_hits:
        consumed.update(normalize(hit).split())
    # "حسب"/"بحسب" only ever introduce a dimension, and the words that
    # signalled the period are not subject matter either.
    consumed.update(("حسب", "بحسب"))
    for marker in _LATEST_MARKERS:
        consumed.update(normalize(marker).split())
    consumed.update(normalize(period.raw).split())

    free = _free_terms(query, consumed, lexicon.stopwords)

    unresolved: list[str] = []
    if topic is None:
        unresolved.append("لم يتم تحديد موضوع واضح للطلب")
    if period.kind is PeriodKind.UNSPECIFIED:
        unresolved.append("لم يتم تحديد سنة أو فترة زمنية")

    return DataRequest(
        raw_query=query.strip(),
        period=period,
        topic=topic,
        dimensions=dimensions,
        free_terms=free,
        typed_phrases=tuple(sorted(set(topic_hits), key=lambda h: (-len(h), h))),
        language=detect_language(query),
        places=places,
        unresolved=tuple(unresolved),
    )


def extract_years(text: str) -> tuple[int, ...]:
    """Every year mentioned in a title or description, as Gregorian years.

    Used to read coverage hints out of dataset titles and filenames. The
    result is only ever a hint: `CoverageOrigin.INFERRED_TITLE` data may not
    be used to promise that a year is present.
    """
    light = normalize_light(text)
    years: set[int] = set()
    for match in _YEAR.finditer(light):
        value = int(match.group(1))
        calendar = _classify_year(value, light, match.start(1))
        if calendar is not None:
            years.update(_expand(value, calendar))
    return tuple(sorted(years))
