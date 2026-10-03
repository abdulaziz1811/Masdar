"""Arabic text normalisation.

Arabic queries reach us with diacritics, inconsistent hamza spelling,
tatweel padding and Arabic-Indic digits. Matching raw user text against a
lexicon fails on all four, so every comparison in Masdar happens on
normalised text.
"""

from __future__ import annotations

import re
import unicodedata

# Tashkeel (harakat), tatweel and the Quranic marks that ride on letters.
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")

# Arabic-Indic (U+0660..) and Eastern Arabic-Indic (U+06F0..) digits.
_DIGIT_MAP = {ord("٠") + i: str(i) for i in range(10)}
_DIGIT_MAP.update({ord("۰") + i: str(i) for i in range(10)})

_LETTER_MAP = {
    "آ": "ا",  # آ -> ا
    "أ": "ا",  # أ -> ا
    "إ": "ا",  # إ -> ا
    "ٱ": "ا",  # ٱ -> ا
    "ى": "ي",  # ى -> ي
    "ئ": "ي",  # ئ -> ي
    "ؤ": "و",  # ؤ -> و
    "ة": "ه",  # ة -> ه
}
_LETTER_TABLE = {ord(k): v for k, v in _LETTER_MAP.items()}

# Arabic punctuation plus the ASCII set, mapped to space so token
# boundaries survive ("الطاقة،الكهربائية").
_PUNCT = re.compile(r"[،؛؟٪-٭۔!-/:-@\[-`{-~]")

_WS = re.compile(r"\s+")

# «و» fused to «فئة/فئات» ("and the categories of"). Folding ئ into ي would
# turn «وفئات» into «وفيات» -- deaths -- so the conjunction is split off
# while the hamza still tells the two apart.
_AND_CATEGORIES = re.compile(r"(?<!\w)و(?=فئ)")

# Definite article and common prepositions fused to nouns. Stripped only
# when a bare match fails, never destructively.
_PREFIXES = ("وال", "بال", "لل", "ال", "و")


def fold_digits(text: str) -> str:
    """Turn ٢٠٢٦ / ۲۰۲۶ into 2026, leaving everything else alone."""
    return text.translate(_DIGIT_MAP)


def normalize(text: str) -> str:
    """Canonical form used for all lexicon and title matching."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = fold_digits(text)
    text = _DIACRITICS.sub("", text)
    text = _AND_CATEGORIES.sub("و ", text)
    text = text.translate(_LETTER_TABLE)
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip().lower()


def strip_article(token: str) -> str:
    """Drop a leading ال/بال/وال so "بالمناطق" matches "مناطق"."""
    for prefix in _PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= 3:
            return token[len(prefix):]
    return token


def tokens(text: str) -> list[str]:
    return [t for t in normalize(text).split(" ") if t]


def stems(text: str) -> list[str]:
    """Tokens with articles stripped, for looser matching."""
    return [strip_article(t) for t in tokens(text)]


def strip_phrase_articles(phrase: str) -> str:
    """Article-stripped form of a whole phrase."""
    return " ".join(strip_article(t) for t in tokens(phrase))


def contains_phrase(haystack: str, phrase: str) -> bool:
    """Whether `phrase` appears in `haystack` on token boundaries.

    Both sides are normalised AND article-stripped before comparing, because
    the article can sit on either side: a lexicon entry "الصادرات" has to
    match a query that says "صادرات", and vice versa.
    """
    needles = {normalize(phrase), strip_phrase_articles(phrase)}
    needles.discard("")
    if not needles:
        return False
    haystacks = []
    for variant in (tokens(haystack), stems(haystack)):
        if variant:
            haystacks.append(" " + " ".join(variant) + " ")
            # How often a series is published is not what it is about:
            # «الرقم القياسي السنوي للإنتاج الصناعي» names «الرقم القياسي
            # للإنتاج الصناعي».
            bare = [w for w in variant if strip_article(w) not in _FREQUENCY]
            if len(bare) < len(variant):
                haystacks.append(" " + " ".join(bare) + " ")
    return any(f" {needle} " in hay for needle in needles for hay in haystacks)


_FREQUENCY = frozenset({"سنوي", "شهري", "ربعي", "اسبوعي", "يومي"})

# Endings that turn a noun into its adjective or plural: «الكهرباء» and
# «الكهربائية», «مستشفى» and «المستشفيات», «السكان» and «السكانية». Longest
# first; one is removed at most, and never below three letters.
_ENDINGS = ("ييه", "يات", "يه", "ات", "ين", "ون", "ي")


def root(token: str) -> str:
    """A token reduced to what its word family shares, for loose matching."""
    return _reduce(strip_article(token))


def _roots(token: str) -> set[str]:
    """A token's family, read with and without a leading «و» or «ل».

    «و» is sometimes "and" («والسكان») and sometimes the word's own first
    letter («وفيات», «وحدات»): stripped, «وفيات» would never meet «الوفيات».
    «ل» is "for" in «لمنصة إحسان», which must meet «منصة»; it is never
    stripped alone, since «لقاحات» begins with it.
    """
    found = {root(token)}
    bare = token[2:] if token.startswith("ال") else token
    if bare.startswith("و") and len(bare) >= 4:
        found.add(_reduce(bare))
    if token.startswith("ل") and not token.startswith("لل") and len(token) >= 5:
        found.add(_reduce(token[1:]))
    return found


# Words an ending cannot be taken from without changing what they mean:
# «الجنسية» (nationality) is not «الجنس» (sex), and a table by nationality
# was once chosen for a question by sex.
_WHOLE = {"جنسيه": "جنسيه", "جنسيات": "جنسيه"}


def _reduce(stem: str) -> str:
    if stem in _WHOLE:
        return _WHOLE[stem]
    for ending in _ENDINGS:
        if stem.endswith(ending) and len(stem) - len(ending) >= 3:
            stem = stem[: -len(ending)]
            break
    if len(stem) > 4 and stem.endswith("ء"):
        stem = stem[:-1]
    if len(stem) > 4 and stem.endswith("ا"):
        stem = stem[:-1]
    return stem


def mentions(haystack: str, phrase: str) -> bool:
    """Whether every word of `phrase` occurs in `haystack`, in any form.

    Looser than `contains_phrase`: a title says «استهلاك الطاقة الكهربائية»
    where the question says «استهلاك الكهرباء». Used to judge whether a
    result is about what was asked, not to rank it.
    """
    wanted = [_roots(t) for t in tokens(phrase)]
    if not wanted:
        return False
    present = set().union(*(_roots(t) for t in tokens(haystack)))
    return all(family & present for family in wanted)


def normalize_light(text: str) -> str:
    """Like `normalize` but keeps punctuation.

    Date parsing needs the dash in "2019-2024" and the suffix in "1447هـ",
    both of which full normalisation throws away.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = fold_digits(text)
    text = _DIACRITICS.sub("", text)
    text = _AND_CATEGORIES.sub("و ", text)
    text = text.translate(_LETTER_TABLE)
    return _WS.sub(" ", text).strip()


_DIGITS_APART = re.compile(r"\d+|[^\W\d_]+")
# Which quarter or month a release covers is its period, not its subject.
_PERIOD_WORDS = frozenset(normalize(w) for w in (
    "ربع", "للربع", "سنوي", "اول", "أول", "ثاني", "ثالث", "رابع",
    "يناير", "فبراير", "مارس", "أبريل", "ابريل", "مايو", "يونيو", "يوليو",
    "أغسطس", "اغسطس", "سبتمبر", "أكتوبر", "اكتوبر", "نوفمبر", "ديسمبر",
))


def word_fit(question: str, title: str, stopwords: frozenset[str] = frozenset()) -> float:
    """How closely a title is worded like the question, from 0 to 1.

    The share of the question's words the title has, and of the title's words
    the question has: asked by its exact name, a dataset scores 1, and a
    longer title that merely shares the topic («تعداد 2022 السكان» for
    «تعداد 2022 الحالة الاجتماعية») scores less. Years and filler are not
    counted.
    """
    def content(text: str) -> list[str]:
        # «لسنة 2026حسب»: a year glued to the next word is still two words.
        words = _DIGITS_APART.findall(normalize(text))
        return [
            w for w in words
            if not w.isdigit() and w not in stopwords
            and strip_article(w) not in stopwords and strip_article(w) not in _PERIOD_WORDS
        ]

    asked, named = content(question), content(title)
    if not asked or not named:
        return 0.0
    recall = sum(1 for w in asked if mentions(title, w)) / len(asked)
    precision = sum(1 for w in named if mentions(question, w)) / len(named)
    return (2 * recall + precision) / 3


_TITLE_YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20\d\d)(?!\d)")
# «الربع الرابع», «للربع الثاني», «الربع السنوي الاول», «Q3» -- after normalize().
_QUARTER = re.compile(r"ربع\s+(?:السنوي\s+)?(الاول|الثاني|الثانيه|الثالث|الرابع)|\bq([1-4])\b")
_QUARTERS = {"الاول": 1, "الثاني": 2, "الثانيه": 2, "الثالث": 3, "الرابع": 4}
_MONTHS = {normalize(name): number for number, names in enumerate((
    ("يناير", "january"), ("فبراير", "february"), ("مارس", "march"), ("أبريل", "ابريل", "april"),
    ("مايو", "may"), ("يونيو", "june"), ("يوليو", "july"), ("أغسطس", "اغسطس", "august"),
    ("سبتمبر", "september"), ("أكتوبر", "اكتوبر", "october"), ("نوفمبر", "november"),
    ("ديسمبر", "december"),
), start=1) for name in names}


MONTH_WORDS = frozenset(_MONTHS)


def period_grain(text: str) -> str | None:
    """'month', 'quarter' or 'year': what span of time a text names.

    «الرقم القياسي السنوي للإنتاج الصناعي لعام 2025» is a year,
    «... لشهر ديسمبر 2025» a month. None when it names no year.
    """
    folded = normalize(fold_digits(text))
    if not _TITLE_YEAR.search(folded):
        return None
    if any(w in _MONTHS for w in tokens(folded)):
        return "month"
    if _QUARTER.search(folded.lower()):
        return "quarter"
    return "year"


def title_period(text: str) -> float | None:
    """The latest point in time a title names, as a year with a fraction.

    «لشهر أغسطس 2026» is 2026 and eight twelfths, «للربع الثاني 2026» 2026
    and a half, «لعام 2025» the end of 2025. None when no year is named.
    """
    folded = normalize(fold_digits(text))
    years = [int(y) for y in _TITLE_YEAR.findall(folded)]
    if not years:
        return None
    months = [_MONTHS[w] for w in tokens(folded) if w in _MONTHS]
    quarters = [_QUARTERS[w] if w else int(d) for w, d in _QUARTER.findall(folded.lower())]
    month = max(months) if months else 3 * max(quarters) if quarters else 12
    return max(years) + month / 12
