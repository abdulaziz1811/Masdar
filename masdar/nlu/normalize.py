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
    return any(f" {needle} " in hay for needle in needles for hay in haystacks)

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
    text = text.translate(_LETTER_TABLE)
    return _WS.sub(" ", text).strip()
