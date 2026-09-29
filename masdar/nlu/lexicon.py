"""Loads the declarative topic lexicon into domain objects."""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

from masdar.domain.models import Dimension, Topic
from masdar.nlu.normalize import normalize, strip_article

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
TOPICS_FILE = CONFIG_DIR / "topics.yaml"


class Lexicon:
    """Topics, dimension vocabulary and stopwords, ready for matching."""

    def __init__(
        self,
        topics: tuple[Topic, ...],
        dimension_words: dict[Dimension, tuple[str, ...]],
        stopwords: frozenset[str] = frozenset(),
        qualifiers: tuple[str, ...] = (),
        places: dict[str, tuple[str, ...]] | None = None,
    ):
        self.topics = topics
        self.dimension_words = dimension_words
        self.stopwords = stopwords
        self.qualifiers = qualifiers
        # Region name -> the ways a question or a table may write it.
        self.places = places or {}
        self._by_id = {t.id: t for t in topics}

    def get(self, topic_id: str) -> Topic | None:
        return self._by_id.get(topic_id)

    @property
    def topic_ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)


def _load(path: Path) -> Lexicon:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    topics = tuple(
        Topic(
            id=entry["id"],
            label_ar=entry["label_ar"],
            label_en=entry["label_en"],
            keywords_ar=tuple(entry.get("keywords_ar", ())),
            keywords_en=tuple(entry.get("keywords_en", ())),
            preferred_sources=tuple(entry.get("preferred_sources", ())),
        )
        for entry in raw.get("topics", [])
    )

    dimension_words: dict[Dimension, tuple[str, ...]] = {}
    for name, words in (raw.get("dimensions") or {}).items():
        try:
            dimension = Dimension(name)
        except ValueError:  # a dimension in config we have no enum for yet
            continue
        dimension_words[dimension] = tuple(words)

    words = [word for word in normalize(raw.get("stopwords") or "").split() if word]
    # "بالسعودية" and "بالمملكة" are the listed words with a fused preposition;
    # free terms are compared without their article, so the list is too.
    stopwords = frozenset(words) | frozenset(strip_article(word) for word in words)

    qualifiers = tuple(str(q) for q in (raw.get("qualifiers") or []) if q)

    places = {
        str(name): tuple(str(v) for v in variants)
        for name, variants in (raw.get("places") or {}).items()
    }

    return Lexicon(
        topics=topics,
        dimension_words=dimension_words,
        stopwords=stopwords,
        qualifiers=qualifiers,
        places=places,
    )


@functools.lru_cache(maxsize=1)
def load_lexicon(path: Path | None = None) -> Lexicon:
    return _load(path or TOPICS_FILE)
