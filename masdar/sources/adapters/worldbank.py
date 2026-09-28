"""The World Bank's World Development Indicators, for Saudi Arabia.

An international source, and treated as one: reliable and long (1960 to the
latest release), but second to the Saudi publisher of the same figure. The
bank compiles most of these values from national agencies and UN bodies,
and each indicator names the organisation it came from; the answer says so.

* **Catalogue: local.** The API has no search, and its indicator list is
  large, so the list is kept in the repository
  (`config/worldbank/wdi_sau.json`): the 1,037 indicators with at least one
  value for the Kingdom in 2019-2024, with the bank's own Arabic and English
  names. `masdar worldbank-catalogue` rebuilds it.
* **Data: live.** The values are fetched from the API at question time, and
  its `lastupdated` becomes the answer's update date.
* **A null is not a value.** The API lists years it has no figure for with
  `"value": null` (2024 and 2025 for electricity use, when checked). Those
  years are left out, so they cannot be reported as available.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from datetime import date
from pathlib import Path

from masdar.domain.models import Coverage, DataRequest, DatasetCandidate, Resource
from masdar.nlu.lexicon import load_lexicon
from masdar.nlu.normalize import normalize, stems
from masdar.nlu.parser import topics_in_text
from masdar.sources.base import SourceAdapter, SourceError

CATALOGUE = Path(__file__).resolve().parent.parent.parent / "config" / "worldbank" / "wdi_sau.json"
API = "https://api.worldbank.org/v2"
COUNTRY = "SAU"
LANDING = "https://data.albankaldawli.org/indicator/{id}?locations=SA"
YEAR_COLUMN = "السنة"
# The bank's own series start in 1960; 200 rows covers them in one page.
PER_PAGE = 200
FIRST_RECENT, LAST_RECENT = 2019, 2024

# Words that say how a thing is measured, not what: a match on these alone
# does not make an indicator relevant.
_GENERIC = frozenset(normalize(
    "عدد تعداد اعداد معدل نسبه اجمالي متوسط حجم قيمه مؤشر كم number rate total share"
).split())
# Spellings the bank uses for a word people type differently.
_EQUIVALENT = {"عدد": ("تعداد", "اعداد")}
# The indicator's main name (before any parenthesis) says what is measured;
# the parenthesis says in what unit or as a share of what. "Trade (% of
# GDP)" mentions GDP but is not GDP, so a word found only in the
# parenthesis counts half, and an unasked word in the main name costs more
# than one in the parenthesis. That also makes the plainest variant win:
# "population, total" over "access to electricity (% of population)".
IN_QUALIFIER = 0.5
# A measure word asked for ("number of", "rate") weighs half a subject word:
# it separates "number of arrivals" from "tourism services", but a name that
# says "unemployment" without "rate" is still the unemployment rate.
MEASURE_WORD = 0.5
EXTRA_IN_NAME = 0.08
EXTRA_IN_QUALIFIER = 0.02
# Below this an indicator is only loosely related, and an international
# source should not be offered for it.
MIN_RELEVANCE = 0.5


def _same_word(a: str, b: str) -> bool:
    """Whether two stems are one word: equal, or one inflects the other.

    "كهرباء" and "كهربائية", "electricity" and "electric" share all but the
    ending; a four-letter common start keeps short words from colliding.
    """
    if a == b or b in _EQUIVALENT.get(a, ()) or a in _EQUIVALENT.get(b, ()):
        return True
    common = len(os.path.commonprefix([a, b]))
    return common >= 4 and common >= min(len(a), len(b)) - 1


def _content_stems(
    text: str, stopwords: frozenset[str], keep: frozenset[str] = frozenset()
) -> list[str]:
    seen: list[str] = []
    for stem in stems(text):
        if len(stem) < 3 or re.fullmatch(r"\d+\D{0,3}", stem):
            continue
        if stem in stopwords and stem not in keep:
            continue
        if stem not in seen:
            seen.append(stem)
    return seen


def _parse_date(text: object) -> date | None:
    try:
        return date.fromisoformat(str(text)[:10])
    except ValueError:
        return None


class WorldBankAdapter(SourceAdapter):
    def __init__(self, descriptor, http=None):
        super().__init__(descriptor, http)
        self._catalogue: dict | None = None
        # Candidates by data URL, so a fetch can record the release date the
        # response carries on the candidate the answer cites.
        self._candidates: dict[str, DatasetCandidate] = {}

    # -- catalogue -----------------------------------------------------
    def _load(self) -> dict:
        if self._catalogue is None:
            path = Path(self.descriptor.api.get("catalogue") or CATALOGUE)
            try:
                self._catalogue = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SourceError(self.id, f"فهرس البنك الدولي غير متاح: {exc}") from exc
        return self._catalogue

    def _indicators(self) -> list[dict]:
        return [i for i in self._load().get("indicators") or [] if i.get("id")]

    def _data_url(self, indicator_id: str) -> str:
        return (
            f"{API}/ar/country/{COUNTRY}/indicator/{indicator_id}"
            f"?format=json&per_page={PER_PAGE}"
        )

    # -- search --------------------------------------------------------
    def relevance(self, request: DataRequest, indicator: dict) -> float:
        """How closely an indicator's name fits the question, from 0 to 1.

        The share of the question's words the name contains, less a little
        for every word the name adds: the catalogue holds a dozen variants of
        most measures (total, female, male, youth, % of ...), and the plainest
        one that says everything asked is the one meant.
        """
        stopwords = load_lexicon().stopwords
        # "عدد" and "إجمالي" are stopwords for the portals' title search, but
        # here they tell "number of tourist arrivals" from "tourism services".
        asked = _content_stems(
            " ".join((request.raw_query, *request.free_terms)), stopwords, keep=_GENERIC
        )
        if not asked:
            return 0.0
        name_ar, name_en = indicator.get("ar") or "", indicator.get("en") or ""
        shown = name_ar if request.language == "ar" and name_ar else name_en or name_ar
        name, _, qualifier = shown.partition("(")
        in_name = _content_stems(name, stopwords, keep=_GENERIC)
        in_qualifier = _content_stems(qualifier, stopwords, keep=_GENERIC)

        def found(word: str, words: list[str]) -> bool:
            return any(_same_word(word, w) for w in words)

        weight, total, specific = 0.0, 0.0, False
        for word in asked:
            value = MEASURE_WORD if word in _GENERIC else 1.0
            total += value
            if found(word, in_name):
                weight += value
            elif found(word, in_qualifier):
                weight += value * IN_QUALIFIER
            else:
                continue
            specific = specific or word not in _GENERIC
        if not specific:
            return 0.0
        # Measure words ("total", "number") say nothing the question lacks.
        extra_name = sum(1 for w in in_name if w not in _GENERIC and not found(w, asked))
        extra_qualifier = sum(
            1 for w in in_qualifier if w not in _GENERIC and not found(w, asked)
        )
        score = (
            weight / total
            - EXTRA_IN_NAME * extra_name
            - EXTRA_IN_QUALIFIER * extra_qualifier
        )
        return max(0.0, min(1.0, score))

    def search(self, request: DataRequest, limit: int = 10) -> list[DatasetCandidate]:
        scored = [(self.relevance(request, i), i) for i in self._indicators()]
        ranked = sorted(
            (p for p in scored if p[0] >= MIN_RELEVANCE),
            key=lambda p: (-p[0], -(p[1].get("latest") or 0), p[1]["id"]),
        )
        candidates = []
        for relevance, indicator in ranked[:limit]:
            candidate = self._candidate(indicator)
            candidate.relevance = round(relevance, 3)
            self._candidates[candidate.resources[0].url] = candidate
            candidates.append(candidate)
        return candidates

    def _candidate(self, indicator: dict) -> DatasetCandidate:
        org = indicator.get("org_ar") or indicator.get("org_en") or ""
        caveat = "مصدر دولي: البنك الدولي (مؤشرات التنمية العالمية)"
        caveat += f"، والقيم مصدرها: {org}" if org else ""
        caveat += ". قد تختلف عن أحدث ما تنشره الجهة السعودية المختصة."
        caveats = [caveat]
        if indicator.get("en"):
            # The bank's Arabic names are translations and a few are wrong
            # (IT.NET.SECR.P6 is secure servers, named "internet users" in
            # Arabic); its English name is the definition, so it is shown.
            caveats.append(f"اسم المؤشر لدى البنك: {indicator['en']} ({indicator['id']})")
        url = self._data_url(indicator["id"])
        return DatasetCandidate(
            source_id=self.id,
            dataset_id=indicator["id"],
            title_ar=indicator.get("ar") or indicator.get("en") or indicator["id"],
            title_en=indicator.get("en") or "",
            keywords=" ".join(indicator.get("topics") or ()),
            landing_url=LANDING.format(id=indicator["id"]),
            publisher_ar="البنك الدولي",
            publisher_en="World Bank",
            resources=(Resource(url=url, format="JSON", title="World Bank API"),),
            claimed_coverage=Coverage.unknown(),
            last_updated=_parse_date(self._load().get("lastupdated")),
            license_name="CC BY 4.0",
            caveats=tuple(caveats),
        )

    # -- fetch ---------------------------------------------------------
    def fetch(self, url: str):
        fetched = super().fetch(url)
        if not url.startswith(f"{API}/ar/country/{COUNTRY}/indicator/"):
            return fetched
        try:
            payload = json.loads(fetched.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceError(self.id, f"استجابة غير صالحة: {exc}") from exc
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[0], dict):
            # The API reports errors as [{"message": [...]}].
            raise SourceError(self.id, f"رد غير متوقع من البنك الدولي: {str(payload)[:200]}")
        meta, rows = payload[0], payload[1] or []
        if int(meta.get("pages") or 1) > 1:
            raise SourceError(self.id, "السلسلة أطول من صفحة واحدة؛ لم تُجلب كاملة")

        table = []
        for row in rows:
            value = row.get("value")
            if value is None:
                continue  # a year the bank lists but has no figure for
            table.append({
                YEAR_COLUMN: int(row["date"]),
                "المؤشر": (row.get("indicator") or {}).get("value", ""),
                "القيمة": value,
                "رمز المؤشر": (row.get("indicator") or {}).get("id", ""),
            })
        table.sort(key=lambda r: r[YEAR_COLUMN])
        if not table:
            raise SourceError(self.id, "لا توجد قيم للمملكة في هذا المؤشر")

        released = _parse_date(meta.get("lastupdated"))
        candidate = self._candidates.get(url)
        if candidate is not None and released is not None:
            candidate.last_updated = released
        body = json.dumps(table, ensure_ascii=False).encode("utf-8")
        return replace(fetched, content=body, media_type="application/json")

    def probe(self) -> str:
        count = len(self._indicators())
        sample = self._data_url("SP.POP.TOTL")
        self.fetch(sample)
        return f"{count} مؤشراً في الفهرس المحلي، والواجهة تستجيب (عدد السكان)"


def build_catalogue(http, lexicon=None) -> dict:
    """Rebuild the local catalogue from the API (run where it is reachable)."""
    from collections import defaultdict

    def get(path: str):
        return json.loads(http.get(f"{API}{path}", source_id="worldbank").content)

    names_ar = get("/ar/source/2/indicators?format=json&per_page=2000")[1]
    names_en = {i["id"]: i for i in get("/en/source/2/indicators?format=json&per_page=2000")[1]}
    span = ";".join(f"yr{y}" for y in range(FIRST_RECENT, LAST_RECENT + 1))
    bulk = get(f"/sources/2/country/{COUNTRY}/series/all/time/{span}?format=json&per_page=20000")

    years: dict[str, set[int]] = defaultdict(set)
    for row in bulk["source"]["data"]:
        concepts = {v["concept"]: v["id"] for v in row["variable"]}
        if row.get("value") is not None:
            years[concepts["Series"]].add(int(concepts["Time"][2:]))

    def clean(text: object) -> str:
        text = " ".join(str(text or "").replace("‎", "").split())
        # Some Arabic names arrive with a stray "اسم المؤشر" glued on.
        return text.removeprefix("اسم المؤشر").strip()

    indicators = []
    for item in names_ar:
        if not years.get(item["id"]):
            continue
        english = names_en.get(item["id"], {})
        name_ar, name_en = clean(item.get("name")), clean(english.get("name"))
        indicators.append({
            "id": item["id"], "ar": name_ar, "en": name_en,
            "org_ar": clean(item.get("sourceOrganization"))[:160],
            "org_en": clean(english.get("sourceOrganization"))[:160],
            "topics": topics_in_text(f"{name_ar} {name_en}", lexicon)[:3],
            "latest": max(years[item["id"]]),
        })
    return {
        "source": "World Bank — World Development Indicators (source 2)",
        "country": COUNTRY,
        "lastupdated": bulk.get("lastupdated"),
        "built": date.today().isoformat(),
        "how": (
            f"Indicators of source 2 (Arabic and English names) that hold at least one "
            f"value for Saudi Arabia in {FIRST_RECENT}-{LAST_RECENT}. "
            "Rebuild with `masdar worldbank-catalogue`."
        ),
        "indicators": indicators,
    }


def write_catalogue(catalogue: dict, path: Path = CATALOGUE) -> None:
    """One indicator per line, so a rebuild reads as a reviewable diff."""
    head = [f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},"
            for k, v in catalogue.items() if k != "indicators"]
    rows = ",\n".join(
        "    " + json.dumps(i, ensure_ascii=False, separators=(",", ":"))
        for i in catalogue.get("indicators") or []
    )
    text = "\n".join(["{", *head, '  "indicators": [', rows, "  ]", "}"]) + "\n"
    path.write_text(text, encoding="utf-8")
