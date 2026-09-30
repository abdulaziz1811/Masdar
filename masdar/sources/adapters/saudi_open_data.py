"""Adapter for the national open data platform, open.data.gov.sa.

Verified live on 2026-09-28 against the platform's own developer guide
(/ar/pages/developers-api). The earlier guess at `/data/api/v1/datasets` was
wrong -- the WAF rejected it because it does not exist. The real endpoints:

* `GET /data/api/organizations?version=-1&organization=<id or Arabic name>`
  returns the publisher and *every* dataset it holds: `{id, titleEn,
  titleAr}`. There is no endpoint listing all organisations and none that
  searches datasets, so this is the way into the catalogue: the publishers
  are configured, and their dataset lists form a local catalogue, searched
  here. The titles are Arabic, which suits Arabic questions.
* `GET /data/api/datasets?version=-1&dataset=<id>` returns the metadata:
  provider, `updatedAt` (a real last-update date), `timePeriod` (declared
  coverage, e.g. "2022-01-01 - 2022-12-31"), `updateFrequency`, categories
  and Arabic tags such as "حسب المناطق".
* `GET /data/api/datasets/resources?version=-1&dataset=<id>` lists the files,
  CSV and XLSX, each with a `downloadUrl` and its columns.

Declared coverage is a claim: it becomes METADATA_CLAIM and never exhaustive,
because portal metadata goes stale (KAPSARC's descriptions do). The
orchestrator opens the file and reads the years from it.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

from masdar.domain.models import (
    Coverage,
    CoverageOrigin,
    DataRequest,
    DatasetCandidate,
    Dimension,
    Resource,
)
from masdar.nlu.lexicon import load_lexicon
from masdar.nlu.normalize import contains_phrase, fold_digits, normalize, word_fit
from masdar.nlu.parser import typed_phrase_score
from masdar.sources.base import SourceAdapter, SourceError, SourceUnreachable
from masdar.sources.discover import DISCOVERED_FILE, load_discovered

TABULAR = {"CSV", "XLSX", "XLS", "JSON", "TSV"}
MAX_DETAIL_FETCHES = 5
# Through the Saudi exit every request costs a Firecrawl credit (two per
# dataset's details), so fewer are opened.
MAX_DETAIL_FETCHES_VIA_EXIT = 3
# Checked 2026-09-30: through Firecrawl's Saudi exit the JSON API answers,
# but the files (/odp-public/...) are refused ("document_antibot"). A reader
# in the Kingdom downloads them from the link.
LINK_ONLY_NOTE = (
    "ملفات منصة البيانات المفتوحة الوطنية لا تُنزَّل من خادم خارج المملكة، "
    "فالحكم مبني على الفترة التي تعلنها الجهة. نزّل الملف من زر «الملف الأصلي»."
)
# Publishers fetched at once when the catalogue is not cached.
CATALOGUE_WORKERS = 8
_DATE = re.compile(r"(\d{4})-\d{2}-\d{2}")


# Publishers often split one series into a dataset per year or quarter
# («الأسماء التجارية المحجوزة لعام 2022 الربع الثالث»). Only a few results are
# opened, so the year in the title has to steer which ones: without it, a
# question about 2022 opened the 2020 files and reported 2022 as missing.
_TITLE_YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20\d\d)(?!\d)")
# «الربع الرابع»، «للربع الثاني»، «الربع السنوي الاول»، «Q3» -- after normalize().
_QUARTER = re.compile(r"ربع\s+(?:السنوي\s+)?(الاول|الثاني|الثانيه|الثالث|الرابع)|\bq([1-4])\b")
_QUARTER_NUMBERS = {"الاول": 1, "الثاني": 2, "الثانيه": 2, "الثالث": 3, "الرابع": 4}
W_WORD_FIT = 6.0
W_TITLE_YEAR_MATCH = 4.0
W_TITLE_YEAR_OTHER = -3.0
W_PER_YEAR = 0.05
# Strictly less than a year's step at its largest (4 x 0.01 < 0.05), so a
# later quarter never outranks a later year.
W_PER_QUARTER = 0.01


def title_years(text: str) -> set[int]:
    return {int(y) for y in _TITLE_YEAR.findall(fold_digits(text))}


def _title_quarter(text: str) -> int:
    found = [
        _QUARTER_NUMBERS[word] if word else int(digit)
        for word, digit in _QUARTER.findall(normalize(text).lower())
    ]
    return max(found, default=0)


def _period_fit(request: DataRequest, text: str) -> float:
    """How well the years a title names fit the years asked for.

    A title naming a requested year moves up; one naming only other years
    moves down but stays -- it may be the latest year there is, which the
    answer should be able to offer. A title naming no year is a series of
    unknown span and is left alone. For "the latest", newer titles win, and
    within a year a later quarter wins, since it runs closer to the present.
    """
    years = title_years(text)
    if not years:
        return 0.0
    quarter = _title_quarter(text) * W_PER_QUARTER
    requested = set(request.period.years)
    if requested:
        if years & requested:
            return W_TITLE_YEAR_MATCH + quarter
        return W_TITLE_YEAR_OTHER
    newest = max(years)
    return max(0, min(newest - 2000, 60)) * W_PER_YEAR + quarter


def coverage_from_period(period: str | None) -> Coverage:
    """Turn "2015-01-01 - 2023-12-31" into the years it declares.

    A declared range covers every year between its ends, not just the two
    ends, so the years are expanded. It is a claim, never exhaustive.
    """
    years = [int(y) for y in _DATE.findall(str(period or ""))]
    if not years:
        return Coverage.unknown()
    span = range(min(years), max(years) + 1)
    return Coverage(
        years=frozenset(span),
        origin=CoverageOrigin.METADATA_CLAIM,
        is_exhaustive=False,
        note=f"الفترة المعلنة في المنصة: {period}",
    )


def _as_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    for parse in (
        lambda t: datetime.fromisoformat(t.replace("Z", "+00:00")).date(),
        lambda t: datetime.strptime(t[:10], "%Y-%m-%d").date(),
    ):
        try:
            return parse(text)
        except ValueError:
            continue
    return None


def _encode_url(url: str) -> str:
    """Download URLs carry raw spaces and Arabic in their file names."""
    parts = urlsplit(url.strip())
    return urlunsplit(parts._replace(path=quote(parts.path, safe="/%")))


class SaudiOpenDataAdapter(SourceAdapter):
    """Catalogue from configured publishers; details and files on demand."""

    def _api(self, key: str, default: str, **values: str) -> str:
        template = str(self.descriptor.api.get(key) or default)
        path, _, query = template.partition("?")
        params = [
            (k, v.format(**values) if "{" in v else v)
            for k, v in (pair.split("=", 1) for pair in query.split("&") if "=" in pair)
        ]
        url = self.descriptor.url(path)
        return f"{url}?{urlencode(params)}" if params else url

    @property
    def _via_exit(self) -> bool:
        return getattr(getattr(self.http, "transport", None), "name", "") == "firecrawl"

    def _get_json(self, url: str):
        fetched = self.fetch(url)
        try:
            return json.loads(fetched.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceError(self.id, f"استجابة غير صالحة: {exc}") from exc

    def _discovered_file(self) -> Path:
        configured = self.descriptor.api.get("discovered_file")
        if not configured:
            return DISCOVERED_FILE
        path = Path(configured)
        return path if path.is_absolute() else DISCOVERED_FILE.parent / path

    def _organizations(self, topic_id: str | None = None) -> list[dict]:
        """Configured publishers, narrowed to those that publish the topic.

        A publisher may list the topics it covers; one that does is skipped
        for questions on other topics, which saves requests and keeps, say,
        court statistics out of an electricity search. A publisher with no
        list -- or a question with no topic -- is always searched.
        """
        configured = self.descriptor.api.get("organizations") or []
        orgs = [o for o in configured if isinstance(o, dict) and (o.get("id") or o.get("name_ar"))]
        # Publishers found by `masdar publishers`, after the configured ones,
        # which win on a clash.
        known = {str(v) for o in orgs for v in (o.get("id"), o.get("name_ar")) if v}
        for entry in load_discovered(self._discovered_file()):
            if not ({str(v) for v in (entry.get("id"), entry.get("name_ar")) if v} & known):
                orgs.append(entry)
        if topic_id is None:
            return orgs
        return [o for o in orgs if not o.get("topics") or topic_id in o["topics"]]

    # -- catalogue -----------------------------------------------------
    def catalogue(self, topic_id: str | None = None) -> tuple[list[dict], list[str]]:
        """Every dataset of the relevant publishers, with any failures.

        One request per publisher, made in parallel and cached on disk by the
        HTTP layer. A publisher that fails is reported, not fatal: the others
        still answer.
        """
        orgs = self._organizations(topic_id)

        def request(key: str):
            return self._get_json(self._api(
                "organization",
                "/data/api/organizations?version=-1&organization={id}",
                id=key,
            ))

        def load(org: dict):
            # By id first: it survives a renaming. The API also accepts the
            # exact Arabic name, which survives a mistyped or reissued id --
            # so a publisher configured with both is found if either holds.
            keys = [str(k) for k in (org.get("id"), org.get("name_ar")) if k]
            problem = None
            for key in keys:
                try:
                    return org, request(key), None
                except SourceUnreachable as exc:
                    return org, None, exc.reason  # the platform is down; a retry won't help
                except SourceError as exc:
                    problem = exc.reason
            return org, None, problem

        with ThreadPoolExecutor(max_workers=CATALOGUE_WORKERS) as pool:
            results = list(pool.map(load, orgs))  # map keeps the configured order

        entries: list[dict] = []
        problems: list[str] = []
        for org, payload, problem in results:
            if problem is not None:
                problems.append(f"{org.get('name_ar') or org.get('id')}: {problem}")
                continue
            for item in payload.get("datasets") or [] if isinstance(payload, dict) else []:
                if isinstance(item, dict) and item.get("id"):
                    entries.append({
                        "id": str(item["id"]),
                        "title_ar": str(item.get("titleAr") or "").strip(),
                        "title_en": str(item.get("titleEn") or "").strip(),
                        "org_ar": str(payload.get("nameAr") or org.get("name_ar") or ""),
                        "org_en": str(payload.get("nameEn") or ""),
                    })
        return entries, problems

    def _score(self, request: DataRequest, entry: dict, lexicon) -> float:
        text = f"{entry['title_ar']} {entry['title_en']}"
        score, _ = typed_phrase_score(request, text)
        if request.topic:
            keywords = (*request.topic.keywords_ar, *request.topic.keywords_en,
                        request.topic.label_ar, request.topic.label_en)
            # Being on the topic, not how many of its words a title repeats:
            # «GDP and National Accounts» must not outrank the growth rate
            # asked for by name.
            score += min(2.0, sum(1.0 for k in keywords if k and contains_phrase(text, k)))
        score += sum(2.0 for t in request.free_terms if contains_phrase(text, t))
        if score <= 0:
            return 0.0  # a breakdown word alone is not a match
        # Worded like the question: a publisher's catalogue holds dozens of
        # titles sharing a topic, and the one asked by name must come first.
        score += W_WORD_FIT * word_fit(request.raw_query, entry["title_ar"], lexicon.stopwords)
        for dimension in request.dimensions:
            if any(contains_phrase(text, w) for w in lexicon.dimension_words.get(dimension, ())):
                score += 2.0
        return score + _period_fit(request, text)

    # -- search --------------------------------------------------------
    def search(self, request: DataRequest, limit: int = 10) -> list[DatasetCandidate]:
        if self.http is None:
            raise SourceError(self.id, "no HTTP client configured")
        entries, problems = self.catalogue(request.topic.id if request.topic else None)
        if not entries and problems:
            raise SourceUnreachable(self.id, "؛ ".join(problems[:3]))

        lexicon = load_lexicon()
        scored = [(self._score(request, e, lexicon), e) for e in entries]
        ranked = sorted((p for p in scored if p[0] > 0), key=lambda p: (-p[0], p[1]["title_ar"]))

        opened = MAX_DETAIL_FETCHES_VIA_EXIT if self._via_exit else MAX_DETAIL_FETCHES

        def open_one(entry: dict) -> DatasetCandidate | None:
            try:
                return self._candidate(entry, lexicon)
            except (SourceError, SourceUnreachable):
                return None

        # In parallel: through the Saudi exit each request takes seconds, and
        # one after another they outlast the answer's time limit.
        chosen = [entry for _, entry in ranked[: min(limit, opened)]]
        with ThreadPoolExecutor(max_workers=max(1, len(chosen))) as pool:
            opened_ones = list(pool.map(open_one, chosen))  # map keeps the ranking
        return [c for c in opened_ones if c is not None]

    def _candidate(self, entry: dict, lexicon) -> DatasetCandidate | None:
        dataset_id = entry["id"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            asked = [pool.submit(self._get_json, self._api(key, default, id=dataset_id))
                     for key, default in (
                         ("dataset", "/data/api/datasets?version=-1&dataset={id}"),
                         ("resources", "/data/api/datasets/resources?version=-1&dataset={id}"),
                     )]
            details, listing = (future.result() for future in asked)
        if not isinstance(details, dict):
            return None

        files = listing.get("resources") if isinstance(listing, dict) else None
        resources: list[Resource] = []
        columns: list[str] = []
        for item in files or []:
            if not isinstance(item, dict) or not item.get("downloadUrl"):
                continue
            fmt = str(item.get("format") or "").upper().strip(".")
            resources.append(Resource(
                url=_encode_url(str(item["downloadUrl"])),
                format=fmt,
                title=str(item.get("name") or ""),
            ))
            columns.extend(str(c.get("name", "")) for c in item.get("columns") or []
                           if isinstance(c, dict))
        # Spreadsheets first: they are what the answer delivers.
        order = {"XLSX": 0, "XLS": 1, "CSV": 2}
        resources.sort(key=lambda r: order.get(r.format, 9 if r.format in TABULAR else 99))

        tags = [str(t.get("name") or "") for t in details.get("tags") or [] if isinstance(t, dict)]
        categories = [
            str(c.get("titleAr") or c.get("titleEn") or "")
            for c in details.get("categories") or [] if isinstance(c, dict)
        ]
        title_ar = str(details.get("titleAr") or entry["title_ar"]).strip()
        title_en = str(details.get("titleEn") or entry["title_en"]).strip()
        frequency = str(details.get("updateFrequency") or "").strip()
        description = str(details.get("descriptionAr") or details.get("descriptionEn") or "")
        if frequency:
            description = f"{description} — تكرار التحديث: {frequency}".strip(" —")

        # Breakdowns only from the files' declared columns -- their schema.
        # A title or tag saying "حسب المناطق" describes the data; a column
        # proves it is there. The orchestrator also checks the file itself.
        present: list[Dimension] = []
        column_text = " ".join(columns)
        for dimension, words in lexicon.dimension_words.items():
            if column_text and any(contains_phrase(column_text, w) for w in words):
                present.append(dimension)

        return DatasetCandidate(
            source_id=self.id,
            dataset_id=dataset_id,
            title_ar=title_ar or title_en,
            title_en=title_en,
            description=description,
            keywords=" ".join([*tags, *categories]),
            landing_url=self.descriptor.url(f"/ar/datasets/view/{dataset_id}"),
            publisher_ar=str(details.get("providerNameAr") or entry["org_ar"]),
            publisher_en=str(details.get("providerNameEn") or entry["org_en"]),
            resources=tuple(resources),
            claimed_coverage=coverage_from_period(details.get("timePeriod")),
            last_updated=_as_date(details.get("updatedAt")),
            provided_dimensions=tuple(present),
            download_note=LINK_ONLY_NOTE if self._via_exit and resources else "",
        ) if resources or title_ar else None

    def probe(self) -> str:
        entries, problems = self.catalogue()
        orgs = len(self._organizations())
        failed = f"، {len(problems)} جهة لم تُجب" if problems else ""
        return f"{len(entries)} مجموعة بيانات من {orgs} جهة{failed}"
