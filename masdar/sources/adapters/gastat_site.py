"""Adapter for GASTAT's own website search, www.stats.gov.sa.

Most of what GASTAT publishes -- some four thousand data tables and fourteen
hundred bulletins -- is on its website and in neither of its APIs, so a
reader who has seen a table there and asks for it by name must not be told
nothing exists. Checked on 2026-09-30, from a German exit as well as a Saudi
one:

* `GET /ar/search?q=<words>` is server-rendered HTML listing each result as a
  type badge, a link with the title, a snippet and a date. The search stems
  Arabic («المعتمرين» finds «المعتمرون»).
* News are under `/ar/w/news/`.
* A bulletin (e.g. «الرقم القياسي لأسعار المستهلك لشهر أغسطس 2026») is a page
  under `/ar/w/` whose snippet names its attachments, and the page links them
  under `/documents/`: a spreadsheet there is downloaded and read like any
  other file, so its years are observed, not guessed.
* A data table is a page under `/ar/w/` that embeds a Tableau view (its
  snippet shows the embed). It has no file to download, so it is offered as
  a link: its year is at best read from its title -- `INFERRED_TITLE`, which
  can never confirm availability. The answer says "found on GASTAT's site,
  not verified" and gives the page, instead of "not found".
"""

from __future__ import annotations

import html
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from urllib.parse import quote, unquote_plus, urljoin, urlsplit, urlunsplit

from masdar.domain.models import (
    Coverage,
    CoverageOrigin,
    DataRequest,
    DatasetCandidate,
    Resource,
)
from masdar.nlu.lexicon import load_lexicon
from masdar.nlu.normalize import fold_digits, mentions, normalize, strip_article
from masdar.nlu.parser import LATEST_MARKERS, extract_years
from masdar.sources.base import SourceAdapter, SourceError, SourceUnreachable

DEFAULT_SEARCH = "/ar/search?q={query}&delta=40"
MAX_QUERY_WORDS = 8
# Bulletin pages opened per question to find their spreadsheets.
MAX_PAGES = 3
DATA_SUFFIXES = ("xlsx", "xls", "csv")

TABLE_NOTE = (
    "هذا الجدول منشور في موقع الهيئة العامة للإحصاء ويُعرض فيه تفاعلياً، ولا يُتاح "
    "ملفه لهذه المنصة؛ افتحه من زر «صفحة البيانات لدى المصدر» ونزّل بياناته من هناك."
)
PAGE_NOTE = (
    "لم يُعثر في صفحة هذه النشرة على ملف بيانات يمكن قراءته؛ افتحها من زر "
    "«صفحة البيانات لدى المصدر»."
)

# One result: the type badge, the titled link, the snippet, then the date.
_RESULT = re.compile(
    r'<span class="badge[^"]*">\s*(?P<kind>[^<]*?)\s*</span>\s*'
    r'<a href="(?P<href>[^"]+)"[^>]*class="dl-link[^"]*"[^>]*>(?P<title>.*?)</a>\s*'
    r'<p[^>]*>(?P<snippet>.*?)</p>'
    r'(?:\s*<p[^>]*>(?P<date>[^<]*)</p>)?',
    re.S,
)
_TAG = re.compile(r"<[^>]+>")
_WORD = re.compile(r"[\w؀-ۿ]+")
_DOCUMENT = re.compile(r'href="(?P<href>[^"]*/documents/[^"]+)"')
# "15‏/09‏/26، 8:26 ص" -- day, month, two-digit year, with direction marks.
_DATE = re.compile(r"(\d{1,2})\D{1,3}(\d{1,2})\D{1,3}(\d{2,4})")
_FILE_NAME = re.compile(r"\.(?:xlsx|xls|csv)\b", re.I)
# A table's Tableau workbook is named for its survey: «umrah_survey_2018_0_AR».
_WORKBOOK = re.compile(r"/views/([^/\s'\"]+)/")
# The site's search is loose (any word matches), so its order is a hint for
# ties, kept below the level at which a source's reading of the question
# would stand in for the result naming the subject (resolve.on_subject).
MAX_RELEVANCE = 0.4
_LATEST = frozenset(normalize(m) for m in LATEST_MARKERS)


def _text(fragment: str) -> str:
    return " ".join(html.unescape(_TAG.sub(" ", fragment)).split())


def _clean_link(href: str) -> str:
    """The page itself, without the search page's back-link parameters."""
    parts = urlsplit(html.unescape(href))
    return urlunsplit(parts._replace(query="", fragment=""))


def _published(text: str | None) -> date | None:
    match = _DATE.search(fold_digits(text or ""))
    if not match:
        return None
    day, month, year = (int(g) for g in match.groups())
    year = year + 2000 if year < 100 else year
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_results(page: str) -> list[dict]:
    """Every result on a search page: kind, link, title, snippet, date."""
    return [
        {
            "kind": _text(m.group("kind")),
            "url": _clean_link(m.group("href")),
            "title": _text(m.group("title")),
            "snippet": _text(m.group("snippet")),
            "date": _published(m.group("date")),
        }
        for m in _RESULT.finditer(page)
    ]


def classify(result: dict) -> str | None:
    """'table', 'bulletin', or None for news and navigation pages."""
    path = urlsplit(result["url"]).path
    if "/w/" not in path or "/w/news/" in path:
        return None
    if "tableau" in result["snippet"].lower():
        return "table"
    if _FILE_NAME.search(result["snippet"]):
        return "bulletin"
    return None


def is_data_table(result: dict) -> bool:
    return classify(result) == "table"


def document_links(page: str, base: str) -> list[Resource]:
    """Spreadsheets a bulletin page links under /documents/, first listed first."""
    found: list[Resource] = []
    for match in _DOCUMENT.finditer(page):
        url = urljoin(base, html.unescape(match.group("href")))
        # /documents/<group>/<folder>/<name>.xlsx/<uuid>: the name is the
        # segment before the uuid.
        name = unquote_plus(next(
            (s for s in reversed(urlsplit(url).path.split("/")) if "." in s), ""))
        suffix = name.rsplit(".", 1)[-1].lower()
        if suffix in DATA_SUFFIXES and all(r.url != url for r in found):
            found.append(Resource(url=url, format=suffix.upper(), title=name))
    return found


def _years(result: dict, kind: str) -> frozenset[int]:
    """Years a result names: its title's, else a table's workbook name."""
    years = extract_years(result["title"])
    if not years and kind == "table":
        workbook = _WORKBOOK.search(result["snippet"])
        years = extract_years(workbook.group(1).replace("_", " ")) if workbook else []
    return frozenset(years)


def _fit(request: DataRequest, words: list[str], result: dict) -> float:
    """How well a result's title answers the question, to pick what to open."""
    title = result["title"]
    score = float(sum(1 for w in words if mentions(title, w)))
    asked = set(request.period.years)
    named = set(extract_years(title))
    if asked and named & asked:
        score += 3.0
    elif asked and named:
        score -= 1.0
    # Newer first among equals: "the latest" means the latest release.
    released = result["date"]
    return score + (released.toordinal() / 1e7 if released else 0.0)


def query_words(request: DataRequest, stopwords: frozenset[str]) -> list[str]:
    """The asker's own words, as typed: the site's search stems them itself.

    Years and filler are left out -- "2023" would pull in every news item of
    that year -- and the words keep their original letters, since the site
    does not know this project's normalised spellings.
    """
    words: list[str] = []
    for token in _WORD.findall(request.raw_query):
        folded = normalize(token)
        if not folded or folded.isdigit() or len(folded) < 3:
            continue
        if folded in stopwords or strip_article(folded) in stopwords:
            continue
        if folded in ("حسب", "بحسب") or folded in _LATEST or token in words:
            continue
        words.append(token)
    return words[:MAX_QUERY_WORDS]


def site_search_url(request: DataRequest, base_url: str = "https://www.stats.gov.sa") -> str | None:
    """GASTAT's own search for the question's words, for a reader to open."""
    words = query_words(request, load_lexicon().stopwords)
    if not words:
        return None
    return base_url.rstrip("/") + DEFAULT_SEARCH.format(query=quote(" ".join(words)))


class GastatSiteAdapter(SourceAdapter):
    """GASTAT's bulletins and data tables, found by the website's own search."""

    def search(self, request: DataRequest, limit: int = 10) -> list[DatasetCandidate]:
        if self.http is None:
            raise SourceError(self.id, "no HTTP client configured")
        words = query_words(request, load_lexicon().stopwords)
        if not words:
            return []
        template = str(self.descriptor.api.get("search_path") or DEFAULT_SEARCH)
        url = self.descriptor.url(template.format(query=quote(" ".join(words))))
        page = self.fetch(url).content.decode("utf-8", errors="replace")

        results: list[tuple[dict, str, frozenset[int]]] = []
        seen: set[tuple[str, frozenset[int]]] = set()
        for result in parse_results(page):
            kind = classify(result)
            if kind is None:
                continue
            years = _years(result, kind)
            # The same table is often published twice under one title.
            if (result["title"], years) in seen:
                continue
            seen.add((result["title"], years))
            results.append((result, kind, years))
        results = results[:limit]

        # Bulletins are opened for their spreadsheets, the likeliest first:
        # «لشهر أغسطس 2026» among a year of monthly releases.
        bulletins = sorted(
            (r for r, kind, years in results if kind == "bulletin"),
            key=lambda r: -_fit(request, words, r),
        )[:MAX_PAGES]
        with ThreadPoolExecutor(max_workers=max(1, len(bulletins))) as pool:
            files = dict(zip(
                (r["url"] for r in bulletins),
                pool.map(self._files, (r["url"] for r in bulletins)),
                strict=True,
            ))

        candidates = []
        for index, (result, kind, years) in enumerate(results):
            resources = tuple(files.get(result["url"], ()))
            note = "" if resources else (TABLE_NOTE if kind == "table" else PAGE_NOTE)
            candidates.append(DatasetCandidate(
                source_id=self.id,
                dataset_id=urlsplit(result["url"]).path,
                title_ar=result["title"],
                description=result["snippet"] if kind == "bulletin" else "",
                landing_url=result["url"],
                publisher_ar=self.descriptor.operator_ar or self.descriptor.name_ar,
                publisher_en=self.descriptor.operator_en or self.descriptor.name_en,
                resources=resources,
                claimed_coverage=(
                    Coverage(years=years, origin=CoverageOrigin.INFERRED_TITLE,
                             note="السنة من عنوان الجدول أو اسمه فقط")
                    if years else Coverage.unknown()
                ),
                # The date GASTAT shows for the release.
                last_updated=result["date"],
                download_note=note,
                relevance=MAX_RELEVANCE * (1.0 - index / max(len(results), 1)),
            ))
        return candidates

    def _files(self, page_url: str) -> list[Resource]:
        try:
            page = self.fetch(page_url).content.decode("utf-8", errors="replace")
        except (SourceError, SourceUnreachable):
            return []  # the result is still offered, as a link
        return document_links(page, page_url)

    def probe(self) -> str:
        page = self.fetch(self.descriptor.url(DEFAULT_SEARCH.format(query=quote("السكان"))))
        found = parse_results(page.content.decode("utf-8", errors="replace"))
        kinds = [classify(r) for r in found]
        return (f"{len(found)} نتيجة في صفحة البحث: {kinds.count('bulletin')} نشرة، "
                f"{kinds.count('table')} جدول بيانات")
