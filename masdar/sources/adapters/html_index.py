"""Adapter for publishers that list releases as HTML pages.

Most Saudi ministries and regulators publish statistics as files linked from
a publications page rather than through an API. This adapter reads such a
page and turns the file links into candidates.

Coverage found this way is always `INFERRED_TITLE`: a filename saying 2024
is a hint, not proof, so the verifier will insist on opening the file before
reporting the year as available.
"""

from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import urljoin

from masdar.domain.models import (
    Coverage,
    CoverageOrigin,
    DataRequest,
    DatasetCandidate,
    Resource,
)
from masdar.nlu.normalize import contains_phrase, normalize
from masdar.nlu.parser import extract_years
from masdar.sources.base import SourceAdapter, SourceError

DATA_SUFFIXES = ("xlsx", "xls", "csv", "tsv", "json")
DOCUMENT_SUFFIXES = ("pdf",)


class _LinkExtractor(HTMLParser):
    """Collects (href, visible text) pairs."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self._href = href
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join(self._text).strip()))
            self._href = None
            self._text = []


def _suffix_of(url: str) -> str:
    tail = url.split("?", 1)[0].split("#", 1)[0].rsplit("/", 1)[-1]
    return tail.rsplit(".", 1)[-1].lower() if "." in tail else ""


class HtmlIndexAdapter(SourceAdapter):
    """Scrapes a configured publications index for data files."""

    def _index_url(self) -> str:
        path = self.descriptor.api.get("publications_index", "/")
        return self.descriptor.url(path)

    def search(self, request: DataRequest, limit: int = 10) -> list[DatasetCandidate]:
        if self.http is None:
            raise SourceError(self.id, "no HTTP client configured")

        index_url = self._index_url()
        fetched = self.http.get(index_url, source_id=self.id)
        parser = _LinkExtractor()
        parser.feed(fetched.text())

        terms = request.search_terms()
        scored: list[tuple[float, DatasetCandidate]] = []
        seen: set[str] = set()

        for href, text in parser.links:
            suffix = _suffix_of(href)
            if suffix not in DATA_SUFFIXES + DOCUMENT_SUFFIXES:
                continue
            absolute = urljoin(fetched.final_url or index_url, href)
            if absolute in seen:
                continue

            label = text or href.rsplit("/", 1)[-1]
            haystack = f"{label} {href}"
            # Without a term match this is just a link on a page, not an
            # answer to the question.
            hits = sum(1 for term in terms if term and contains_phrase(haystack, term))
            if not hits:
                continue
            seen.add(absolute)

            years = extract_years(haystack)
            candidate = DatasetCandidate(
                source_id=self.id,
                dataset_id=absolute.rsplit("/", 1)[-1] or normalize(label)[:60],
                title_ar=label,
                description=f"مستخرج من صفحة النشرات: {index_url}",
                landing_url=index_url,
                publisher_ar=self.descriptor.name_ar,
                publisher_en=self.descriptor.name_en,
                resources=(
                    Resource(url=absolute, format=suffix.upper(), title=label),
                ),
                claimed_coverage=Coverage(
                    years=frozenset(years),
                    origin=CoverageOrigin.INFERRED_TITLE,
                    note="مستنتجة من اسم الملف أو نص الرابط" if years else "غير معروفة",
                ),
            )
            # Prefer tabular files: a PDF cannot become a spreadsheet.
            bonus = 1.0 if suffix in DATA_SUFFIXES else 0.0
            scored.append((hits + bonus, candidate))

        scored.sort(key=lambda pair: -pair[0])
        return [candidate for _, candidate in scored[:limit]]

    def probe(self) -> str:
        fetched = self.http.get(self._index_url(), source_id=self.id)
        parser = _LinkExtractor()
        parser.feed(fetched.text())
        data_links = [h for h, _ in parser.links if _suffix_of(h) in DATA_SUFFIXES]
        return (
            f"HTTP {fetched.status}, {len(parser.links)} link(s), "
            f"{len(data_links)} tabular file link(s)"
        )
