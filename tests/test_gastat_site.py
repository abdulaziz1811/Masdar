"""GASTAT's website search: bulletins with spreadsheets, and data tables.

The search pages under fixtures/gastat_site were recorded on 2026-09-30 from
a German exit (the site answers from outside the Kingdom) and trimmed to the
result list; the bulletin page keeps only its /documents/ links. The
spreadsheet behind the bulletin could not be fetched from the build
environment, so the one served here is SYNTHETIC, built in `_synthetic_cpi`.
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

from openpyxl import Workbook

from masdar.domain.models import CoverageOrigin, Verdict
from masdar.nlu.lexicon import load_lexicon
from masdar.nlu.parser import parse
from masdar.pipeline.orchestrator import Agent, AgentConfig
from masdar.sources.adapters.gastat_site import (
    PAGE_NOTE,
    TABLE_NOTE,
    classify,
    document_links,
    parse_results,
    query_words,
)
from masdar.sources.http import HttpClient
from masdar.sources.registry import Registry, load_descriptors
from masdar.sources.transport import Response

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gastat_site"
RECORDED = {
    ("المعتمرين",): "search_umrah.html",
    ("البطالة",): "search_unemployment.html",
    ("التقديرات", "السكانية"): "search_population_estimates.html",
    ("الرقم", "القياسي", "لأسعار", "المستهلك"): "search_cpi.html",
    ("الحجاج",): "search_pilgrims.html",
}
CPI_AUGUST = "/ar/w/consumer-price-index-august-2026"


def _page(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _synthetic_cpi() -> bytes:
    """A stand-in for the bulletin's spreadsheet: formulaic, not GASTAT's."""
    book = Workbook()
    sheet = book.active
    sheet.append(["السنة", "الشهر", "الرقم القياسي العام (بيانات تجريبية)"])
    for month in range(1, 9):
        sheet.append([2026, month, 100 + month])
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


class SiteTransport:
    """GASTAT's site as recorded: a search answers when its words were recorded."""

    name = "recorded"

    def __init__(self):
        self.urls: list[str] = []

    def get(self, url, source_id, params=None, headers=None):
        self.urls.append(url)
        parts = urlsplit(url)
        if parts.path == "/ar/search":
            asked = set(dict(parse_qsl(parts.query)).get("q", "").split())
            body = next((_page(name) for words, name in RECORDED.items()
                         if set(words) <= asked), "")
            return Response(url, url, 200, body.encode(), "text/html", {})
        if unquote(parts.path) == CPI_AUGUST:
            return Response(url, url, 200, _page("bulletin_cpi_august_2026.html").encode(),
                            "text/html", {})
        if "/documents/" in parts.path and ".xlsx" in parts.path:
            return Response(url, url, 200, _synthetic_cpi(),
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            {})
        if parts.path.startswith("/ar/w/"):
            return Response(url, url, 200, b"<html></html>", "text/html", {})
        raise AssertionError(f"unexpected request: {url}")

    def describe(self):
        return self.name


def agent(tmp_path, transport=None):
    http = HttpClient(use_cache=False, retries=1, transport=transport or SiteTransport())
    registry = Registry(tuple(d for d in load_descriptors() if d.id == "gastat"), http)
    return Agent(registry=registry, http=http,
                 config=AgentConfig(out_dir=tmp_path / "out", today=date(2026, 9, 30)))


class TestReadingTheSearchPage:
    def test_every_result_is_read(self):
        results = parse_results(_page("search_umrah.html"))
        assert len(results) == 40
        assert all(r["title"] and r["url"].startswith("https://www.stats.gov.sa/") for r in results)
        # The search page's back-link parameters are not part of the link.
        assert not any("p_l_back_url" in r["url"] for r in results)
        assert results[1]["date"] == date(2026, 4, 12)

    def test_tables_and_bulletins_are_told_from_news(self):
        umrah = [classify(r) for r in parse_results(_page("search_umrah.html"))]
        assert umrah.count("table") == 39
        cpi = [classify(r) for r in parse_results(_page("search_cpi.html"))]
        assert cpi.count("bulletin") == 31 and cpi.count("table") == 0
        news = [r for r in parse_results(_page("search_cpi.html"))
                if "/w/news/" in r["url"]]
        assert news and all(classify(r) is None for r in news)

    def test_a_bulletins_spreadsheet_is_found_and_its_pdf_is_not(self):
        files = document_links(_page("bulletin_cpi_august_2026.html"),
                               "https://www.stats.gov.sa" + CPI_AUGUST)
        assert [f.format for f in files] == ["XLSX"]
        assert files[0].title == "CPI Tables-Aug 2026-AR-EN (1).xlsx"
        assert files[0].url.startswith("https://www.stats.gov.sa/documents/20117/")


class TestTheQuery:
    def test_the_askers_words_without_years_or_filler(self):
        words = query_words(parse("كم عدد المعتمرين من الخارج في 2023"), load_lexicon().stopwords)
        assert "المعتمرين" in words and "الخارج" in words
        assert "2023" not in words and "من" not in words and "في" not in words

    def test_spelling_is_kept_for_the_sites_own_search(self):
        words = query_words(parse("الرقم القياسي لأسعار المستهلك"), load_lexicon().stopwords)
        assert words == ["الرقم", "القياسي", "لأسعار", "المستهلك"]


class TestAnswers:
    def test_a_table_seen_on_the_site_is_found_not_denied(self, tmp_path):
        answer = agent(tmp_path).answer("المعتمرين من الخارج حسب منفذ الدخول")
        assert answer.verdict is Verdict.UNVERIFIED
        primary = answer.primary
        assert "منفذ الدخول" in primary.candidate.title_ar
        assert primary.provenance.landing_url.startswith("https://www.stats.gov.sa/ar/w/")
        assert TABLE_NOTE in primary.notes
        assert "لم يُعثر" not in answer.message_ar

    def test_a_bulletins_spreadsheet_is_opened_and_read(self, tmp_path):
        answer = agent(tmp_path).answer("الرقم القياسي لأسعار المستهلك لشهر أغسطس 2026")
        assert answer.verdict is Verdict.AVAILABLE
        primary = answer.primary
        assert "أغسطس 2026" in primary.candidate.title_ar
        assert primary.coverage.origin is CoverageOrigin.OBSERVED_DATA
        assert primary.export_path
        assert primary.provenance.last_updated == date(2026, 9, 15)

    def test_the_unemployment_table_is_found_by_its_subject(self, tmp_path):
        # «معدل» is left out of the site's query: it matches every rate on
        # the site (fertility, productivity...), and the search takes any word.
        answer = agent(tmp_path).answer("معدل البطالة")
        titles = [f.candidate.title_ar for f in answer.findings]
        assert titles and all("البطالة" in t for t in titles)

    def test_the_same_title_twice_is_kept_apart_by_its_survey_year(self, tmp_path):
        answer = agent(tmp_path).answer("المعتمرين من الخارج حسب منفذ الدخول")
        entry = [f for f in answer.findings if "منفذ الدخول" in f.candidate.title_ar]
        years = [f.coverage.years for f in entry]
        assert len(entry) == 2 and len(set(years)) == 2
        assert all(f.coverage.origin is CoverageOrigin.INFERRED_TITLE for f in entry)

    def test_the_site_order_never_stands_in_for_the_subject(self):
        from masdar.pipeline.resolve import SUBJECT_RELEVANCE
        from masdar.sources.adapters.gastat_site import MAX_RELEVANCE

        assert MAX_RELEVANCE < SUBJECT_RELEVANCE

    def test_nothing_found_points_to_gastats_own_search(self, tmp_path):
        answer = agent(tmp_path).answer("كم عدد الحجاج")
        assert answer.verdict is Verdict.NO_SOURCE
        (label, url), = answer.elsewhere
        assert "موقع الهيئة العامة للإحصاء" in label
        assert url.startswith("https://www.stats.gov.sa/ar/search?q=")
        assert url in answer.message_ar

    def test_a_bulletin_without_a_file_is_still_a_link(self, tmp_path):
        answer = agent(tmp_path).answer("الرقم القياسي لأسعار المستهلك لشهر مايو 2026")
        may = next(f for f in answer.findings if "مايو 2026" in f.candidate.title_ar)
        assert PAGE_NOTE in may.notes
        assert may.provenance.landing_url.endswith("consumer-price-index-may-2026-1")
