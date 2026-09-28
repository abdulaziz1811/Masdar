"""Tests for the national open data platform adapter.

Recorded live on 2026-09-28: `organization_energy.json` is eight entries
copied verbatim from the Ministry of Energy's catalogue (which held about a
hundred); `dataset_134b9ff3.json` and `resources_134b9ff3.json` are
verbatim. The data file itself could not be downloaded from the build
environment, so `SYNTHETIC_consumption_by_area.csv` stands in for it: it has
the platform's declared column (السنة) and formulaic numbers, and is named
so it cannot be mistaken for the real file.
"""

from datetime import date
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest

from masdar.domain.models import CoverageOrigin, Verdict
from masdar.pipeline.orchestrator import Agent, AgentConfig
from masdar.sources.adapters.saudi_open_data import coverage_from_period
from masdar.sources.base import SourceError
from masdar.sources.http import HttpClient
from masdar.sources.registry import Registry, load_descriptors
from masdar.sources.transport import Response

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "open_data"
CONSUMPTION = "134b9ff3-ef7c-4543-a245-72373fcf9531"
ENERGY = "886f6d71-8034-47f7-914d-9613825d9153"


class PortalTransport:
    name = "recorded"

    def __init__(self, missing_orgs=(), broken_xlsx=False):
        self.urls: list[str] = []
        self.missing_orgs = set(missing_orgs)
        self.broken_xlsx = broken_xlsx

    def get(self, url, source_id, params=None, headers=None):
        self.urls.append(url)
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query))
        if parts.path.endswith("/organizations"):
            org = query.get("organization")
            # The platform accepts the exact Arabic name in place of the id.
            if org == "وزارة الطاقة":
                org = ENERGY
            if org != ENERGY or org in self.missing_orgs:
                raise SourceError(source_id, "not found: Publisher Not Found")
            body = (FIXTURES / "organization_energy.json").read_bytes()
        elif parts.path.endswith("/datasets/resources"):
            if query.get("dataset") != CONSUMPTION:
                raise SourceError(source_id, "not found")
            body = (FIXTURES / "resources_134b9ff3.json").read_bytes()
        elif parts.path.endswith("/datasets"):
            if query.get("dataset") != CONSUMPTION:
                raise SourceError(source_id, "not found")
            body = (FIXTURES / "dataset_134b9ff3.json").read_bytes()
        elif "/odp-public/" in parts.path and parts.path.endswith(".xlsx"):
            if self.broken_xlsx:
                return Response(url, url, 200, b"not a spreadsheet", "application/octet-stream", {})
            body = (FIXTURES / "SYNTHETIC_consumption_by_area.xlsx").read_bytes()
            return Response(url, url, 200, body, "application/vnd.ms-excel", {})
        elif "/odp-public/" in parts.path:
            body = (FIXTURES / "SYNTHETIC_consumption_by_area.csv").read_bytes()
            return Response(url, url, 200, body, "text/csv", {})
        else:
            raise AssertionError(f"unexpected request: {url}")
        return Response(url, url, 200, body, "application/json", {})

    def describe(self):
        return self.name


def registry(transport):
    http = HttpClient(use_cache=False, retries=1, transport=transport)
    return Registry(tuple(d for d in load_descriptors() if d.id == "saudi_open_data"), http)


def agent(tmp_path, transport):
    reg = registry(transport)
    return Agent(registry=reg, http=reg._http,
                 config=AgentConfig(out_dir=tmp_path / "out", today=date(2026, 9, 28)))


class TestCatalogue:
    def test_one_failing_publisher_does_not_sink_the_others(self):
        adapter = registry(PortalTransport()).adapter("saudi_open_data")
        entries, problems = adapter.catalogue()
        # Only the energy fixture answers; every other configured publisher
        # fails and is reported, not fatal.
        assert len(entries) == 8
        assert len(problems) == len(adapter._organizations()) - 1

    def test_titles_arrive_in_arabic(self):
        entries, _ = registry(PortalTransport()).adapter("saudi_open_data").catalogue()
        consumption = next(e for e in entries if e["id"] == CONSUMPTION)
        assert "حسب المناطق" in consumption["title_ar"]
        assert consumption["org_ar"] == "وزارة الطاقة"


class TestSearch:
    def test_the_original_question_finds_regional_consumption(self):
        from masdar.nlu.parser import parse

        found = registry(PortalTransport()).adapter("saudi_open_data").search(
            parse("ابي احصاءات استهلاك الكهرباء لسنه 2022 حسب المناطق")
        )
        assert found and found[0].dataset_id == CONSUMPTION

    def test_metadata_is_the_platforms(self):
        from masdar.nlu.parser import parse

        candidate = registry(PortalTransport()).adapter("saudi_open_data").search(
            parse("استهلاك الكهرباء 2022 حسب المناطق")
        )[0]
        assert candidate.publisher_ar == "وزارة الطاقة"
        assert candidate.last_updated == date(2024, 7, 14)
        assert candidate.landing_url.endswith(f"/ar/datasets/view/{CONSUMPTION}")
        assert "حسب المناطق" in candidate.keywords

    def test_declared_period_is_a_claim_not_proof(self):
        from masdar.nlu.parser import parse

        candidate = registry(PortalTransport()).adapter("saudi_open_data").search(
            parse("استهلاك الكهرباء 2022")
        )[0]
        assert candidate.claimed_coverage.origin is CoverageOrigin.METADATA_CLAIM
        assert candidate.claimed_coverage.is_exhaustive is False

    def test_spreadsheet_first_and_spaces_encoded(self):
        from masdar.nlu.parser import parse

        candidate = registry(PortalTransport()).adapter("saudi_open_data").search(
            parse("استهلاك الكهرباء 2022")
        )[0]
        assert candidate.resources[0].format == "XLSX"
        assert " " not in candidate.resources[0].url
        assert "%20" in candidate.resources[0].url

    def test_unrelated_question_finds_nothing(self):
        from masdar.nlu.parser import parse

        # No topic, so every publisher is asked; the energy catalogue answers
        # and holds nothing about giraffes.
        assert registry(PortalTransport()).adapter("saudi_open_data").search(
            parse("عدد الزرافات 2022")
        ) == []

    def test_only_publishers_of_the_topic_are_asked(self):
        from masdar.nlu.parser import parse

        transport = PortalTransport()
        registry(transport).adapter("saudi_open_data").search(parse("استهلاك الكهرباء 2022"))
        asked = {
            dict(parse_qsl(urlsplit(u).query)).get("organization")
            for u in transport.urls if urlsplit(u).path.endswith("/organizations")
        }
        assert ENERGY in asked
        assert "35c63412-c4ae-4303-8fef-56cfd71303cf" not in asked  # the courts
        assert "bd9ff32e-4956-4dbf-bdde-1c1e4d786640" in asked  # GASTAT: no topic list

    def test_a_topicless_question_asks_everyone(self):
        from masdar.nlu.parser import parse

        transport = PortalTransport()
        adapter = registry(transport).adapter("saudi_open_data")
        adapter.search(parse("عدد الزرافات 2022"))
        asked = {
            dict(parse_qsl(urlsplit(u).query)).get("organization")
            for u in transport.urls if urlsplit(u).path.endswith("/organizations")
        }
        assert {str(o["id"]) for o in adapter._organizations()} <= asked


class TestPublisherIdentity:
    """A publisher configured by id and name is found if either still holds."""

    @staticmethod
    def adapter_with(orgs, transport):
        from dataclasses import replace

        base = next(d for d in load_descriptors() if d.id == "saudi_open_data")
        descriptor = replace(base, api={**base.api, "organizations": orgs})
        http = HttpClient(use_cache=False, retries=1, transport=transport)
        return Registry((descriptor,), http).adapter("saudi_open_data")

    def test_a_stale_id_falls_back_to_the_exact_name(self):
        adapter = self.adapter_with(
            [{"id": "00000000-dead-beef-0000-000000000000", "name_ar": "وزارة الطاقة"}],
            PortalTransport(),
        )
        entries, problems = adapter.catalogue()
        assert len(entries) == 8 and problems == []

    def test_a_name_alone_is_enough(self):
        adapter = self.adapter_with([{"name_ar": "وزارة الطاقة"}], PortalTransport())
        entries, _ = adapter.catalogue()
        assert len(entries) == 8

    def test_when_neither_holds_it_is_reported(self):
        adapter = self.adapter_with(
            [{"id": "00000000-dead-beef-0000-000000000000", "name_ar": "جهة لا وجود لها"}],
            PortalTransport(),
        )
        entries, problems = adapter.catalogue()
        assert entries == [] and len(problems) == 1


class TestPeriods:
    def test_a_range_covers_every_year_between_its_ends(self):
        assert coverage_from_period("2015-01-01 - 2023-12-31").years == frozenset(range(2015, 2024))

    def test_a_single_year(self):
        assert coverage_from_period("2022-01-01 - 2022-12-31").years == frozenset({2022})

    def test_no_period_is_unknown(self):
        assert coverage_from_period(None).is_empty


class TestEndToEnd:
    def test_available_year_is_confirmed_from_the_file(self, tmp_path):
        answer = agent(tmp_path, PortalTransport()).answer(
            "ابي احصاءات استهلاك الكهرباء لسنه 2022 حسب المناطق"
        )
        assert answer.verdict is Verdict.AVAILABLE
        assert answer.primary.coverage.origin.is_authoritative
        assert answer.primary.provenance.last_updated == date(2024, 7, 14)
        assert answer.primary.export_path

    def test_the_original_2026_question_is_refused_with_the_real_year(self, tmp_path):
        answer = agent(tmp_path, PortalTransport()).answer(
            "ابي احصاءات الطاقه الكهربائيه لسنه 2026 حسب المناطق"
        )
        assert answer.verdict is Verdict.NOT_AVAILABLE
        assert [s.year for s in answer.suggestions] == [2022]


class TestFileFallback:
    def test_a_damaged_spreadsheet_falls_back_to_the_csv(self, tmp_path):
        answer = agent(tmp_path, PortalTransport(broken_xlsx=True)).answer(
            "استهلاك الكهرباء لسنه 2022 حسب المناطق"
        )
        assert answer.verdict is Verdict.AVAILABLE
        assert answer.primary.coverage.origin.is_authoritative
        assert any("بعد تعذّر غيره" in n for n in answer.primary.notes)

    def test_provenance_cites_the_file_actually_read(self, tmp_path):
        import hashlib

        answer = agent(tmp_path, PortalTransport(broken_xlsx=True)).answer(
            "استهلاك الكهرباء لسنه 2022 حسب المناطق"
        )
        provenance = answer.primary.provenance
        # The XLSX was ranked first but failed; the CSV was read.
        assert provenance.resource_url.endswith(".csv")
        csv_bytes = (FIXTURES / "SYNTHETIC_consumption_by_area.csv").read_bytes()
        assert provenance.sha256 == hashlib.sha256(csv_bytes).hexdigest()


class TestUnreachable:
    def test_all_publishers_failing_is_unreachable_not_empty(self):
        from masdar.nlu.parser import parse
        from masdar.sources.base import SourceUnreachable

        transport = PortalTransport(missing_orgs={ENERGY})
        with pytest.raises(SourceUnreachable):
            registry(transport).adapter("saudi_open_data").search(parse("الكهرباء 2022"))


class TestYearsInTitles:
    """Publishers split a series into one dataset per year or quarter.

    Recorded live on 2026-09-28: the Ministry of Commerce publishes
    «الأسماء التجارية المحجوزة لعام 2022 الربع الثالث» and its siblings as
    separate datasets. Only the top few results are opened, so the year in
    the title must steer the ranking, or a question about 2022 opens the 2020
    files and reports 2022 as missing.
    """

    @staticmethod
    def ranked(question):
        import json

        from masdar.nlu.lexicon import load_lexicon
        from masdar.nlu.parser import parse
        from masdar.sources.adapters.saudi_open_data import SaudiOpenDataAdapter

        payload = json.loads(
            (FIXTURES / "organization_commerce.json").read_text(encoding="utf-8")
        )
        entries = [
            {"id": d["id"], "title_ar": d["titleAr"].strip(), "title_en": d["titleEn"].strip()}
            for d in payload["datasets"]
        ]
        lexicon = load_lexicon()
        request = parse(question, lexicon)
        adapter = SaudiOpenDataAdapter.__new__(SaudiOpenDataAdapter)
        scored = [(adapter._score(request, e, lexicon), e["title_ar"]) for e in entries]
        return [t for s, t in sorted(scored, key=lambda p: (-p[0], p[1])) if s > 0]

    def test_the_requested_year_is_opened_first(self):
        top = self.ranked("الأسماء التجارية المحجوزة 2022")[:4]
        assert all("2022" in title for title in top)

    def test_other_years_stay_available_below(self):
        titles = self.ranked("الأسماء التجارية المحجوزة 2022")
        assert any("2021" in t for t in titles)

    def test_latest_means_the_newest_year_then_quarter(self):
        top = self.ranked("اخر بيانات السجلات التجارية القائمة")[0]
        assert "2026" in top and "الثاني" in top

    def test_a_later_quarter_never_outranks_a_later_year(self):
        from masdar.nlu.parser import parse
        from masdar.sources.adapters.saudi_open_data import _period_fit

        latest = parse("اخر البيانات")
        assert _period_fit(latest, "2026 Q1") > _period_fit(latest, "2025 Q4")

    def test_quarters_are_read_however_they_are_written(self):
        from masdar.sources.adapters.saudi_open_data import _title_quarter

        assert _title_quarter("لسنة 2026حسب الربع السنوي الاول") == 1
        assert _title_quarter("تعداد 2025 للربع الثاني") == 2
        assert _title_quarter("المحجوزة لعام 2022 الربع الأول") == 1
        assert _title_quarter("Active registrations 2024 By Q4") == 4
        assert _title_quarter("نسب الإشغال 2024") == 0

    def test_titles_without_a_year_are_left_alone(self):
        from masdar.nlu.parser import parse
        from masdar.sources.adapters.saudi_open_data import _period_fit

        assert _period_fit(parse("السجلات التجارية 2024"), "السجلات التجارية") == 0.0
