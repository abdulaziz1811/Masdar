"""Tests for the GASTAT cdata API adapter.

Both fixtures are responses captured verbatim from the live API on
2026-09-27, so the parsing under test faces the real shapes -- including
the real coverage of this dataset: 2017-2019 and 2021-2022, with 2020
genuinely absent.
"""

from datetime import date
from pathlib import Path
from urllib.parse import unquote

import pytest
from openpyxl import load_workbook

from masdar.domain.models import Verdict
from masdar.pipeline.orchestrator import Agent, AgentConfig
from masdar.sources.http import HttpClient
from masdar.sources.registry import Registry, load_descriptors
from masdar.sources.transport import Response, Transport

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gastat_cdata"
TODAY = date(2026, 9, 27)

# What the live dataset actually holds.
REAL_COVERAGE = {2017, 2018, 2019, 2021, 2022}


class RoutingTransport(Transport):
    """Serves the recorded responses, routed by query string.

    A coverage probe and a data request differ only in their query, which is
    exactly the distinction that matters here.
    """

    name = "recorded"

    def __init__(self):
        self.requests: list[str] = []
        self.headers: list[dict] = []

    def get(self, url, source_id, params=None, headers=None):
        self.requests.append(url)
        self.headers.append(dict(headers or {}))
        decoded = unquote(url)
        consumption = "HES0303" in decoded
        if "dimensions[]=REGION" in decoded:
            name = "HES0303_by_region_season.json" if consumption else "by_region.json"
        elif "dimensions[]=YEAR" in decoded:
            name = "HES0303_years.json" if consumption else "coverage_years.json"
        else:
            raise AssertionError(f"unexpected request: {url}")
        body = (FIXTURES / name).read_bytes()
        return Response(
            url=url,
            final_url=url,
            status=200,
            content=body,
            media_type="application/json",
            headers={},
        )


@pytest.fixture
def transport():
    return RoutingTransport()


@pytest.fixture
def registry(transport):
    http = HttpClient(use_cache=False, retries=1, transport=transport)
    descriptors = tuple(d for d in load_descriptors() if d.id == "gastat_cdata")
    assert descriptors, "gastat_cdata must be registered in sources.yaml"
    return Registry(descriptors, http)


@pytest.fixture
def agent(registry, tmp_path):
    return Agent(
        registry=registry,
        http=registry._http,
        config=AgentConfig(out_dir=tmp_path / "out", today=TODAY),
    )


class TestRequestConstruction:
    def test_groups_by_the_requested_dimension_server_side(self, registry):
        from masdar.nlu.parser import parse

        candidate = registry.adapter("gastat_cdata").search(
            parse("الكهرباء 2022 حسب المناطق")
        )[0]
        url = unquote(candidate.resources[0].url)
        assert "dimensions[]=REGION" in url
        assert "dimensions[]=YEAR" in url
        assert "format=JSON" in url

    def test_brackets_are_sent_literally(self, registry):
        from masdar.nlu.parser import parse

        candidate = registry.adapter("gastat_cdata").search(parse("الكهرباء 2022"))[0]
        # The literal form is what was verified to work against the API.
        assert "dimensions[]=" in candidate.resources[0].url

    def test_uses_the_verified_path_template(self, registry):
        from masdar.nlu.parser import parse

        candidate = registry.adapter("gastat_cdata").search(parse("الكهرباء 2022"))[0]
        assert "/v1/stats/DPV_HES_EHE_IT_HES" in candidate.resources[0].url

    def test_consumption_query_prefers_the_consumption_dataset(self, registry):
        from masdar.nlu.parser import parse

        # HES0301 counts connected dwellings; HES0303 holds kWh consumed.
        candidate = registry.adapter("gastat_cdata").search(
            parse("ابي استهلاك الطاقه الكهربائيه لسنه 2022 حسب المناطق")
        )[0]
        assert candidate.dataset_id == "DPV_HES_EHE_IT_HES0303"

    def test_season_request_groups_by_the_period_dimension(self, registry):
        from masdar.nlu.parser import parse

        candidate = registry.adapter("gastat_cdata").search(
            parse("استهلاك الكهرباء 2022 حسب المناطق والفصل")
        )[0]
        url = unquote(candidate.resources[0].url)
        assert "dimensions[]=CONSUMP_OPERATION_PERIOD" in url
        assert "dimensions[]=REGION" in url

    def test_unrelated_query_matches_nothing(self, registry):
        from masdar.nlu.parser import parse

        assert registry.adapter("gastat_cdata").search(parse("التضخم 2022")) == []


class TestCoverageProbe:
    def test_coverage_comes_from_the_source_not_a_guess(self, registry):
        from masdar.nlu.parser import parse

        candidate = registry.adapter("gastat_cdata").search(parse("الكهرباء 2022"))[0]
        assert candidate.claimed_coverage.origin.is_authoritative
        assert candidate.claimed_coverage.years == frozenset(REAL_COVERAGE)

    def test_probe_is_a_year_only_query(self, registry, transport):
        from masdar.nlu.parser import parse

        registry.adapter("gastat_cdata").search(parse("الكهرباء 2022"))
        probes = [unquote(u) for u in transport.requests if "REGION" not in unquote(u)]
        assert probes, "expected a year-only coverage probe"
        # Cheap by construction: it asks for the year axis alone.
        assert "dimensions[]=YEAR" in probes[0]


class TestVerdictsAgainstRealCoverage:
    def test_available_year_is_confirmed(self, agent):
        answer = agent.answer("ابي الكهرباء لسنه 2022 حسب المناطق")
        assert answer.verdict is Verdict.AVAILABLE
        assert answer.primary.matched_years == (2022,)

    def test_year_after_the_last_published_is_refused(self, agent):
        answer = agent.answer("ابي الكهرباء لسنه 2026 حسب المناطق")
        assert answer.verdict is Verdict.NOT_AVAILABLE
        assert answer.primary.export_path is None
        assert [s.year for s in answer.suggestions] == [2022]

    def test_a_real_gap_in_the_middle_is_refused_with_both_neighbours(self, agent):
        # 2020 is genuinely missing between 2019 and 2021.
        answer = agent.answer("ابي الكهرباء لسنه 2020 حسب المناطق")
        assert answer.verdict is Verdict.NOT_AVAILABLE
        offered = {s.year for s in answer.suggestions}
        assert 2019 in offered and 2021 in offered

    def test_partial_range_reports_the_gap(self, agent):
        answer = agent.answer("الكهرباء من 2019 الى 2022 حسب المناطق")
        assert answer.verdict is Verdict.PARTIAL
        assert answer.primary.missing_years == (2020,)


class TestExport:
    def test_workbook_carries_arabic_region_labels(self, agent):
        answer = agent.answer("الكهرباء لسنه 2022 حسب المناطق")
        sheet = load_workbook(answer.primary.export_path)["البيانات"]
        text = "\n".join(
            str(c) for row in sheet.iter_rows(values_only=True) for c in row if c
        )
        assert "الرياض" in text and "جازان" in text

    def test_workbook_holds_only_the_requested_year(self, agent):
        answer = agent.answer("الكهرباء لسنه 2022 حسب المناطق")
        sheet = load_workbook(answer.primary.export_path)["البيانات"]
        header = [c.value for c in sheet[1]]
        year_column = header.index("YEAR_TIME")
        years = {row[year_column] for row in sheet.iter_rows(min_row=2, values_only=True)}
        assert years == {2022}

    def test_measures_are_numeric(self, agent):
        # The API suffixes every measure with _OBSV; the name itself varies
        # per dataset, so the convention is what to rely on.
        answer = agent.answer("الكهرباء لسنه 2022 حسب المناطق")
        sheet = load_workbook(answer.primary.export_path)["البيانات"]
        header = [str(c.value) for c in sheet[1]]
        measures = [i for i, name in enumerate(header) if name.endswith("_OBSV")]
        assert measures, f"no measure column in {header}"
        first = next(sheet.iter_rows(min_row=2, max_row=2, values_only=True))
        assert all(isinstance(first[i], (int, float)) for i in measures)

    def test_source_sheet_cites_the_api_url(self, agent):
        answer = agent.answer("الكهرباء لسنه 2022 حسب المناطق")
        sheet = load_workbook(answer.primary.export_path)["المصدر"]
        text = "\n".join(
            str(c) for row in sheet.iter_rows(values_only=True) for c in row if c
        )
        assert "api.stats.gov.sa" in text
        assert answer.primary.provenance.sha256 in text


class TestTruncatedDownloadsCannotNarrowCoverage:
    """A partial page must never turn a present year into a missing one."""

    def test_enumerated_coverage_survives_a_short_response(self, agent):
        # by_region.json holds only 2022 and 2017 rows, while the source
        # enumerated 2017-2019 and 2021-2022. The enumeration must win.
        answer = agent.answer("الكهرباء لسنه 2019 حسب المناطق")
        assert answer.primary.coverage.years == frozenset(REAL_COVERAGE)
        assert answer.verdict is Verdict.AVAILABLE

    def test_union_keeps_years_only_the_download_reveals(self):
        from masdar.domain.models import Coverage, CoverageOrigin
        from masdar.pipeline.orchestrator import _merge_coverage

        enumerated = Coverage(
            years=frozenset({2020, 2021}),
            origin=CoverageOrigin.OBSERVED_DATA,
            is_exhaustive=True,
        )
        observed = Coverage(
            years=frozenset({2021, 2022}), origin=CoverageOrigin.OBSERVED_DATA
        )
        merged = _merge_coverage(enumerated, observed)
        assert merged.years == frozenset({2020, 2021, 2022})

    def test_weak_declared_coverage_is_replaced_by_observation(self):
        from masdar.domain.models import Coverage, CoverageOrigin
        from masdar.pipeline.orchestrator import _merge_coverage

        # A title hint carries no authority, so observation simply wins.
        hinted = Coverage(years=frozenset({2030}), origin=CoverageOrigin.INFERRED_TITLE)
        observed = Coverage(
            years=frozenset({2021, 2022}), origin=CoverageOrigin.OBSERVED_DATA
        )
        assert _merge_coverage(hinted, observed).years == frozenset({2021, 2022})


class TestConsumptionDataset:
    """The dataset the original question was really about: kWh by region."""

    def test_coverage_matches_the_live_series(self, agent):
        answer = agent.answer("ابي استهلاك الطاقه الكهربائيه لسنه 2022 حسب المناطق")
        assert answer.primary.candidate.dataset_id == "DPV_HES_EHE_IT_HES0303"
        assert answer.primary.coverage.years == frozenset(REAL_COVERAGE)

    def test_exports_seasonal_regional_consumption(self, agent):
        answer = agent.answer(
            "ابي استهلاك الطاقه الكهربائيه لسنه 2022 حسب المناطق والفصل"
        )
        assert answer.verdict is Verdict.AVAILABLE
        sheet = load_workbook(answer.primary.export_path)["البيانات"]
        text = "\n".join(
            str(c) for row in sheet.iter_rows(values_only=True) for c in row if c
        )
        assert "خلال فصل الشتاء" in text
        assert "خلال باقي السنة" in text
        assert "الرياض" in text

    def test_measure_column_is_numeric(self, agent):
        answer = agent.answer("استهلاك الكهرباء 2022 حسب المناطق")
        sheet = load_workbook(answer.primary.export_path)["البيانات"]
        header = [c.value for c in sheet[1]]
        column = header.index("OBSVALUE_OBSV")
        first = next(sheet.iter_rows(min_row=2, max_row=2, values_only=True))
        assert isinstance(first[column], float)

    def test_2020_is_still_refused_for_this_dataset(self, agent):
        answer = agent.answer("استهلاك الكهرباء 2020 حسب المناطق")
        assert answer.verdict is Verdict.NOT_AVAILABLE
        assert "ناقصة: 2020" in answer.message_ar


class TestKeyedRoutes:
    """A route that needs a key must say so, not imply the data is missing."""

    def test_missing_key_is_reported_as_a_key_problem(self):
        from masdar.sources.transport import describe_refusal

        class Refused:
            status_code = 401
            content = (FIXTURES / "missing_key.json").read_bytes()

        reason = describe_refusal(Refused(), "https://api.stats.gov.sa/v1/stats/X")
        assert "مفتاح API" in reason
        assert "لا يعني عدم وجود البيانات" in reason

    def test_a_block_is_not_described_as_a_key_problem(self):
        from masdar.sources.transport import describe_refusal

        class Blocked:
            status_code = 403
            content = b"<html><body>Access denied by policy</body></html>"

        assert "مفتاح API" not in describe_refusal(Blocked(), "x")


class TestProvidedDimensions:
    """A breakdown the server grouped by must not be reported as missing."""

    def test_no_false_warning_for_an_api_named_column(self, agent):
        # The season column is CONSUMP_OPERATION_PERIOD_ARAB, which matches no
        # word in the lexicon; the adapter declaring it is what prevents a
        # warning that the file lacks data it plainly contains.
        answer = agent.answer("استهلاك الكهرباء 2022 حسب المناطق والفصل")
        assert not any("لا يحتوي عمود" in note for note in answer.primary.notes)

    def test_candidate_declares_what_it_grouped_by(self, registry):
        from masdar.domain.models import Dimension
        from masdar.nlu.parser import parse

        candidate = registry.adapter("gastat_cdata").search(
            parse("استهلاك الكهرباء 2022 حسب المناطق والفصل")
        )[0]
        assert set(candidate.provided_dimensions) == {Dimension.REGION, Dimension.SEASON}

    def test_a_genuinely_absent_breakdown_is_still_reported(self, agent):
        # This dataset has no gender axis, so the warning must appear.
        answer = agent.answer("استهلاك الكهرباء 2022 حسب الجنس")
        assert any("لا يحتوي عمود" in note for note in answer.primary.notes)


class TestKeyReachesTheWire:
    """The bug this guards: the key was read but never sent.

    `_auth_headers()` existed and fed the doctor's "with key" label, while no
    request ever carried the header -- so setting the key could not have
    unlocked a keyed route. Nothing asserted on the outgoing request, which
    is exactly what these tests do.
    """

    KEY = "test-key-0000"

    def test_coverage_probe_sends_the_key(self, registry, transport, monkeypatch):
        from masdar.nlu.parser import parse

        monkeypatch.setenv("GASTAT_API_KEY", self.KEY)
        registry.adapter("gastat_cdata").search(parse("الكهرباء 2022"))
        assert transport.headers, "no request was made"
        assert all(h.get("apikey") == self.KEY for h in transport.headers)

    def test_data_download_sends_the_key(self, agent, transport, monkeypatch):
        monkeypatch.setenv("GASTAT_API_KEY", self.KEY)
        agent.answer("استهلاك الكهرباء 2022 حسب المناطق")
        downloads = [
            h for u, h in zip(transport.requests, transport.headers, strict=True)
            if "REGION" in unquote(u)
        ]
        assert downloads, "the data itself was never downloaded"
        assert all(h.get("apikey") == self.KEY for h in downloads)

    def test_no_header_without_a_key(self, agent, transport, monkeypatch):
        monkeypatch.delenv("GASTAT_API_KEY", raising=False)
        agent.answer("استهلاك الكهرباء 2022 حسب المناطق")
        assert all("apikey" not in h for h in transport.headers)

    def test_key_never_appears_in_the_answer(self, agent, monkeypatch):
        import json

        from masdar.cli import _answer_to_dict

        monkeypatch.setenv("GASTAT_API_KEY", self.KEY)
        answer = agent.answer("استهلاك الكهرباء 2022 حسب المناطق")
        dumped = json.dumps(_answer_to_dict(answer), ensure_ascii=False)
        assert self.KEY not in dumped

    def test_key_never_appears_in_the_workbook(self, agent, monkeypatch):
        monkeypatch.setenv("GASTAT_API_KEY", self.KEY)
        answer = agent.answer("استهلاك الكهرباء 2022 حسب المناطق")
        workbook = load_workbook(answer.primary.export_path)
        for sheet in workbook.worksheets:
            for row in sheet.iter_rows(values_only=True):
                assert all(self.KEY not in str(c) for c in row if c is not None)
