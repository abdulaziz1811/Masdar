"""Tests for reading GASTAT OpenAPI specifications.

`energy_electrical.json` is reconstructed from the two portal pages for the
Electrical Energy Statistics API: its summaries, descriptions, dimension
enums, filter parameters and response fields are verbatim, laid out in the
standard OpenAPI 3.0 structure. `synthetic_refs.json` is invented to
exercise what the real one does not: `$ref` indirection, a quarterly time
axis, a server on a foreign host and an embedded key.
"""

import json
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from masdar.domain.models import Verdict
from masdar.sources.openapi import (
    datasets_from_dir,
    datasets_from_spec,
    find_secrets,
    load_spec,
    split_bilingual,
)
from tests.fakes import paged

SPECS = Path(__file__).resolve().parent / "fixtures" / "specs"
BASE = "https://api.stats.gov.sa"


def energy():
    return datasets_from_spec(load_spec(SPECS / "energy_electrical.json"), BASE)


def synthetic():
    return datasets_from_spec(load_spec(SPECS / "synthetic_refs.json"), BASE)


class TestBilingualTitles:
    def test_splits_by_script(self):
        assert split_bilingual("إحصاءات الطاقة / Energy Statistics") == (
            "إحصاءات الطاقة",
            "Energy Statistics",
        )

    def test_order_does_not_matter(self):
        assert split_bilingual("Energy Statistics / إحصاءات الطاقة")[0] == "إحصاءات الطاقة"

    def test_single_language(self):
        assert split_bilingual("Energy Statistics") == ("", "Energy Statistics")


class TestRealSpecShape:
    def test_one_dataset_per_stats_path(self):
        assert [d["id"] for d in energy()] == [
            "DPV_HES_EHE_IT_HES0301",
            "DPV_HES_EHE_IT_HES0303",
        ]

    def test_dimensions_come_from_the_enum(self):
        consumption = energy()[1]
        assert consumption["dimensions"] == ["CONSUMP_OPERATION_PERIOD", "REGION", "YEAR"]

    def test_year_is_the_time_axis(self):
        assert all(d["time_dimension"] == "YEAR" for d in energy())

    def test_measures_come_from_the_response_schema(self):
        connections, consumption = energy()
        assert connections["measures"] == ["HH_CONNECTED_TO_ELECTRICITY_OBSV", "TOTAL_HH_OBSV"]
        assert consumption["measures"] == ["OBSVALUE_OBSV"]

    def test_titles_are_split(self):
        consumption = energy()[1]
        assert consumption["title_ar"].startswith("استهلاك الطاقة الكهربائية")
        assert consumption["title_en"].startswith("Housing units electrical energy")

    def test_topics_are_inferred_from_the_titles(self):
        assert "electricity" in energy()[1]["topics"]

    def test_same_host_server_is_kept(self):
        assert energy()[0]["server"] == BASE

    def test_security_scheme_naming_the_header_is_not_a_secret(self):
        # It says where a key goes -- "apikey" -- not what the key is.
        assert find_secrets(load_spec(SPECS / "energy_electrical.json")) == []


class TestStructuralRobustness:
    def test_resolves_refs_through_several_levels(self):
        (dataset,) = synthetic()
        assert dataset["dimensions"] == ["REGION", "QUARTER"]
        assert dataset["measures"] == ["PRICE_OBSV"]

    def test_a_non_annual_time_axis_is_found(self):
        (dataset,) = synthetic()
        assert dataset["time_dimension"] == "QUARTER"

    def test_templated_paths_are_not_datasets(self):
        assert all("{" not in d["id"] for d in synthetic())

    def test_dimensions_fall_back_to_filter_parameters(self):
        spec = {
            "openapi": "3.0.1",
            "paths": {"/v1/stats/X1": {"get": {
                "summary": "a / a",
                "parameters": [
                    {"name": "REGION_CODE", "in": "query"},
                    {"name": "YEAR_TIME", "in": "query"},
                ],
            }}},
        }
        (dataset,) = datasets_from_spec(spec, BASE)
        assert set(dataset["dimensions"]) == {"REGION", "YEAR"}
        assert dataset["time_dimension"] == "YEAR"


class TestSecurity:
    def test_foreign_server_is_ignored(self):
        # The spec names evil.example.com; requests carry the API key to
        # wherever `server` points, so it must not be honoured.
        (dataset,) = synthetic()
        assert "server" not in dataset

    def test_embedded_key_is_detected(self):
        found = find_secrets(load_spec(SPECS / "synthetic_refs.json"))
        assert found == ["/components/securitySchemes/k/x-example-apikey"]

    def test_prose_is_not_mistaken_for_a_secret(self):
        assert find_secrets({"token_description": "the token goes in the apikey header"}) == []


class TestLoading:
    def test_rejects_a_non_openapi_file(self, tmp_path):
        from masdar.sources.openapi import SpecError

        bogus = tmp_path / "x.json"
        bogus.write_text('{"hello": "world"}', encoding="utf-8")
        with pytest.raises(SpecError):
            load_spec(bogus)

    def test_reads_yaml_too(self, tmp_path):
        import yaml

        spec = json.loads((SPECS / "energy_electrical.json").read_text(encoding="utf-8"))
        target = tmp_path / "energy.yaml"
        target.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
        assert len(datasets_from_spec(load_spec(target), BASE)) == 2

    def test_a_bad_file_is_reported_not_fatal(self, tmp_path):
        (tmp_path / "good.json").write_text(
            (SPECS / "energy_electrical.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        datasets, problems = datasets_from_dir(tmp_path, BASE)
        assert len(datasets) == 2
        assert len(problems) == 1 and "broken.json" in problems[0]

    def test_non_spec_files_in_the_directory_are_ignored(self, tmp_path):
        (tmp_path / "README.md").write_text("# notes", encoding="utf-8")
        assert datasets_from_dir(tmp_path, BASE) == ([], [])


# -- through the adapter and the agent ------------------------------------------
SPEC_ONLY = {
    "openapi": "3.0.1",
    "info": {"title": "إحصاءات الطاقة / Energy Statistics", "version": "1.0.0"},
    "servers": [{"url": "https://evil.example.com"}],
    "paths": {"/v1/stats/TEST_ELEC_PRODUCTION": {"get": {
        "summary": "إنتاج الكهرباء حسب المنطقة الإدارية / Electricity production by region",
        "parameters": [{"name": "dimensions[]", "in": "query", "schema": {
            "type": "array", "items": {"type": "string", "enum": ["REGION", "YEAR"]}}}],
    }}},
}


class SpecOnlyTransport:
    """Serves coverage and data for the spec-only dataset, recording hosts."""

    name = "recorded"

    def __init__(self):
        self.urls: list[str] = []
        self.headers: list[dict] = []

    def get(self, url, source_id, params=None, headers=None):
        from masdar.sources.transport import Response

        self.urls.append(url)
        self.headers.append(dict(headers or {}))
        decoded = unquote(url)
        if "dimensions[]=REGION" in decoded:
            rows = [
                {"REGION_ARAB": "الرياض", "YEAR_TIME": "2022", "OBSVALUE_OBSV": 10.5},
                {"REGION_ARAB": "جازان", "YEAR_TIME": "2022", "OBSVALUE_OBSV": 2.5},
            ]
        else:
            rows = [{"YEAR_TIME": "2021"}, {"YEAR_TIME": "2022"}]
        body = json.dumps({"value": rows}, ensure_ascii=False).encode("utf-8")
        return Response(url, url, 200, paged(body, url), "application/json", {})

    def describe(self):
        return self.name


@pytest.fixture
def spec_agent(tmp_path):
    from dataclasses import replace

    from masdar.pipeline.orchestrator import Agent, AgentConfig
    from masdar.sources.http import HttpClient
    from masdar.sources.registry import Registry, load_descriptors

    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    (spec_dir / "energy.json").write_text(json.dumps(SPEC_ONLY), encoding="utf-8")

    base = next(d for d in load_descriptors() if d.id == "gastat_cdata")
    # No hand-declared datasets: everything must come from the spec.
    descriptor = replace(base, api={**base.api, "spec_dir": str(spec_dir), "datasets": []})

    transport = SpecOnlyTransport()
    http = HttpClient(use_cache=False, retries=1, transport=transport)
    agent = Agent(
        registry=Registry((descriptor,), http),
        http=http,
        config=AgentConfig(out_dir=tmp_path / "out", today=date(2026, 9, 28)),
    )
    return agent, transport


class TestSpecDrivenSearch:
    def test_a_dataset_known_only_from_a_spec_is_answered(self, spec_agent):
        agent, _ = spec_agent
        answer = agent.answer("ابي انتاج الكهرباء لسنه 2022 حسب المناطق")
        assert answer.verdict is Verdict.AVAILABLE
        assert answer.primary.candidate.dataset_id == "TEST_ELEC_PRODUCTION"
        assert answer.primary.export_path

    def test_requests_stay_on_the_configured_host(self, spec_agent):
        agent, transport = spec_agent
        agent.answer("ابي انتاج الكهرباء لسنه 2022 حسب المناطق")
        assert transport.urls
        assert {urlparse(u).netloc for u in transport.urls} == {"api.stats.gov.sa"}

    def test_the_key_never_reaches_a_host_named_by_a_spec(self, spec_agent, monkeypatch):
        agent, transport = spec_agent
        monkeypatch.setenv("GASTAT_API_KEY", "k-0000")
        agent.answer("ابي انتاج الكهرباء لسنه 2022 حسب المناطق")
        for url, headers in zip(transport.urls, transport.headers, strict=True):
            if "apikey" in headers:
                assert urlparse(url).netloc == "api.stats.gov.sa"


class TestImportCommand:
    def _run(self, *argv):
        from masdar.cli import main

        return main(["import-spec", *argv])

    def test_imports_a_valid_spec(self, tmp_path, capsys):
        assert self._run(str(SPECS / "energy_electrical.json"), "--dest", str(tmp_path)) == 0
        (written,) = tmp_path.glob("*.json")
        assert len(datasets_from_spec(load_spec(written), BASE)) == 2

    def test_refuses_a_spec_with_an_embedded_key(self, tmp_path, capsys):
        assert self._run(str(SPECS / "synthetic_refs.json"), "--dest", str(tmp_path)) == 1
        assert not list(tmp_path.glob("*.json"))
        assert "x-example-apikey" in capsys.readouterr().out

    def test_reimport_is_idempotent(self, tmp_path, capsys):
        self._run(str(SPECS / "energy_electrical.json"), "--dest", str(tmp_path))
        self._run(str(SPECS / "energy_electrical.json"), "--dest", str(tmp_path))
        assert len(list(tmp_path.glob("*.json"))) == 1
