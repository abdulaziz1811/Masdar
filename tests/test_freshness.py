"""A new year published at the source must reach the next answer.

Sources are asked live so that publication is seen without anyone editing
the configuration. The one thing that could defeat that is a cached copy
standing in for the source indefinitely.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from masdar.sources.http import DEFAULT_CACHE_MAX_AGE, HttpClient, cache_max_age
from masdar.sources.transport import Response, Transport

URL = "https://api.example.invalid/v1/stats/X?dimensions[]=YEAR"


class Publishing(Transport):
    """A source whose answer changes when a new year is published."""

    name = "publishing"

    def __init__(self):
        self.years = [2021, 2022]
        self.calls = 0

    def get(self, url, source_id, params=None, headers=None):
        self.calls += 1
        body = json.dumps({"value": [{"YEAR_TIME": str(y)} for y in self.years]})
        return Response(
            url=url, final_url=url, status=200, content=body.encode(),
            media_type="application/json", headers={},
        )


def years(fetched) -> list[int]:
    return [int(row["YEAR_TIME"]) for row in fetched.json()["value"]]


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MASDAR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MASDAR_CACHE_MAX_AGE_HOURS", raising=False)


def age_cache(client: HttpClient, url: str, by: timedelta) -> None:
    meta_path = client._cache_path(url).with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["retrieved_at"] = (datetime.now(UTC) - by).isoformat()
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


class TestCacheExpiry:
    def test_a_recent_copy_is_reused(self):
        source = Publishing()
        client = HttpClient(transport=source, retries=1)
        client.get(URL)
        again = client.get(URL)
        assert source.calls == 1 and again.from_cache

    def test_a_year_published_after_the_copy_expired_is_seen(self):
        source = Publishing()
        client = HttpClient(transport=source, retries=1)
        assert years(client.get(URL)) == [2021, 2022]

        source.years.append(2023)  # the publisher adds a year
        age_cache(client, URL, DEFAULT_CACHE_MAX_AGE + timedelta(minutes=1))

        fresh = client.get(URL)
        assert years(fresh) == [2021, 2022, 2023]
        assert not fresh.from_cache and source.calls == 2

    def test_zero_max_age_always_asks_the_source(self):
        source = Publishing()
        client = HttpClient(transport=source, retries=1, max_age=timedelta(0))
        client.get(URL)
        client.get(URL)
        assert source.calls == 2

    def test_offline_runs_still_use_an_old_copy_with_its_own_date(self):
        source = Publishing()
        HttpClient(transport=source, retries=1).get(URL)
        offline = HttpClient(transport=source, retries=1, offline=True)
        age_cache(offline, URL, timedelta(days=30))
        copy = offline.get(URL)
        assert copy.from_cache
        assert datetime.now(UTC) - copy.retrieved_at > timedelta(days=29)


class TestConfiguredMaxAge:
    def test_default(self):
        assert cache_max_age() == DEFAULT_CACHE_MAX_AGE

    def test_environment_override(self, monkeypatch):
        monkeypatch.setenv("MASDAR_CACHE_MAX_AGE_HOURS", "0.5")
        assert cache_max_age() == timedelta(minutes=30)

    def test_nonsense_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("MASDAR_CACHE_MAX_AGE_HOURS", "soon")
        assert cache_max_age() == DEFAULT_CACHE_MAX_AGE
