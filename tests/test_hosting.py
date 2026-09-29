"""Running as a hosted demo: a public link instead of a server on the
presenter's machine.

A host abroad cannot see the geo-restricted portal and must say so once,
not time out on every question; a host that goes quiet must not stall the
presentation; and a server that wakes with an empty cache warms itself up.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from masdar.chat.overview import overview
from masdar.chat.warmup import Warmup
from masdar.cli import _env_port
from masdar.domain.models import Verdict
from masdar.nlu.lexicon import load_lexicon
from masdar.sources.base import SourceRejected, SourceUnreachable
from masdar.sources.http import HttpClient
from masdar.sources.registry import Registry, load_descriptors, outside_ksa
from masdar.sources.transport import Response, Transport

URL = "https://api.example.invalid/v1/stats/X"


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    for name in ("MASDAR_OUTSIDE_KSA", "MASDAR_HOST_COOLDOWN_SECONDS", "PORT",
                 "MASDAR_WARMUP_ON_START"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MASDAR_CACHE_DIR", str(tmp_path / "cache"))


class TestOutsideTheKingdom:
    def test_off_by_default(self):
        assert not outside_ksa()

    def test_the_geo_restricted_portal_is_not_asked(self, monkeypatch):
        registry = Registry(load_descriptors())
        assert "saudi_open_data" in {d.id for d in registry.for_topic("electricity")}
        monkeypatch.setenv("MASDAR_OUTSIDE_KSA", "1")
        planned = {d.id for d in registry.for_topic("electricity")}
        assert "saudi_open_data" not in planned
        assert {"gastat_cdata", "kapsarc", "worldbank"} <= planned

    def test_the_landing_page_says_so_and_counts_only_what_answers(self, monkeypatch):
        home = overview(Registry(load_descriptors()), load_lexicon())
        monkeypatch.setenv("MASDAR_OUTSIDE_KSA", "1")
        abroad = overview(Registry(load_descriptors()), load_lexicon())
        assert home["unavailable"] == []
        assert [u["id"] for u in abroad["unavailable"]] == ["saudi_open_data"]
        assert "saudi_open_data" not in {s["id"] for s in abroad["sources"]}
        assert abroad["stats"]["publishers"] < home["stats"]["publishers"]
        assert abroad["stats"]["datasets"] < home["stats"]["datasets"]


class Counting(Transport):
    name = "counting"

    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def get(self, url, source_id, params=None, headers=None):
        self.calls += 1
        if self.error:
            raise self.error(source_id, "انتهت مهلة الاتصال")
        return Response(url, url, 200, b"{}", "application/json", {})


class TestHostCooldown:
    def test_a_host_that_failed_is_not_asked_again_at_once(self):
        down = Counting(SourceUnreachable)
        client = HttpClient(transport=down, retries=1, use_cache=False)
        for _ in range(3):
            with pytest.raises(SourceUnreachable):
                client.get(URL)
        assert down.calls == 1

    def test_other_paths_on_the_host_are_skipped_too(self):
        down = Counting(SourceUnreachable)
        client = HttpClient(transport=down, retries=1, use_cache=False)
        with pytest.raises(SourceUnreachable):
            client.get(URL)
        with pytest.raises(SourceUnreachable, match="تعذّر قبل قليل"):
            client.get(URL.replace("/X", "/Y"))
        assert down.calls == 1

    def test_a_refusal_concerns_one_path_not_the_host(self):
        refused = Counting(SourceRejected)
        client = HttpClient(transport=refused, retries=1, use_cache=False)
        for _ in range(2):
            with pytest.raises(SourceRejected):
                client.get(URL)
        assert refused.calls == 2

    def test_clients_of_one_server_share_what_they_learnt(self):
        # The start-up warm-up and the presenter's questions use separate
        # clients; a host the warm-up found dead must not cost a question.
        shared: dict = {}
        down = Counting(SourceUnreachable)
        warmup = HttpClient(transport=down, retries=1, use_cache=False, down_hosts=shared)
        presenter = HttpClient(transport=down, retries=1, use_cache=False, down_hosts=shared)
        with pytest.raises(SourceUnreachable):
            warmup.get(URL)
        with pytest.raises(SourceUnreachable):
            presenter.get(URL)
        assert down.calls == 1

    def _expire(self, client):
        from datetime import UTC, datetime, timedelta

        host, (_, reason, failures) = next(iter(client._down.items()))
        client._down[host] = (datetime.now(UTC) - timedelta(seconds=1), reason, failures)

    def test_a_host_that_keeps_failing_waits_longer_each_time(self):
        from datetime import UTC, datetime, timedelta

        down = Counting(SourceUnreachable)
        client = HttpClient(transport=down, retries=1, use_cache=False)
        waits = []
        for _ in range(3):
            with pytest.raises(SourceUnreachable):
                client.get(URL)
            until = next(iter(client._down.values()))[0]
            waits.append(until - datetime.now(UTC))
            self._expire(client)
        assert down.calls == 3
        assert waits[0] < waits[1] < waits[2] <= timedelta(minutes=30)

    def test_a_host_that_answers_again_is_forgiven(self):
        flaky = Counting(SourceUnreachable)
        client = HttpClient(transport=flaky, retries=1, use_cache=False)
        with pytest.raises(SourceUnreachable):
            client.get(URL)
        self._expire(client)
        flaky.error = None  # back up
        client.get(URL)
        assert client._down == {}

    def test_zero_turns_it_off(self, monkeypatch):
        monkeypatch.setenv("MASDAR_HOST_COOLDOWN_SECONDS", "0")
        down = Counting(SourceUnreachable)
        client = HttpClient(transport=down, retries=1, use_cache=False)
        for _ in range(2):
            with pytest.raises(SourceUnreachable):
                client.get(URL)
        assert down.calls == 2


class TestWarmup:
    def agent(self, verdicts):
        def answer(question):
            outcome = verdicts[question]
            if isinstance(outcome, Exception):
                raise outcome
            return SimpleNamespace(verdict=outcome)
        return SimpleNamespace(answer=answer)

    def test_progress_and_what_no_source_answered(self):
        questions = ["الكهرباء 2026", "عدد الإبل", "عدد الحجاج 2023", "سؤال يكسر"]
        verdicts = {
            questions[0]: Verdict.NOT_AVAILABLE,  # an honest "not published" is fine
            questions[1]: Verdict.AVAILABLE,
            questions[2]: Verdict.SOURCE_UNREACHABLE,
            questions[3]: RuntimeError("bug"),
        }
        warm = Warmup(questions, retry_rounds=0)
        assert warm.status() == {"total": 4, "done": 0, "running": False, "failed": []}
        warm.start(lambda: self.agent(verdicts)).join(timeout=10)
        status = warm.status()
        assert status["done"] == 4 and not status["running"]
        assert status["failed"] == [questions[2], questions[3]]

    def test_a_source_back_later_brings_its_question_back(self):
        questions = ["عدد الحجاج 2023", "معدل البطالة 2023"]
        replies = {questions[0]: [Verdict.SOURCE_UNREACHABLE, Verdict.AVAILABLE],
                   questions[1]: [Verdict.SOURCE_UNREACHABLE] * 3}
        asked = []

        def answer(question):
            asked.append(question)
            return SimpleNamespace(verdict=replies[question].pop(0))

        warm = Warmup(questions, retry_rounds=2, retry_delay=0)
        warm.start(lambda: SimpleNamespace(answer=answer)).join(timeout=10)
        assert warm.status()["failed"] == [questions[1]]
        # Asked once, then only while it still failed.
        assert asked.count(questions[0]) == 2 and asked.count(questions[1]) == 3


class TestPort:
    def test_the_platform_port_is_used(self, monkeypatch):
        assert _env_port() == 8000
        monkeypatch.setenv("PORT", "10000")
        assert _env_port() == 10000
        monkeypatch.setenv("PORT", "nonsense")
        assert _env_port() == 8000
