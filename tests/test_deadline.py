"""An audience waits on every answer: a slow source must not hold it up.

With a time limit (the chat server sets one), sources are asked in
parallel; one still silent when its share of the time is up is reported as
not answering, rests for a while so the next question does not wait on it
again, and keeps working in the background so its reply is cached.
"""

from __future__ import annotations

import time
from datetime import date

from masdar.domain.models import Verdict
from masdar.nlu.understand import understand
from masdar.pipeline import orchestrator
from masdar.pipeline.orchestrator import Agent, AgentConfig
from masdar.sources.base import SourceUnreachable
from masdar.sources.http import HttpClient
from masdar.sources.registry import DEMO_SOURCES_FILE, Registry, load_descriptors
from masdar.sources.transport import Transport
from tests.fakes import cdata_descriptor

QUESTION = "ابي احصاءات الطاقه الكهربائيه لسنه 2026 حسب المناطق"
SLOW_SECONDS = 3.0


class Slow(Transport):
    """A host that accepts the connection and then takes its time."""

    name = "slow"

    def __init__(self):
        self.calls = 0

    def get(self, url, source_id, params=None, headers=None):
        self.calls += 1
        time.sleep(SLOW_SECONDS)
        raise SourceUnreachable(source_id, "انتهت مهلة القراءة")


def agent_with_slow_cdata(tmp_path, answer_seconds=1.0, slow=None):
    transport = Slow()
    http = HttpClient(use_cache=False, retries=1, transport=transport)
    descriptors = (cdata_descriptor(), *load_descriptors(DEMO_SOURCES_FILE))
    agent = Agent(
        registry=Registry(descriptors, http),
        http=http,
        config=AgentConfig(out_dir=tmp_path / "out", today=date(2026, 9, 27),
                           answer_seconds=answer_seconds),
        slow=slow,
    )
    return agent, transport


def test_the_answer_does_not_wait_for_a_slow_source(tmp_path):
    agent, _ = agent_with_slow_cdata(tmp_path)
    started = time.monotonic()
    answer = agent.answer(QUESTION)
    assert time.monotonic() - started < SLOW_SECONDS - 0.5
    # The sources that answered still decide the verdict.
    assert answer.verdict is Verdict.NOT_AVAILABLE
    assert [s.year for s in answer.suggestions] == [2024]


def test_the_slow_source_is_reported_as_not_answering(tmp_path):
    agent, _ = agent_with_slow_cdata(tmp_path)
    answer = agent.answer(QUESTION)
    errors = dict(answer.source_errors)
    assert "لم يُجب خلال" in errors["gastat_cdata"]
    consulted = {c.source_id: c for c in answer.consulted}
    assert consulted["gastat_cdata"].found is None


def test_the_next_question_does_not_wait_on_it_again(tmp_path):
    agent, transport = agent_with_slow_cdata(tmp_path)
    agent.answer(QUESTION)
    calls = transport.calls
    started = time.monotonic()
    answer = agent.answer(QUESTION)
    assert time.monotonic() - started < 1.0
    assert "سؤال سابق" in dict(answer.source_errors)["gastat_cdata"]
    assert transport.calls == calls


def test_what_one_agent_learns_spares_the_other(tmp_path):
    # The warm-up and the page are separate agents sharing this record.
    shared: dict = {}
    first, _ = agent_with_slow_cdata(tmp_path, slow=shared)
    second, transport = agent_with_slow_cdata(tmp_path, slow=shared)
    first.answer(QUESTION)
    second.answer(QUESTION)
    assert transport.calls == 0


def test_a_late_reply_brings_the_source_back(tmp_path):
    agent, _ = agent_with_slow_cdata(tmp_path)
    demo = next(d for d in agent.registry.descriptors if d.synthetic)
    agent._slow[demo.id] = time.monotonic() + 60
    found, error, _ = agent._search_one(demo, understand(QUESTION, agent.lexicon).request)
    assert error is None and found is not None
    assert demo.id not in agent._slow


def test_a_file_that_does_not_arrive_in_time(tmp_path, monkeypatch):
    agent, _ = agent_with_slow_cdata(tmp_path, answer_seconds=0.5)
    monkeypatch.setattr(orchestrator, "MIN_FETCH_SECONDS", 0.3)

    def slow_observe(candidate, question=""):
        time.sleep(SLOW_SECONDS)
        return None, None, None, None

    monkeypatch.setattr(agent, "_observe", slow_observe)
    started = time.monotonic()
    answer = agent.answer(QUESTION)
    assert time.monotonic() - started < SLOW_SECONDS - 0.5
    assert any("لم يصل الملف" in " ".join(f.notes) for f in answer.findings)




def slow_cdata_alone(tmp_path):
    """The slow source with no other to answer: nothing on the subject in time."""
    agent, transport = agent_with_slow_cdata(tmp_path)
    agent.registry = Registry((cdata_descriptor(),), agent.http)
    return agent, transport


def test_with_nothing_on_the_subject_the_slow_source_is_waited_for(tmp_path, monkeypatch):
    # «معسكرات سدايا» came back «لم يُعثر» at fifteen seconds while the
    # national platform, the one source with the answer, was still replying.
    monkeypatch.setattr(orchestrator, "PATIENCE_SECONDS", SLOW_SECONDS + 2)
    agent, _ = slow_cdata_alone(tmp_path)
    started = time.monotonic()
    answer = agent.answer(QUESTION)
    assert time.monotonic() - started >= SLOW_SECONDS - 0.5
    # It was heard out: its own reply, not "did not answer in time".
    assert "لم يُجب خلال" not in dict(answer.source_errors)["gastat_cdata"]
    assert "gastat_cdata" not in agent._slow


def test_patience_has_a_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "PATIENCE_SECONDS", 0.5)
    agent, _ = slow_cdata_alone(tmp_path)
    started = time.monotonic()
    answer = agent.answer(QUESTION)
    assert time.monotonic() - started < SLOW_SECONDS - 0.5
    assert "لم يُجب خلال 2 ثانية" in dict(answer.source_errors)["gastat_cdata"]
