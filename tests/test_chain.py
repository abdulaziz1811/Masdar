"""Several models behind one reader (nlu/chain.py).

Fake readers and a fake clock: which model is asked, which is rested after
a passing failure and for how long, and that the whole attempt keeps to its
time budget. No key and no network are needed.
"""

from __future__ import annotations

import pytest

from masdar.domain.models import DataRequest
from masdar.nlu.chain import MIN_CALL_SECONDS, REST_SECONDS, ModelChain
from masdar.nlu.lexicon import load_lexicon
from masdar.nlu.llm import TIMEOUT_SECONDS, LlmReading, LlmUnavailable
from masdar.nlu.understand import LLM, RULES, understand

UNPLACED = "ودي اعرف كم كيلوات صرفته البيوت بكل منطقة"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Reader:
    """Answers, or fails, as told; records each call's time limit."""

    def __init__(self, name: str, fails: list | None = None, takes: float = 0.0,
                 clock: Clock | None = None, works: bool = True):
        self.NAME = name
        self.model = f"{name.lower()}-model"
        self.label = f"{name} · {self.model}"
        self.fails = list(fails or [])
        self.takes = takes
        self.clock = clock
        self.works = works
        self.timeouts: list[float] = []

    def read(self, message, previous=None, timeout=None):
        self.timeouts.append(timeout)
        if self.clock is not None:
            self.clock.now += self.takes
        if self.fails:
            failure = self.fails.pop(0)
            if failure is not None:
                raise failure
        return LlmReading(DataRequest(raw_query=message, free_terms=("الكهرباء",)), False,
                          model=self.label)

    def verify(self):
        return (True, "") if self.works else (False, "مفتاح غير صالح")


def busy(after: float | None = None) -> LlmUnavailable:
    return LlmUnavailable("مزدحم", transient=True, retry_after=after)


@pytest.fixture
def clock():
    return Clock()


class TestTheOrder:
    def test_the_first_valid_reading_is_used(self, clock):
        first, second = Reader("A"), Reader("B")
        reading = ModelChain([first, second], clock=clock).read("سؤال")
        assert reading.model == first.label
        assert second.timeouts == []

    def test_a_failure_moves_on_to_the_next(self, clock):
        first = Reader("A", fails=[LlmUnavailable("رد غير صالح")])
        second = Reader("B")
        assert ModelChain([first, second], clock=clock).read("سؤال").model == second.label

    def test_a_bug_in_one_reader_does_not_stop_the_others(self, clock):
        first, second = Reader("A", fails=[KeyError("x")]), Reader("B")
        assert ModelChain([first, second], clock=clock).read("سؤال").model == second.label

    def test_when_all_fail_each_reason_is_named(self, clock):
        chain = ModelChain([Reader("A", fails=[busy()]),
                            Reader("B", fails=[LlmUnavailable("رفض النموذج الطلب")])],
                           clock=clock)
        with pytest.raises(LlmUnavailable) as caught:
            chain.read("سؤال")
        assert "A · a-model: مزدحم" in caught.value.reason
        assert "B · b-model: رفض" in caught.value.reason

    def test_one_reader_needs_no_name_in_its_reason(self, clock):
        with pytest.raises(LlmUnavailable) as caught:
            ModelChain([Reader("A", fails=[busy()])], clock=clock).read("سؤال")
        assert caught.value.reason == "مزدحم"


class TestResting:
    def test_a_passing_failure_rests_that_model(self, clock):
        first, second = Reader("A", fails=[busy(), None]), Reader("B")
        chain = ModelChain([first, second], clock=clock)
        chain.read("الأول")
        chain.read("الثاني")
        assert len(first.timeouts) == 1  # not asked again while resting
        clock.now += REST_SECONDS
        assert chain.read("الثالث").model == first.label

    def test_the_providers_own_wait_is_kept(self, clock):
        first, second = Reader("A", fails=[busy(after=300), None]), Reader("B")
        chain = ModelChain([first, second], clock=clock)
        chain.read("الأول")
        clock.now += REST_SECONDS
        assert chain.read("الثاني").model == second.label
        clock.now += 300
        assert chain.read("الثالث").model == first.label

    def test_any_other_failure_does_not(self, clock):
        first = Reader("A", fails=[LlmUnavailable("رد غير صالح"), None])
        chain = ModelChain([first, Reader("B")], clock=clock)
        chain.read("الأول")
        assert chain.read("الثاني").model == first.label

    def test_all_resting_is_said(self, clock):
        chain = ModelChain([Reader("A", fails=[busy()])], clock=clock)
        with pytest.raises(LlmUnavailable):
            chain.read("الأول")
        with pytest.raises(LlmUnavailable) as caught:
            chain.read("الثاني")
        assert "استراحة" in caught.value.reason


class TestTheBudget:
    def test_each_call_gets_at_most_the_time_left(self, clock):
        first = Reader("A", fails=[busy()], takes=7.0, clock=clock)
        second = Reader("B")
        ModelChain([first, second], budget=12.0, clock=clock).read("سؤال")
        assert first.timeouts == [TIMEOUT_SECONDS]
        assert second.timeouts == [pytest.approx(5.0)]

    def test_no_call_is_started_without_time_for_it(self, clock):
        first = Reader("A", fails=[busy()], takes=12.0 - MIN_CALL_SECONDS / 2, clock=clock)
        second = Reader("B")
        with pytest.raises(LlmUnavailable) as caught:
            ModelChain([first, second], budget=12.0, clock=clock).read("سؤال")
        assert second.timeouts == [] and "المهلة" in caught.value.reason

    def test_a_shorter_limit_can_be_asked_for(self, clock):
        first = Reader("A")
        ModelChain([first], budget=12.0, clock=clock).read("سؤال", timeout=4.0)
        assert first.timeouts == [4.0]


class TestTheKeyCheck:
    def test_readers_whose_key_fails_are_dropped(self, clock):
        good, bad = Reader("A"), Reader("B", works=False)
        chain = ModelChain([bad, good], clock=clock)
        ok, why = chain.verify()
        assert ok and chain.readers == [good]
        assert "B · b-model: مفتاح غير صالح" in why

    def test_none_working_is_not_ok(self, clock):
        chain = ModelChain([Reader("A", works=False)], clock=clock)
        assert chain.verify()[0] is False


class TestThroughTheUnderstandingStep:
    def test_the_rules_answer_and_say_why_when_every_model_fails(self, clock):
        lexicon = load_lexicon()
        chain = ModelChain([Reader("A", fails=[busy()]), Reader("B", fails=[busy()])],
                           clock=clock)
        result = understand(UNPLACED, lexicon, llm=chain)
        assert result.method == RULES and result.model_asked
        assert "A · a-model" in result.note and "B · b-model" in result.note

    def test_a_reading_from_the_second_model(self, clock):
        lexicon = load_lexicon()
        chain = ModelChain([Reader("A", fails=[busy()]), Reader("B")], clock=clock)
        result = understand(UNPLACED, lexicon, llm=chain)
        assert result.method == LLM and result.model == "B · b-model"
