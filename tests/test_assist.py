"""When a model is asked, and what of its reading is kept (nlu/understand.py).

The rules read a question first. A model is asked when they could not
place it, or -- in the default `assist` role -- when they placed it but
left words unread; its reading then only adds to theirs. A fake reader
stands in for the model.
"""

from __future__ import annotations

import pytest

from masdar.domain.models import (
    DataRequest,
    DatasetCandidate,
    Dimension,
    Period,
    PeriodKind,
    Verdict,
)
from masdar.nlu.lexicon import load_lexicon
from masdar.nlu.llm import LlmReading, LlmUnavailable
from masdar.nlu.parser import parse
from masdar.nlu.understand import (
    ALWAYS,
    ASSIST_SECONDS,
    ASSISTED,
    FALLBACK,
    LLM,
    RULES,
    model_role,
    rephrasings,
    understand,
    unread_words,
)
from masdar.pipeline import resolve

UNPLACED = "ودي اعرف كم كيلوات صرفته البيوت بكل منطقة"
# The rules place it under population («الأسر») and leave «تملك» unread.
OWNERSHIP = "نسبة تملك المساكن للأسر السعودية 2022"


class Model:
    """Reads every message the same way; records how it was asked."""

    label = "Fake · fake-1"

    def __init__(self, topic: str | None = "housing", phrases=("ملكية المساكن",), years=(),
                 dimensions=(), alternatives=("نسبة الأسر المالكة لمساكنها",), fails=None):
        self.topic, self.phrases, self.years = topic, tuple(phrases), tuple(years)
        self.dimensions, self.alternatives, self.fails = tuple(dimensions), alternatives, fails
        self.calls: list[tuple[str, float | None]] = []

    def read(self, message, previous=None, timeout=None):
        self.calls.append((message, timeout))
        if self.fails is not None:
            raise self.fails
        lexicon = load_lexicon()
        request = DataRequest(
            raw_query=message,
            topic=lexicon.get(self.topic) if self.topic else None,
            period=Period(years=self.years, kind=(PeriodKind.SINGLE if self.years
                                                  else PeriodKind.UNSPECIFIED)),
            dimensions=self.dimensions,
            free_terms=self.phrases,
        )
        return LlmReading(request, False, "إعادة صياغة", tuple(self.alternatives), self.label)


@pytest.fixture
def lexicon():
    return load_lexicon()


class TestWhenTheModelIsAsked:
    def test_a_question_the_rules_read_completely_stays_on_the_machine(self, lexicon):
        model = Model()
        for question in ("عدد السياح 2024", "معدل البطالة 2023",
                         "احصاءات الطاقة الكهربائية 2024 حسب المناطق"):
            assert understand(question, lexicon, llm=model).method == RULES
        assert model.calls == []

    def test_words_the_rules_left_unread_ask_it(self, lexicon):
        model = Model()
        result = understand(OWNERSHIP, lexicon, llm=model)
        assert result.method == ASSISTED and len(model.calls) == 1
        # Adding to a reading that is already enough, it is given less time.
        assert model.calls[0][1] == ASSIST_SECONDS

    def test_a_measure_word_is_not_an_unread_word(self, lexicon):
        assert unread_words(parse("متوسط دخل الأسرة الشهري", lexicon)) == ()

    def test_under_fallback_only_an_unplaced_question_asks_it(self, lexicon):
        model = Model()
        assert understand(OWNERSHIP, lexicon, llm=model, role=FALLBACK).method == RULES
        assert understand(UNPLACED, lexicon, llm=model, role=FALLBACK).method == LLM

    def test_under_always_every_question_asks_it(self, lexicon):
        model = Model(topic="tourism", phrases=("السياح", "السياحة الوافدة"))
        assert understand("عدد السياح 2024", lexicon, llm=model, role=ALWAYS).method == ASSISTED

    @pytest.mark.parametrize("value, role", [
        ("", "assist"), ("fallback", "fallback"), ("ALWAYS", "always"), ("sometimes", "assist"),
    ])
    def test_the_role_comes_from_the_environment(self, monkeypatch, value, role):
        monkeypatch.setenv("MASDAR_LLM_ROLE", value)
        assert model_role() == role


class TestWhatIsKept:
    def test_the_rules_topic_phrases_and_years_stand(self, lexicon):
        rules = parse(OWNERSHIP, lexicon)
        result = understand(OWNERSHIP, lexicon, llm=Model(topic="housing", years=(2019,)))
        request = result.request
        assert request.topic == rules.topic
        assert request.typed_phrases == rules.typed_phrases
        assert request.free_terms == rules.free_terms
        assert request.period.years == (2022,)

    def test_its_wordings_widen_the_search(self, lexicon):
        result = understand(OWNERSHIP, lexicon, llm=Model(phrases=("ملكية المساكن", "تملك")))
        # What the asker already typed is not suggested back.
        assert result.request.expansions == ("ملكية المساكن",)

    def test_it_adds_a_breakdown_and_a_period_the_rules_missed(self, lexicon):
        question = "نسبة تملك المساكن للأسر السعودية"
        model = Model(years=(2022,), dimensions=(Dimension.REGION,))
        request = understand(question, lexicon, llm=model).request
        assert request.period.years == (2022,)
        assert Dimension.REGION in request.dimensions

    def test_standing_in_for_the_rules_its_phrases_are_the_search(self, lexicon):
        model = Model(topic="electricity", phrases=("استهلاك الكهرباء",))
        request = understand(UNPLACED, lexicon, llm=model).request
        assert request.free_terms == ("استهلاك الكهرباء",)
        assert request.expansions == ("استهلاك الكهرباء",)
        # Counted once in a score, as the asker's subject.
        assert request.expansion_hints() == ()

    def test_a_model_with_nothing_to_add_leaves_the_rules_reading(self, lexicon):
        model = Model(topic="tourism", phrases=("السياح",))
        result = understand("عدد السياح 2024", lexicon, llm=model, role=ALWAYS)
        assert result.method == RULES and result.model_asked and model.calls
        assert result.request == parse("عدد السياح 2024", lexicon)
        # Its other wordings are kept, for an answer that finds nothing.
        assert result.alternatives

    def test_what_it_added_is_said(self, lexicon):
        model = Model(dimensions=(Dimension.REGION,))
        note = understand(OWNERSHIP, lexicon, llm=model).note
        assert "«ملكية المساكن»" in note and Dimension.REGION.label_ar in note

    def test_standing_in_for_the_rules_it_keeps_the_places_they_read(self, lexicon):
        model = Model(topic="agriculture", phrases=("الخيول",))
        request = understand("كم عدد الخيول في الرياض", lexicon, llm=model).request
        assert request.places and Dimension.REGION in request.dimensions

    def test_a_failure_leaves_the_rules_reading_and_says_why(self, lexicon):
        model = Model(fails=LlmUnavailable("نموذج Google مزدحم الآن", transient=True))
        result = understand(OWNERSHIP, lexicon, llm=model)
        assert result.method == RULES and result.model_asked
        assert result.request == parse(OWNERSHIP, lexicon)
        assert "مزدحم" in result.note


def candidate(title: str, **kwargs) -> DatasetCandidate:
    return DatasetCandidate(source_id="s", dataset_id=title, title_ar=title, **kwargs)


class TestWhatTheWordingsDecide:
    def test_a_wording_alone_never_makes_a_result_on_the_subject(self, lexicon):
        request = understand(OWNERSHIP, lexicon, llm=Model()).request
        assert not resolve.on_subject(request, candidate("ملكية المساكن في المدن"))

    def test_among_results_on_the_subject_it_adds_a_little(self, lexicon):
        assisted = understand(OWNERSHIP, lexicon, llm=Model()).request
        alone = parse(OWNERSHIP, lexicon)
        named = candidate("ملكية المساكن للأسر السعودية")
        assert resolve.on_subject(assisted, named)

        def score(request):
            return resolve.score_candidate(request, named, None, lexicon).score

        assert score(assisted) - score(alone) == pytest.approx(resolve.W_EXPANSION)
        assert resolve.W_EXPANSION < resolve.W_TERM  # less than one of the asker's words

    def test_a_topic_from_the_model_still_needs_one_of_its_phrases(self, lexicon):
        model = Model(topic="electricity", phrases=("استهلاك الكهرباء",))
        request = understand(UNPLACED, lexicon, llm=model).request
        assert resolve.on_subject(request, candidate("استهلاك الطاقة الكهربائية حسب المنطقة"))
        assert not resolve.on_subject(request, candidate("عدد محطات توليد الكهرباء"))


class TestOtherWaysToAsk:
    def test_the_models_own_when_it_read_the_question(self, lexicon):
        model = Model()
        result = understand(UNPLACED, lexicon, llm=model)
        assert rephrasings(result, UNPLACED, model) == ("نسبة الأسر المالكة لمساكنها",)
        assert len(model.calls) == 1  # not asked again

    def test_asked_for_once_when_the_rules_read_it(self, lexicon):
        model = Model()
        result = understand("عدد السياح 2024", lexicon, llm=model)
        assert rephrasings(result, "عدد السياح 2024", model)
        assert len(model.calls) == 1

    def test_not_asked_again_after_it_failed(self, lexicon):
        model = Model(fails=LlmUnavailable("مزدحم", transient=True))
        result = understand(OWNERSHIP, lexicon, llm=model)
        assert rephrasings(result, OWNERSHIP, model) == ()
        assert len(model.calls) == 1

    def test_none_without_a_model(self, lexicon):
        result = understand("عدد السياح 2024", lexicon)
        assert rephrasings(result, "عدد السياح 2024", None) == ()

    def test_a_failure_now_is_no_suggestions(self, lexicon):
        result = understand("عدد السياح 2024", lexicon)
        assert rephrasings(result, "عدد السياح 2024", Model(fails=RuntimeError("x"))) == ()


class TestTheAnswer:
    def agent(self, tmp_path, model):
        from datetime import date

        from masdar.pipeline.orchestrator import Agent, AgentConfig
        from masdar.sources.http import HttpClient
        from masdar.sources.registry import DEMO_SOURCES_FILE, Registry, load_descriptors

        http = HttpClient(offline=True, use_cache=False, retries=1)
        return Agent(
            registry=Registry(load_descriptors(DEMO_SOURCES_FILE), http), http=http,
            lexicon=load_lexicon(), llm=model,
            config=AgentConfig(out_dir=tmp_path / "out", today=date(2026, 10, 17)),
        )

    def test_nothing_found_offers_the_models_wordings(self, tmp_path):
        model = Model(topic="agriculture", phrases=("الخيول",),
                      alternatives=("أعداد الخيول المسجلة", "إحصاءات الخيل العربية"))
        answer = self.agent(tmp_path, model).answer("كم عدد الخيول المسجلة")
        assert answer.verdict is Verdict.NO_SOURCE
        assert answer.rephrasings == ("أعداد الخيول المسجلة", "إحصاءات الخيل العربية")
        assert "نموذج لغوي" in answer.audit[0] and "Fake · fake-1" in answer.audit[0]

    def test_an_assisted_reading_says_what_it_added(self, tmp_path):
        answer = self.agent(tmp_path, Model()).answer(OWNERSHIP)
        assert "ملكية المساكن" in answer.audit[0]

    def test_the_chat_carries_them(self, tmp_path):
        from masdar.chat.server import ChatApp

        model = Model(topic="agriculture", phrases=("الخيول",),
                      alternatives=("أعداد الخيول المسجلة",))
        payload = ChatApp.build(self.agent(tmp_path, model)).ask(None, "كم عدد الخيول المسجلة")
        assert payload["rephrasings"] == ["أعداد الخيول المسجلة"]
        assert payload["understood"]["method"] == "llm"
