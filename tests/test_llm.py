"""The optional Claude reading of questions the rules cannot place.

A fake client stands in for the API: these tests pin down what is sent,
what is accepted back, and that every failure leaves the rules' reading in
place. No key and no network are needed.
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest

from masdar.domain.models import Dimension, PeriodKind, Verdict
from masdar.nlu.lexicon import load_lexicon
from masdar.nlu.llm import BETAS, DEFAULT_MODEL, LlmUnderstanding
from masdar.nlu.understand import LLM, RULES, understand
from masdar.pipeline.orchestrator import Agent, AgentConfig
from masdar.sources.http import HttpClient
from masdar.sources.registry import DEMO_SOURCES_FILE, Registry, load_descriptors

TODAY = date(2026, 9, 27)
# Names no topic the lexicon knows, so the rules alone cannot place it.
UNPLACED = "ودي اعرف كم كيلوات صرفته البيوت بكل منطقة"


def reading(**overrides) -> dict:
    base = {
        "is_data_request": True,
        "follow_up": False,
        "topic": "electricity",
        "keywords_ar": ["استهلاك الكهرباء", "الكيلوواط"],
        "keywords_en": ["electricity consumption"],
        "period_kind": "single",
        "years": [2024],
        "dimensions": ["region"],
        "restatement_ar": "استهلاك الكهرباء في المساكن لسنة 2024 حسب المناطق",
    }
    base.update(overrides)
    return base


class FakeClient:
    def __init__(self, payload=None, stop_reason="end_turn", raises=None):
        self.payload = payload if payload is not None else reading()
        self.stop_reason = stop_reason
        self.raises = raises
        self.calls: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text=text)],
        )


@pytest.fixture
def lexicon():
    return load_lexicon()


def llm_with(lexicon, **kwargs) -> tuple[LlmUnderstanding, FakeClient]:
    client = FakeClient(**kwargs)
    return LlmUnderstanding(client, lexicon, today=TODAY), client


class TestWhenTheModelIsAsked:
    def test_a_question_the_rules_place_never_leaves_the_machine(self, lexicon):
        llm, client = llm_with(lexicon)
        result = understand("احصاءات الطاقة الكهربائية 2024 حسب المناطق", lexicon, llm=llm)
        assert result.method == RULES and client.calls == []

    def test_a_question_the_rules_cannot_place_is_read_by_the_model(self, lexicon):
        llm, client = llm_with(lexicon)
        result = understand(UNPLACED, lexicon, llm=llm)
        assert len(client.calls) == 1
        assert result.method == LLM
        request = result.request
        assert request.topic.id == "electricity"
        assert request.period.years == (2024,) and request.period.kind is PeriodKind.SINGLE
        assert request.dimensions == (Dimension.REGION,)
        assert "استهلاك الكهرباء" in request.free_terms
        assert result.note.startswith("استهلاك الكهرباء")

    def test_without_a_model_the_rules_stand(self, lexicon):
        result = understand(UNPLACED, lexicon, llm=None)
        assert result.method == RULES and result.request.topic is None


class TestWhatIsSent:
    def test_request_shape(self, lexicon):
        llm, client = llm_with(lexicon)
        llm.read(UNPLACED)
        call = client.calls[0]
        assert call["model"] == DEFAULT_MODEL == "claude-opus-5-5"
        assert call["betas"] == BETAS and call["fallbacks"] == "default"
        assert call["output_config"]["effort"] == "low"
        schema = call["output_config"]["format"]["schema"]
        assert call["output_config"]["format"]["type"] == "json_schema"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        assert set(schema["properties"]["topic"]["enum"]) == {*lexicon.topic_ids, "none"}

    def test_the_topic_list_is_in_the_stable_system_prompt(self, lexicon):
        llm, client = llm_with(lexicon)
        llm.read(UNPLACED)
        system = client.calls[0]["system"]
        assert all(topic_id in system for topic_id in lexicon.topic_ids)
        assert "2026" not in system  # the date is volatile and goes in the message

    def test_the_message_carries_today_and_the_previous_request(self, lexicon):
        from masdar.nlu.parser import parse

        llm, client = llm_with(lexicon)
        llm.read("وحسب الجنس؟", parse("السكان 2022", lexicon))
        content = client.calls[0]["messages"][0]["content"]
        assert "2026-09-27" in content and "السكان 2022" in content


class TestWhatIsAccepted:
    def test_values_outside_the_contract_are_dropped(self, lexicon):
        llm, _ = llm_with(lexicon, payload=reading(
            topic="astrology", years=[2024, 3000, 1066, True], dimensions=["region", "planet"],
        ))
        request = llm.read(UNPLACED).request
        assert request.topic is None
        assert request.period.years == (2024,)
        assert request.dimensions == (Dimension.REGION,)

    def test_a_claimed_year_that_did_not_survive_is_not_a_period(self, lexicon):
        llm, _ = llm_with(lexicon, payload=reading(years=[3000], period_kind="single"))
        assert llm.read(UNPLACED).request.period.kind is PeriodKind.UNSPECIFIED

    def test_phrases_are_trimmed_deduplicated_and_capped(self, lexicon):
        llm, _ = llm_with(lexicon, payload=reading(
            keywords_ar=["  الكهرباء ", "الكهرباء", "x" * 200, *[f"عبارة {i}" for i in range(9)]],
            keywords_en=[],
        ))
        terms = llm.read(UNPLACED).request.free_terms
        assert terms[0] == "الكهرباء" and terms.count("الكهرباء") == 1
        assert all(len(t) <= 60 for t in terms) and len(terms) <= 5

    def test_rule_years_win_because_their_hijri_conversion_is_exact(self, lexicon):
        llm, _ = llm_with(lexicon, payload=reading(years=[2025]))
        result = understand("ودي اعرف كم كيلوات صرفته البيوت عام 1446هـ", lexicon, llm=llm)
        assert result.request.period.years == (2024, 2025)

    def test_not_a_data_request(self, lexicon):
        llm, _ = llm_with(lexicon, payload=reading(is_data_request=False, topic="none"))
        result = understand("مرحبا كيف حالك", lexicon, llm=llm)
        assert not result.request.is_answerable

    def test_a_follow_up_continues_the_previous_question(self, lexicon):
        from masdar.nlu.parser import parse

        previous = parse("احصاءات الطاقة الكهربائية 2026 حسب المناطق", lexicon)
        llm, _ = llm_with(lexicon, payload=reading(
            follow_up=True, topic="none", years=[2021], dimensions=[], keywords_ar=[],
            keywords_en=[],
        ))
        result = understand("وش صار قبلها بخمس سنين", lexicon, previous, "الأول", llm)
        assert result.follow_up and result.method == LLM
        assert result.request.topic.id == "electricity"
        assert result.request.period.years == (2021,)


class TestFailuresFallBackToTheRules:
    def test_refusal(self, lexicon):
        llm, _ = llm_with(lexicon, stop_reason="refusal")
        result = understand(UNPLACED, lexicon, llm=llm)
        assert result.method == RULES and "رفض" in result.note

    def test_malformed_output(self, lexicon):
        llm, _ = llm_with(lexicon, payload="not json at all")
        result = understand(UNPLACED, lexicon, llm=llm)
        assert result.method == RULES and "JSON" in result.note

    def test_truncated_output(self, lexicon):
        llm, _ = llm_with(lexicon, stop_reason="max_tokens")
        assert understand(UNPLACED, lexicon, llm=llm).method == RULES

    def test_api_error_is_described_in_arabic(self, lexicon):
        anthropic = pytest.importorskip("anthropic")
        import httpx2

        error = anthropic.APIConnectionError(
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        )
        llm, _ = llm_with(lexicon, raises=error)
        result = understand(UNPLACED, lexicon, llm=llm)
        assert result.method == RULES and "تعذّر الاتصال" in result.note

    def test_an_unexpected_error_cannot_sink_the_answer(self, lexicon):
        llm, _ = llm_with(lexicon, raises=RuntimeError("boom"))
        result = understand(UNPLACED, lexicon, llm=llm)
        assert result.method == RULES and "RuntimeError" in result.note


class TestConfiguration:
    @pytest.fixture(autouse=True)
    def no_gemini(self, monkeypatch):
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "MASDAR_LLM_PROVIDER"):
            monkeypatch.delenv(name, raising=False)

    def test_off_without_a_key(self, lexicon, monkeypatch):
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "MASDAR_LLM"):
            monkeypatch.delenv(name, raising=False)
        llm, status = LlmUnderstanding.from_environment(lexicon)
        assert llm is None and "ANTHROPIC_API_KEY" in status

    def test_explicitly_disabled(self, lexicon, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
        monkeypatch.setenv("MASDAR_LLM", "off")
        llm, _ = LlmUnderstanding.from_environment(lexicon)
        assert llm is None

    def test_on_with_a_key_and_the_key_is_never_echoed(self, lexicon, monkeypatch):
        pytest.importorskip("anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
        monkeypatch.delenv("MASDAR_LLM", raising=False)
        monkeypatch.setenv("MASDAR_LLM_MODEL", "claude-sonnet-5")
        llm, status = LlmUnderstanding.from_environment(lexicon)
        assert llm is not None and llm.model == "claude-sonnet-5"
        assert "sk-ant" not in status


class TestKeyCheck:
    """The key is checked once at start-up, with the real SDK, so a wrong or
    unfunded key shows on the page instead of failing quietly per question."""

    def client(self, status, body):
        anthropic = pytest.importorskip("anthropic")
        import httpx2

        seen: list[str] = []

        def handler(request):
            seen.append(f"{request.method} {request.url.path}")
            return httpx2.Response(status, json=body)

        client = anthropic.Anthropic(
            api_key="sk-ant-test-not-a-real-key", max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        )
        return client, seen

    def test_a_working_key_passes_without_generating_anything(self, lexicon):
        model = {"type": "model", "id": DEFAULT_MODEL, "display_name": "Claude",
                 "created_at": "2026-01-01T00:00:00Z"}
        client, seen = self.client(200, model)
        assert LlmUnderstanding(client, lexicon).verify() == (True, "")
        assert seen == [f"GET /v1/models/{DEFAULT_MODEL}"]

    def test_a_wrong_key_is_named(self, lexicon):
        error = {"type": "error", "error": {"type": "authentication_error",
                                            "message": "invalid x-api-key"}}
        client, _ = self.client(401, error)
        ok, why = LlmUnderstanding(client, lexicon).verify()
        assert not ok and "غير صالح" in why

    def test_a_model_the_key_cannot_use_is_named(self, lexicon):
        error = {"type": "error", "error": {"type": "not_found_error", "message": "model"}}
        client, _ = self.client(404, error)
        ok, why = LlmUnderstanding(client, lexicon).verify()
        assert not ok and "النموذج غير متاح" in why


    def test_an_unfunded_key_is_named(self, lexicon):
        # A key with no credit passes the free start-up check, then every
        # question is refused with a 400 that says so.
        from masdar.nlu.llm import LlmUnavailable

        error = {"type": "error", "error": {
            "type": "invalid_request_error",
            "message": "Your credit balance is too low to access the Anthropic API. "
                       "Please go to Plans & Billing to upgrade or purchase credits.",
        }}
        client, _ = self.client(400, error)
        with pytest.raises(LlmUnavailable) as caught:
            LlmUnderstanding(client, lexicon, today=TODAY).read("كم عدد الخيول")
        assert "رصيد" in caught.value.reason

    def test_any_other_refusal_carries_the_api_explanation(self, lexicon):
        from masdar.nlu.llm import LlmUnavailable

        error = {"type": "error", "error": {"type": "invalid_request_error",
                                            "message": "something about the request"}}
        client, _ = self.client(400, error)
        with pytest.raises(LlmUnavailable) as caught:
            LlmUnderstanding(client, lexicon, today=TODAY).read("كم عدد الخيول")
        assert "(400)" in caught.value.reason
        assert "something about the request" in caught.value.reason

    def test_the_request_the_real_sdk_sends(self, lexicon):
        anthropic = pytest.importorskip("anthropic")
        import httpx2

        sent: list = []
        reading = {
            "is_data_request": True, "follow_up": False, "topic": "agriculture",
            "keywords_ar": ["الخيول"], "keywords_en": ["horses"], "period_kind": "latest",
            "years": [], "dimensions": [], "restatement_ar": "عدد الخيول في المملكة",
        }

        def handler(request):
            sent.append(request)
            return httpx2.Response(200, json={
                "id": "msg_test", "type": "message", "role": "assistant",
                "model": DEFAULT_MODEL, "stop_reason": "end_turn", "stop_sequence": None,
                "content": [{"type": "text", "text": json.dumps(reading, ensure_ascii=False)}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            })

        client = anthropic.Anthropic(
            api_key="sk-ant-test-not-a-real-key", max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        )
        result = LlmUnderstanding(client, lexicon, today=TODAY).read("كم عدد الخيول")
        assert result.request.topic.id == "agriculture"

        (request,) = sent
        body = json.loads(request.content)
        assert request.url.path == "/v1/messages"
        assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
        assert body["model"] == DEFAULT_MODEL
        assert body["fallbacks"] == "default"
        assert body["output_config"]["effort"] == "low"
        assert body["output_config"]["format"]["type"] == "json_schema"
        # Opus 5.5 rejects both; thinking is controlled by effort alone.
        assert "thinking" not in body and "temperature" not in body


class TestEndToEnd:
    def test_the_agent_answers_a_question_only_the_model_could_place(self, lexicon, tmp_path):
        http = HttpClient(offline=True, use_cache=False, retries=1)
        llm, _ = llm_with(lexicon)
        agent = Agent(
            registry=Registry(load_descriptors(DEMO_SOURCES_FILE), http),
            http=http,
            lexicon=lexicon,
            llm=llm,
            config=AgentConfig(out_dir=tmp_path / "out", today=TODAY),
        )
        answer = agent.answer(UNPLACED)
        assert answer.verdict is Verdict.AVAILABLE
        assert answer.primary.export_path is not None
        assert "نموذج لغوي" in answer.audit[0]

    def test_the_chat_says_the_model_helped(self, lexicon, tmp_path):
        from masdar.chat.server import ChatApp

        http = HttpClient(offline=True, use_cache=False, retries=1)
        llm, _ = llm_with(lexicon)
        agent = Agent(
            registry=Registry(load_descriptors(DEMO_SOURCES_FILE), http),
            http=http,
            lexicon=lexicon,
            llm=llm,
            config=AgentConfig(out_dir=tmp_path / "out", today=TODAY),
        )
        payload = ChatApp.build(agent).ask(None, UNPLACED)
        assert payload["understood"]["method"] == "llm"
        assert payload["verdict"] == "available"
