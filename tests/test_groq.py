"""The reading of a question by an open model served by Groq.

A fake session stands in for the API, as in test_gemini.py: what is sent,
what is accepted back, and that every failure is named -- and marked as
passing when it will pass, so the chain rests Groq instead of waiting on it.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
import requests

from masdar.nlu.groq import API, DEFAULT_MODEL, GroqUnderstanding
from masdar.nlu.lexicon import load_lexicon
from masdar.nlu.llm import LlmUnavailable
from masdar.nlu.understand import LLM, understand

TODAY = date(2026, 10, 17)
KEY = "test-groq-key-not-real"
UNPLACED = "ودي اعرف كم كيلوات صرفته البيوت بكل منطقة"


def reading(**overrides) -> dict:
    base = {
        "is_data_request": True, "follow_up": False, "topic": "electricity",
        "keywords_ar": ["استهلاك الكهرباء"], "keywords_en": ["electricity consumption"],
        "period_kind": "unspecified", "years": [], "dimensions": ["region"],
        "restatement_ar": "استهلاك الكهرباء في المساكن حسب المناطق",
        "alternatives_ar": ["الاستهلاك السكني للكهرباء حسب المناطق"],
    }
    base.update(overrides)
    return base


class Response:
    def __init__(self, status: int, body: dict, headers: dict | None = None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def json(self):
        return self._body


def completion(payload: dict | str, finish: str = "stop") -> dict:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return {"choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": text}}]}


def groq_error(message: str, code: str = "") -> dict:
    return {"error": {"message": message, "type": "invalid_request_error", "code": code}}


class FakeSession:
    def __init__(self, response: Response | None = None, raises: Exception | None = None):
        self.response = response or Response(200, completion(reading()))
        self.raises = raises
        self.calls: list[dict] = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if self.raises is not None:
            raise self.raises
        return self.response


@pytest.fixture
def lexicon():
    return load_lexicon()


def groq(lexicon, model=DEFAULT_MODEL, **kwargs):
    session = FakeSession(**kwargs)
    return GroqUnderstanding(session, lexicon, KEY, model=model, today=TODAY), session


class TestTheRequest:
    def test_what_is_sent(self, lexicon):
        llm, session = groq(lexicon)
        llm.read(UNPLACED, timeout=4.0)
        (call,) = session.calls
        assert call["method"] == "POST" and call["url"] == f"{API}/chat/completions"
        assert call["timeout"] == 4.0
        body = call["json"]
        assert body["model"] == DEFAULT_MODEL == "openai/gpt-oss-120b"
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        assert UNPLACED in body["messages"][1]["content"]
        schema = body["response_format"]["json_schema"]
        assert body["response_format"]["type"] == "json_schema"
        assert schema["strict"] is True
        assert set(schema["schema"]["required"]) == set(schema["schema"]["properties"])
        # gpt-oss reasons briefly, and keeps the reasoning out of the reply.
        assert body["reasoning_effort"] == "low" and body["include_reasoning"] is False

    def test_the_key_travels_in_a_header_never_in_the_address(self, lexicon):
        llm, session = groq(lexicon)
        llm.read(UNPLACED)
        (call,) = session.calls
        assert call["headers"]["Authorization"] == f"Bearer {KEY}"
        assert KEY not in call["url"]

    def test_other_models_get_no_reasoning_fields_and_no_strict_mode(self, lexicon):
        llm, session = groq(lexicon, model="qwen/qwen3.8-27b")
        llm.read(UNPLACED)
        body = session.calls[0]["json"]
        assert "reasoning_effort" not in body and "include_reasoning" not in body
        assert body["response_format"]["json_schema"]["strict"] is False


class TestTheReply:
    def test_it_reads_like_the_others(self, lexicon):
        llm, _ = groq(lexicon)
        result = llm.read(UNPLACED)
        assert result.request.topic.id == "electricity"
        assert result.alternatives == ("الاستهلاك السكني للكهرباء حسب المناطق",)
        assert result.model == f"Groq · {DEFAULT_MODEL}"

    def test_through_the_understanding_step(self, lexicon):
        llm, _ = groq(lexicon)
        result = understand(UNPLACED, lexicon, llm=llm)
        assert result.method == LLM and result.request.topic.id == "electricity"


class TestWhenGroqCannotHelp:
    @pytest.mark.parametrize("response, words, transient", [
        (Response(401, groq_error("Invalid API Key", "invalid_api_key")), "غير صالح", False),
        (Response(404, groq_error("The model does not exist", "model_not_found")),
         "غير متاح", False),
        (Response(429, groq_error("Rate limit reached"), {"retry-after": "7"}),
         "حد الاستخدام", True),
        (Response(503, groq_error("Service unavailable")), "مؤقت", True),
        (Response(498, groq_error("Flex tier capacity exceeded")), "سعتها", True),
        (Response(400, groq_error("Generated JSON does not match", "json_validate_failed")),
         "الشكل المطلوب", False),
        (Response(400, groq_error("something about the request")),
         "something about the request", False),
        (Response(200, completion("{", finish="length")), "انقطع", False),
        (Response(200, completion("", finish="content_filter")), "رفض", False),
        (Response(200, completion("not json")), "JSON", False),
        (Response(200, {"choices": []}), "لم يُرجع", False),
    ])
    def test_the_reason_is_named(self, lexicon, response, words, transient):
        llm, _ = groq(lexicon, response=response)
        with pytest.raises(LlmUnavailable) as caught:
            llm.read(UNPLACED)
        assert words in caught.value.reason
        assert caught.value.transient is transient
        assert KEY not in caught.value.reason

    def test_the_wait_groq_asks_for_is_kept(self, lexicon):
        llm, _ = groq(lexicon, response=Response(429, groq_error("Rate limit"),
                                                  {"retry-after": "7"}))
        with pytest.raises(LlmUnavailable) as caught:
            llm.read(UNPLACED)
        assert caught.value.retry_after == 7.0

    def test_a_timeout_passes(self, lexicon):
        llm, _ = groq(lexicon, raises=requests.ReadTimeout("slow"))
        with pytest.raises(LlmUnavailable) as caught:
            llm.read(UNPLACED)
        assert "مهلة" in caught.value.reason and caught.value.transient


class TestKeyCheck:
    def test_a_working_key_that_serves_the_model(self, lexicon):
        models = {"object": "list", "data": [{"id": DEFAULT_MODEL}, {"id": "openai/gpt-oss-20b"}]}
        llm, session = groq(lexicon, response=Response(200, models))
        assert llm.verify() == (True, "")
        assert session.calls[0]["method"] == "GET"
        assert session.calls[0]["url"] == f"{API}/models"

    def test_a_model_groq_does_not_serve(self, lexicon):
        llm, _ = groq(lexicon, response=Response(200, {"data": [{"id": "other"}]}))
        ok, why = llm.verify()
        assert not ok and "غير متاح" in why

    def test_a_wrong_key(self, lexicon):
        llm, _ = groq(lexicon, response=Response(401, groq_error("Invalid API Key",
                                                                 "invalid_api_key")))
        ok, why = llm.verify()
        assert not ok and "غير صالح" in why


class TestConfiguration:
    @pytest.fixture(autouse=True)
    def clean(self, monkeypatch):
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY",
                     "GOOGLE_API_KEY", "GROQ_API_KEY", "MASDAR_LLM", "MASDAR_LLM_PROVIDER",
                     "MASDAR_GROQ_MODEL"):
            monkeypatch.delenv(name, raising=False)

    def test_a_groq_key_alone_turns_groq_on(self, lexicon, monkeypatch):
        from masdar.nlu.chain import from_environment

        monkeypatch.setenv("GROQ_API_KEY", KEY)
        llm, status = from_environment(lexicon)
        assert [r.label for r in llm.readers] == [f"Groq · {DEFAULT_MODEL}"]
        assert "Groq" in status and KEY not in status

    def test_after_gemini(self, lexicon, monkeypatch):
        from masdar.nlu.chain import from_environment

        monkeypatch.setenv("GROQ_API_KEY", KEY)
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-not-real")
        llm, _ = from_environment(lexicon)
        assert [r.NAME for r in llm.readers] == ["Gemini", "Gemini", "Groq"]

    def test_another_groq_model_can_be_chosen(self, lexicon, monkeypatch):
        from masdar.nlu.chain import from_environment

        monkeypatch.setenv("GROQ_API_KEY", KEY)
        monkeypatch.setenv("MASDAR_GROQ_MODEL", "openai/gpt-oss-20b")
        llm, _ = from_environment(lexicon)
        assert llm.model == "openai/gpt-oss-20b"
