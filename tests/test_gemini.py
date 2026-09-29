"""Gemini as the language model: the same reading as Claude, a different call.

No test reaches Google. A fake session records what would be sent and
answers the way the Gemini API does, error bodies included.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
import requests

from masdar.nlu.gemini import DEFAULT_MODEL, GeminiUnderstanding
from masdar.nlu.lexicon import load_lexicon
from masdar.nlu.llm import LlmUnavailable, LlmUnderstanding
from masdar.nlu.understand import LLM, understand

TODAY = date(2026, 9, 29)
KEY = "test-gemini-key-not-real"
UNPLACED = "ودي اعرف كم كيلوات صرفته البيوت بكل منطقة"


def reading(**overrides) -> dict:
    base = {
        "is_data_request": True, "follow_up": False, "topic": "electricity",
        "keywords_ar": ["استهلاك الكهرباء"], "keywords_en": ["electricity consumption"],
        "period_kind": "single", "years": [2024], "dimensions": ["region"],
        "restatement_ar": "استهلاك الكهرباء في المساكن لسنة 2024 حسب المناطق",
    }
    base.update(overrides)
    return base


class Response:
    def __init__(self, status: int, body: dict):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


def answer(payload: dict | str, finish: str = "STOP", thought: str | None = None) -> dict:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    parts = [{"text": thought, "thought": True}] if thought else []
    parts.append({"text": text})
    return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}]}


def google_error(code: int, status: str, message: str, reason: str = "") -> dict:
    info = "type.googleapis.com/google.rpc.ErrorInfo"
    details = [{"@type": info, "reason": reason}] if reason else []
    return {"error": {"code": code, "message": message, "status": status, "details": details}}


class FakeSession:
    def __init__(self, response: Response | None = None, raises: Exception | None = None):
        self.response = response or Response(200, answer(reading()))
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


def gemini(lexicon, model=DEFAULT_MODEL, **kwargs):
    session = FakeSession(**kwargs)
    return GeminiUnderstanding(session, lexicon, KEY, model=model, today=TODAY), session


class TestTheRequest:
    def test_what_is_sent(self, lexicon):
        llm, session = gemini(lexicon)
        llm.read(UNPLACED)
        (call,) = session.calls
        assert call["method"] == "POST"
        host = "https://generativelanguage.googleapis.com/v1beta/models"
        assert call["url"] == f"{host}/{DEFAULT_MODEL}:generateContent"
        body = call["json"]
        config = body["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        assert config["responseJsonSchema"]["required"]
        assert config["thinkingConfig"] == {"thinkingLevel": "LOW"}
        assert "الطاقة الكهربائية" in body["systemInstruction"]["parts"][0]["text"]
        assert UNPLACED in body["contents"][0]["parts"][0]["text"]

    def test_the_key_travels_in_a_header_never_in_the_address(self, lexicon):
        llm, session = gemini(lexicon)
        llm.read(UNPLACED)
        (call,) = session.calls
        assert call["headers"]["x-goog-api-key"] == KEY
        assert KEY not in call["url"]

    def test_older_models_get_no_thinking_level(self, lexicon):
        llm, session = gemini(lexicon, model="gemini-2.5-flash")
        llm.read(UNPLACED)
        assert "thinkingConfig" not in session.calls[0]["json"]["generationConfig"]


class TestTheReply:
    def test_it_reads_like_claude(self, lexicon):
        llm, _ = gemini(lexicon)
        result = llm.read(UNPLACED)
        assert result.request.topic.id == "electricity"
        assert result.request.period.years == (2024,)

    def test_a_thought_summary_is_not_the_answer(self, lexicon):
        llm, _ = gemini(lexicon, response=Response(200, answer(reading(), thought="thinking...")))
        assert llm.read(UNPLACED).request.topic.id == "electricity"

    def test_an_unknown_topic_is_dropped_not_trusted(self, lexicon):
        llm, _ = gemini(lexicon, response=Response(200, answer(reading(topic="astrology"))))
        assert llm.read(UNPLACED).request.topic is None

    def test_through_the_understanding_step(self, lexicon):
        llm, _ = gemini(lexicon)
        result = understand(UNPLACED, lexicon, llm=llm)
        assert result.method == LLM and result.request.topic.id == "electricity"


class TestWhenGeminiCannotHelp:
    @pytest.mark.parametrize("response, words", [
        (Response(400, google_error(400, "INVALID_ARGUMENT", "API key not valid.",
                                    "API_KEY_INVALID")), "غير صالح"),
        (Response(429, google_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded")), "حد الاستخدام"),
        (Response(404, google_error(404, "NOT_FOUND", "no such model")), "النموذج غير متاح"),
        (Response(400, google_error(400, "INVALID_ARGUMENT", "something about the request")),
         "something about the request"),
        (Response(200, answer("{", finish="MAX_TOKENS")), "انقطع"),
        (Response(200, answer("", finish="SAFETY")), "رفض"),
        (Response(200, {"promptFeedback": {"blockReason": "OTHER"}}), "رفض"),
        (Response(200, answer("not json")), "JSON"),
    ])
    def test_the_reason_is_named(self, lexicon, response, words):
        llm, _ = gemini(lexicon, response=response)
        with pytest.raises(LlmUnavailable) as caught:
            llm.read(UNPLACED)
        assert words in caught.value.reason
        assert KEY not in caught.value.reason

    def test_network_failures_are_named(self, lexicon):
        llm, _ = gemini(lexicon, raises=requests.ConnectTimeout("slow"))
        with pytest.raises(LlmUnavailable) as caught:
            llm.read(UNPLACED)
        assert "مهلة" in caught.value.reason

    def test_the_rules_answer_when_gemini_fails(self, lexicon):
        quota = google_error(429, "RESOURCE_EXHAUSTED", "Quota")
        llm, _ = gemini(lexicon, response=Response(429, quota))
        result = understand(UNPLACED, lexicon, llm=llm)
        assert result.method != LLM and "حد الاستخدام" in result.note


class TestKeyCheck:
    def test_a_working_key(self, lexicon):
        llm, session = gemini(lexicon, response=Response(200, {"name": f"models/{DEFAULT_MODEL}"}))
        assert llm.verify() == (True, "")
        assert session.calls[0]["method"] == "GET"
        assert session.calls[0]["url"].endswith(f"/models/{DEFAULT_MODEL}")

    def test_a_wrong_key(self, lexicon):
        llm, _ = gemini(lexicon, response=Response(
            400, google_error(400, "INVALID_ARGUMENT", "API key not valid.", "API_KEY_INVALID")))
        ok, why = llm.verify()
        assert not ok and "غير صالح" in why


class TestChoosingTheProvider:
    NAMES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY",
             "MASDAR_LLM", "MASDAR_LLM_PROVIDER", "MASDAR_LLM_MODEL")

    @pytest.fixture(autouse=True)
    def clean(self, monkeypatch):
        for name in self.NAMES:
            monkeypatch.delenv(name, raising=False)

    def test_a_gemini_key_turns_gemini_on(self, lexicon, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", KEY)
        llm, status = LlmUnderstanding.from_environment(lexicon)
        assert isinstance(llm, GeminiUnderstanding) and llm.model == DEFAULT_MODEL
        assert "Gemini" in status and KEY not in status

    def test_with_both_keys_gemini_is_used_unless_claude_is_chosen(self, lexicon, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", KEY)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
        llm, _ = LlmUnderstanding.from_environment(lexicon)
        assert isinstance(llm, GeminiUnderstanding)
        pytest.importorskip("anthropic")
        monkeypatch.setenv("MASDAR_LLM_PROVIDER", "claude")
        llm, status = LlmUnderstanding.from_environment(lexicon)
        assert not isinstance(llm, GeminiUnderstanding) and "Claude" in status

    def test_a_claude_model_name_is_not_sent_to_gemini(self, lexicon, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", KEY)
        monkeypatch.setenv("MASDAR_LLM_MODEL", "claude-opus-5-5")
        llm, _ = LlmUnderstanding.from_environment(lexicon)
        assert llm.model == DEFAULT_MODEL

    def test_a_gemini_model_can_be_chosen(self, lexicon, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", KEY)
        monkeypatch.setenv("MASDAR_LLM_MODEL", "gemini-3.5-flash-lite")
        llm, _ = LlmUnderstanding.from_environment(lexicon)
        assert llm.model == "gemini-3.5-flash-lite"

    def test_a_placeholder_is_no_key(self, lexicon, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "-")
        llm, status = LlmUnderstanding.from_environment(lexicon)
        assert llm is None and "GEMINI_API_KEY" in status

    def test_off_means_off(self, lexicon, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", KEY)
        monkeypatch.setenv("MASDAR_LLM", "off")
        llm, _ = LlmUnderstanding.from_environment(lexicon)
        assert llm is None
