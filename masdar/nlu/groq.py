"""The same reading of a question as `llm.py`, by an open model served by Groq.

Groq serves open-weight models behind an OpenAI-compatible API with a free
plan (for `openai/gpt-oss-120b`: 30 requests a minute, 1,000 a day), which
makes it the second free reader after Gemini: another company, so another
quota and another outage. The contract is the one in `llm.py`; the reply
is held to the same JSON schema by Groq's strict structured outputs.

The key goes in the `Authorization` header, never in the URL.
"""

from __future__ import annotations

import os

import requests

from masdar.nlu.http_model import TRANSIENT_STATUS, HttpUnderstanding
from masdar.nlu.lexicon import Lexicon
from masdar.nlu.llm import API_MESSAGE_CHARS, MAX_TOKENS, LlmUnavailable, retry_after

DEFAULT_MODEL = "openai/gpt-oss-120b"
API = "https://api.groq.com/openai/v1"
KEY_NAMES = ("GROQ_API_KEY",)
# Chooses another model Groq serves («openai/gpt-oss-20b»).
MODEL_VARIABLE = "MASDAR_GROQ_MODEL"
# gpt-oss reasons before it answers; mapping a question to a handful of
# fields needs little of it, and the reasoning is not wanted in the reply.
REASONING_EFFORT = "low"


def api_key() -> str:
    from masdar.envfile import secret_from_env

    for name in KEY_NAMES:
        value = secret_from_env(name)
        if value:
            return value
    return ""


def _is_gpt_oss(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


def _error_message(response) -> tuple[str, str]:
    """(code, message) from an OpenAI-style error body, as far as present."""
    try:
        error = response.json().get("error") or {}
    except (ValueError, AttributeError):
        return "", ""
    if not isinstance(error, dict):
        return "", ""
    message = " ".join(str(error.get("message") or "").split())[:API_MESSAGE_CHARS]
    return str(error.get("code") or ""), message


def describe_response_error(response) -> str:
    code = response.status_code
    error_code, message = _error_message(response)
    if code == 401 or error_code == "invalid_api_key":
        return "مفتاح Groq غير صالح"
    if code == 403:
        return "مفتاح Groq لا يملك صلاحية هذا النموذج"
    if code == 404 or error_code == "model_not_found":
        return "النموذج غير متاح لدى Groq"
    if code == 429:
        return "تجاوز حد الاستخدام المجاني لدى Groq"
    if code == 498:
        return "خدمة Groq بلغت سعتها الآن"
    if code >= 500:
        return "خطأ مؤقت لدى Groq"
    # The reply broke the schema: this message failed, not the provider.
    if error_code == "json_validate_failed":
        return "رد النموذج لم يطابق الشكل المطلوب"
    suffix = f": {message}" if message else ""
    return f"خطأ من Groq ({code}){suffix}"


def unavailable(response) -> LlmUnavailable:
    """A failed reply as the reason, marked transient when it will pass."""
    return LlmUnavailable(
        describe_response_error(response),
        # 498: Groq's "flex tier at capacity", which passes like a 503.
        transient=response.status_code in TRANSIENT_STATUS or response.status_code == 498,
        retry_after=retry_after(response),
    )


def readers(lexicon: Lexicon) -> list[GroqUnderstanding]:
    key = api_key()
    if not key:
        return []
    model = os.environ.get(MODEL_VARIABLE, "").strip() or DEFAULT_MODEL
    return [GroqUnderstanding(requests.Session(), lexicon, key, model=model)]


class GroqUnderstanding(HttpUnderstanding):
    """Maps a free-form question to a `DataRequest` with a model served by Groq."""

    NAME = "Groq"
    VENDOR = "Groq"

    def __init__(self, session, lexicon: Lexicon, key: str, model: str = DEFAULT_MODEL,
                 today=None):
        super().__init__(session, lexicon, key, model=model, today=today)

    # -- transport -----------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}

    def verify(self) -> tuple[bool, str]:
        """Whether the key works and serves this model; listing models is free."""
        try:
            response = self._request("GET", f"{API}/models")
        except LlmUnavailable as exc:
            return False, exc.reason
        if response.status_code != 200:
            return False, describe_response_error(response)
        try:
            served = {str(m.get("id")) for m in response.json().get("data") or ()
                      if isinstance(m, dict)}
        except (ValueError, AttributeError):
            return False, "رد Groq ليس JSON صالحاً"
        if self.model not in served:
            return False, "النموذج غير متاح لدى Groq"
        return True, ""

    def _body(self, user: str) -> dict:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": MAX_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "data_request",
                    # Strict decoding guarantees the schema; Groq offers it
                    # on gpt-oss and validates other models' replies instead.
                    "strict": _is_gpt_oss(self.model),
                    "schema": self._schema,
                },
            },
        }
        if _is_gpt_oss(self.model):
            body["reasoning_effort"] = REASONING_EFFORT
            body["include_reasoning"] = False
        return body

    def _ask(self, user: str, timeout: float | None = None) -> str:
        response = self._request("POST", f"{API}/chat/completions", timeout,
                                 json=self._body(user))
        if response.status_code != 200:
            raise unavailable(response)
        try:
            choice = (response.json().get("choices") or [])[0]
        except (ValueError, AttributeError, IndexError) as exc:
            raise LlmUnavailable("لم يُرجع النموذج رداً") from exc
        if not isinstance(choice, dict):
            raise LlmUnavailable("لم يُرجع النموذج رداً")
        finish = str(choice.get("finish_reason") or "")
        if finish == "length":
            raise LlmUnavailable("انقطع رد النموذج قبل اكتماله")
        if finish == "content_filter":
            raise LlmUnavailable("رفض النموذج الطلب")
        message = choice.get("message") or {}
        if message.get("refusal"):
            raise LlmUnavailable("رفض النموذج الطلب")
        return str(message.get("content") or "")
