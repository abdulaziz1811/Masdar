"""The same reading of a question as `llm.py`, by Google's Gemini.

Everything that decides what the reading may contain is shared with the
Claude version: the instructions, the JSON schema, and the validation in
`LlmUnderstanding._to_reading`. Only the call differs, and it is made over
plain HTTPS with `requests`, so no Google library is needed.

The key goes in the `x-goog-api-key` header, never in the URL, so it cannot
end up in a log line or an error message that quotes the address.
"""

from __future__ import annotations

import requests

from masdar.nlu.lexicon import Lexicon
from masdar.nlu.llm import (
    API_MESSAGE_CHARS,
    MAX_TOKENS,
    TIMEOUT_SECONDS,
    LlmUnavailable,
    LlmUnderstanding,
    model_from_env,
)

DEFAULT_MODEL = "gemini-3.8-flash"
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}"
KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
# Mapping a question to a handful of fields is a light task. Gemini 3 models
# take a thinking level; earlier ones reject the field, so it is only sent
# to them.
THINKING_LEVEL = "LOW"

# finishReason values meaning the model declined rather than ran out.
_DECLINED = ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION")


def api_key() -> str:
    from masdar.envfile import secret_from_env

    for name in KEY_NAMES:
        value = secret_from_env(name)
        if value:
            return value
    return ""


def _error_details(response) -> tuple[str, str, str]:
    """(status, reason, message) from a Google API error body, as far as present."""
    try:
        error = response.json().get("error") or {}
    except ValueError:
        return "", "", ""
    reason = ""
    for detail in error.get("details") or ():
        if isinstance(detail, dict) and detail.get("reason"):
            reason = str(detail["reason"])
            break
    message = " ".join(str(error.get("message") or "").split())[:API_MESSAGE_CHARS]
    return str(error.get("status") or ""), reason, message


def describe_response_error(response) -> str:
    code = response.status_code
    status, reason, message = _error_details(response)
    if reason == "API_KEY_INVALID" or "api key not valid" in message.lower():
        return "مفتاح Gemini غير صالح"
    if code == 403 or status == "PERMISSION_DENIED":
        return "مفتاح Gemini لا يملك صلاحية هذا النموذج"
    if code == 404 or status == "NOT_FOUND":
        return "النموذج غير متاح لهذا المفتاح"
    if code == 429 or status == "RESOURCE_EXHAUSTED":
        return "تجاوز حد الاستخدام لدى Google؛ أعد المحاولة بعد دقيقة"
    # "This model is currently experiencing high demand": Google's side, and
    # passing; said in Arabic, not in the English it comes in.
    if code == 503 or status == "UNAVAILABLE":
        return "نموذج Google مزدحم الآن"
    if code >= 500:
        return "خطأ مؤقت لدى Google"
    suffix = f": {message}" if message else ""
    return f"خطأ من Google ({code}){suffix}"


class GeminiUnderstanding(LlmUnderstanding):
    """Maps a free-form question to a `DataRequest` with Gemini."""

    def __init__(
        self,
        session,
        lexicon: Lexicon,
        key: str,
        model: str = DEFAULT_MODEL,
        today=None,
    ):
        super().__init__(None, lexicon, model=model, today=today)
        self._session = session
        self._key = key

    @classmethod
    def from_environment(cls, lexicon: Lexicon) -> tuple[GeminiUnderstanding | None, str]:
        key = api_key()
        if not key:
            return None, "غير مفعّل — أضف GEMINI_API_KEY لتفعيله"
        model = model_from_env("gemini") or DEFAULT_MODEL
        return cls(requests.Session(), lexicon, key, model=model), f"مفعّل (Gemini · {model})"

    # -- transport -----------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._key, "Content-Type": "application/json"}

    def _request(self, method: str, url: str, **kwargs):
        try:
            return self._session.request(
                method, url, headers=self._headers(), timeout=TIMEOUT_SECONDS, **kwargs
            )
        except requests.Timeout as exc:
            raise LlmUnavailable("انتهت مهلة الاتصال بـ Google") from exc
        except requests.RequestException as exc:
            raise LlmUnavailable("تعذّر الاتصال بـ Google") from exc

    def verify(self) -> tuple[bool, str]:
        """Whether the key works for this model; looking a model up is free."""
        try:
            response = self._request("GET", API.format(model=self.model))
        except LlmUnavailable as exc:
            return False, exc.reason
        if response.status_code != 200:
            return False, describe_response_error(response)
        return True, ""

    def _body(self, user: str) -> dict:
        config: dict = {
            "maxOutputTokens": MAX_TOKENS,
            "responseMimeType": "application/json",
            "responseJsonSchema": self._schema,
        }
        if self.model.startswith("gemini-3"):
            config["thinkingConfig"] = {"thinkingLevel": THINKING_LEVEL}
        return {
            "systemInstruction": {"parts": [{"text": self._system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": config,
        }

    def _ask(self, user: str) -> str:
        url = API.format(model=self.model) + ":generateContent"
        response = self._request("POST", url, json=self._body(user))
        if response.status_code != 200:
            raise LlmUnavailable(describe_response_error(response))
        try:
            data = response.json()
        except ValueError as exc:
            raise LlmUnavailable("رد Google ليس JSON صالحاً") from exc

        if (data.get("promptFeedback") or {}).get("blockReason"):
            raise LlmUnavailable("رفض النموذج الطلب")
        candidates = data.get("candidates") or []
        if not candidates:
            raise LlmUnavailable("لم يُرجع النموذج رداً")
        candidate = candidates[0]
        finish = str(candidate.get("finishReason") or "")
        if finish == "MAX_TOKENS":
            raise LlmUnavailable("انقطع رد النموذج قبل اكتماله")
        if finish in _DECLINED:
            raise LlmUnavailable("رفض النموذج الطلب")
        parts = (candidate.get("content") or {}).get("parts") or []
        # Thought summaries, when a model returns them, are not the answer.
        return "".join(
            str(part.get("text") or "") for part in parts
            if isinstance(part, dict) and not part.get("thought")
        )
