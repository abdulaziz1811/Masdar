"""Optional understanding by Claude, for questions the rules do not catch.

The rule-based parser is fast, free and deterministic, but it only knows
the words in its lexicon: «كم صيدلية في الرياض؟» names no topic it knows.
Here Claude turns such a question into the same `DataRequest` the rules
produce -- a topic from the fixed list, years, breakdowns and search
phrases -- and nothing else.

What it is never asked to do matters as much:

* It never answers the question and never produces a number. Availability,
  coverage and every figure still come from the sources and the evidence
  rules in `pipeline/verify.py`.
* Its output is constrained to a JSON schema and then validated again here:
  an unknown topic, a breakdown outside the enum or an implausible year is
  dropped, not trusted.
* Any failure -- no key, no library, network, refusal, malformed output --
  falls back to the rules, and the reason is recorded for the audit trail.

It is used only when the rules find no topic, so a question the lexicon
understands never leaves the machine. When it is used, the question text is
sent to Anthropic's API.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import date

from masdar.domain.models import DataRequest, Dimension, Period, PeriodKind
from masdar.nlu.lexicon import Lexicon

DEFAULT_MODEL = "claude-opus-5-5"
# Server-side fallback: if a request is declined, the API re-runs it on the
# model Anthropic recommends for that case instead of returning a refusal.
BETAS = ["server-side-fallback-2026-07-01"]
# Mapping a question to a handful of fields is a light task.
EFFORT = "low"
MAX_TOKENS = 4096
TIMEOUT_SECONDS = 30.0

MAX_PHRASES = 5
MAX_PHRASE_CHARS = 60
EARLIEST_YEAR = 1900

_PERIOD_KINDS = ("single", "range", "latest", "unspecified")


class LlmUnavailable(Exception):
    """The model could not help with this message; `reason` says why, in Arabic."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class LlmReading:
    request: DataRequest
    follow_up: bool
    # One Arabic sentence saying how the question was read; shown to the user
    # so a misreading is visible before anyone relies on the file.
    restatement: str = ""


def _schema(lexicon: Lexicon) -> dict:
    return {
        "type": "object",
        "properties": {
            "is_data_request": {"type": "boolean"},
            "follow_up": {"type": "boolean"},
            "topic": {"type": "string", "enum": [*lexicon.topic_ids, "none"]},
            "keywords_ar": {"type": "array", "items": {"type": "string"}},
            "keywords_en": {"type": "array", "items": {"type": "string"}},
            "period_kind": {"type": "string", "enum": list(_PERIOD_KINDS)},
            "years": {"type": "array", "items": {"type": "integer"}},
            "dimensions": {
                "type": "array",
                "items": {"type": "string", "enum": [d.value for d in Dimension]},
            },
            "restatement_ar": {"type": "string"},
        },
        "required": [
            "is_data_request", "follow_up", "topic", "keywords_ar", "keywords_en",
            "period_kind", "years", "dimensions", "restatement_ar",
        ],
        "additionalProperties": False,
    }


def _system_prompt(lexicon: Lexicon) -> str:
    topics = "\n".join(f"- {t.id}: {t.label_ar} / {t.label_en}" for t in lexicon.topics)
    dimensions = "\n".join(f"- {d.value}: {d.label_ar}" for d in Dimension)
    return f"""You read questions about Saudi official open data, written in Arabic \
(any dialect) or English, and turn each into a structured search request for a \
system that looks the data up in official sources.

You do not answer the question and you never state figures: the system fetches \
the data and checks which years exist. Your job is only to say what is being \
asked for.

Topics (use the id, or "none" if none fits):
{topics}

Breakdowns the user may ask for (use the value):
{dimensions}

Fill the fields as follows.
- is_data_request: false for greetings, thanks, or anything that is not a request \
for statistics or data.
- topic: the single best topic id for what is measured.
- keywords_ar: 1 to {MAX_PHRASES} short Arabic phrases naming what is measured, as \
they would appear in a dataset title (for «كم صيدلية بالرياض» give «الصيدليات» and \
«عدد الصيدليات»). Do not include years, places, or words like «احصائيات» or «بيانات».
- keywords_en: the same phrases in English.
- years: Gregorian years the user asked for, including years implied relative to \
today's date given in the message («العام الماضي»). A Hijri year spans two \
Gregorian years: include both. Empty when no year is asked for.
- period_kind: single (one year), range (several), latest (the user wants the \
newest available), unspecified (no period mentioned).
- dimensions: only breakdowns the user explicitly asked for.
- follow_up: true only when the message cannot be understood on its own and \
continues the previous request shown in the message (a new year, an added \
breakdown, "the same but ..."). A message naming its own subject is not a \
follow-up.
- restatement_ar: one short Arabic sentence restating the request as you \
understood it."""


def _describe_previous(previous: DataRequest | None) -> str:
    if previous is None or not previous.is_answerable:
        return "لا يوجد"
    parts = [f"السؤال: {previous.raw_query}"]
    if previous.topic:
        parts.append(f"الموضوع: {previous.topic.id}")
    parts.append(f"الفترة: {previous.period.label()}")
    if previous.dimensions:
        parts.append("التفصيل: " + "، ".join(d.value for d in previous.dimensions))
    return " | ".join(parts)


def _clean_phrases(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    phrases: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        phrase = " ".join(value.split())
        if phrase and len(phrase) <= MAX_PHRASE_CHARS and phrase not in phrases:
            phrases.append(phrase)
    return phrases[:MAX_PHRASES]


def _api_errors() -> tuple[type[BaseException], ...]:
    try:
        import anthropic
    except ImportError:
        return ()
    return (anthropic.APIError,)


def _describe_error(exc: BaseException) -> str:
    try:
        import anthropic
    except ImportError:
        return type(exc).__name__
    if isinstance(exc, anthropic.AuthenticationError):
        return "مفتاح Anthropic غير صالح أو منتهي"
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "المفتاح لا يملك صلاحية هذا النموذج"
    if isinstance(exc, anthropic.NotFoundError):
        return "النموذج غير متاح لهذا المفتاح"
    if isinstance(exc, anthropic.RateLimitError):
        return "تجاوز حد الطلبات لدى Anthropic"
    if isinstance(exc, anthropic.APITimeoutError):
        return "انتهت مهلة الاتصال بـ Anthropic"
    if isinstance(exc, anthropic.APIConnectionError):
        return "تعذّر الاتصال بـ Anthropic"
    if isinstance(exc, anthropic.APIStatusError):
        return f"خطأ من Anthropic ({exc.status_code})"
    return type(exc).__name__


class LlmUnderstanding:
    """Maps a free-form question to a `DataRequest` with Claude."""

    def __init__(
        self,
        client,
        lexicon: Lexicon,
        model: str = DEFAULT_MODEL,
        today: date | None = None,
    ):
        self._client = client
        self._lexicon = lexicon
        self.model = model
        self._today = today
        self._system = _system_prompt(lexicon)
        self._schema = _schema(lexicon)

    # -- construction --------------------------------------------------
    @classmethod
    def from_environment(cls, lexicon: Lexicon) -> tuple[LlmUnderstanding | None, str]:
        """The configured instance, or None with the reason in Arabic.

        Enabled by an Anthropic credential in the environment (or `.env`), or
        explicitly with MASDAR_LLM=on; disabled with MASDAR_LLM=off. The key
        itself is read by the SDK and never passes through this code.
        """
        switch = os.environ.get("MASDAR_LLM", "").strip().lower()
        if switch in ("0", "off", "false", "no"):
            return None, "معطّل (MASDAR_LLM=off)"
        has_credential = bool(
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        )
        if not has_credential and switch not in ("1", "on", "true", "yes"):
            return None, "غير مفعّل — أضف ANTHROPIC_API_KEY في ملف .env لتفعيله"
        try:
            import anthropic
        except ImportError:
            return None, "مكتبة anthropic غير مثبتة — pip install -e '.[ai]'"
        model = os.environ.get("MASDAR_LLM_MODEL", "").strip() or DEFAULT_MODEL
        client = anthropic.Anthropic(timeout=TIMEOUT_SECONDS, max_retries=1)
        return cls(client, lexicon, model=model), f"مفعّل ({model})"

    def verify(self) -> tuple[bool, str]:
        """Whether the key works for this model, without generating anything.

        Asked once when the server starts, so a wrong or unfunded key shows on
        the page at once instead of every question quietly falling back to
        the rules. Looking the model up is free and checks both the key and
        the model name.
        """
        try:
            self._client.models.retrieve(self.model)
        except _api_errors() as exc:
            return False, _describe_error(exc)
        return True, ""

    # -- reading -------------------------------------------------------
    def read(self, message: str, previous: DataRequest | None = None) -> LlmReading:
        """The model's reading of `message`, or LlmUnavailable with the reason.

        Holds no per-call state, so concurrent conversations can share it.
        """
        today = self._today or date.today()
        user = (
            f"تاريخ اليوم: {today.isoformat()}\n"
            f"الطلب السابق في المحادثة: {_describe_previous(previous)}\n"
            f"الرسالة: {message}"
        )
        try:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                betas=BETAS,
                fallbacks="default",
                system=self._system,
                output_config={
                    "effort": EFFORT,
                    "format": {"type": "json_schema", "schema": self._schema},
                },
                messages=[{"role": "user", "content": user}],
            )
        except _api_errors() as exc:
            raise LlmUnavailable(_describe_error(exc)) from exc

        if getattr(response, "stop_reason", None) == "refusal":
            raise LlmUnavailable("رفض النموذج الطلب")
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise LlmUnavailable("انقطع رد النموذج قبل اكتماله")
        text = next(
            (b.text for b in getattr(response, "content", ()) if getattr(b, "type", "") == "text"),
            "",
        )
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LlmUnavailable("رد النموذج ليس JSON صالحاً") from exc
        if not isinstance(data, dict):
            raise LlmUnavailable("رد النموذج ليس كائناً")
        return self._to_reading(message, data, today)

    def _to_reading(self, message: str, data: dict, today: date) -> LlmReading:
        restatement = str(data.get("restatement_ar") or "").strip()[:300]
        if data.get("is_data_request") is False:
            return LlmReading(DataRequest(raw_query=message), False, restatement)

        topic = self._lexicon.get(str(data.get("topic") or ""))

        years = sorted({
            y for y in (data.get("years") or [])
            if isinstance(y, int) and not isinstance(y, bool)
            and EARLIEST_YEAR <= y <= today.year + 1
        })
        kind_name = data.get("period_kind")
        kind = PeriodKind(kind_name) if kind_name in _PERIOD_KINDS else PeriodKind.UNSPECIFIED
        if years:
            kind = PeriodKind.SINGLE if len(years) == 1 else PeriodKind.RANGE
        elif kind in (PeriodKind.SINGLE, PeriodKind.RANGE):
            kind = PeriodKind.UNSPECIFIED  # a year was claimed but none survived

        dimensions: list[Dimension] = []
        for value in data.get("dimensions") or []:
            try:
                dimension = Dimension(value)
            except ValueError:
                continue
            if dimension not in dimensions:
                dimensions.append(dimension)

        phrases = _clean_phrases(data.get("keywords_ar")) + _clean_phrases(
            data.get("keywords_en")
        )
        request = DataRequest(
            raw_query=message,
            period=Period(years=tuple(years), kind=kind, raw=" ".join(map(str, years))),
            topic=topic,
            dimensions=tuple(dimensions),
            free_terms=tuple(phrases),
        )
        return LlmReading(request, bool(data.get("follow_up")), restatement)


def with_rule_years(reading: LlmReading, rules: DataRequest) -> LlmReading:
    """Prefer years the rules parsed: their Hijri conversion is exact."""
    if not rules.period.years:
        return reading
    return replace(reading, request=replace(reading.request, period=rules.period))
