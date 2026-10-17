"""What a language model is asked about a question, and what is kept of it.

The rule-based parser is fast, free and deterministic, but it only knows
the words in its lexicon: «كم عدد الخيول المسجلة؟» names no topic it knows.
A model turns such a question into the same `DataRequest` the rules
produce -- a topic from the fixed list, years, breakdowns and search
phrases -- plus other wordings of the question to offer if nothing is found,
and nothing else.

This module holds the contract every model is held to -- the instructions,
the JSON schema, the validation in `LlmUnderstanding._to_reading` -- and the
Claude reader. `gemini.py` and `groq.py` read through the same contract over
HTTPS; `chain.py` tries them in turn, free ones first; `understand.py`
decides when a model is asked and how its reading is merged with the rules'.

What it is never asked to do matters as much:

* It never answers the question and never produces a number. Availability,
  coverage and every figure still come from the sources and the evidence
  rules in `pipeline/verify.py`.
* Its output is constrained to a JSON schema and then validated again here:
  an unknown topic, a breakdown outside the enum or an implausible year is
  dropped, not trusted.
* Any failure -- no key, no library, network, refusal, malformed output --
  falls back to the rules, and the reason is recorded for the audit trail.

A question the rules read completely never leaves the machine. When a
model is asked, the question text (never a key or a file) is sent to that
model's provider.
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
# Reading a question takes the model a second or two. Waiting longer only
# delays the answer: the rules' reading is used instead (one question once
# took 86 s, 30 of them waiting on a stalled model before any search began).
TIMEOUT_SECONDS = 10.0

MAX_PHRASES = 5
MAX_ALTERNATIVES = 3
API_MESSAGE_CHARS = 160
MAX_PHRASE_CHARS = 60
MAX_ALTERNATIVE_CHARS = 120
EARLIEST_YEAR = 1900

# Longest a provider's own Retry-After is honoured: beyond that it is rested
# for this long and asked again.
MAX_RETRY_AFTER = 600.0

_PERIOD_KINDS = ("single", "range", "latest", "unspecified")

NO_CLAUDE_KEY = "غير مفعّل — أضف ANTHROPIC_API_KEY لتفعيله"


def model_from_env(family: str) -> str:
    """MASDAR_LLM_MODEL, if it names a model of this family («claude», «gemini»).

    A model left over from the other provider is ignored rather than sent
    where it cannot exist.
    """
    model = os.environ.get("MASDAR_LLM_MODEL", "").strip()
    return model if model.lower().startswith(family) else ""


class LlmUnavailable(Exception):
    """The model could not help with this message; `reason` says why, in Arabic.

    `transient` marks a failure that passes on its own -- a rate limit, a busy
    or unreachable server, a timeout -- as opposed to a wrong key or a reply
    that broke the contract. `retry_after` is the provider's own wait, in
    seconds, when it gives one. `chain.ModelChain` uses both to rest a
    provider for a while instead of making every question wait on it.
    """

    def __init__(self, reason: str, transient: bool = False, retry_after: float | None = None):
        super().__init__(reason)
        self.reason = reason
        self.transient = transient
        self.retry_after = retry_after


@dataclass(frozen=True)
class LlmReading:
    request: DataRequest
    follow_up: bool
    # One Arabic sentence saying how the question was read; shown to the user
    # so a misreading is visible before anyone relies on the file.
    restatement: str = ""
    # Other ways to ask for the same data, offered when nothing is found.
    alternatives: tuple[str, ...] = ()
    # Which model read it («Gemini · gemini-3.8-flash»), for the audit trail.
    model: str = ""


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
            "alternatives_ar": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "is_data_request", "follow_up", "topic", "keywords_ar", "keywords_en",
            "period_kind", "years", "dimensions", "restatement_ar", "alternatives_ar",
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
the publisher would title the dataset, most specific first. Official titles name \
the measured thing in Modern Standard Arabic, often as a plural or a verbal noun, \
whatever the dialect of the question: for «كم صيدلية بالرياض» give «الصيدليات» and \
«عدد الصيدليات»; for «كم صرفت البيوت كهرباء» give «استهلاك الكهرباء» and «الاستهلاك \
السكني للكهرباء»; for «تملك البيوت» give «ملكية المساكن». GASTAT names a field by \
its subject: «الحج» for pilgrims, «القوى العاملة» for workers, «التجارة الدولية» for \
exports and imports. Do not include years, places, or words like «احصائيات» or \
«بيانات».
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
understood it.
- alternatives_ar: up to {MAX_ALTERNATIVES} short Arabic questions asking for the \
same need in other words an official statistics publisher would use, each complete \
on its own (subject, and the period if one was asked). They are offered to the \
user when the search finds nothing, so make them likely to be found, not \
paraphrases of the message. Empty when the message is not a data request."""


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


def _clean_alternatives(values: object, message: str) -> tuple[str, ...]:
    """Other questions to offer: plain sentences, not the message itself again."""
    if not isinstance(values, list):
        return ()
    asked = " ".join(message.split())
    kept: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        question = " ".join(value.split())
        if 4 <= len(question) <= MAX_ALTERNATIVE_CHARS and question not in (asked, *kept):
            kept.append(question)
    return tuple(kept[:MAX_ALTERNATIVES])


def _api_errors() -> tuple[type[BaseException], ...]:
    try:
        import anthropic
    except ImportError:
        return ()
    return (anthropic.APIError,)


def _unavailable(exc: BaseException) -> LlmUnavailable:
    """An Anthropic SDK error as the reason a model could not help, in Arabic."""
    try:
        import anthropic
    except ImportError:
        return LlmUnavailable(type(exc).__name__)
    if isinstance(exc, anthropic.AuthenticationError):
        return LlmUnavailable("مفتاح Anthropic غير صالح أو منتهي")
    if isinstance(exc, anthropic.PermissionDeniedError):
        return LlmUnavailable("المفتاح لا يملك صلاحية هذا النموذج")
    if isinstance(exc, anthropic.NotFoundError):
        return LlmUnavailable("النموذج غير متاح لهذا المفتاح")
    if isinstance(exc, anthropic.RateLimitError):
        return LlmUnavailable("تجاوز حد الطلبات لدى Anthropic", transient=True,
                              retry_after=retry_after(getattr(exc, "response", None)))
    if isinstance(exc, anthropic.APITimeoutError):
        return LlmUnavailable("انتهت مهلة الاتصال بـ Anthropic", transient=True)
    if isinstance(exc, anthropic.APIConnectionError):
        return LlmUnavailable("تعذّر الاتصال بـ Anthropic", transient=True)
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code >= 500:
            # 529 is "overloaded": Anthropic's side, and passing.
            return LlmUnavailable("خدمة Anthropic مشغولة الآن", transient=True)
        detail = _api_message(exc)
        # The commonest 400 for a new key: looking the model up is free, so
        # the start-up check passes, and every paid request is then refused.
        if "credit balance" in detail.lower():
            return LlmUnavailable(
                "رصيد حساب Anthropic لا يكفي — اشحنه من صفحة Billing في حسابك")
        suffix = f": {detail}" if detail else ""
        return LlmUnavailable(f"خطأ من Anthropic ({exc.status_code}){suffix}")
    return LlmUnavailable(type(exc).__name__)


def retry_after(response) -> float | None:
    """The seconds a provider asked to wait (its Retry-After header), if any."""
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        seconds = float(headers.get("retry-after") or "")
    except (TypeError, ValueError):
        return None
    return min(seconds, MAX_RETRY_AFTER) if seconds > 0 else None


def _api_message(exc: BaseException) -> str:
    """The API's own one-line explanation of an error, shortened for display.

    Error bodies describe the request, never the credential, so they are safe
    to show; the length cap keeps a long one from flooding the answer.
    """
    body = getattr(exc, "body", None)
    message = ""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "")
    if not message:
        message = str(getattr(exc, "message", "") or "")
    return " ".join(message.split())[:API_MESSAGE_CHARS]


class LlmUnderstanding:
    """Maps a free-form question to a `DataRequest` with Claude.

    The base of the other readers too: they change how the model is called
    (`_ask`, `verify`), never what is asked or what is kept.
    """

    # Shown in the status line and the audit trail; VENDOR in error messages.
    NAME = "Claude"
    VENDOR = "Anthropic"

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

    @property
    def label(self) -> str:
        return f"{self.NAME} · {self.model}"

    # -- construction --------------------------------------------------
    @classmethod
    def from_environment(cls, lexicon: Lexicon) -> tuple[LlmUnderstanding | None, str]:
        """The Claude reader, or None with the reason in Arabic.

        Only Claude: which models are used, and in what order, is
        `chain.from_environment`'s to decide. The key is read from the
        environment (or `.env`) by the SDK and never appears in any message;
        MASDAR_LLM=on uses the SDK's other credentials without a key here.
        """
        from masdar.envfile import secret_from_env

        has_credential = bool(
            secret_from_env("ANTHROPIC_API_KEY") or secret_from_env("ANTHROPIC_AUTH_TOKEN")
        )
        switch = os.environ.get("MASDAR_LLM", "").strip().lower()
        if not has_credential and switch not in ("1", "on", "true", "yes"):
            return None, NO_CLAUDE_KEY
        try:
            import anthropic
        except ImportError:
            return None, "مكتبة anthropic غير مثبتة — pip install -e '.[ai]'"
        model = model_from_env("claude") or DEFAULT_MODEL
        client = anthropic.Anthropic(timeout=TIMEOUT_SECONDS, max_retries=1)
        reader = cls(client, lexicon, model=model)
        return reader, f"مفعّل ({reader.label})"

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
            return False, _unavailable(exc).reason
        return True, ""

    # -- reading -------------------------------------------------------
    def read(
        self, message: str, previous: DataRequest | None = None, timeout: float | None = None
    ) -> LlmReading:
        """The model's reading of `message`, or LlmUnavailable with the reason.

        `timeout` caps the wait for this one call (the chain's time left).
        Holds no per-call state, so concurrent conversations can share it.
        """
        today = self._today or date.today()
        user = (
            f"تاريخ اليوم: {today.isoformat()}\n"
            f"الطلب السابق في المحادثة: {_describe_previous(previous)}\n"
            f"الرسالة: {message}"
        )
        text = self._ask(user, timeout)
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LlmUnavailable("رد النموذج ليس JSON صالحاً") from exc
        if not isinstance(data, dict):
            raise LlmUnavailable("رد النموذج ليس كائناً")
        return self._to_reading(message, data, today)

    def _ask(self, user: str, timeout: float | None = None) -> str:
        """The model's JSON reply to one message, or LlmUnavailable with the reason."""
        options = {"timeout": timeout} if timeout else {}
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
                **options,
            )
        except _api_errors() as exc:
            raise _unavailable(exc) from exc

        if getattr(response, "stop_reason", None) == "refusal":
            raise LlmUnavailable("رفض النموذج الطلب")
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise LlmUnavailable("انقطع رد النموذج قبل اكتماله")
        return next(
            (b.text for b in getattr(response, "content", ()) if getattr(b, "type", "") == "text"),
            "",
        )

    def _to_reading(self, message: str, data: dict, today: date) -> LlmReading:
        restatement = str(data.get("restatement_ar") or "").strip()[:300]
        if data.get("is_data_request") is False:
            return LlmReading(DataRequest(raw_query=message), False, restatement,
                              model=self.label)

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
        return LlmReading(
            request,
            bool(data.get("follow_up")),
            restatement,
            _clean_alternatives(data.get("alternatives_ar"), message),
            self.label,
        )


def with_rule_years(reading: LlmReading, rules: DataRequest) -> LlmReading:
    """Prefer years the rules parsed: their Hijri conversion is exact."""
    if not rules.period.years:
        return reading
    return replace(reading, request=replace(reading.request, period=rules.period))
