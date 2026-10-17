"""From a message to a `DataRequest`: rules first, a language model where it helps.

Three things are decided here, for the command line and the chat alike:

* **Follow-ups.** In a conversation, "and 2021?" or "by gender too" is a
  complete question only together with what came before it. The rule is
  deliberately narrow, so a new question is never mistaken for a follow-up:
  a message that names a topic, or carries search words of its own, is a new
  question; a message that only changes the period and/or adds a breakdown
  (after fillers such as «طيب» or «نفس الشي» are set aside) continues the
  previous one.
* **When a model is asked** (MASDAR_LLM_ROLE, if one is configured; see
  `llm.py`, `chain.py`):

  - ``fallback``: only when the rules find no topic;
  - ``assist`` (the default): also when they found one but left words of
    the question unread -- a measure the lexicon lacks («تملك»), a dialect
    word («يشتغلون عن بعد») -- since those are what the search may miss;
  - ``always``: every question.

  A question the rules read completely never leaves the machine (except
  under ``always``).
* **How the readings are merged.** Without a topic from the rules, the
  model's reading is the request, with the rules' years and typed phrases;
  its phrases are then both the subject and, as `expansions`, the wording
  the sources are searched with. With a topic from the rules, their reading
  stands -- its topic, phrases and years are deterministic and auditable --
  and the model only adds: breakdowns the rules missed, a period if they
  found none, and its publisher-style phrases as `expansions`, which widen
  the search but never decide whether a result is on the subject.

Any failure of the model leaves the rules' reading in place, and says why.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import date

from masdar.domain.models import DataRequest, Dimension, PeriodKind
from masdar.nlu.lexicon import Lexicon
from masdar.nlu.llm import LlmReading, LlmUnavailable, with_rule_years
from masdar.nlu.normalize import normalize
from masdar.nlu.parser import parse

# Words that carry no search meaning in a follow-up. Normalised form.
_FILLERS = frozenset(
    normalize(word)
    for word in (
        "طيب", "طب", "اوكي", "تمام", "نفس", "الشي", "الشيء", "الشئ", "بس", "كذا",
        "هذا", "هذي", "هذه", "نفسها", "نفسه", "ايضا", "أيضاً", "كمان", "برضو", "بعد",
        "وش", "ايش", "ماذا", "عن", "ولو", "لو", "يعني", "البيانات", "بيانات",
        "اعطني", "أعطني", "عطني", "هات", "ابي", "أبي", "ابغى", "أبغى", "اريد", "أريد",
    )
)
# What is measured, which the ranking reads from the question itself
# (`resolve.shared_measures`): left over, it is not a word the rules missed.
_MEASURES = frozenset(
    normalize(word)
    for word in ("عدد", "متوسط", "إجمالي", "اجمالي", "مجموع", "نسبة", "معدل", "حجم",
                 "كمية", "قيمة", "غير", "number", "total", "rate", "average", "share",
                 "percentage", "percent", "count", "amount", "value", "volume")
)

# How the request was read: by the rules alone; by a model standing in for
# them (no topic); or by the rules with a model's additions.
RULES = "rules"
LLM = "llm"
ASSISTED = "assisted"

FALLBACK = "fallback"
ASSIST = "assist"
ALWAYS = "always"
ROLES = (FALLBACK, ASSIST, ALWAYS)
DEFAULT_ROLE = ASSIST
# A model only adding to the rules' reading gets less time than one standing
# in for it: the rules' reading is already enough to search with.
ASSIST_SECONDS = 6.0
# Asking for other wordings once a search found nothing.
REPHRASE_SECONDS = 6.0

READ_BY_RULES = "قُرئ السؤال بالمحلل العربي"


@dataclass(frozen=True)
class Understanding:
    request: DataRequest
    follow_up: bool
    method: str = RULES
    # The model's restatement, or why the model could not help. For display
    # and the audit trail; never used to decide anything.
    note: str = ""
    # Other ways to ask for the same data, from the model; offered when
    # nothing is found.
    alternatives: tuple[str, ...] = ()
    # Whether a model was asked at all (it may have failed).
    model_asked: bool = False
    # Which model read it, when one did.
    model: str = ""


def model_role() -> str:
    """MASDAR_LLM_ROLE, or the default for an unset or unknown value."""
    role = os.environ.get("MASDAR_LLM_ROLE", "").strip().lower()
    return role if role in ROLES else DEFAULT_ROLE


def is_follow_up(request: DataRequest) -> bool:
    if request.topic is not None:
        return False
    meaningful = [t for t in request.free_terms if normalize(t) not in _FILLERS]
    if meaningful:
        return False
    return bool(request.period.kind is not PeriodKind.UNSPECIFIED or request.dimensions)


def unread_words(request: DataRequest) -> tuple[str, ...]:
    """Words of the question the rules kept but could not place."""
    return tuple(
        t for t in request.free_terms
        if len(normalize(t)) >= 3 and not t.isdigit()
        and normalize(t) not in _FILLERS and normalize(t) not in _MEASURES
    )


def needs_model(request: DataRequest, role: str) -> bool:
    if role == ALWAYS or request.topic is None:
        return True
    return role == ASSIST and bool(unread_words(request))


def continue_request(previous: DataRequest, current: DataRequest, root: str) -> DataRequest:
    """The previous question with this message's period and breakdowns applied."""
    period = (
        current.period if current.period.kind is not PeriodKind.UNSPECIFIED else previous.period
    )
    dimensions = previous.dimensions + tuple(
        d for d in current.dimensions if d not in previous.dimensions
    )
    return replace(
        previous,
        # The workbook records the question "as asked"; for a follow-up that
        # is both what was typed and what it continued.
        raw_query=f"{current.raw_query} (متابعة لـ «{root}»)",
        period=period,
        dimensions=dimensions,
        places=current.places or previous.places,
    )


def assisted(rules: DataRequest, model: DataRequest) -> DataRequest:
    """The rules' reading, with what the model adds and nothing it replaces."""
    known = {normalize(t) for t in (*rules.typed_phrases, *rules.free_terms)}
    expansions = tuple(t for t in model.free_terms if normalize(t) not in known)
    period = rules.period if rules.period.kind is not PeriodKind.UNSPECIFIED else model.period
    dimensions = rules.dimensions + tuple(d for d in model.dimensions if d not in rules.dimensions)
    return replace(rules, period=period, dimensions=dimensions, expansions=expansions)


def describe_additions(rules: DataRequest, merged: DataRequest) -> str:
    """What the model added to the rules' reading, in Arabic; empty if nothing."""
    parts = []
    if merged.expansions:
        parts.append("صياغات للبحث: " + "، ".join(f"«{t}»" for t in merged.expansions))
    added = [d.label_ar for d in merged.dimensions if d not in rules.dimensions]
    if added:
        parts.append("التفصيل: " + "، ".join(added))
    if merged.period != rules.period:
        parts.append(f"الفترة: {merged.period.label()}")
    return "؛ ".join(parts)


def understand(
    message: str,
    lexicon: Lexicon,
    previous: DataRequest | None = None,
    root: str = "",
    llm=None,
    today: date | None = None,
    role: str | None = None,
) -> Understanding:
    current = parse(message, lexicon, today)
    has_context = previous is not None and previous.is_answerable

    if has_context and is_follow_up(current):
        return Understanding(continue_request(previous, current, root), True)
    role = role or model_role()
    if llm is None or not needs_model(current, role):
        return Understanding(current, False)

    standing_in = current.topic is None
    try:
        reading: LlmReading = llm.read(
            message, previous if has_context else None,
            timeout=None if standing_in else ASSIST_SECONDS,
        )
    # The model is an aid: without it the question is read by the rules, and
    # the answer says so without alarm.
    except LlmUnavailable as exc:
        return Understanding(current, False, note=f"{READ_BY_RULES}؛ {exc.reason}",
                             model_asked=True)
    except Exception:  # its failure must not sink the answer
        return Understanding(current, False, note=f"{READ_BY_RULES}؛ تعذّر الاتصال بالنموذج",
                             model_asked=True)

    def understood(request: DataRequest, follow_up: bool = False, method: str = LLM,
                   note: str | None = None) -> Understanding:
        return Understanding(request, follow_up, method,
                             reading.restatement if note is None else note,
                             reading.alternatives, True, reading.model)

    if not standing_in:
        merged = assisted(current, reading.request)
        added = describe_additions(current, merged)
        # Asked, and nothing to add: the rules' reading, as it was.
        return (understood(merged, method=ASSISTED, note=added) if added
                else understood(current, method=RULES, note=""))

    reading = with_rule_years(reading, current)
    # The rules' places stand: the model is not asked for them.
    dimensions = reading.request.dimensions
    if current.places and Dimension.REGION not in dimensions:
        dimensions = (*dimensions, Dimension.REGION)
    request = replace(reading.request, typed_phrases=current.typed_phrases,
                      expansions=reading.request.free_terms, places=current.places,
                      dimensions=dimensions)
    if reading.follow_up and has_context and request.topic is None:
        return understood(continue_request(previous, request, root), True)
    return understood(request)


def rephrasings(understood: Understanding, message: str, llm) -> tuple[str, ...]:
    """Other ways to ask, for an answer that found nothing.

    The model's own, if it read the question; otherwise it is asked now --
    once, briefly -- unless it was already asked and failed, or the message
    was a follow-up (too short to reword on its own). Never raises.
    """
    if understood.alternatives or understood.model_asked or understood.follow_up:
        return understood.alternatives
    if llm is None:
        return ()
    try:
        return llm.read(message, None, timeout=REPHRASE_SECONDS).alternatives
    except Exception:  # an aid only: no suggestions is a fine answer
        return ()
