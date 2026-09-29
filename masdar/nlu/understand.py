"""From a message to a `DataRequest`: rules first, a language model only
when the rules cannot tell what is being asked about.

Two things are decided here, for the command line and the chat alike:

* **Follow-ups.** In a conversation, "and 2021?" or "by gender too" is a
  complete question only together with what came before it. The rule is
  deliberately narrow, so a new question is never mistaken for a follow-up:
  a message that names a topic, or carries search words of its own, is a new
  question; a message that only changes the period and/or adds a breakdown
  (after fillers such as «طيب» or «نفس الشي» are set aside) continues the
  previous one.
* **The fallback to Claude.** Only when the rules find no topic, and only if
  it is configured (see `llm.py`). Its reading goes through the same
  follow-up rule, and any failure leaves the rules' reading in place.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from masdar.domain.models import DataRequest, PeriodKind
from masdar.nlu.lexicon import Lexicon
from masdar.nlu.llm import LlmUnavailable, with_rule_years
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

RULES = "rules"
LLM = "llm"


@dataclass(frozen=True)
class Understanding:
    request: DataRequest
    follow_up: bool
    method: str = RULES
    # The model's restatement, or why the model could not help. For display
    # and the audit trail; never used to decide anything.
    note: str = ""


def is_follow_up(request: DataRequest) -> bool:
    if request.topic is not None:
        return False
    meaningful = [t for t in request.free_terms if normalize(t) not in _FILLERS]
    if meaningful:
        return False
    return bool(request.period.kind is not PeriodKind.UNSPECIFIED or request.dimensions)


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


def understand(
    message: str,
    lexicon: Lexicon,
    previous: DataRequest | None = None,
    root: str = "",
    llm=None,
) -> Understanding:
    current = parse(message, lexicon)
    has_context = previous is not None and previous.is_answerable

    if has_context and is_follow_up(current):
        return Understanding(continue_request(previous, current, root), True)
    if current.topic is not None or llm is None:
        return Understanding(current, False)

    try:
        reading = llm.read(message, previous if has_context else None)
    except LlmUnavailable as exc:
        return Understanding(current, False, note=f"تعذّر الفهم بالنموذج: {exc.reason}")
    except Exception as exc:  # the model is an aid; its failure must not sink the answer
        return Understanding(current, False, note=f"تعذّر الفهم بالنموذج: {type(exc).__name__}")

    reading = with_rule_years(reading, current)
    request = replace(reading.request, typed_phrases=current.typed_phrases)
    if reading.follow_up and has_context and request.topic is None:
        return Understanding(
            continue_request(previous, request, root), True, LLM, reading.restatement
        )
    return Understanding(request, False, LLM, reading.restatement)
