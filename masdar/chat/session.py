"""A conversation: each message is understood in the light of the last one.

In a chat, "and 2021?" or "by gender too" is a complete question only
together with what came before it. The rule is deliberately narrow, so a
new question is never mistaken for a follow-up:

* a message that names a topic, or carries search words of its own, is a new
  question;
* a message that only changes the period and/or adds a breakdown (after
  conversational fillers such as «طيب» or «نفس الشي» are set aside) continues
  the previous question.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, replace

from masdar.domain.models import DataRequest, PeriodKind
from masdar.nlu.lexicon import Lexicon
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


def _is_follow_up(request: DataRequest) -> bool:
    if request.topic is not None:
        return False
    meaningful = [t for t in request.free_terms if normalize(t) not in _FILLERS]
    if meaningful:
        return False
    return bool(
        request.period.kind is not PeriodKind.UNSPECIFIED or request.dimensions
    )


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
    )


@dataclass
class Turn:
    request: DataRequest
    follow_up: bool


@dataclass
class Conversation:
    lexicon: Lexicon
    last: DataRequest | None = None
    # The message that opened the current line of questioning.
    root: str = ""
    turns: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def understand(self, message: str) -> Turn:
        current = parse(message, self.lexicon)
        self.turns += 1
        if self.last is not None and self.last.is_answerable and _is_follow_up(current):
            request = continue_request(self.last, current, self.root)
            # Chained follow-ups build on each other: "2021", then "by gender".
            self.last = request
            return Turn(request=request, follow_up=True)
        if current.is_answerable:
            self.last = current
            self.root = message.strip()
        return Turn(request=current, follow_up=False)

    def reset(self) -> None:
        self.last = None
        self.root = ""


class Sessions:
    """Conversations by id, the oldest forgotten beyond `capacity`."""

    def __init__(self, lexicon: Lexicon, capacity: int = 200):
        self._lexicon = lexicon
        self._capacity = capacity
        self._items: OrderedDict[str, Conversation] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str | None) -> tuple[str, Conversation]:
        with self._lock:
            if session_id and session_id in self._items:
                self._items.move_to_end(session_id)
                return session_id, self._items[session_id]
            new_id = uuid.uuid4().hex
            self._items[new_id] = Conversation(self._lexicon)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)
            return new_id, self._items[new_id]
