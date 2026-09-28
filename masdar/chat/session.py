"""A conversation: each message is understood in the light of the last one.

The rules for what counts as a follow-up live in `nlu/understand.py`, shared
with the command line; this module only keeps the state between messages.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from masdar.domain.models import DataRequest
from masdar.nlu.lexicon import Lexicon
from masdar.nlu.understand import RULES, understand


@dataclass
class Turn:
    request: DataRequest
    follow_up: bool
    method: str = RULES
    note: str = ""


@dataclass
class Conversation:
    lexicon: Lexicon
    llm: object | None = None
    last: DataRequest | None = None
    # The message that opened the current line of questioning.
    root: str = ""
    turns: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def understand(self, message: str) -> Turn:
        self.turns += 1
        result = understand(message, self.lexicon, self.last, self.root, self.llm)
        if result.follow_up:
            # Chained follow-ups build on each other: "2021", then "by gender".
            self.last = result.request
        elif result.request.is_answerable:
            self.last = result.request
            self.root = message.strip()
        return Turn(result.request, result.follow_up, result.method, result.note)

    def reset(self) -> None:
        self.last = None
        self.root = ""


class Sessions:
    """Conversations by id, the oldest forgotten beyond `capacity`."""

    def __init__(self, lexicon: Lexicon, capacity: int = 200, llm: object | None = None):
        self._lexicon = lexicon
        self._llm = llm
        self._capacity = capacity
        self._items: OrderedDict[str, Conversation] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str | None) -> tuple[str, Conversation]:
        with self._lock:
            if session_id and session_id in self._items:
                self._items.move_to_end(session_id)
                return session_id, self._items[session_id]
            new_id = uuid.uuid4().hex
            self._items[new_id] = Conversation(self._lexicon, self._llm)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)
            return new_id, self._items[new_id]
