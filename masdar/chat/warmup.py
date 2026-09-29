"""Ask the showcase questions when the server starts, in the background.

A hosted demo sleeps when nobody uses it and wakes with an empty cache, so
the first questions of a presentation would each wait on every source. With
MASDAR_WARMUP_ON_START=1 the server asks them itself as soon as it is up;
the page shows the progress, and marks any example no source answered, so
the presenter knows before the audience does.

The warm-up has its own agent (and HTTP client): the agent is not built for
concurrent use, and a presenter's question must not queue behind it. Both
share the disk cache, which is what makes the later answers immediate.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from masdar.domain.models import Verdict

# An honest "not published" is a good demonstration; a failure is not.
GOOD = (Verdict.AVAILABLE, Verdict.PARTIAL, Verdict.NOT_AVAILABLE)

# A source that was slow at start-up is often back minutes later, so the
# questions it failed are asked again; one it answers is offered again.
RETRY_ROUNDS = 3
RETRY_DELAY_SECONDS = 120.0


def enabled() -> bool:
    return os.environ.get("MASDAR_WARMUP_ON_START", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


@dataclass
class Warmup:
    questions: list[str]
    results: dict[str, bool] = field(default_factory=dict)
    running: bool = False
    retry_rounds: int = RETRY_ROUNDS
    retry_delay: float = RETRY_DELAY_SECONDS
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def status(self) -> dict:
        with self._lock:
            return {
                "total": len(self.questions),
                "done": len(self.results),
                "running": self.running,
                "failed": [q for q, ok in self.results.items() if not ok],
            }

    def run(self, make_agent: Callable[[], object]) -> None:
        with self._lock:
            self.running = True
        try:
            agent = make_agent()
            for question in self.questions:
                ok = _answered(agent, question)
                with self._lock:
                    self.results[question] = ok
        finally:
            with self._lock:
                self.running = False
        # The page shows the first pass as done; these only turn failures
        # into successes, never the other way.
        for _ in range(self.retry_rounds):
            failed = self.status()["failed"]
            if not failed:
                return
            time.sleep(self.retry_delay)
            for question in failed:
                if _answered(agent, question):
                    with self._lock:
                        self.results[question] = True

    def start(self, make_agent: Callable[[], object]) -> threading.Thread:
        thread = threading.Thread(target=self.run, args=(make_agent,), daemon=True,
                                  name="masdar-warmup")
        thread.start()
        return thread


def _answered(agent, question: str) -> bool:
    try:
        return agent.answer(question).verdict in GOOD
    except Exception:  # one broken question must not stop the rest
        return False
