"""Several models behind one reader: the free ones first, each in turn.

A free tier is a quota and a queue: Gemini Flash answers «مزدحم» at busy
hours and runs out of requests in a long session. One model is therefore
not relied on. The configured readers are asked in order -- Gemini Flash,
Gemini Flash-Lite (same key, its own quota), Groq (another company), then
Claude (paid) -- and the first valid reading is used:

* **A passing failure rests that model** (`transient`: a rate limit, a busy
  server, a timeout) for the wait the provider asked for, or a minute, so
  the questions after it do not each wait on it again.
* **Any other failure** (a refusal, a reply outside the schema) moves on
  to the next model for this message only.
* **A time budget** bounds the whole attempt: the rules' reading is always
  there, and a slow model must not delay an answer by more than a few
  seconds.

The chain reads like a single model (`read`, `verify`, `label`), so the rest
of the code does not know how many there are.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable

from masdar.domain.models import DataRequest
from masdar.nlu.lexicon import Lexicon
from masdar.nlu.llm import TIMEOUT_SECONDS, LlmReading, LlmUnavailable, LlmUnderstanding

# How long one message may wait on the models, in total, before the rules'
# reading is used.
BUDGET_SECONDS = 12.0
# Below this, a call is not worth starting.
MIN_CALL_SECONDS = 2.0
# How long a model is rested after a passing failure without its own wait.
REST_SECONDS = 60.0

# Values of MASDAR_LLM_PROVIDER, each naming one provider to use alone.
PROVIDERS = {
    "gemini": "Gemini", "google": "Gemini",
    "groq": "Groq",
    "claude": "Claude", "anthropic": "Claude",
}


class ModelChain:
    """Asks each reader in turn until one gives a valid reading."""

    def __init__(
        self,
        readers: list[LlmUnderstanding],
        budget: float = BUDGET_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not readers:
            raise ValueError("a chain needs at least one reader")
        self.readers = list(readers)
        self.budget = budget
        self._clock = clock
        # Reader label -> when it may be asked again. Shared by every
        # conversation on the page, hence the lock.
        self._resting: dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def label(self) -> str:
        return " ← ".join(r.label for r in self.readers)

    @property
    def model(self) -> str:
        return self.readers[0].model

    # -- start-up check ------------------------------------------------
    def verify(self) -> tuple[bool, str]:
        """Check each reader's key; keep those that work.

        A reader whose key or model does not work is dropped, so no question
        waits on it; the chain works if any is left. The reasons name each
        dropped reader, never a key.
        """
        working, problems = [], []
        for reader in self.readers:
            ok, why = reader.verify()
            if ok:
                working.append(reader)
            else:
                problems.append(f"{reader.label}: {why}")
        if working:
            self.readers = working
        return bool(working), "؛ ".join(problems)

    # -- reading -------------------------------------------------------
    def read(
        self, message: str, previous: DataRequest | None = None, timeout: float | None = None
    ) -> LlmReading:
        """The first valid reading, or LlmUnavailable naming why each failed.

        `timeout` replaces the chain's budget for this message.
        """
        deadline = self._clock() + (timeout or self.budget)
        reasons: list[str] = []
        for reader in self.readers:
            if self._is_resting(reader):
                continue
            left = deadline - self._clock()
            if left < MIN_CALL_SECONDS:
                reasons.append("انتهت المهلة المخصصة للنموذج")
                break
            try:
                return reader.read(message, previous, timeout=min(TIMEOUT_SECONDS, left))
            except LlmUnavailable as exc:
                if exc.transient:
                    self._rest(reader, exc.retry_after)
                reasons.append(self._reason(reader, exc.reason))
            except Exception:  # one reader's fault must not stop the others
                reasons.append(self._reason(reader, "خطأ غير متوقع"))
        if not reasons:
            reasons.append("كل النماذج في استراحة قصيرة بعد ضغط أو انقطاع")
        raise LlmUnavailable("؛ ".join(reasons), transient=True)

    def _reason(self, reader: LlmUnderstanding, why: str) -> str:
        return f"{reader.label}: {why}" if len(self.readers) > 1 else why

    def _is_resting(self, reader: LlmUnderstanding) -> bool:
        with self._lock:
            until = self._resting.get(reader.label)
            if until is None:
                return False
            if self._clock() >= until:
                del self._resting[reader.label]
                return False
            return True

    def _rest(self, reader: LlmUnderstanding, seconds: float | None) -> None:
        with self._lock:
            self._resting[reader.label] = self._clock() + (seconds or REST_SECONDS)


def configured_readers(lexicon: Lexicon) -> tuple[list[LlmUnderstanding], dict[str, str]]:
    """Every reader a key is set for, free ones first; and, by provider name,
    why one whose key is set cannot be used (a library not installed)."""
    from masdar.nlu import gemini, groq
    from masdar.nlu.llm import NO_CLAUDE_KEY

    readers: list[LlmUnderstanding] = [*gemini.readers(lexicon), *groq.readers(lexicon)]
    problems: dict[str, str] = {}
    claude, why = LlmUnderstanding.from_environment(lexicon)
    if claude is not None:
        readers.append(claude)
    elif why != NO_CLAUDE_KEY:
        problems[LlmUnderstanding.NAME] = why
    return readers, problems


def from_environment(lexicon: Lexicon) -> tuple[ModelChain | None, str]:
    """The configured models as one reader, or None; the status in Arabic.

    Every model a key is set for (GEMINI_API_KEY, GROQ_API_KEY,
    ANTHROPIC_API_KEY), free ones first. MASDAR_LLM_PROVIDER (gemini, groq or
    claude) uses that one alone; MASDAR_LLM=off disables them all. Keys are
    read from the environment (or `.env`) and never appear in any message.
    """
    switch = os.environ.get("MASDAR_LLM", "").strip().lower()
    if switch in ("0", "off", "false", "no"):
        return None, "معطّل (MASDAR_LLM=off)"

    readers, problems = configured_readers(lexicon)
    chosen = os.environ.get("MASDAR_LLM_PROVIDER", "").strip().lower()
    if chosen in PROVIDERS:
        provider = PROVIDERS[chosen]
        readers = [r for r in readers if provider == r.NAME]
        problems = {n: why for n, why in problems.items() if n == provider}
    if not readers:
        if problems:
            return None, next(iter(problems.values()))
        if chosen in PROVIDERS:
            return None, f"غير مفعّل — اختير {PROVIDERS[chosen]} (MASDAR_LLM_PROVIDER) ولا مفتاح له"
        return None, "غير مفعّل — أضف GEMINI_API_KEY أو GROQ_API_KEY (مجانيان) لتفعيله"
    chain = ModelChain(readers)
    return chain, f"مفعّل ({chain.label})"
