from collections.abc import Callable
from logging import getLogger
from threading import Lock

from speech_to_speech.pipeline.transcript_logging import log_exception

logger = getLogger(__name__)


class CancelScope:
    """Unified cancellation signal for the speech-to-speech pipeline.

    Uses a generation counter so pipeline threads (LLM, TTS) can detect
    cancellation without brief-pulse timing games, and an internal
    ``discarding`` flag so the async send loop can drop stale output.

    Thread safety: the generation counter and ``discarding`` flag have one
    writer (the asyncio router thread) and multiple readers (pipeline handler
    threads).  Python's GIL makes int/bool reads and writes atomic at the
    bytecode level, so those need no lock.

    Abort callbacks are the exception, and carry ``_abort_lock``: a pipeline
    thread arms one from *its* thread (``register_abort``) while the router
    thread runs and clears them from ``cancel``, so the two races.  Polling
    ``is_stale`` cannot interrupt a read already blocked inside a provider
    stream that has gone quiet; a registered abort closes that stream out from
    under the read instead, the way ``ResponsePrefetchTransaction.discard``
    does for a speculative prefetch.
    """

    def __init__(self) -> None:
        self._gen: int = 0
        self._discarding: bool = False
        self._discarded_generation: int | None = None
        self._abort_lock = Lock()
        self._abort_callbacks: list[Callable[[], None]] = []

    @property
    def generation(self) -> int:
        """Current generation number.  Pipeline threads capture this at
        the start of each response and compare with ``is_stale``."""
        return self._gen

    def cancel(self) -> None:
        """Cancel the current response.

        Increments the generation (so pipeline threads see their captured
        generation as stale), enables the send-loop discard guard, and runs
        every armed abort so a read blocked inside a provider stream is closed
        at once rather than lingering until the provider's next event.
        """
        with self._abort_lock:
            # prevent overflow... after 4 billion generations, we'll wrap around xD...
            self._discarded_generation = self._gen
            self._gen = (self._gen + 1) & 0xFFFFFFFF
            self._discarding = True
            # Snapshot under the lock so a concurrent register_abort either
            # lands in this batch or sees the bumped generation and runs itself.
            callbacks = self._abort_callbacks
            self._abort_callbacks = []
        for abort in callbacks:
            self._run_abort(abort)

    def register_abort(self, gen: int, abort: Callable[[], None]) -> None:
        """Arm an abort for an in-flight read tagged with generation ``gen``.

        Called from the pipeline thread that owns the read, once its provider
        response exists.  If ``gen`` was already superseded between the caller
        capturing it and arming here, the abort runs immediately so a read that
        has only just begun is still interrupted; otherwise ``cancel`` runs it.
        """
        run_now = False
        with self._abort_lock:
            if gen != self._gen:
                run_now = True
            else:
                self._abort_callbacks.append(abort)
        if run_now:
            self._run_abort(abort)

    def unregister_abort(self, abort: Callable[[], None]) -> None:
        """Disarm an abort whose read completed or failed on its own."""
        with self._abort_lock:
            self._abort_callbacks = [cb for cb in self._abort_callbacks if cb is not abort]

    @staticmethod
    def _run_abort(abort: Callable[[], None]) -> None:
        try:
            abort()
        except Exception as exc:
            # Cancellation must still reach the rest of teardown even when a
            # provider raises while releasing its stream.
            log_exception(logger, "Failed to abort cancelled in-flight response", exc)

    def response_done(self, generation: int | None = None) -> None:
        """Pipeline acknowledged completion.  Clears the discard guard."""
        if (
            generation is not None
            and self._discarded_generation is not None
            and generation not in {self._discarded_generation, self._gen}
        ):
            return
        self._discarding = False
        self._discarded_generation = None

    def new_response(self) -> None:
        """An explicit ``response.create`` starts a new response.
        Clears the discard guard."""
        self._discarding = False
        self._discarded_generation = None

    def is_stale(self, gen: int) -> bool:
        """Return True if *gen* has been superseded by a ``cancel`` call."""
        return gen != self._gen

    @property
    def discarding(self) -> bool:
        """Whether the send loop should silently drop stale output."""
        return self._discarding

    def reset(self) -> None:
        """Clear discard state (e.g. on new session connect)."""
        self._discarding = False
        self._discarded_generation = None
