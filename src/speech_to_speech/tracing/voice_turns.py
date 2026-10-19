"""Per-turn voice traces: the config gate, the tracer holder and the export registry.

One trace per spoken turn, session = call id (Ian, 2026-10-19; SOP-838). Each
traced turn gets one parentless ``voice_turn`` root span whose ids Sophie
derives too (:mod:`speech_to_speech.tracing.voice_ids`), and its stage spans
(VAD, STT, LLM, TTS, e2e) as direct children. Sophie's LLM proxy hangs its
own spans under the same root through the ``traceparent`` the model request
carries.

This module decides what to export and when. :mod:`.otel_backend` is the only
module that imports OpenTelemetry, and it is imported only once all three
``LANGFUSE_*`` variables are set and valid. With tracing off nothing from the
OTel SDK is loaded, no ref is ever built, and so no turn header is sent.

Threading: a :class:`VoiceTurnTraces` is used from the realtime event loop
(bind, response terminal, unregister), from whichever thread clears a pooled
unit's call id, and from shutdown. Its lock guards bookkeeping only; spans
are built outside it, and building one only enqueues it on the backend's
bounded batch queue, so nothing here waits on the network. Every public
method swallows its own failures: tracing must never take a turn down.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlsplit

from speech_to_speech.tracing.voice_ids import (
    TURN_ID_RE,
    format_traceparent,
    mint_epoch,
    normalize_call_uuid,
    voice_turn_root_span_id,
    voice_turn_trace_id,
    wire_turn_id,
)

if TYPE_CHECKING:
    from speech_to_speech.pipeline.turn_latency import TurnLatencyTracker

logger = logging.getLogger(__name__)

# The variable names Sophie reads (ops/LANGFUSE.md); the values are never logged.
ENV_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
ENV_SECRET_KEY = "LANGFUSE_SECRET_KEY"
ENV_BASE_URL = "LANGFUSE_BASE_URL"
ENV_ENVIRONMENT = "SOPHIE_ENVIRONMENT"
OTLP_TRACES_PATH = "/api/public/otel/v1/traces"

# Sophie's environment rule (internal/tracing/tracing.go), so an s2s span and
# the proxy spans of the same turn land in the same Langfuse environment.
DEFAULT_ENVIRONMENT = "dev"
UNKNOWN_ENVIRONMENT = "unknown"
_ENVIRONMENT_RE = re.compile(r"[a-z0-9_-]{1,40}")
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# Each names variables, never their values or the URL.
PARTIAL_CONFIG_LINE = "Voice tracing off: %s unset (set all three LANGFUSE_* variables, or none)"
BAD_BASE_URL_LINE = (
    "Voice tracing off: LANGFUSE_BASE_URL must be an https URL without credentials (http only for localhost)"
)
BAD_ENVIRONMENT_LINE = (
    "Voice tracing: SOPHIE_ENVIRONMENT must be 1-40 of [a-z0-9_-] and not start with langfuse; using %s"
)
MISSING_EXTRA_LINE = "Voice tracing off: LANGFUSE_* is set but the [tracing] extra is not installed"
BACKEND_FAILED_LINE = "Voice tracing off: the exporter could not be built (%s)"
TRACING_ON_LINE = "Voice tracing on: per-turn spans export to Langfuse (environment %s)"
EMIT_FAILED_LINE = "Voice tracing: a span export failed (%s); later failures on this unit are not logged"

# Span and attribute names. Sophie's proxy names its turn spans the same way.
TRACE_NAME = "voice_turn"
ROOT_SPAN_NAME = "voice_turn"
KEY_SESSION_ID = "langfuse.session.id"
KEY_TRACE_NAME = "langfuse.trace.name"
KEY_ENVIRONMENT = "langfuse.environment"
KEY_TURN_ID = "sophie.voice.turn_id"
KEY_TURN_REVISION = "s2s.turn_revision"
KEY_SOURCE = "s2s.source"
KEY_STATUS = "s2s.status"


def validate_endpoint_base(raw: str) -> str | None:
    """Sophie's ``ValidateEndpointBase``, ported: the base with trailing '/' trimmed, or None.

    https only, so credentials never cross a network in clear text; http only
    for a loopback host; no userinfo; a host is required.
    """
    trimmed = raw.rstrip("/")
    try:
        parts = urlsplit(trimmed)
        host = parts.hostname
        _ = parts.port  # a malformed port raises, as Go's url.Parse rejects it
    except ValueError:
        return None
    if not host or "@" in parts.netloc:
        return None
    if parts.scheme == "https" or (parts.scheme == "http" and host in _LOOPBACK_HOSTS):
        return trimmed
    return None


def _environment(raw: str | None) -> str:
    value = (raw or "").strip()
    if not value:
        return DEFAULT_ENVIRONMENT
    if _ENVIRONMENT_RE.fullmatch(value) and not value.startswith("langfuse"):
        return value
    logger.warning(BAD_ENVIRONMENT_LINE, UNKNOWN_ENVIRONMENT)
    return UNKNOWN_ENVIRONMENT


@dataclass(frozen=True)
class TracingConfig:
    """Where and as whom to export. Built only from a complete, valid environment."""

    public_key: str = field(repr=False)
    secret_key: str = field(repr=False)
    endpoint: str
    environment: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> TracingConfig | None:
        """None when tracing is off. Empty or blank counts as unset.

        All three unset is the ordinary "off" and logs nothing. A partial set or
        an unusable base URL logs exactly one WARNING and is also off.
        """
        values = {name: (environ.get(name) or "").strip() for name in (ENV_PUBLIC_KEY, ENV_SECRET_KEY, ENV_BASE_URL)}
        missing = [name for name, value in values.items() if not value]
        if len(missing) == len(values):
            return None
        if missing:
            logger.warning(PARTIAL_CONFIG_LINE, ", ".join(missing))
            return None
        base = validate_endpoint_base(values[ENV_BASE_URL])
        if base is None:
            logger.warning(BAD_BASE_URL_LINE)
            return None
        return cls(
            public_key=values[ENV_PUBLIC_KEY],
            secret_key=values[ENV_SECRET_KEY],
            endpoint=base + OTLP_TRACES_PATH,
            environment=_environment(environ.get(ENV_ENVIRONMENT)),
        )


@dataclass(frozen=True)
class TurnTraceRef:
    """One response's place in its turn's trace, fixed when the response is bound.

    It carries its own call id, so a response that finishes after its pooled
    unit moved to another call still lands on its original call's trace, and
    the request path can refuse to send it under a different call's id.
    """

    call_uuid: str
    wire_turn_id: str
    turn_id: str
    revision: int | None
    source: str
    trace_id: bytes
    root_span_id: bytes
    traceparent: str


@dataclass(frozen=True)
class StageSpan:
    """A finished stage, in epoch nanoseconds, ready to export."""

    name: str
    start_ns: int
    end_ns: int
    attributes: dict[str, Any]


@dataclass(frozen=True)
class StageInterval:
    """A tracker measurement: its duration and where on perf_counter it ended."""

    name: str
    end_perf_s: float
    duration_s: float
    attributes: dict[str, Any]


class VoiceTracer(Protocol):
    @property
    def enabled(self) -> bool: ...

    def emit_root(self, ref: TurnTraceRef, start_ns: int, end_ns: int) -> None: ...

    def emit_stages(self, ref: TurnTraceRef, stages: Sequence[StageSpan], status: str) -> None: ...

    def shutdown(self, timeout_s: float) -> None: ...


class NoopVoiceTracer:
    """Tracing off: emits nothing, holds nothing."""

    enabled = False

    def emit_root(self, ref: TurnTraceRef, start_ns: int, end_ns: int) -> None:
        return None

    def emit_stages(self, ref: TurnTraceRef, stages: Sequence[StageSpan], status: str) -> None:
        return None

    def shutdown(self, timeout_s: float) -> None:
        return None


_current: VoiceTracer = NoopVoiceTracer()
_state_lock = threading.Lock()
_registries: weakref.WeakSet[VoiceTurnTraces] = weakref.WeakSet()


def current_voice_tracer() -> VoiceTracer:
    return _current


def install(tracer: VoiceTracer) -> None:
    """Make *tracer* the process's voice tracer (tests install in-memory ones)."""
    global _current
    with _state_lock:
        _current = tracer


def uninstall() -> None:
    """Back to tracing off, without shutting the previous tracer down."""
    install(NoopVoiceTracer())


def init_tracing(environ: Mapping[str, str] | None = None) -> VoiceTracer:
    """Turn voice tracing on if the environment configures it. Never raises.

    Called once by each serve entry point, after logging is set up. Calling it
    again while a tracer is installed keeps that tracer.
    """
    if _current.enabled:
        return _current
    config = TracingConfig.from_env(os.environ if environ is None else environ)
    if config is None:
        return _current
    try:
        from speech_to_speech.tracing.otel_backend import OtelVoiceTracer
    except ImportError:
        logger.warning(MISSING_EXTRA_LINE)
        return _current
    try:
        tracer = OtelVoiceTracer(config)
    except Exception as exc:
        logger.warning(BACKEND_FAILED_LINE, type(exc).__name__)
        return _current
    install(tracer)
    logger.info(TRACING_ON_LINE, config.environment)
    return tracer


def shutdown_tracing(timeout_s: float = 3.0) -> None:
    """Export every open turn root, then stop the exporter, waiting at most *timeout_s*.

    Idempotent and never raises. The SDK's own shutdown joins its export thread
    for up to 30 s, which an unreachable Langfuse would spend in full, so it
    runs on a daemon thread and a process stop waits for it no longer than
    *timeout_s*. Spans still queued after that are lost.
    """
    global _current
    with _state_lock:
        tracer = _current
        if not tracer.enabled:
            return
        _current = NoopVoiceTracer()
        registries = list(_registries)
    for registry in registries:
        registry.finalize_all(tracer)
    worker = threading.Thread(
        target=_shutdown_quietly, args=(tracer, timeout_s), name="voice-tracing-shutdown", daemon=True
    )
    worker.start()
    worker.join(timeout_s)


def _shutdown_quietly(tracer: VoiceTracer, timeout_s: float) -> None:
    try:
        tracer.shutdown(timeout_s)
    except Exception as exc:
        logger.debug("Voice tracing shutdown failed (%s)", type(exc).__name__)


# (field on TurnLatencyTracker, span name), in the order a turn runs.
_STAGES = (
    ("vad_decision_s", "vad"),
    ("stt_s", "stt"),
    ("llm_ttft_s", "llm.ttft"),
    ("llm_s", "llm"),
    ("tts_ttfa_s", "tts.ttfa"),
    ("e2e_s", "e2e"),
)


def build_stage_intervals(tracker: TurnLatencyTracker) -> list[StageInterval]:
    """The tracker's measured stages; one with no value or no end anchor is skipped.

    Durations are the tracker's own fields, the same numbers format_log_line()
    prints. Ids, enums and durations only: no content ever reaches a span.
    """
    intervals: list[StageInterval] = []
    for field_name, span_name in _STAGES:
        seconds = getattr(tracker, field_name)
        ended_at_s = tracker.ended_at_s(field_name)
        if seconds is None or ended_at_s is None:
            continue
        attributes: dict[str, Any] = {}
        if span_name == "vad":
            for key, value in (
                ("s2s.vad_settle_s", tracker.vad_settle_s),
                ("s2s.smart_turn_status", tracker.smart_turn_status),
                ("s2s.smart_turn_wait_s", tracker.smart_turn_wait_s),
            ):
                if value is not None:
                    attributes[key] = value
        intervals.append(StageInterval(span_name, ended_at_s, seconds, attributes))
    return intervals


def _perf_ns(perf_s: float) -> int:
    return round(perf_s * 1e9)


def _clock_offset_ns() -> int:
    """Epoch ns minus perf_counter ns: turns a perf_counter reading into wall time."""
    return time.time_ns() - time.perf_counter_ns()


@dataclass
class _OpenRoot:
    ref: TurnTraceRef
    # perf_counter nanoseconds; integers, so a root and its stages compare exactly.
    start_ns: int
    end_ns: int


class VoiceTurnTraces:
    """One pipeline unit's turn traces: the bound call, its epoch, and which roots are out.

    A turn's root is exported exactly once, because Langfuse upserts by span
    id and a second export would overwrite the first. It stays open while its
    turn can still grow (revisions, tool follow-ups) and is exported when the
    call's next turn binds, when the call id changes or clears (pooled claim
    and release), when the session unregisters, or at process shutdown.
    """

    EXPORTED_CAP = 4096

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._call_uuid: str | None = None
        self._epoch: str | None = None
        self._epoch_key: tuple[str, int] | None = None
        self._open: dict[tuple[str, str], _OpenRoot] = {}
        self._exported: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._offset_ns = _clock_offset_ns()
        self._failure_logged = False
        with _state_lock:
            _registries.add(self)

    def bind_call(self, raw: str | None) -> None:
        """Follow the unit's Sophie call id; empty or non-uuid means no traced call.

        A different id exports the previous call's open roots first. The same id
        again (a repeated session.update) changes nothing, so one turn is never
        split across two epochs.
        """
        try:
            call_uuid = normalize_call_uuid(raw or "")
            with self._lock:
                if call_uuid == self._call_uuid:
                    return
                closing = self._close_locked()
                offset = self._offset_ns
                self._call_uuid = call_uuid
                self._epoch = None
                self._epoch_key = None
                self._offset_ns = _clock_offset_ns()
            self._emit_roots(closing, current_voice_tracer(), offset)
        except Exception as exc:
            self._log_failure(exc)

    def ref_for(
        self,
        turn_id: str | None,
        revision: int | None,
        source: str,
        est_start_perf: float,
        now_perf: float,
        *,
        reset_count: int = 0,
    ) -> TurnTraceRef | None:
        """The trace ref for a response of *turn_id*, or None when it is not traced.

        None when tracing is off, no uuid call is bound, or there is no turn.
        The epoch is re-minted whenever *reset_count* (the speculative tracker's
        reset counter) moves, so ``turn_1`` after a reset is a new wire id.
        Binding a turn exports any other open root; binding a turn whose root is
        already out (a late follow-up) returns its ref and reopens nothing.
        """
        tracer = current_voice_tracer()
        if turn_id is None or not tracer.enabled:
            return None
        try:
            with self._lock:
                call_uuid = self._call_uuid
                if call_uuid is None:
                    return None
                epoch_key = (call_uuid, reset_count)
                if self._epoch is None or self._epoch_key != epoch_key:
                    self._epoch = mint_epoch()
                    self._epoch_key = epoch_key
                wire_id = wire_turn_id(turn_id, self._epoch)
                if not TURN_ID_RE.fullmatch(wire_id):
                    raise ValueError("turn id is not a safe header value")
                trace_id = voice_turn_trace_id(call_uuid, wire_id)
                root_span_id = voice_turn_root_span_id(call_uuid, wire_id)
                ref = TurnTraceRef(
                    call_uuid=call_uuid,
                    wire_turn_id=wire_id,
                    turn_id=turn_id,
                    revision=revision,
                    source=source,
                    trace_id=trace_id,
                    root_span_id=root_span_id,
                    traceparent=format_traceparent(trace_id, root_span_id),
                )
                key = (call_uuid, wire_id)
                if key in self._exported:
                    return ref
                closing = self._close_locked(keep=key)
                entry = self._open.get(key)
                start_ns, end_ns = _perf_ns(est_start_perf), _perf_ns(now_perf)
                if entry is None:
                    self._open[key] = _OpenRoot(ref, start_ns, end_ns)
                else:
                    entry.start_ns = min(entry.start_ns, start_ns)
                    entry.end_ns = max(entry.end_ns, end_ns)
                offset = self._offset_ns
            self._emit_roots(closing, tracer, offset)
            return ref
        except Exception as exc:
            self._log_failure(exc)
            return None

    def record_response(self, ref: TurnTraceRef, tracker: TurnLatencyTracker, status: str) -> None:
        """Export one finished response's stage spans under its turn's root.

        Runs on the event loop at the response terminal: at most six spans,
        each only enqueued. Widens the open root's window to cover them.
        """
        try:
            tracer = current_voice_tracer()
            if not tracer.enabled:
                return
            intervals = build_stage_intervals(tracker)
            if not intervals:
                return
            spans_perf = [
                (interval, _perf_ns(interval.end_perf_s) - _perf_ns(interval.duration_s), _perf_ns(interval.end_perf_s))
                for interval in intervals
            ]
            with self._lock:
                entry = self._open.get((ref.call_uuid, ref.wire_turn_id))
                if entry is not None:
                    entry.start_ns = min(entry.start_ns, *(start for _, start, _ in spans_perf))
                    entry.end_ns = max(entry.end_ns, *(end for _, _, end in spans_perf))
                offset = self._offset_ns
            stages = [
                StageSpan(interval.name, start + offset, end + offset, interval.attributes)
                for interval, start, end in spans_perf
            ]
            tracer.emit_stages(ref, stages, status)
        except Exception as exc:
            self._log_failure(exc)

    def finalize_all(self, tracer: VoiceTracer | None = None) -> None:
        """Export every open root now (session end, call change, shutdown)."""
        try:
            with self._lock:
                closing = self._close_locked()
                offset = self._offset_ns
            self._emit_roots(closing, tracer or current_voice_tracer(), offset)
        except Exception as exc:
            self._log_failure(exc)

    def _close_locked(self, keep: tuple[str, str] | None = None) -> list[_OpenRoot]:
        """Claim every open root but *keep* for export; each is claimed exactly once."""
        closing: list[_OpenRoot] = []
        for key in [key for key in self._open if key != keep]:
            closing.append(self._open.pop(key))
            self._exported[key] = None
        while len(self._exported) > self.EXPORTED_CAP:
            self._exported.popitem(last=False)
        return closing

    def _emit_roots(self, closing: list[_OpenRoot], tracer: VoiceTracer, offset_ns: int) -> None:
        for entry in closing:
            try:
                tracer.emit_root(entry.ref, entry.start_ns + offset_ns, entry.end_ns + offset_ns)
            except Exception as exc:
                self._log_failure(exc)

    def _log_failure(self, exc: BaseException) -> None:
        if not self._failure_logged:
            self._failure_logged = True
            logger.warning(EMIT_FAILED_LINE, type(exc).__name__)
