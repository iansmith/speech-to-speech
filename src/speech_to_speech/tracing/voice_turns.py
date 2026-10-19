"""RED stub (SOP-805): API surface only."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class TracingConfig:
    public_key: str = field(repr=False)
    secret_key: str = field(repr=False)
    endpoint: str
    environment: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> TracingConfig | None:
        return None


@dataclass(frozen=True)
class TurnTraceRef:
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
    name: str
    start_ns: int
    end_ns: int
    attributes: dict[str, Any]


class VoiceTracer(Protocol):
    @property
    def enabled(self) -> bool: ...

    def emit_root(self, ref: TurnTraceRef, start_ns: int, end_ns: int) -> None: ...

    def emit_stages(self, ref: TurnTraceRef, stages: Sequence[StageSpan], status: str) -> None: ...

    def shutdown(self, timeout_s: float) -> None: ...


class NoopVoiceTracer:
    enabled = False

    def emit_root(self, ref: TurnTraceRef, start_ns: int, end_ns: int) -> None:
        return None

    def emit_stages(self, ref: TurnTraceRef, stages: Sequence[StageSpan], status: str) -> None:
        return None

    def shutdown(self, timeout_s: float) -> None:
        return None


_current: VoiceTracer = NoopVoiceTracer()


def current_voice_tracer() -> VoiceTracer:
    return _current


def install(tracer: VoiceTracer) -> None:
    global _current
    _current = tracer


def uninstall() -> None:
    install(NoopVoiceTracer())


def init_tracing(environ: Mapping[str, str] | None = None) -> VoiceTracer:
    TracingConfig.from_env(os.environ if environ is None else environ)
    return _current


def shutdown_tracing(timeout_s: float = 3.0) -> None:
    return None


class VoiceTurnTraces:
    def bind_call(self, raw: str | None) -> None:
        return None

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
        return None

    def record_response(self, ref: TurnTraceRef, tracker: Any, status: str) -> None:
        return None

    def finalize_all(self) -> None:
        return None
