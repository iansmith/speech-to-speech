"""RED stub (SOP-805): API surface only."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter

from speech_to_speech.tracing.voice_turns import StageSpan, TracingConfig, TurnTraceRef

__all__ = ["OTLPSpanExporter", "OtelVoiceTracer"]


class OtelVoiceTracer:
    enabled = True

    def __init__(
        self,
        config: TracingConfig,
        *,
        exporter: SpanExporter | None = None,
        processor: SpanProcessor | None = None,
    ) -> None:
        self.provider = TracerProvider()
        self.processor: Any = processor

    def emit_root(self, ref: TurnTraceRef, start_ns: int, end_ns: int) -> None:
        return None

    def emit_stages(self, ref: TurnTraceRef, stages: Sequence[StageSpan], status: str) -> None:
        return None

    def shutdown(self, timeout_s: float) -> None:
        return None
