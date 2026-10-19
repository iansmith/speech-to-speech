"""An in-memory voice tracer for tests: the production tracer, minus the network."""

from __future__ import annotations

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from speech_to_speech.tracing import voice_turns
from speech_to_speech.tracing.otel_backend import OtelVoiceTracer
from speech_to_speech.tracing.voice_turns import TracingConfig

# Stand-in values only. Nothing is ever sent to this endpoint: the exporter
# below is in-memory.
STAND_IN_CONFIG = TracingConfig(
    public_key="pk-stand-in",
    secret_key="sk-stand-in",
    endpoint="https://langfuse.invalid/api/public/otel/v1/traces",
    environment="test",
)


def install_memory_tracer() -> InMemorySpanExporter:
    """Install a real OtelVoiceTracer whose spans land in memory, synchronously."""
    exporter = InMemorySpanExporter()
    voice_turns.install(OtelVoiceTracer(STAND_IN_CONFIG, processor=SimpleSpanProcessor(exporter)))
    return exporter


def roots(spans: tuple[ReadableSpan, ...] | list[ReadableSpan]) -> list[ReadableSpan]:
    return [span for span in spans if span.parent is None]


def children(spans: tuple[ReadableSpan, ...] | list[ReadableSpan]) -> list[ReadableSpan]:
    return [span for span in spans if span.parent is not None]
