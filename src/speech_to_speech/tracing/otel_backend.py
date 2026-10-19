"""The OpenTelemetry half of voice tracing: the only module that imports OTel.

:func:`speech_to_speech.tracing.voice_turns.init_tracing` imports it lazily,
and only when ``LANGFUSE_*`` is complete and valid; the packages come from the
optional ``[tracing]`` extra.

Everything here is private to the process's voice tracing. The provider is
never made global (``trace.set_tracer_provider``), no propagator is installed
and nothing is instrumented, so no other library's spans (FastAPI's included)
reach Langfuse through it and no ``traceparent`` is stamped on anyone's
requests. The only ``traceparent`` s2s sends is the one
``BaseOpenAICompatibleHandler.sophie_call_headers`` builds for Sophie's proxy.

Spans are built after the fact, with explicit start and end times, from
measurements the pipeline already took. Ending one only enqueues it on a
bounded BatchSpanProcessor queue that drops when full; the HTTP export runs
on the processor's own daemon thread.
"""

from __future__ import annotations

import base64
import threading
from collections.abc import Sequence
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from speech_to_speech.tracing.voice_turns import (
    KEY_ENVIRONMENT,
    KEY_SESSION_ID,
    KEY_SOURCE,
    KEY_STATUS,
    KEY_TRACE_NAME,
    KEY_TURN_ID,
    KEY_TURN_REVISION,
    ROOT_SPAN_NAME,
    TRACE_NAME,
    StageSpan,
    TracingConfig,
    TurnTraceRef,
)

SERVICE_NAME = "speech-to-speech"
INSTRUMENTATION_SCOPE = "speech_to_speech.tracing"
INGESTION_VERSION_HEADER = "x-langfuse-ingestion-version"
INGESTION_VERSION = "4"
EXPORT_TIMEOUT_S = 5.0
MAX_QUEUE_SIZE = 2048
MAX_EXPORT_BATCH_SIZE = 256
SCHEDULE_DELAY_MS = 1000
EXPORT_TIMEOUT_MS = 5000


class _ForcedIdGenerator(RandomIdGenerator):
    """Random ids, except for a root span whose ids Sophie derives too.

    The SDK has no public way to choose a span's ids. ``Tracer.start_span``
    calls ``generate_trace_id`` (a span with no valid parent) and then
    ``generate_span_id`` synchronously on the calling thread, so a
    thread-local slot set just before ``start_span`` is read by exactly that
    span. Roots are emitted from the realtime event loop and, at shutdown,
    from the main thread, which is why the slot is thread-local rather than
    shared.
    """

    def __init__(self) -> None:
        self._slot = threading.local()

    def force(self, trace_id: int, span_id: int) -> None:
        self._slot.ids = (trace_id, span_id)

    def clear(self) -> None:
        self._slot.ids = None

    def _forced(self) -> tuple[int, int] | None:
        forced: tuple[int, int] | None = getattr(self._slot, "ids", None)
        return forced

    def generate_trace_id(self) -> int:
        forced = self._forced()
        return forced[0] if forced is not None else super().generate_trace_id()

    def generate_span_id(self) -> int:
        forced = self._forced()
        if forced is None:
            return super().generate_span_id()
        self._slot.ids = None
        return forced[1]

    def is_trace_id_random(self) -> bool:
        return self._forced() is None


def langfuse_exporter(config: TracingConfig) -> OTLPSpanExporter:
    """OTLP over HTTP (Langfuse takes no gRPC) with Basic auth and ingestion v4."""
    token = base64.b64encode(f"{config.public_key}:{config.secret_key}".encode()).decode("ascii")
    return OTLPSpanExporter(
        endpoint=config.endpoint,
        headers={"Authorization": f"Basic {token}", INGESTION_VERSION_HEADER: INGESTION_VERSION},
        timeout=EXPORT_TIMEOUT_S,
    )


class OtelVoiceTracer:
    """Exports turn roots and stage spans through a private TracerProvider.

    Production passes only *config*. Tests pass an *exporter* (wrapped in the
    production batch processor) or a whole *processor*.
    """

    enabled = True

    def __init__(
        self,
        config: TracingConfig,
        *,
        exporter: SpanExporter | None = None,
        processor: SpanProcessor | None = None,
    ) -> None:
        self._environment = config.environment
        self._ids = _ForcedIdGenerator()
        # Resource(...) rather than Resource.create(): nothing from
        # OTEL_RESOURCE_ATTRIBUTES or OTEL_SERVICE_NAME leaks onto the spans.
        self.provider = TracerProvider(
            sampler=ALWAYS_ON,
            resource=Resource(
                {
                    "service.name": SERVICE_NAME,
                    "deployment.environment": config.environment,
                    "deployment.environment.name": config.environment,
                    KEY_ENVIRONMENT: config.environment,
                }
            ),
            # No atexit flush: shutdown_tracing bounds it instead.
            shutdown_on_exit=False,
            id_generator=self._ids,
        )
        if processor is None:
            processor = BatchSpanProcessor(
                exporter if exporter is not None else langfuse_exporter(config),
                max_queue_size=MAX_QUEUE_SIZE,
                schedule_delay_millis=SCHEDULE_DELAY_MS,
                max_export_batch_size=MAX_EXPORT_BATCH_SIZE,
                export_timeout_millis=EXPORT_TIMEOUT_MS,
            )
        self.processor = processor
        self.provider.add_span_processor(processor)
        self._tracer = self.provider.get_tracer(INSTRUMENTATION_SCOPE)

    def _common(self, ref: TurnTraceRef) -> dict[str, Any]:
        return {
            KEY_SESSION_ID: ref.call_uuid,
            KEY_TRACE_NAME: TRACE_NAME,
            KEY_ENVIRONMENT: self._environment,
            KEY_TURN_ID: ref.wire_turn_id,
        }

    def emit_root(self, ref: TurnTraceRef, start_ns: int, end_ns: int) -> None:
        """The turn's parentless root, under the ids Sophie derives for it."""
        self._ids.force(int.from_bytes(ref.trace_id, "big"), int.from_bytes(ref.root_span_id, "big"))
        try:
            span = self._tracer.start_span(
                ROOT_SPAN_NAME, context=Context(), start_time=start_ns, attributes=self._common(ref)
            )
        finally:
            self._ids.clear()
        span.end(end_time=end_ns)

    def emit_stages(self, ref: TurnTraceRef, stages: Sequence[StageSpan], status: str) -> None:
        """Stage spans as direct children of the turn root, which may not be exported yet."""
        parent = trace.set_span_in_context(
            NonRecordingSpan(
                SpanContext(
                    trace_id=int.from_bytes(ref.trace_id, "big"),
                    span_id=int.from_bytes(ref.root_span_id, "big"),
                    is_remote=True,
                    trace_flags=TraceFlags(TraceFlags.SAMPLED),
                )
            ),
            Context(),
        )
        common = self._common(ref)
        common[KEY_SOURCE] = ref.source
        common[KEY_STATUS] = status
        if ref.revision is not None:
            common[KEY_TURN_REVISION] = ref.revision
        for stage in stages:
            span = self._tracer.start_span(
                stage.name, context=parent, start_time=stage.start_ns, attributes={**common, **stage.attributes}
            )
            span.end(end_time=stage.end_ns)

    def shutdown(self, timeout_s: float) -> None:
        """Flush and stop. The caller bounds how long it waits (shutdown_tracing)."""
        self.provider.shutdown()
