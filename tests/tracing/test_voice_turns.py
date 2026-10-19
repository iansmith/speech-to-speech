"""SOP-805: the voice tracer's config gate, its export registry and its backend."""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest
from opentelemetry import propagate, trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

from speech_to_speech.pipeline.turn_latency import TurnLatencyTracker
from speech_to_speech.tracing import otel_backend, voice_turns
from speech_to_speech.tracing.otel_backend import OtelVoiceTracer
from speech_to_speech.tracing.voice_ids import voice_turn_root_span_id, voice_turn_trace_id
from speech_to_speech.tracing.voice_turns import (
    TurnTraceRef,
    VoiceTurnTraces,
    current_voice_tracer,
    init_tracing,
    shutdown_tracing,
)
from tests.tracing.memory import STAND_IN_CONFIG, children, install_memory_tracer, roots

CALL = "0f8e4c2a-6b1d-4e3f-9a7c-5d2b8e1f4a60"
TRACING_LOGGER = "speech_to_speech.tracing.voice_turns"
PK, SK = "pk-stand-in-value", "sk-stand-in-value"
GOOD_BASE = "https://langfuse.invalid"


def _env(pk: str | None = PK, sk: str | None = SK, base: str | None = GOOD_BASE) -> dict[str, str]:
    environ: dict[str, str] = {}
    for name, value in (
        ("LANGFUSE_PUBLIC_KEY", pk),
        ("LANGFUSE_SECRET_KEY", sk),
        ("LANGFUSE_BASE_URL", base),
    ):
        if value is not None:
            environ[name] = value
    return environ


class _MustNotBuild:
    """Stands in for a constructor that an off configuration must never reach.

    It records rather than raises: init_tracing contains a failing backend
    build (and logs it), so a raise here could be swallowed into a pass.
    """

    calls: list[str] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        type(self).calls.append(type(self).__name__)


class _RecordingExporter(SpanExporter):
    """Replaces OTLPSpanExporter: records how it was built, sends nothing."""

    built: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).built.append(kwargs)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


@pytest.fixture
def recording_exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_RecordingExporter]]:
    _RecordingExporter.built = []
    monkeypatch.setattr(otel_backend, "OTLPSpanExporter", _RecordingExporter)
    yield _RecordingExporter
    shutdown_tracing(timeout_s=1.0)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == TRACING_LOGGER and r.levelno >= logging.WARNING]


# (environment, WARNING lines expected). All-unset is the ordinary "off" and
# says nothing; anything half-configured says exactly one thing.
OFF_CASES = [
    pytest.param({}, 0, id="unset"),
    pytest.param(_env("", "", ""), 0, id="all-empty"),
    pytest.param(_env("  ", "\t", " "), 0, id="all-whitespace"),
    pytest.param(_env(PK, None, None), 1, id="pk-only"),
    pytest.param(_env(None, SK, None), 1, id="sk-only"),
    pytest.param(_env(None, None, GOOD_BASE), 1, id="base-only"),
    pytest.param(_env(PK, SK, None), 1, id="no-base"),
    pytest.param(_env(PK, None, GOOD_BASE), 1, id="no-sk"),
    pytest.param(_env(None, SK, GOOD_BASE), 1, id="no-pk"),
    pytest.param(_env(PK, "", GOOD_BASE), 1, id="empty-sk"),
    pytest.param(_env(base="http://langfuse.example.com"), 1, id="http-not-loopback"),
    pytest.param(_env(base="https://user:pw@langfuse.example.com"), 1, id="userinfo"),
    pytest.param(_env(base="https://"), 1, id="hostless"),
    pytest.param(_env(base="ftp://langfuse.example.com"), 1, id="wrong-scheme"),
    pytest.param(_env(base="langfuse.example.com"), 1, id="no-scheme"),
]


@pytest.mark.parametrize(("environ", "expected_warnings"), OFF_CASES)
def test_config_env_matrix(
    environ: Mapping[str, str],
    expected_warnings: int,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _MustNotBuild.calls = []
    monkeypatch.setattr(otel_backend, "OtelVoiceTracer", _MustNotBuild)
    monkeypatch.setattr(otel_backend, "OTLPSpanExporter", _MustNotBuild)
    caplog.set_level(logging.DEBUG, logger=TRACING_LOGGER)

    tracer = init_tracing(environ)

    assert _MustNotBuild.calls == []
    assert tracer.enabled is False
    assert current_voice_tracer().enabled is False
    assert len(_warnings(caplog)) == expected_warnings
    for record in caplog.records:
        message = record.getMessage()
        # A leak check, not a wording check: no key or URL value is logged.
        for value in (PK, SK, "langfuse.example.com", "user:pw"):
            assert value not in message


@pytest.mark.parametrize("base", ["http://localhost:3000", "http://127.0.0.1:3000/", "http://[::1]:3000"])
def test_loopback_http_base_is_allowed(base: str, recording_exporter: type[_RecordingExporter]) -> None:
    tracer = init_tracing(_env(base=base))
    assert tracer.enabled is True
    assert recording_exporter.built[0]["endpoint"] == base.rstrip("/") + "/api/public/otel/v1/traces"


def test_valid_env_builds_langfuse_http_exporter(
    recording_exporter: type[_RecordingExporter], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "leaked.attr=1")
    tracer = init_tracing({**_env(base=GOOD_BASE + "/"), "SOPHIE_ENVIRONMENT": "demo"})

    assert tracer.enabled is True
    assert current_voice_tracer() is tracer
    assert len(recording_exporter.built) == 1
    built = recording_exporter.built[0]
    assert built["endpoint"] == "https://langfuse.invalid/api/public/otel/v1/traces"
    expected_auth = "Basic " + base64.b64encode(f"{PK}:{SK}".encode()).decode()
    assert built["headers"] == {"Authorization": expected_auth, "x-langfuse-ingestion-version": "4"}
    assert isinstance(tracer, OtelVoiceTracer)
    assert isinstance(tracer.processor, BatchSpanProcessor)
    # shutdown_on_exit=False: no atexit hook that could stall a systemd stop.
    assert tracer.provider._atexit_handler is None
    attributes = dict(tracer.provider.resource.attributes)
    assert attributes["service.name"] == "speech-to-speech"
    assert attributes["deployment.environment"] == "demo"
    assert "leaked.attr" not in attributes


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "dev"),
        ("", "dev"),
        ("demo", "demo"),
        ("Demo", "unknown"),
        ("langfuse-x", "unknown"),
        ("a" * 41, "unknown"),
    ],
)
def test_environment_follows_sophies_rule(
    raw: str | None, expected: str, recording_exporter: type[_RecordingExporter]
) -> None:
    environ = _env()
    if raw is not None:
        environ["SOPHIE_ENVIRONMENT"] = raw
    tracer = init_tracing(environ)
    assert isinstance(tracer, OtelVoiceTracer)
    assert tracer.provider.resource.attributes["deployment.environment"] == expected


def test_env_set_but_extra_missing_runs_untraced(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The [tracing] extra is absent: the SDK and exporter cannot be imported.
    # opentelemetry-api alone may be present (fastapi depends on it).
    monkeypatch.delitem(sys.modules, "speech_to_speech.tracing.otel_backend", raising=False)
    hidden = [m for m in sys.modules if m.startswith(("opentelemetry.sdk", "opentelemetry.exporter"))]
    for name in {"opentelemetry.sdk", "opentelemetry.exporter", *hidden}:
        monkeypatch.setitem(sys.modules, name, None)
    caplog.set_level(logging.DEBUG, logger=TRACING_LOGGER)

    tracer = init_tracing(_env())

    assert tracer.enabled is False
    assert len(_warnings(caplog)) == 1


def test_import_and_init_with_env_unset_load_no_otel_sdk() -> None:
    script = textwrap.dedent(
        """
        import socket, sys
        attempts = []
        real_create_connection = socket.create_connection
        def counting_create_connection(*args, **kwargs):
            attempts.append(args)
            return real_create_connection(*args, **kwargs)
        socket.create_connection = counting_create_connection
        import speech_to_speech.s2s_pipeline
        import speech_to_speech.configured_runtime
        import speech_to_speech.LLM.base_openai_compatible_language_model
        import speech_to_speech.api.openai_realtime.websocket_router
        from speech_to_speech.tracing import voice_turns
        tracer = voice_turns.init_tracing()
        voice_turns.shutdown_tracing()
        loaded = sorted(
            m for m in sys.modules
            if m.startswith(("opentelemetry.sdk", "opentelemetry.exporter", "speech_to_speech.tracing.otel_backend"))
        )
        print("ENABLED", tracer.enabled)
        print("LOADED", loaded)
        print("CONNECTS", len(attempts))
        """
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LANGFUSE_", "OTEL_"))}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-2000:]
    lines = dict(line.split(" ", 1) for line in result.stdout.strip().splitlines() if " " in line)
    assert lines["ENABLED"] == "False"
    assert lines["LOADED"] == "[]"
    assert lines["CONNECTS"] == "0"


def _tracker(*, end: float = 100.0, **fields: float) -> TurnLatencyTracker:
    tracker = TurnLatencyTracker(turn_id="turn_1", turn_revision=0, source="speech")
    for name, seconds in fields.items():
        setattr(tracker, name, seconds)
        tracker._ended_at_s[name] = end
    return tracker


def test_late_followup_after_next_turn_does_not_close_new_turn() -> None:
    exporter = install_memory_tracer()
    registry = VoiceTurnTraces()
    registry.bind_call(CALL)

    first = registry.ref_for("turn_1", 0, "speech", 1.0, 2.0)
    second = registry.ref_for("turn_2", 0, "speech", 3.0, 4.0)
    assert first is not None and second is not None
    assert [s.context.span_id for s in roots(exporter.get_finished_spans())] == [
        int.from_bytes(first.root_span_id, "big")
    ]

    late = registry.ref_for("turn_1", 0, "tool", 5.0, 6.0)
    assert late is not None
    assert late.wire_turn_id == first.wire_turn_id
    assert late.trace_id == first.trace_id
    # turn_1's root is not exported again and turn_2 stays open.
    assert len(roots(exporter.get_finished_spans())) == 1

    registry.record_response(late, _tracker(llm_s=0.4, end=6.0), "completed")
    stage = children(exporter.get_finished_spans())
    assert len(stage) == 1
    assert stage[0].parent is not None
    assert stage[0].parent.span_id == int.from_bytes(first.root_span_id, "big")
    assert stage[0].context.trace_id == int.from_bytes(first.trace_id, "big")

    registry.finalize_all()
    root_ids = [s.context.span_id for s in roots(exporter.get_finished_spans())]
    assert root_ids == [int.from_bytes(first.root_span_id, "big"), int.from_bytes(second.root_span_id, "big")]
    registry.finalize_all()
    assert len(roots(exporter.get_finished_spans())) == 2


def test_root_window_covers_its_stages() -> None:
    exporter = install_memory_tracer()
    registry = VoiceTurnTraces()
    registry.bind_call(CALL)
    ref = registry.ref_for("turn_1", 0, "speech", 10.0, 10.5)
    assert ref is not None
    registry.record_response(ref, _tracker(e2e_s=2.0, end=12.0), "completed")
    registry.finalize_all()
    (root,) = roots(exporter.get_finished_spans())
    (stage,) = children(exporter.get_finished_spans())
    assert root.start_time is not None and root.end_time is not None
    assert stage.start_time is not None and stage.end_time is not None
    assert root.start_time <= stage.start_time
    assert root.end_time >= stage.end_time
    assert stage.end_time - stage.start_time == 2_000_000_000


def test_ref_needs_tracing_a_uuid_call_and_a_turn() -> None:
    registry = VoiceTurnTraces()
    registry.bind_call(CALL)
    assert registry.ref_for("turn_1", 0, "speech", 1.0, 2.0) is None  # tracing off

    install_memory_tracer()
    assert registry.ref_for(None, None, "client", 1.0, 2.0) is None
    registry.bind_call("not-a-uuid")
    assert registry.ref_for("turn_1", 0, "speech", 1.0, 2.0) is None
    registry.bind_call(CALL.upper())
    ref = registry.ref_for("turn_1", 0, "speech", 1.0, 2.0)
    assert ref is not None
    assert ref.call_uuid == CALL


class _HungExporter(SpanExporter):
    def __init__(self) -> None:
        self.release = threading.Event()
        self.received = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.received += len(spans)
        self.release.wait()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def test_hung_exporter_never_blocks_turns(caplog: pytest.LogCaptureFixture) -> None:
    # The BSP logs one "queue full" warning per dropped span; that is expected here.
    caplog.set_level(logging.CRITICAL, logger="opentelemetry")
    exporter = _HungExporter()
    processor = BatchSpanProcessor(exporter, max_queue_size=8, schedule_delay_millis=1, max_export_batch_size=4)
    voice_turns.install(OtelVoiceTracer(STAND_IN_CONFIG, processor=processor))
    try:
        registry = VoiceTurnTraces()
        registry.bind_call(CALL)
        emitted = 0
        slowest = 0.0
        started = time.perf_counter()
        for n in range(1, 201):
            call_started = time.perf_counter()
            ref = registry.ref_for(f"turn_{n}", 0, "speech", float(n), float(n) + 0.5)
            assert ref is not None
            registry.record_response(
                ref, _tracker(stt_s=0.1, llm_s=0.2, tts_ttfa_s=0.3, e2e_s=0.9, end=float(n) + 0.9), "completed"
            )
            emitted += 5  # 4 stages, plus the previous turn's root
            slowest = max(slowest, time.perf_counter() - call_started)
        registry.finalize_all()
        total = time.perf_counter() - started

        # A blocking export would park here for the exporter's lifetime.
        assert slowest < 0.5
        assert total < 3.0
        assert exporter.received < emitted

        shutdown_started = time.perf_counter()
        shutdown_tracing(timeout_s=0.2)
        assert time.perf_counter() - shutdown_started < 1.0
        assert current_voice_tracer().enabled is False
    finally:
        exporter.release.set()


def test_forced_root_ids_are_thread_isolated() -> None:
    exporter = install_memory_tracer()
    tracer = current_voice_tracer()
    barrier = threading.Barrier(8)
    failures: list[BaseException] = []

    def emit(worker: int) -> None:
        try:
            barrier.wait()
            for n in range(25):
                turn = f"turn_{worker}_{n}"
                trace_id = voice_turn_trace_id(CALL, turn)
                root_id = voice_turn_root_span_id(CALL, turn)
                ref = TurnTraceRef(
                    call_uuid=CALL,
                    wire_turn_id=turn,
                    turn_id=turn,
                    revision=0,
                    source="speech",
                    trace_id=trace_id,
                    root_span_id=root_id,
                    traceparent="",
                )
                tracer.emit_root(ref, 1_000, 2_000)
                tracer.emit_stages(ref, [voice_turns.StageSpan("llm", 1_100, 1_900, {})], "completed")
        except BaseException as exc:  # surfaced below
            failures.append(exc)

    threads = [threading.Thread(target=emit, args=(w,)) for w in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert not failures

    spans = exporter.get_finished_spans()
    assert len(roots(spans)) == 200
    for root in roots(spans):
        turn = str(root.attributes["sophie.voice.turn_id"]) if root.attributes else ""
        assert root.context.trace_id == int.from_bytes(voice_turn_trace_id(CALL, turn), "big")
        assert root.context.span_id == int.from_bytes(voice_turn_root_span_id(CALL, turn), "big")
    root_ids = {root.context.span_id for root in roots(spans)}
    for child in children(spans):
        assert child.context.span_id not in root_ids


def test_forced_id_slot_is_per_thread() -> None:
    ids = otel_backend._ForcedIdGenerator()
    forced_elsewhere = threading.Event()

    def other_thread() -> None:
        ids.force(3, 4)
        forced_elsewhere.set()

    ids.force(1, 2)
    worker = threading.Thread(target=other_thread)
    worker.start()
    assert forced_elsewhere.wait(5)
    worker.join(5)
    # This thread still reads its own ids, once each, then random ones.
    assert ids.is_trace_id_random() is False
    assert ids.generate_trace_id() == 1
    assert ids.generate_span_id() == 2
    assert ids.is_trace_id_random() is True
    assert ids.generate_span_id() != 2


def test_no_global_provider_propagator_or_instrumentation(recording_exporter: type[_RecordingExporter]) -> None:
    provider_before = trace.get_tracer_provider()
    textmap_before = propagate.get_global_textmap()

    tracer = init_tracing(_env())

    assert tracer.enabled is True
    assert trace.get_tracer_provider() is provider_before
    assert propagate.get_global_textmap() is textmap_before
    assert not [m for m in sys.modules if m.startswith("opentelemetry.instrumentation")]
