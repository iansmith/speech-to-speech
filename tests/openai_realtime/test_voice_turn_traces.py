"""SOP-805: one trace per spoken turn, carried from the realtime service to the
model request (X-Sophie-Turn-Id + traceparent) and exported as a turn root with
stage spans.

The LLM handler here is the production ChatCompletionsApiModelHandler (the
backend demo1 runs) with only its OpenAI client replaced by a recorder, so the
headers asserted are the kwargs the real request path hands the SDK.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Iterator
from queue import Queue
from threading import Event
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from openai.types.realtime import ConversationItemCreateEvent, ResponseCreateEvent
from openai.types.realtime.conversation_item import RealtimeConversationItemFunctionCall
from opentelemetry.sdk.trace import ReadableSpan

import speech_to_speech.api.openai_realtime.service as service_module
import speech_to_speech.LLM.base_openai_compatible_language_model as base_mod
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.sophie_call import (
    CLIENT_SESSION_HEADER,
    TRACEPARENT_HEADER,
    TURN_ID_HEADER,
    clear_sophie_call_id,
    set_sophie_call_id,
)
from speech_to_speech.LLM.chat import make_user_message
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.pipeline import turn_latency
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    ResponseGenerationDoneEvent,
    SpeechStartedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import (
    AssistantToolCallPart,
    GenerateResponseRequest,
    ResponsePrefetchTransaction,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.tracing import voice_turns
from speech_to_speech.tracing.voice_ids import TURN_ID_RE, voice_turn_root_span_id, voice_turn_trace_id
from speech_to_speech.tracing.voice_turns import StageSpan, TurnTraceRef, VoiceTurnTraces
from tests.test_sophie_call_id import _handler, _Recorder, _responses_handler
from tests.tracing.memory import children, install_memory_tracer, roots
from tests.turns import reopen

CALL = "0f8e4c2a-6b1d-4e3f-9a7c-5d2b8e1f4a60"
OTHER_CALL = "7b1e0c55-2d4a-4f8e-9c3b-1a2b3c4d5e6f"
TURN_HEADERS = (TURN_ID_HEADER, TRACEPARENT_HEADER)


# ── fakes and drivers ───────────────────────────────────────────────────────


class _Completions:
    """Records each create() call's kwargs and the thread that issued it."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.threads: list[str] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        self.threads.append(threading.current_thread().name)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Okay.", tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )


class _FakeOpenAI:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.chat = SimpleNamespace(completions=_Completions())

    def with_options(self, **_kwargs: Any) -> _FakeOpenAI:
        return self

    def close(self) -> None:
        return None


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> ChatCompletionsApiModelHandler:
    monkeypatch.setattr(base_mod, "OpenAI", _FakeOpenAI)
    handler = ChatCompletionsApiModelHandler(
        Event(),
        Queue(),
        Queue(),
        setup_kwargs=dict(
            model_name="m",
            base_url="http://127.0.0.1:9743/v1",
            api_key="k",
            stream=False,
            compact_history=False,
        ),
    )
    _completions(handler).calls.clear()  # forget the warm-up
    return handler


def _completions(handler: ChatCompletionsApiModelHandler) -> _Completions:
    completions = handler.client.chat.completions
    assert isinstance(completions, _Completions)
    return completions


@pytest.fixture
def unit(
    runtime_config: RuntimeConfig, should_listen: Event, llm: ChatCompletionsApiModelHandler
) -> Iterator[SimpleNamespace]:
    """One pooled unit: a realtime service with speculative turns, plus the LLM."""
    runtime_config.session.instructions = "You are Sophie."
    service = RealtimeService(
        text_prompt_queue=Queue(),
        should_listen=should_listen,
        speculative_turns=SpeculativeTurnTracker(),
    )
    conn = service.register()
    service._state(conn).runtime_config = runtime_config
    llm.turn_latency_store = service.turn_latency_store
    pooled = SimpleNamespace(handlers=[llm], service=service, conn=conn, llm=llm)
    set_sophie_call_id(pooled, CALL)
    yield pooled
    service.unregister(conn)


def _speak(
    unit: SimpleNamespace,
    turn_id: str = "turn_1",
    revision: int = 0,
    *,
    reopened: bool = False,
    transcript: str = "Hello",
    stt_s: float = 0.12,
) -> GenerateResponseRequest:
    service: RealtimeService = unit.service
    service.dispatch_pipeline_event(
        unit.conn,
        SpeechStartedEvent(turn_id=turn_id, turn_revision=revision, interrupt_response=True, reopened=reopened),
    )
    pending = service.turn_latency_store.get_or_create_for_turn(turn_id, revision)
    assert pending is not None
    pending.record_stt(stt_s)
    service.dispatch_pipeline_event(
        unit.conn,
        TranscriptionCompletedEvent(transcript=transcript, turn_id=turn_id, turn_revision=revision),
    )
    queue = service.text_prompt_queue
    assert queue is not None
    request = queue.get_nowait()
    assert isinstance(request, GenerateResponseRequest)
    return request


def _send(unit: SimpleNamespace, request: GenerateResponseRequest) -> dict[str, str]:
    """Run *request* through the LLM handler; return the one request's headers."""
    completions = _completions(unit.llm)
    before = len(completions.calls)
    list(unit.llm.process(request))
    assert len(completions.calls) == before + 1
    headers = completions.calls[-1].get("extra_headers")
    return dict(headers) if headers else {}


def _finish(unit: SimpleNamespace, request: GenerateResponseRequest, status: str = "completed") -> list[Any]:
    """Open *request*'s response (as its first output would) and end it."""
    service: RealtimeService = unit.service
    service.response._ensure_response(unit.conn, request.response_key)
    return service.finish_response(unit.conn, status=status, response_key=request.response_key)  # type: ignore[arg-type]


def _ref(unit: SimpleNamespace, request: GenerateResponseRequest) -> TurnTraceRef:
    tracker = unit.service.turn_latency_store.get_response(request.response_key)
    assert tracker is not None
    assert isinstance(tracker.trace, TurnTraceRef)
    return tracker.trace


def _expected_turn_headers(wire_turn_id: str) -> dict[str, str]:
    trace_hex = voice_turn_trace_id(CALL, wire_turn_id).hex()
    root_hex = voice_turn_root_span_id(CALL, wire_turn_id).hex()
    return {
        CLIENT_SESSION_HEADER: CALL,
        TURN_ID_HEADER: wire_turn_id,
        TRACEPARENT_HEADER: f"00-{trace_hex}-{root_hex}-01",
    }


def _int(raw: bytes) -> int:
    return int.from_bytes(raw, "big")


# ── DoD 4: unique turn ids within a call ────────────────────────────────────


def test_turn_ids_unique_across_reset_within_one_call(unit: SimpleNamespace) -> None:
    install_memory_tracer()
    service: RealtimeService = unit.service
    turns = service.speculative_turns
    assert turns is not None
    service._state(unit.conn).runtime_config.chat.add_item(make_user_message("Hello"))

    def bind_and_send(response_key: str, *, start: bool = True) -> dict[str, str]:
        turn_id, revision = turns.start_turn() if start else ("turn_1", 0)
        assert turn_id == "turn_1"
        request = GenerateResponseRequest(
            runtime_config=service._state(unit.conn).runtime_config,
            turn_id=turn_id,
            turn_revision=revision,
            response_key=response_key,
        )
        service.bind_response_latency_tracker(unit.conn, response_key, turn_id=turn_id, turn_revision=revision)
        return _send(unit, request)

    first = bind_and_send("r1")
    turns.reset()  # the VAD on_session_end path restarts turn_N
    after_reset = bind_and_send("r2")
    # Release, claim and the same call id again, with no reset in between.
    clear_sophie_call_id(unit)
    set_sophie_call_id(unit, CALL)
    after_rebind = bind_and_send("r3", start=False)

    sent = [first, after_reset, after_rebind]
    wire_ids = [headers[TURN_ID_HEADER] for headers in sent]
    assert len(set(wire_ids)) == 3
    for wire_id in wire_ids:
        assert wire_id.startswith("turn_1_")
        assert TURN_ID_RE.fullmatch(wire_id)
    assert len({headers[TRACEPARENT_HEADER].split("-")[1] for headers in sent}) == 3
    for headers in sent:
        assert headers == _expected_turn_headers(headers[TURN_ID_HEADER])


def test_resession_with_same_call_id_keeps_epoch(unit: SimpleNamespace) -> None:
    install_memory_tracer()
    turns = unit.service.speculative_turns
    turns.start_turn()
    first = _ref(unit, _speak(unit))
    set_sophie_call_id(unit, CALL)  # a repeated session.update carrying the same id
    revision = reopen(turns)
    second = _ref(unit, _speak(unit, revision=revision, reopened=True))
    assert second.wire_turn_id == first.wire_turn_id
    assert second.trace_id == first.trace_id


def test_transcription_events_keep_turn_n_format(unit: SimpleNamespace) -> None:
    install_memory_tracer()
    unit.service.speculative_turns.start_turn()
    request = _speak(unit)
    assert request.turn_id == "turn_1"
    tracker = unit.service.turn_latency_store.get_response(request.response_key)
    assert tracker is not None
    assert tracker.turn_id == "turn_1"
    line = tracker.format_log_line()
    assert line is not None and line.startswith("Turn turn_1 rev=0 ")
    assert _ref(unit, request).turn_id == "turn_1"


# ── DoD 2/3: which requests carry the turn headers ──────────────────────────


@pytest.mark.parametrize("kind", ["text", "audio"])
def test_spoken_turn_request_headers(unit: SimpleNamespace, kind: str) -> None:
    exporter = install_memory_tracer()
    unit.service.speculative_turns.start_turn()
    request = _speak(unit)
    if kind == "audio":
        request = request.model_copy(update={"audio": np.zeros(1600, dtype=np.float32), "audio_sample_rate": 16000})

    headers = _send(unit, request)

    wire_id = headers[TURN_ID_HEADER]
    assert TURN_ID_RE.fullmatch(wire_id)
    assert headers == _expected_turn_headers(wire_id)
    unit.service.voice_traces.finalize_all()
    (root,) = roots(exporter.get_finished_spans())
    assert format(root.context.span_id, "016x") == headers[TRACEPARENT_HEADER].split("-")[2]
    assert format(root.context.trace_id, "032x") == headers[TRACEPARENT_HEADER].split("-")[1]


@pytest.mark.parametrize("path", ["chat-text", "chat-audio", "responses-text", "responses-audio"])
def test_every_request_path_carries_turn_headers(path: str) -> None:
    install_memory_tracer()
    recorder = _Recorder()
    handler: Any
    if path.startswith("chat"):
        handler = _handler()
        handler.client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))
        handler.model_name = "m"
        handler.stream = False
        handler._extra_body = None
        handler.request_timeout = 1
    else:
        handler = _responses_handler(recorder)
    request: Callable[[Any, dict[str, Any]], Any] = (
        handler._request_audio if path.endswith("audio") else handler._request
    )
    registry = VoiceTurnTraces()
    registry.bind_call(CALL)
    ref = registry.ref_for("turn_1", 0, "speech", 0.0, 0.0)
    assert ref is not None
    handler.set_sophie_call_id(CALL)

    token = base_mod._REQUEST_TURN_TRACE.set(ref)
    try:
        request([{"role": "user", "content": "hi"}], {})
    finally:
        base_mod._REQUEST_TURN_TRACE.reset(token)
    assert recorder.seen.get("extra_headers") == _expected_turn_headers(ref.wire_turn_id)

    request([{"role": "user", "content": "hi"}], {})  # no turn in scope
    assert recorder.seen.get("extra_headers") == {CLIENT_SESSION_HEADER: CALL}


def test_prefetch_worker_request_carries_turn_headers(unit: SimpleNamespace) -> None:
    install_memory_tracer()
    unit.service.speculative_turns.start_turn()
    speech = _speak(unit)
    speech_headers = _send(unit, speech)
    assert speech_headers == _expected_turn_headers(_ref(unit, speech).wire_turn_id)

    prefetch = GenerateResponseRequest(
        runtime_config=speech.runtime_config,
        turn_id=speech.turn_id,
        turn_revision=speech.turn_revision,
        prefetch_transaction=ResponsePrefetchTransaction(),
    )
    unit.service.bind_response_latency_tracker(
        unit.conn, prefetch.response_key, turn_id=prefetch.turn_id, turn_revision=prefetch.turn_revision, source="tool"
    )
    assert prefetch.prefetch_transaction is not None
    prefetch.prefetch_transaction.claim()
    headers = _send(unit, prefetch)

    assert _completions(unit.llm).threads[-1] == "realtime-tool-prefetch"
    assert headers == speech_headers


def _tool_round_trip(unit: SimpleNamespace, origin: GenerateResponseRequest, followup: str) -> GenerateResponseRequest:
    """Answer *origin* with a tool call and its output; return the follow-up request."""
    service: RealtimeService = unit.service
    conn = unit.conn
    call = RealtimeConversationItemFunctionCall(
        type="function_call", id="fc_lookup", call_id="call_lookup", name="lookup", arguments="{}"
    )
    origin.runtime_config.chat.add_provisional_generation_items(origin.response_key, [call])
    service.dispatch_pipeline_event(
        conn,
        AssistantOutputEvent(
            response_key=origin.response_key,
            turn_id=origin.turn_id,
            turn_revision=origin.turn_revision,
            parts=[
                AssistantToolCallPart(
                    tool={"type": "function_call", **call.model_dump(include={"id", "call_id", "name", "arguments"})}
                )
            ],
        ),
    )
    service.dispatch_pipeline_event(
        conn, ResponseGenerationDoneEvent(response_key=origin.response_key, call_ids=[call.call_id])
    )
    service.handle_conversation_item_create(
        conn,
        ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "call_id": call.call_id, "output": "result"},
        ),
    )
    queue = service.text_prompt_queue
    assert queue is not None
    prefetched = queue.get_nowait()
    assert prefetched.prefetch_transaction is not None
    if followup == "client_response_create":
        service.response.discard_tool_followup_prefetch(conn)
    service.finish_response(conn, response_key=origin.response_key)
    created = service.handle_response_create(conn, ResponseCreateEvent(type="response.create"))
    assert created is not None and created.type == "response.created"
    if followup == "prefetch":
        return prefetched
    request = queue.get_nowait()
    assert request.prefetch_transaction is None
    return request


@pytest.mark.parametrize("followup", ["prefetch", "client_response_create"])
def test_tool_followup_and_reopen_share_turn_trace(unit: SimpleNamespace, followup: str) -> None:
    exporter = install_memory_tracer()
    turns = unit.service.speculative_turns
    turns.start_turn()
    first = _speak(unit)
    first_headers = _send(unit, first)
    _finish(unit, first, "cancelled")  # the caller kept talking over it
    revision = reopen(turns)  # rev 1 of the same turn
    second = _speak(unit, revision=revision, reopened=True)
    second_headers = _send(unit, second)
    followup_request = _tool_round_trip(unit, second, followup)
    followup_headers = _send(unit, followup_request)
    _finish(unit, followup_request)

    sent = [first_headers, second_headers, followup_headers]
    wire_id = first_headers[TURN_ID_HEADER]
    for headers in sent:
        assert headers == _expected_turn_headers(wire_id)

    unit.service.voice_traces.finalize_all()
    spans = exporter.get_finished_spans()
    assert [span.context.span_id for span in roots(spans)] == [_int(voice_turn_root_span_id(CALL, wire_id))]
    span_ids = [span.context.span_id for span in spans]
    assert len(span_ids) == len(set(span_ids))
    assert {span.context.trace_id for span in spans} == {_int(voice_turn_trace_id(CALL, wire_id))}
    assert {span.attributes["s2s.source"] for span in children(spans) if span.attributes} >= {"speech", "tool"}
    # Each stage carries its response's terminal status: the barged-in rev 0's are cancelled.
    stage_status = set()
    for span in children(spans):
        assert span.attributes is not None
        stage_status.add(
            (span.attributes["s2s.turn_revision"], span.attributes["s2s.source"], span.attributes["s2s.status"])
        )
    assert stage_status == {(0, "speech", "cancelled"), (1, "speech", "completed"), (1, "tool", "completed")}


def test_tool_followup_with_unknown_origin_uses_last_traced_turn(unit: SimpleNamespace) -> None:
    install_memory_tracer()
    service: RealtimeService = unit.service
    service.speculative_turns.start_turn()
    speech = _speak(unit)
    speech_headers = _send(unit, speech)
    assert speech_headers == _expected_turn_headers(_ref(unit, speech).wire_turn_id)
    _finish(unit, speech)

    # A tool call this server never produced: its origin turn is unknown, so
    # the follow-up resolves to no turn at all.
    for item in (
        {"type": "function_call", "call_id": "call_unknown", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_unknown", "output": "result"},
    ):
        service.handle_conversation_item_create(
            unit.conn, ConversationItemCreateEvent(type="conversation.item.create", item=item)
        )
    created = service.handle_response_create(unit.conn, ResponseCreateEvent(type="response.create"))
    assert created is not None and created.type == "response.created"
    queue = service.text_prompt_queue
    assert queue is not None
    followup = queue.get_nowait()

    assert followup.turn_id is None  # speculative gating is untouched
    tracker = service.turn_latency_store.get_response(followup.response_key)
    assert tracker is not None and tracker.source == "tool"
    assert _send(unit, followup) == speech_headers


def test_non_turn_requests_send_no_turn_headers(unit: SimpleNamespace) -> None:
    install_memory_tracer()
    service: RealtimeService = unit.service
    llm: ChatCompletionsApiModelHandler = unit.llm
    queue = service.text_prompt_queue
    assert queue is not None

    def create_and_send(event: ResponseCreateEvent) -> dict[str, str]:
        created = service.handle_response_create(unit.conn, event)
        assert created is not None and created.type == "response.created"
        request = queue.get_nowait()
        headers = _send(unit, request)
        _finish(unit, request)
        return headers

    # A greeting: a client response.create with nothing to answer yet.
    assert create_and_send(ResponseCreateEvent(type="response.create")) == {CLIENT_SESSION_HEADER: CALL}

    # A traced spoken turn, so a later request could only inherit it by mistake.
    service.speculative_turns.start_turn()
    speech = _speak(unit)
    ref = _ref(unit, speech)
    assert _send(unit, speech)[TURN_ID_HEADER] == ref.wire_turn_id
    _finish(unit, speech)

    # Typed text is client input with no spoken turn: it must not inherit turn_1.
    service.handle_conversation_item_create(
        unit.conn,
        ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "typed"}]},
        ),
    )
    assert create_and_send(ResponseCreateEvent(type="response.create")) == {CLIENT_SESSION_HEADER: CALL}

    out_of_band = ResponseCreateEvent.model_validate(
        {"type": "response.create", "response": {"conversation": "none", "instructions": "Summarize."}}
    )
    assert create_and_send(out_of_band) == {CLIENT_SESSION_HEADER: CALL}

    completions = _completions(llm)
    token = base_mod._REQUEST_TURN_TRACE.set(ref)  # even with a turn in scope
    try:
        llm.warmup()
        warm = completions.calls[-1]
        llm._build_compaction_generate_fn()("system", "user")
        compaction = completions.calls[-1]
    finally:
        base_mod._REQUEST_TURN_TRACE.reset(token)
    assert "extra_headers" not in warm
    assert "extra_headers" not in compaction


def test_tracing_off_requests_are_unchanged(unit: SimpleNamespace) -> None:
    unit.service.speculative_turns.start_turn()
    request = _speak(unit)
    tracker = unit.service.turn_latency_store.get_response(request.response_key)
    assert tracker is not None and tracker.trace is None
    assert _send(unit, request) == {CLIENT_SESSION_HEADER: CALL}

    clear_sophie_call_id(unit)
    list(unit.llm.process(request))
    assert "extra_headers" not in _completions(unit.llm).calls[-1]


def test_non_uuid_call_id_sends_only_the_session_header(unit: SimpleNamespace) -> None:
    exporter = install_memory_tracer()
    set_sophie_call_id(unit, "call-1")
    unit.service.speculative_turns.start_turn()
    request = _speak(unit)
    assert _send(unit, request) == {CLIENT_SESSION_HEADER: "call-1"}
    _finish(unit, request)
    unit.service.voice_traces.finalize_all()
    assert exporter.get_finished_spans() == ()


def test_late_finalize_after_call_change_stays_on_original_call(unit: SimpleNamespace) -> None:
    exporter = install_memory_tracer()
    service: RealtimeService = unit.service
    service.speculative_turns.start_turn()
    request = _speak(unit)
    ref_a = _ref(unit, request)

    # The unit moves to call B before call A's response reaches its terminal.
    set_sophie_call_id(unit, OTHER_CALL)
    assert _send(unit, request) == {CLIENT_SESSION_HEADER: OTHER_CALL}  # A's turn, B's call: no turn headers
    _finish(unit, request)
    service.voice_traces.finalize_all()

    spans = exporter.get_finished_spans()
    assert spans
    for span in spans:
        assert span.attributes is not None
        assert span.attributes["langfuse.session.id"] == CALL
        assert span.context.trace_id == _int(ref_a.trace_id)


def test_root_exported_for_discarded_response(unit: SimpleNamespace) -> None:
    exporter = install_memory_tracer()
    service: RealtimeService = unit.service
    service.speculative_turns.start_turn()
    wall_before = time.time_ns()
    request = _speak(unit)
    bound_by = time.perf_counter()
    ref = _ref(unit, request)
    tracker = service.turn_latency_store.get_response(request.response_key)
    assert tracker is not None and tracker.stt_s is not None
    stt_end = tracker.ended_at_s("stt_s")
    assert stt_end is not None
    offset = service.voice_traces._offset_ns

    service.unregister(unit.conn)  # drops the tracker without a terminal
    wall_after = time.time_ns()

    spans = exporter.get_finished_spans()
    assert [(s.context.trace_id, s.context.span_id) for s in spans] == [(_int(ref.trace_id), _int(ref.root_span_id))]
    (root,) = spans
    assert root.parent is None
    assert root.start_time is not None and root.end_time is not None
    # With no terminal to widen it, the bind alone sets the window: it opens
    # where the measured stt stage started and closes at the bind.
    assert root.start_time == round((stt_end - tracker.stt_s) * 1e9) + offset
    assert round(stt_end * 1e9) + offset <= root.end_time <= round(bound_by * 1e9) + offset
    # Wall-clock time, read without the registry's offset: the root lies in
    # this test's own window (its stt stage starts 0.12 s before _speak
    # records it), not near perf_counter's arbitrary zero.
    assert wall_before - 5_000_000_000 <= root.start_time <= root.end_time <= wall_after
    service.voice_traces.finalize_all()
    assert len(exporter.get_finished_spans()) == 1


def test_root_window_spans_every_bind_of_its_turn(unit: SimpleNamespace) -> None:
    exporter = install_memory_tracer()
    service: RealtimeService = unit.service
    turns = service.speculative_turns
    turns.start_turn()
    first = _speak(unit)
    first_ref = _ref(unit, first)
    first_tracker = service.turn_latency_store.get_response(first.response_key)
    assert first_tracker is not None and first_tracker.stt_s is not None
    first_stt_end = first_tracker.ended_at_s("stt_s")
    assert first_stt_end is not None
    revision = reopen(turns)
    second = _speak(unit, revision=revision, reopened=True)  # bound after rev 0's bind
    assert _ref(unit, second).wire_turn_id == first_ref.wire_turn_id
    second_tracker = service.turn_latency_store.get_response(second.response_key)
    assert second_tracker is not None
    second_stt_end = second_tracker.ended_at_s("stt_s")
    assert second_stt_end is not None
    offset = service.voice_traces._offset_ns

    service.unregister(unit.conn)  # rev 1 never reaches a terminal

    (root,) = roots(exporter.get_finished_spans())
    assert root.context.span_id == _int(first_ref.root_span_id)
    assert root.start_time is not None and root.end_time is not None
    assert root.start_time == round((first_stt_end - first_tracker.stt_s) * 1e9) + offset
    # Rev 1's bind widened the open root past rev 1's STT, which ended after rev 0's bind.
    assert root.end_time >= round(second_stt_end * 1e9) + offset


# ── DoD 5/6/7: what is exported ─────────────────────────────────────────────


def _by_name(spans: tuple[ReadableSpan, ...]) -> dict[str, ReadableSpan]:
    named = {span.name: span for span in children(spans)}
    assert len(named) == len(children(spans))
    return named


def _duration_ns(span: ReadableSpan) -> int:
    assert span.start_time is not None and span.end_time is not None
    return span.end_time - span.start_time


def test_stage_spans_match_tracker_and_log_line(
    unit: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    exporter = install_memory_tracer()
    clock = [100.0]
    monkeypatch.setattr(turn_latency, "perf_counter", lambda: clock[0])
    service: RealtimeService = unit.service
    service.speculative_turns.start_turn()
    service.dispatch_pipeline_event(unit.conn, SpeechStartedEvent(turn_id="turn_1", turn_revision=0))
    pending = service.turn_latency_store.get_or_create_for_turn("turn_1", 0)
    assert pending is not None
    clock[0] = 100.31
    pending.record_vad_decision(0.31)
    clock[0] = 100.52
    pending.record_vad_settle(0.21)
    clock[0] = 100.95
    pending.record_stt(0.43)
    service.dispatch_pipeline_event(
        unit.conn, TranscriptionCompletedEvent(transcript="Hello", turn_id="turn_1", turn_revision=0)
    )
    queue = service.text_prompt_queue
    assert queue is not None
    request = queue.get_nowait()
    tracker = service.turn_latency_store.get_response(request.response_key)
    assert tracker is not None
    clock[0] = 101.40
    tracker.record_llm_ttft(0.45)
    clock[0] = 101.65
    tracker.record_llm_ttft(0.70)  # every later text delta reports too; TTFT is the first
    clock[0] = 101.90
    tracker.record_tts_ttfa(0.37)
    clock[0] = 102.05
    tracker.record_tts_ttfa(0.15)  # so does each later sentence's first audio
    clock[0] = 102.10
    assert tracker.record_e2e(2.10)
    clock[0] = 102.30
    assert not tracker.record_e2e(2.50)  # and each later sentence's first audio; e2e is the first
    assert tracker.ended_at_s("e2e_s") == 102.10
    clock[0] = 102.60
    tracker.record_llm(1.65)
    ref = _ref(unit, request)

    with caplog.at_level(logging.INFO, logger="speech_to_speech.api.openai_realtime.handlers.response"):
        _finish(unit, request)
    (line,) = [r.getMessage() for r in caplog.records if " latency: " in r.getMessage()]

    stages = _by_name(exporter.get_finished_spans())
    expected = {
        "vad": 0.31,
        "stt": 0.43,
        "llm": 1.65,
        "llm.ttft": 0.45,
        "tts.ttfa": 0.37,
        "e2e": 2.10,
    }
    assert set(stages) == set(expected)
    for name, seconds in expected.items():
        assert _duration_ns(stages[name]) == round(seconds * 1e9)
        parent = stages[name].parent
        assert parent is not None and parent.span_id == _int(ref.root_span_id)
        assert stages[name].context.trace_id == _int(ref.trace_id)
    # The same numbers format_log_line() printed for this turn.
    printed = dict(re.findall(r"(\w+)=(\d+\.\d\d)s", line))
    for name, key in (
        ("stt", "stt"),
        ("llm", "llm"),
        ("tts.ttfa", "tts_ttfa"),
        ("e2e", "e2e"),
        ("vad", "vad_decision"),
    ):
        assert f"{_duration_ns(stages[name]) / 1e9:.2f}" == printed[key]
    # Placement is measured: each stage ends where the tracker recorded it,
    # TTFT and TTFA at the first token and the first audio.
    offset = service.voice_traces._offset_ns
    ends = {"vad": 100.31, "stt": 100.95, "llm.ttft": 101.40, "tts.ttfa": 101.90, "e2e": 102.10, "llm": 102.60}
    for name, end_s in ends.items():
        assert stages[name].end_time == round(end_s * 1e9) + offset
    assert stages["vad"].attributes is not None
    assert stages["vad"].attributes["s2s.vad_settle_s"] == pytest.approx(0.21)
    end_gap = stages["e2e"].end_time - stages["tts.ttfa"].end_time  # type: ignore[operator]
    assert end_gap == pytest.approx(0.20 * 1e9, abs=1_000)


PHONE = "+14708767443"
SENTINELS = ("purple-elephant-sentinel", "system-prompt-sentinel")


def _strings(span: ReadableSpan) -> Iterator[str]:
    yield span.name
    for attributes in [span.attributes, span.resource.attributes, *[e.attributes for e in span.events]]:
        for key, value in (attributes or {}).items():
            yield key
            if isinstance(value, str):
                yield value
    for event in span.events:
        yield event.name


def test_every_span_has_session_and_no_content(unit: SimpleNamespace) -> None:
    exporter = install_memory_tracer()
    service: RealtimeService = unit.service
    service._state(unit.conn).runtime_config.session.instructions = f"You are Sophie. {SENTINELS[1]}"
    service.speculative_turns.start_turn()
    request = _speak(unit, transcript=f"Call me at {PHONE} about the {SENTINELS[0]}")
    _send(unit, request)
    _finish(unit, request)
    service.unregister(unit.conn)

    spans = exporter.get_finished_spans()
    assert len(spans) >= 3
    for span in spans:
        assert span.attributes is not None
        assert span.attributes["langfuse.session.id"] == CALL
        # Langfuse's routing: the proxy's spans of the same turn carry the same two.
        assert span.attributes["langfuse.environment"] == "test"
        assert span.attributes["langfuse.trace.name"] == "voice_turn"
        for text in _strings(span):
            assert re.search(r"\+?\d{10,15}", text) is None
            for sentinel in SENTINELS:
                assert sentinel not in text
            assert not text.startswith(("gen_ai.prompt", "gen_ai.completion"))


class _ExplodingTracer:
    enabled = True

    def emit_root(self, ref: TurnTraceRef, start_ns: int, end_ns: int) -> None:
        raise RuntimeError("export exploded")

    def emit_stages(self, ref: TurnTraceRef, stages: list[StageSpan], status: str) -> None:
        raise RuntimeError("export exploded")

    def shutdown(self, timeout_s: float) -> None:
        return None


def _warnings_from(caplog: pytest.LogCaptureFixture, logger_name: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == logger_name and r.levelno >= logging.WARNING]


def test_emit_failure_never_reaches_pipeline(unit: SimpleNamespace, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=voice_turns.__name__)
    voice_turns.install(_ExplodingTracer())
    service: RealtimeService = unit.service
    service.speculative_turns.start_turn()
    request = _speak(unit)
    tracker = service.turn_latency_store.get_response(request.response_key)
    assert tracker is not None and tracker.trace is not None

    with caplog.at_level(logging.INFO):
        events = _finish(unit, request)  # stage export raises
    assert any(event.type == "response.done" for event in events)
    assert any(" latency: " in record.getMessage() for record in caplog.records)

    service.speculative_turns.start_turn()
    second = _speak(unit, turn_id="turn_2")  # closes turn_1: its root export raises
    assert service.turn_latency_store.get_response(second.response_key) is not None
    service.unregister(unit.conn)  # closes turn_2 the same way
    # Three failed exports on the event loop, one WARNING for the unit.
    assert len(_warnings_from(caplog, voice_turns.__name__)) == 1


def test_bind_failure_logs_once_and_the_request_still_goes(
    unit: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def failing_stage_intervals(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("stage read exploded")

    install_memory_tracer()
    monkeypatch.setattr(service_module, "build_stage_intervals", failing_stage_intervals)
    caplog.set_level(logging.WARNING, logger=service_module.__name__)
    service: RealtimeService = unit.service
    requests = []
    for turn_id in ("turn_1", "turn_2"):
        service.speculative_turns.start_turn()
        requests.append(_speak(unit, turn_id=turn_id))  # each bind raises inside _bind_turn_trace

    for request in requests:
        tracker = service.turn_latency_store.get_response(request.response_key)
        assert tracker is not None and tracker.trace is None
        assert _send(unit, request) == {CLIENT_SESSION_HEADER: CALL}
    assert len(_warnings_from(caplog, service_module.__name__)) == 1


def test_tts_requests_get_no_trace_headers(unit: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """DoD 10: the voice (Sophie's TTS shim on :9744) is not the LLM proxy."""
    from speech_to_speech.pipeline.messages import TTSInput
    from speech_to_speech.TTS import openai_compatible_handler as tts_module
    from speech_to_speech.TTS.openai_compatible_handler import OpenAICompatibleTTSHandler

    install_memory_tracer()
    unit.service.speculative_turns.start_turn()
    speech = _speak(unit)
    ref = _ref(unit, speech)
    seen: list[Any] = []

    def respond(request: Any) -> Any:
        seen.append(request.headers)
        return tts_module.httpx.Response(
            200,
            headers={"Content-Type": "audio/pcm"},
            content=np.arange(2400, dtype="<i2").tobytes(),
            request=request,
        )

    client = tts_module.httpx.AsyncClient(transport=tts_module.httpx.MockTransport(respond))
    monkeypatch.setattr(tts_module.httpx, "AsyncClient", lambda **_kwargs: client)
    monkeypatch.setattr(OpenAICompatibleTTSHandler, "warmup", lambda self: None)
    handler = OpenAICompatibleTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={"sample_rate": 24000, "blocksize": 512},
    )
    token = base_mod._REQUEST_TURN_TRACE.set(ref)  # a traced model request is in flight
    try:
        audio = list(
            handler.process(
                TTSInput(
                    text="Hello",
                    response_key=speech.response_key,
                    turn_id=speech.turn_id,
                    turn_revision=speech.turn_revision,
                )
            )
        )
    finally:
        base_mod._REQUEST_TURN_TRACE.reset(token)

    assert audio
    assert seen
    for headers in seen:
        for name in ("traceparent", "tracestate", "baggage", TURN_ID_HEADER, CLIENT_SESSION_HEADER):
            assert name not in headers
