"""SOP-625: a pooled unit must not keep the previous session's Sophie call id."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from queue import Queue
from threading import Event
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from openai import OpenAI

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.sophie_call import (
    CLIENT_SESSION_HEADER,
    CLIENT_TURN_ID_HEADER,
    CLIENT_TURN_REVISION_HEADER,
    TRACEPARENT_HEADER,
    TURN_ID_HEADER,
    client_session_id,
    set_sophie_call_id,
)
from speech_to_speech.api.openai_realtime.websocket_router import (
    _dispatch_client_event,
    _release_unit_after_drain,
    claim_idle_unit,
)
from speech_to_speech.LLM import base_openai_compatible_language_model as llm_module
from speech_to_speech.LLM.base_openai_compatible_language_model import (
    BaseOpenAICompatibleHandler,
)
from speech_to_speech.LLM.chat import make_user_message
from speech_to_speech.LLM.chat_completions_language_model import (
    ChatCompletionsApiModelHandler,
)
from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler
from speech_to_speech.pipeline.messages import GenerateResponseRequest
from speech_to_speech.utils.request_tracing import REQUEST_ID_HEADER


def _handler() -> ChatCompletionsApiModelHandler:
    return ChatCompletionsApiModelHandler.__new__(ChatCompletionsApiModelHandler)


def test_client_session_header_pins_the_wire_name() -> None:
    # Sophie's LLM proxy reads this exact header (aatoolkit
    # realtime.XClientSessionID, AATK-140). A rename on one side alone drops
    # every voice call's spliced context, so the string is pinned, not derived.
    assert CLIENT_SESSION_HEADER == "X-Client-Session-Id"


def test_turn_trace_headers_pin_the_wire_names() -> None:
    # SOP-838: Sophie's proxy reads tracing.TurnIDHeader and the W3C
    # traceparent by these exact names (iansmith/sophie#589). A rename here
    # alone silently drops every voice turn to the per-request fallback.
    assert TURN_ID_HEADER == "X-Sophie-Turn-Id"
    assert TRACEPARENT_HEADER == "traceparent"


def test_client_turn_headers_pin_the_wire_names() -> None:
    # SOP-853: Sophie's proxy reads the caller turn by these exact names and
    # strips both before forwarding upstream. A rename here alone drops every
    # voice request back to the proxy's anchor heuristic and leaks the new
    # name to the provider.
    assert CLIENT_TURN_ID_HEADER == "X-Client-Turn-Id"
    assert CLIENT_TURN_REVISION_HEADER == "X-Client-Turn-Revision"


def test_session_update_carries_the_client_id() -> None:
    assert client_session_id({"type": "session.update", "session": {"client_session_id": "  abc  "}}) == "abc"
    assert client_session_id({"type": "session.update", "session": {}}) is None
    assert client_session_id({"type": "response.create"}) is None


def test_claim_release_claim_does_not_keep_the_first_id() -> None:
    handler = _handler()
    handler.set_sophie_call_id("first")
    unit = SimpleNamespace(
        index=0,
        handlers=[handler],
        service=SimpleNamespace(unregister=lambda _sid: None),
    )
    session = SimpleNamespace(drained=asyncio.Event(), quarantined_at=None)
    session.drained.set()
    asyncio.run(_release_unit_after_drain(unit, session, "server-session"))
    assert handler.sophie_call_headers() == {}

    handler.set_sophie_call_id("stale-if-claim-forgets")
    unit.session = None
    claimed = claim_idle_unit([unit], None)
    assert claimed is unit
    assert handler.sophie_call_headers() == {}

    handler.set_sophie_call_id("second")
    assert handler.sophie_call_headers() == {CLIENT_SESSION_HEADER: "second"}


def test_chat_request_stamps_the_header_and_omits_it_when_clear() -> None:
    seen: dict[str, object] = {}

    class Completions:
        def create(self, **kwargs: object) -> object:
            seen.clear()
            seen.update(kwargs)
            return object()

    handler = _handler()
    handler.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    handler.model_name = "m"
    handler.stream = False
    handler._extra_body = None
    handler.request_timeout = 1
    handler.set_sophie_call_id("call-1")
    handler._request([{"role": "user", "content": "hi"}], {})
    headers = seen.get("extra_headers")
    assert isinstance(headers, dict)
    assert headers[CLIENT_SESSION_HEADER] == "call-1"

    handler.on_session_end()
    handler._request([{"role": "user", "content": "hi"}], {})
    assert "extra_headers" not in seen


class _Recorder:
    """Stands in for a client endpoint's ``create``; keeps the last call's kwargs."""

    def __init__(self) -> None:
        self.seen: dict[str, object] = {}

    def create(self, **kwargs: object) -> object:
        self.seen.clear()
        self.seen.update(kwargs)
        return object()


def _responses_handler(recorder: _Recorder) -> ResponsesApiModelHandler:
    handler = ResponsesApiModelHandler.__new__(ResponsesApiModelHandler)
    handler.client = SimpleNamespace(
        responses=recorder,
        chat=SimpleNamespace(completions=recorder),
    )
    handler.model_name = "m"
    handler.stream = False
    handler._extra_body = None
    handler.request_timeout = 1
    return handler


def _assert_stamped_then_clear(
    handler: BaseOpenAICompatibleHandler,
    request: Callable[[Any, dict[str, Any]], Any],
    recorder: _Recorder,
) -> None:
    handler.set_sophie_call_id("call-1")
    request([{"role": "user", "content": "hi"}], {})
    headers = recorder.seen.get("extra_headers")
    assert isinstance(headers, dict)
    assert headers[CLIENT_SESSION_HEADER] == "call-1"

    handler.on_session_end()
    request([{"role": "user", "content": "hi"}], {})
    assert "extra_headers" not in recorder.seen


def test_responses_request_stamps_the_header_and_omits_it_when_clear() -> None:
    # --llm_backend responses-api sends text turns through
    # ResponsesApiModelHandler._request -> client.responses.create.
    recorder = _Recorder()
    handler = _responses_handler(recorder)
    _assert_stamped_then_clear(handler, handler._request, recorder)


def test_responses_audio_request_stamps_the_header_and_omits_it_when_clear() -> None:
    # Audio-input turns on the Responses backend go out as Chat Completions
    # (ResponsesApiModelHandler._request_audio), a separate request path.
    recorder = _Recorder()
    handler = _responses_handler(recorder)
    _assert_stamped_then_clear(handler, handler._request_audio, recorder)


def test_chat_audio_request_stamps_the_header_and_omits_it_when_clear() -> None:
    recorder = _Recorder()
    handler = _handler()
    handler.client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))
    handler.model_name = "m"
    handler.stream = False
    handler._extra_body = None
    handler.request_timeout = 1
    _assert_stamped_then_clear(handler, handler._request_audio, recorder)


class _Transport:
    async def send_events(self, events: list[object]) -> None:
        return None


@pytest.mark.asyncio
async def test_dispatch_stores_the_id_from_the_raw_event() -> None:
    from openai.types.realtime import SessionUpdateEvent

    class ParsingService:
        def parse_client_event(self, event: dict[str, object]) -> SessionUpdateEvent:
            return SessionUpdateEvent.model_validate(event)

        def handle_session_update(self, _sid: str, _event: object) -> None:
            return None

        def build_session_updated(self, _sid: str) -> SimpleNamespace:
            return SimpleNamespace(type="session.updated")

    handler = _handler()
    unit = SimpleNamespace(handlers=[handler], service=ParsingService())
    raw = {
        "type": "session.update",
        "session": {"type": "realtime", "model": "unused", "client_session_id": "from-wire"},
    }
    await _dispatch_client_event(unit, "srv", raw, _Transport())
    assert handler.sophie_call_headers() == {CLIENT_SESSION_HEADER: "from-wire"}


def test_claim_release_clears_the_service_turn_traces() -> None:
    """SOP-805: the per-call trace binding joins SOP-625's claim/release clear."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from speech_to_speech.tracing import voice_turns
    from speech_to_speech.tracing.otel_backend import OtelVoiceTracer
    from tests.tracing.memory import STAND_IN_CONFIG

    exporter = InMemorySpanExporter()
    voice_turns.install(OtelVoiceTracer(STAND_IN_CONFIG, processor=SimpleSpanProcessor(exporter)))
    call = "0f8e4c2a-6b1d-4e3f-9a7c-5d2b8e1f4a60"
    service = RealtimeService()
    unit = SimpleNamespace(index=0, handlers=[_handler()], service=service, session=None)
    set_sophie_call_id(unit, call)
    conn = service.register()
    service.bind_response_latency_tracker(conn, "r1", turn_id="turn_1", turn_revision=0)
    assert service.voice_traces.ref_for("turn_1", 0, "speech", 0.0, 0.0) is not None

    session = SimpleNamespace(drained=asyncio.Event(), quarantined_at=None)
    session.drained.set()
    asyncio.run(_release_unit_after_drain(unit, session, conn))

    assert service.voice_traces.ref_for("turn_1", 0, "speech", 0.0, 0.0) is None
    assert len(exporter.get_finished_spans()) == 1  # the open turn root, exactly once

    set_sophie_call_id(unit, call)
    assert claim_idle_unit([unit], None) is unit
    assert service.voice_traces.ref_for("turn_1", 0, "speech", 0.0, 0.0) is None
    assert len(exporter.get_finished_spans()) == 1


_CHAT_REPLY = {
    "id": "provider-response",
    "object": "chat.completion",
    "created": 0,
    "model": "synthetic",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Synthetic."}}],
}
_RESPONSES_REPLY = {
    "id": "provider-response",
    "object": "response",
    "created_at": 0,
    "status": "completed",
    "model": "synthetic",
    "output": [
        {
            "id": "message-one",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Synthetic.", "annotations": []}],
        }
    ],
}


@pytest.mark.parametrize("backend", ["chat", "responses", "responses-audio"])
def test_request_id_and_sophie_headers_reach_one_request(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    # SOP-1113: upstream's request_options puts X-Speech-Request-Id into
    # extra_headers before the backend's _request runs, and the backend adds
    # the Sophie headers. Both must reach the provider on the same request;
    # replacing the dict instead of merging it drops the request id.
    sent: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=_RESPONSES_REPLY if request.url.path.endswith("/responses") else _CHAT_REPLY)

    client = OpenAI(
        api_key="provider-secret-sentinel",
        base_url="https://provider.test/v1",
        # A mount, so ProviderConnectAborter still finds a real default pool.
        http_client=httpx.Client(mounts={"all://": httpx.MockTransport(respond)}),
    )
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: client)
    handler_class = ChatCompletionsApiModelHandler if backend == "chat" else ResponsesApiModelHandler
    monkeypatch.setattr(handler_class, "warmup", lambda self: None)
    handler = handler_class(Event(), Queue(), Queue(), setup_kwargs={"model_name": "synthetic", "stream": False})
    try:
        handler.set_sophie_call_id("call-1113")
        config = RuntimeConfig()
        config.chat.add_item(make_user_message("synthetic input"))
        request = GenerateResponseRequest(runtime_config=config, turn_id="turn_1", turn_revision=0)
        if backend == "responses-audio":
            request.audio = np.zeros(160, dtype=np.float32)
            request.audio_sample_rate = 16_000
        list(handler.process(request))
        assert len(sent) == 1
        assert sent[0].headers.get(REQUEST_ID_HEADER)
        assert sent[0].headers.get(CLIENT_SESSION_HEADER) == "call-1113"
    finally:
        handler.cleanup()
