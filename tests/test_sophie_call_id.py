"""SOP-625: a pooled unit must not keep the previous session's Sophie call id."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from speech_to_speech.api.openai_realtime.sophie_call import (
    SOPHIE_CALL_HEADER,
    client_session_id,
)
from speech_to_speech.api.openai_realtime.websocket_router import (
    _dispatch_client_event,
    _release_unit_after_drain,
    claim_idle_unit,
)
from speech_to_speech.LLM.chat_completions_language_model import (
    ChatCompletionsApiModelHandler,
)


def _handler() -> ChatCompletionsApiModelHandler:
    return ChatCompletionsApiModelHandler.__new__(ChatCompletionsApiModelHandler)


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
    assert handler.sophie_call_headers() == {SOPHIE_CALL_HEADER: "second"}


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
    assert headers[SOPHIE_CALL_HEADER] == "call-1"

    handler.on_session_end()
    handler._request([{"role": "user", "content": "hi"}], {})
    assert "extra_headers" not in seen


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
    assert handler.sophie_call_headers() == {SOPHIE_CALL_HEADER: "from-wire"}
