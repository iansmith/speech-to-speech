"""The /v1/warm endpoint and the prompt-composition it shares with a real turn.

The sharing is the point. A warm only helps if the system message it prefills
is byte-identical to the one the next real turn will send: the backing model
matches a cached prefix from position 0, and the system message IS position 0.
A warm built from a second, drifting copy of the composition logic would match
nothing while still returning 200 -- the most expensive kind of no-op, because
it looks like it is working.

The live mlx path is ChatCompletionsApiModelHandler, whose system message is
composed by BaseOpenAICompatibleHandler._apply_config as
``build_voice_system_prompt(instructions or "")`` with NO tool section: the
tools travel structurally as chat-completions tool params, not as prose in the
prompt. This endpoint composes the same way, so the warmed prefix matches the
first real turn's from position 0.
"""

from threading import Event
from typing import Any

import pytest
from fastapi.testclient import TestClient

from speech_to_speech.LLM.voice_prompt import (
    VOICE_SYSTEM_PROMPT_LEAD,
    build_voice_system_prompt,
)


TOOLS = [
    {
        "type": "function",
        "name": "list_upcoming_actions",
        "description": "List what the caller has asked for so far this call.",
        "parameters": {"type": "object", "properties": {}},
    }
]


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return object()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = type("C", (), {"completions": _FakeCompletions()})()


class _FakeHandler:
    """Stands in for a ChatCompletionsApiModelHandler.

    Carries the four attributes the warm reads off a real handler. A real
    handler carries _extra_body and request_timeout; a warm that omits them
    prefills a prefix rendered under different template kwargs than every real
    request -- succeeds, reports warmed:true, buys nothing.
    """

    def __init__(self) -> None:
        self.client = _FakeClient()
        self.model_name = "test-model"
        self._extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        self.request_timeout = 42.0


@pytest.fixture()
def warm_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, _FakeHandler]:
    from speech_to_speech.api.openai_realtime import websocket_router as wr

    handler = _FakeHandler()
    monkeypatch.setattr(wr, "_first_llm_handler", lambda pool: handler)

    app = wr.create_app([], Event())
    return TestClient(app), handler


def test_warm_composes_with_build_voice_system_prompt(warm_client) -> None:
    """The bytes sent to the model are the live composition, not an approximation.

    The system message must equal ``build_voice_system_prompt(instructions)`` --
    the exact function BaseOpenAICompatibleHandler._apply_config runs -- and so
    must carry the lead skeleton followed by the passed session prompt.
    """
    client, handler = warm_client

    resp = client.post(
        "/v1/warm",
        json={"instructions": "You are Sophie.", "tools": TOOLS},
    )
    assert resp.status_code == 200
    assert resp.json()["warmed"] is True

    sent = handler.client.chat.completions.calls
    assert len(sent) == 1
    system = sent[0]["messages"][0]
    assert system["role"] == "system"

    expected = build_voice_system_prompt("You are Sophie.")
    assert system["content"] == expected
    # The lead skeleton comes first, the session prompt after it: the prefix
    # matching starts at position 0 and must find both.
    assert system["content"].startswith(VOICE_SYSTEM_PROMPT_LEAD.rstrip())
    assert "You are Sophie." in system["content"]

    # Tools travel structurally, exactly as a real chat-completions turn sends
    # them -- NOT as prose in the system message.
    assert "list_upcoming_actions" not in system["content"], (
        "tool prose belongs in the tools array, not the system message"
    )
    assert "tools" in sent[0]
    assert any(t["function"]["name"] == "list_upcoming_actions" for t in sent[0]["tools"])

    # One token: the answer is discarded, the KV state behind it is the point.
    assert sent[0]["max_tokens"] == 1


def test_warm_returns_the_documented_shape(warm_client) -> None:
    """warmed:true plus system_prompt_chars = the composed length Go decodes."""
    client, handler = warm_client
    resp = client.post("/v1/warm", json={"instructions": "You are Sophie.", "tools": TOOLS})
    body = resp.json()
    assert body["warmed"] is True
    assert body["system_prompt_chars"] == len(build_voice_system_prompt("You are Sophie."))


def test_warm_declines_without_a_chat_completions_handler() -> None:
    """A backend with no mlx chat-completions handler declines cleanly.

    warmed:false with a reason, 200 not 500: a warm is an optimisation and must
    never fail a call that has not started. This exercises the real
    _first_llm_handler against an empty pool -- no monkeypatch -- so the
    handler-discovery path is what is under test.
    """
    from speech_to_speech.api.openai_realtime import websocket_router as wr

    client = TestClient(wr.create_app([], Event()))
    resp = client.post("/v1/warm", json={"instructions": "You are Sophie."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["warmed"] is False
    assert body["reason"]


def test_warm_fails_open_without_instructions(warm_client) -> None:
    """No instructions -> declined, and the model is never dialed."""
    client, handler = warm_client
    resp = client.post("/v1/warm", json={})
    assert resp.status_code == 200
    assert resp.json()["warmed"] is False
    assert handler.client.chat.completions.calls == []


def test_warm_fails_open_when_the_model_errors(warm_client, monkeypatch) -> None:
    """A prefill request that raises becomes a reported status, never a 500."""
    client, handler = warm_client

    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("backend down")

    monkeypatch.setattr(handler.client.chat.completions, "create", boom)

    resp = client.post("/v1/warm", json={"instructions": "You are Sophie."})
    assert resp.status_code == 200
    assert resp.json()["warmed"] is False
    assert "backend down" in resp.json()["reason"]


def test_warm_sends_extra_body_because_a_real_turn_does(warm_client) -> None:
    """extra_body is part of the prefix, not decoration.

    _extra_body carries chat_template_kwargs, which the serving stack's chat
    template consumes. A warm that omits it renders the prefix under different
    template kwargs than every real request: it succeeds, reports warmed:true,
    and buys nothing.
    """
    client, handler = warm_client
    client.post("/v1/warm", json={"instructions": "You are Sophie."})

    sent = handler.client.chat.completions.calls
    assert len(sent) == 1
    assert sent[0]["extra_body"] == handler._extra_body


def test_warm_bounds_its_own_request(warm_client) -> None:
    """A warm runs off the event loop; without a timeout it inherits the SDK's
    600s default and a wedged backend holds a thread live calls draw on. The
    real turn passes its handler's request_timeout; so does this."""
    client, handler = warm_client
    client.post("/v1/warm", json={"instructions": "You are Sophie."})

    sent = handler.client.chat.completions.calls
    assert sent[0]["timeout"] == handler.request_timeout


def test_warm_survives_a_handler_without_the_optional_attributes(monkeypatch) -> None:
    """Neither optional field may be required: a handler lacking them still warms."""
    from speech_to_speech.api.openai_realtime import websocket_router as wr

    class _Bare:
        def __init__(self) -> None:
            self.client = _FakeClient()
            self.model_name = "test-model"

    bare = _Bare()
    monkeypatch.setattr(wr, "_first_llm_handler", lambda pool: bare)

    client = TestClient(wr.create_app([], Event()))
    resp = client.post("/v1/warm", json={"instructions": "You are Sophie."})

    assert resp.json()["warmed"] is True
    sent = bare.client.chat.completions.calls
    assert len(sent) == 1
    assert "extra_body" not in sent[0]
    assert "timeout" not in sent[0]
