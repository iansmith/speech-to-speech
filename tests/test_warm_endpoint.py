"""The /v1/warm endpoint and the prompt-composition it shares with a real turn.

The sharing is the point. A warm only helps if the system message it prefills
is byte-identical to the one the next real turn will send: the backing model
matches a cached prefix from position 0, and the system message IS position 0.
A warm built from a second, drifting copy of the composition logic would match
nothing while still returning 200 -- the most expensive kind of no-op, because
it looks like it is working.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from openai.types.realtime import RealtimeFunctionTool

from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.language_model import compose_system_prompt


TOOLS = [
    RealtimeFunctionTool(
        type="function",
        name="list_upcoming_actions",
        description="List what the caller has asked for so far this call.",
        parameters={"type": "object", "properties": {}},
    )
]


def test_compose_is_deterministic_for_the_same_inputs() -> None:
    """Two composes of the same session must be byte-identical.

    If they are not, no warm can ever hit, because the prefix the warm stores
    and the prefix the turn looks up differ at position 0.
    """
    a, *_ = compose_system_prompt("You are Sophie.", TOOLS, None, wants_audio=True)
    b, *_ = compose_system_prompt("You are Sophie.", TOOLS, None, wants_audio=True)
    assert a == b


def test_compose_matches_what_apply_instructions_puts_in_the_chat() -> None:
    """The warm path and the turn path must produce the same system message.

    _apply_instructions is what a real turn runs. This asserts the extracted
    function it now delegates to returns exactly what lands in the chat, so a
    future edit to one path cannot silently stop the other from matching.
    """
    from speech_to_speech.LLM.language_model import LanguageModelHandler

    chat = Chat(size=8)
    chat.add_item(make_user_message("hello"))

    handler = LanguageModelHandler.__new__(LanguageModelHandler)
    handler._apply_instructions(chat, "You are Sophie.", TOOLS, None, None, True)

    composed, *_ = compose_system_prompt("You are Sophie.", TOOLS, None, wants_audio=True)
    assert chat.init_chat_message is not None
    stored = chat.init_chat_message.content
    if not isinstance(stored, str):
        stored = "".join(getattr(p, "text", "") for p in stored)
    assert stored == composed


def test_compose_includes_the_tools_when_they_are_declared() -> None:
    """A zero-argument tool must reach the prompt like any other.

    list_upcoming_actions declares no arguments at all, which is the shape that
    has caused trouble elsewhere in this stack, so it is the one used here.
    """
    with_tools, *_ = compose_system_prompt("You are Sophie.", TOOLS, None, wants_audio=True)
    without, *_ = compose_system_prompt("You are Sophie.", [], None, wants_audio=True)
    assert "list_upcoming_actions" in with_tools
    assert "list_upcoming_actions" not in without
    assert len(with_tools) > len(without)


def test_compose_honours_tool_choice_none() -> None:
    """tool_choice "none" must drop the tool section, as the turn path does."""
    disabled, *_ = compose_system_prompt("You are Sophie.", TOOLS, "none", wants_audio=True)
    assert "list_upcoming_actions" not in disabled


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
    def __init__(self) -> None:
        self.client = _FakeClient()
        self.model_name = "test-model"


class _FakeUnit:
    def __init__(self, handler: Any) -> None:
        self.handlers = [handler]
        self.index = 0
        self.session = None


@pytest.fixture()
def warm_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, _FakeHandler]:
    from speech_to_speech.api.openai_realtime import websocket_router as wr

    handler = _FakeHandler()
    monkeypatch.setattr(wr, "_first_llm_handler", lambda pool: handler)

    from threading import Event

    app = wr.create_app([], Event())
    return TestClient(app), handler


def test_warm_prefills_with_the_same_prompt_a_turn_would_use(warm_client) -> None:
    """The bytes sent to the model are the composed prompt, not an approximation."""
    client, handler = warm_client

    resp = client.post(
        "/v1/warm",
        json={"instructions": "You are Sophie.", "tools": [t.model_dump() for t in TOOLS]},
    )
    assert resp.status_code == 200
    assert resp.json()["warmed"] is True

    sent = handler.client.chat.completions.calls
    assert len(sent) == 1
    system = sent[0]["messages"][0]
    assert system["role"] == "system"

    expected, *_ = compose_system_prompt("You are Sophie.", TOOLS, None, wants_audio=True)
    assert system["content"] == expected

    # One token: the answer is discarded, the KV state behind it is the point.
    assert sent[0]["max_tokens"] == 1


def test_warm_fails_open_without_instructions(warm_client) -> None:
    """A warm is an optimisation; it must never be able to fail a call."""
    client, handler = warm_client
    resp = client.post("/v1/warm", json={})
    assert resp.status_code == 200
    assert resp.json()["warmed"] is False
    assert handler.client.chat.completions.calls == []


def test_warm_fails_open_when_the_model_errors(warm_client, monkeypatch) -> None:
    client, handler = warm_client

    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("backend down")

    monkeypatch.setattr(handler.client.chat.completions, "create", boom)

    resp = client.post("/v1/warm", json={"instructions": "You are Sophie."})
    assert resp.status_code == 200
    assert resp.json()["warmed"] is False
    assert "backend down" in resp.json()["reason"]
