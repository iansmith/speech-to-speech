"""Sophie's call id, carried from session.update onto the language-model request.

aatoolkit's WithSessionID writes ``client_session_id`` into the session.update
handshake. The pooled pipeline unit must stamp that value on the outbound
chat-completions request and clear it when the unit is released, or the next
session that claims the unit is spliced as the previous caller.
"""

from __future__ import annotations

from typing import Any

CLIENT_SESSION_HEADER = "X-Client-Session-Id"
# SOP-805: the turn's wire id and W3C trace context, sent next to the call id
# on each spoken-turn model request only while voice tracing is on. Sophie's
# proxy reads both by these names (SOP-838) and strips all three upstream.
TURN_ID_HEADER = "X-Sophie-Turn-Id"
TRACEPARENT_HEADER = "traceparent"


def client_session_id(raw: dict[str, Any]) -> str | None:
    """The id in a session.update, or None when this event does not carry one.

    None means "this event does not set the id", which is different from an
    empty string. An empty string is a present field and clears the id.
    """
    if raw.get("type") != "session.update":
        return None
    session = raw.get("session")
    if not isinstance(session, dict) or "client_session_id" not in session:
        return None
    value = session.get("client_session_id")
    if not isinstance(value, str):
        return None
    return value.strip()


def set_sophie_call_id(unit: Any, call_id: str) -> None:
    """Stamp *call_id* on every handler that sends it, and on the unit's service.

    The service holds the unit's voice-trace binding (SOP-805), so the clear on
    claim and on release reaches it the same way it reaches the LLM handler.
    """
    for handler in getattr(unit, "handlers", ()):
        setter = getattr(handler, "set_sophie_call_id", None)
        if setter is not None:
            setter(call_id)
    service_setter = getattr(getattr(unit, "service", None), "set_sophie_call_id", None)
    if service_setter is not None:
        service_setter(call_id)


def clear_sophie_call_id(unit: Any) -> None:
    set_sophie_call_id(unit, "")
