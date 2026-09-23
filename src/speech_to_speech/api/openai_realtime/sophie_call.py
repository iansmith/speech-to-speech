"""Sophie's call id, carried from session.update onto the language-model request.

aatoolkit's WithSessionID writes ``client_session_id`` into the session.update
handshake. The pooled pipeline unit must stamp that value on the outbound
chat-completions request and clear it when the unit is released, or the next
session that claims the unit is spliced as the previous caller.
"""

from __future__ import annotations

from typing import Any

SOPHIE_CALL_HEADER = "X-Sophie-Call-Id"


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
    for handler in getattr(unit, "handlers", ()):
        setter = getattr(handler, "set_sophie_call_id", None)
        if setter is not None:
            setter(call_id)


def clear_sophie_call_id(unit: Any) -> None:
    set_sophie_call_id(unit, "")
