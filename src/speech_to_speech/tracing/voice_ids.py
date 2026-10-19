"""RED stub (SOP-805): API surface only."""

from __future__ import annotations

import re

TURN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def normalize_call_uuid(raw: str) -> str | None:
    return raw


def voice_turn_trace_id(call_uuid: str, turn_id: str) -> bytes:
    return bytes(16)


def voice_turn_root_span_id(call_uuid: str, turn_id: str) -> bytes:
    return bytes(8)


def mint_epoch() -> str:
    return "0"


def wire_turn_id(turn_id: str, epoch: str) -> str:
    return turn_id


def format_traceparent(trace_id: bytes, span_id: bytes) -> str:
    return ""
