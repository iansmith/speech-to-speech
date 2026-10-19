"""Sophie's per-turn voice trace ids, derived exactly as Sophie derives them.

SOP-838 (iansmith/sophie#589, ``internal/tracing/voiceturn.go``) is the
contract: Sophie's LLM proxy validates an inbound ``traceparent`` by checking
that its trace id equals ``VoiceTurnTraceID(call, turn)``, and expects s2s to
export the turn's parentless root span under ``VoiceTurnRootSpanID(call,
turn)``. Both are first-bytes-of-sha256 over FROZEN namespaces, so this module
is stdlib only and has no OTel dependency.

The turn id that goes on the wire is not the internal ``turn_N``: that
restarts on every ``SpeculativeTurnTracker.reset()``, and Sophie requires
the id to be unique within a call. :func:`wire_turn_id` appends a per-epoch
nonce of lowercase letters, so it can never add a digit run that Sophie's
PII check (``redact.go``) reads as a phone number.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import string
import uuid

# FROZEN: changing either breaks the match with Sophie's proxy.
TRACE_NAMESPACE = b"sophie/voice-turn/trace/v1\x00"
ROOT_SPAN_NAMESPACE = b"sophie/voice-turn/root-span/v1\x00"

# Sophie's TurnRef.ForVoiceTurn accepts a turn id only in this shape.
TURN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

EPOCH_LENGTH = 8


def normalize_call_uuid(raw: str) -> str | None:
    """The call id as Sophie spells it (lowercase, dashed), or None if not a uuid."""
    candidate = raw.strip()
    if not candidate:
        return None
    try:
        return str(uuid.UUID(candidate))
    except ValueError:
        return None


def _derive(namespace: bytes, call_uuid: str, turn_id: str, size: int) -> bytes:
    digest = hashlib.sha256(namespace + call_uuid.encode() + b"\x00" + turn_id.encode()).digest()[:size]
    if not any(digest):
        # An all-zero id is invalid in W3C trace context; Sophie sets the last byte.
        digest = digest[:-1] + b"\x01"
    return digest


def voice_turn_trace_id(call_uuid: str, turn_id: str) -> bytes:
    """Sophie's ``VoiceTurnTraceID``: 16 bytes."""
    return _derive(TRACE_NAMESPACE, call_uuid, turn_id, 16)


def voice_turn_root_span_id(call_uuid: str, turn_id: str) -> bytes:
    """Sophie's ``VoiceTurnRootSpanID``: 8 bytes."""
    return _derive(ROOT_SPAN_NAMESPACE, call_uuid, turn_id, 8)


def mint_epoch() -> str:
    """A fresh nonce of lowercase letters only (no digits, see the module doc)."""
    return "".join(secrets.choice(string.ascii_lowercase) for _ in range(EPOCH_LENGTH))


def wire_turn_id(turn_id: str, epoch: str) -> str:
    """The ``X-Sophie-Turn-Id`` value for internal *turn_id* under *epoch*."""
    return f"{turn_id}_{epoch}"


def format_traceparent(trace_id: bytes, span_id: bytes) -> str:
    """A W3C version-00 ``traceparent`` with the sampled flag."""
    return f"00-{trace_id.hex()}-{span_id.hex()}-01"
