"""SOP-805: s2s derives the same per-turn trace ids as Sophie (SOP-838)."""

from __future__ import annotations

import hashlib
import re

import pytest

from speech_to_speech.tracing.voice_ids import (
    TURN_ID_RE,
    format_traceparent,
    mint_epoch,
    normalize_call_uuid,
    voice_turn_root_span_id,
    voice_turn_trace_id,
    wire_turn_id,
)

CALL = "0f8e4c2a-6b1d-4e3f-9a7c-5d2b8e1f4a60"

# Hard-coded from iansmith/sophie#589's PR body (TestVoiceTurnIDs_Golden), not
# recomputed here: a test that derives its expectation with the code under
# test cannot catch a namespace or separator drift.
GOLDEN = {
    "turn_1": ("be303d8a7f3d658574ee0624bd6484d3", "1f43552473080135"),
    "turn_2": ("77892be8c2591b0f6a6feef08dbbfd78", "2bc4f1c3e4fed3e3"),
}


@pytest.mark.parametrize("raw", [CALL, CALL.upper(), f"  {CALL}\n"])
def test_golden_vector_matches_sop_838(raw: str) -> None:
    call_uuid = normalize_call_uuid(raw)
    assert call_uuid == CALL
    for turn_id, (trace_hex, root_hex) in GOLDEN.items():
        assert voice_turn_trace_id(call_uuid, turn_id).hex() == trace_hex
        assert voice_turn_root_span_id(call_uuid, turn_id).hex() == root_hex


def test_all_zero_digest_sets_last_byte_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ZeroDigest:
        def digest(self) -> bytes:
            return bytes(32)

    monkeypatch.setattr(hashlib, "sha256", lambda *_args: _ZeroDigest())
    assert voice_turn_trace_id(CALL, "turn_1") == bytes(15) + b"\x01"
    assert voice_turn_root_span_id(CALL, "turn_1") == bytes(7) + b"\x01"


@pytest.mark.parametrize("raw", ["not-a-uuid", "", "   ", "+14708767443", "0f8e4c2a-6b1d-4e3f-9a7c"])
def test_non_uuid_call_id_is_rejected(raw: str) -> None:
    assert normalize_call_uuid(raw) is None


def test_wire_turn_id_is_safe() -> None:
    epochs = [mint_epoch() for _ in range(1000)]
    for epoch in epochs:
        # Letters only: Sophie's redact.go rejects a turn id whose digit run
        # is shaped like a phone number, so the nonce must never add digits.
        assert re.fullmatch(r"[a-z]{8}", epoch)
        wire = wire_turn_id("turn_9999", epoch)
        assert TURN_ID_RE.fullmatch(wire)
        assert re.search(r"\d{10,}", wire) is None
    assert len(set(epochs)) > 990


def test_traceparent_is_w3c_version_00_sampled() -> None:
    trace_hex, root_hex = GOLDEN["turn_1"]
    assert format_traceparent(bytes.fromhex(trace_hex), bytes.fromhex(root_hex)) == f"00-{trace_hex}-{root_hex}-01"
