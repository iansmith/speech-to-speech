"""G.711 (``audio/pcmu``, ``audio/pcma``) support on the realtime audio path.

Expected values are frozen in ``testdata/g711_golden.json``, taken from CPython's
``audioop`` (the ITU-T G.711 reference). The library under test cannot use
``audioop`` itself: it was removed in Python 3.13, which this project's
``requires-python = ">=3.10"`` admits.
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

import pytest
from openai.types.realtime import InputAudioBufferAppendEvent, RealtimeSessionCreateRequest
from openai.types.realtime.realtime_audio_config import RealtimeAudioConfig
from openai.types.realtime.realtime_audio_config_input import RealtimeAudioConfigInput
from openai.types.realtime.realtime_audio_config_output import RealtimeAudioConfigOutput
from openai.types.realtime.realtime_audio_formats import AudioPCM, AudioPCMA, AudioPCMU

from speech_to_speech.api.openai_realtime.handlers.audio import CHUNK_SIZE_BYTES
from speech_to_speech.api.openai_realtime.utils import (
    alaw_to_pcm16,
    pcm16_to_alaw,
    pcm16_to_ulaw,
    ulaw_to_pcm16,
)

GOLDEN = json.loads((Path(__file__).parent / "testdata" / "g711_golden.json").read_text())

# One 20 ms G.711 telephony frame: 160 samples at 8 kHz, one byte each.
FRAME_BYTES = 160
# The same frame after decoding to PCM16 (x2) and upsampling 8k -> 16k (x2).
FRAME_PIPELINE_BYTES = FRAME_BYTES * 4


def _unpack(pcm: bytes) -> list[int]:
    return [v for (v,) in struct.iter_unpack("<h", pcm)]


def _pack(values: list[int]) -> bytes:
    return b"".join(struct.pack("<h", v) for v in values)


def _session(fmt) -> RealtimeSessionCreateRequest:
    return RealtimeSessionCreateRequest.model_construct(
        type="realtime",
        audio=RealtimeAudioConfig.model_construct(
            input=RealtimeAudioConfigInput.model_construct(format=fmt),
            output=RealtimeAudioConfigOutput.model_construct(format=fmt),
        ),
    )


def _use_format(service, conn_id, fmt) -> None:
    service._state(conn_id).runtime_config.session = _session(fmt)


def _append(service, conn_id, payload: bytes) -> list[bytes]:
    evt = InputAudioBufferAppendEvent(type="input_audio_buffer.append", audio=base64.b64encode(payload).decode("ascii"))
    return service.handle_audio_append(conn_id, evt)


def _delta_payload(events) -> bytes:
    deltas = [e for e in events if getattr(e, "type", None) == "response.output_audio.delta"]
    assert deltas, "expected a response.output_audio.delta event"
    return base64.b64decode(deltas[0].delta)


class TestCodecAgainstGolden:
    """The codec itself, against the frozen ITU reference vectors."""

    def test_ulaw_decode_matches_golden_table(self):
        assert _unpack(ulaw_to_pcm16(bytes(range(256)))) == GOLDEN["ulaw_decode"]

    def test_alaw_decode_matches_golden_table(self):
        assert _unpack(alaw_to_pcm16(bytes(range(256)))) == GOLDEN["alaw_decode"]

    def test_ulaw_encode_matches_golden_probes(self):
        probes = GOLDEN["encode_probes"]
        assert list(pcm16_to_ulaw(_pack(probes["linear"]))) == probes["ulaw"]

    def test_alaw_encode_matches_golden_probes(self):
        probes = GOLDEN["encode_probes"]
        assert list(pcm16_to_alaw(_pack(probes["linear"]))) == probes["alaw"]

    def test_named_ulaw_codepoints(self):
        """The named codepoints from ITU-T G.711, pinned as the contract."""
        decoded = _unpack(ulaw_to_pcm16(bytes(range(256))))
        assert decoded[0xFF] == 0, "0xFF is the mu-law silence codepoint"
        assert decoded[0x7F] == 0, "0x7F also decodes to zero"
        assert decoded[0x00] == -32124
        assert decoded[0x80] == 32124
        assert pcm16_to_ulaw(_pack([0])) == b"\xff"
        assert pcm16_to_ulaw(_pack([32767])) == b"\x80"
        assert pcm16_to_ulaw(_pack([-32768])) == b"\x00"

    def test_ulaw_roundtrip_is_identity_except_0x7f(self):
        """0x7F normalizes to 0xFF: both decode to linear zero, and the encoder
        emits 0xFF for zero. Every other codepoint round-trips unchanged."""
        codes = bytes(range(256))
        roundtripped = pcm16_to_ulaw(ulaw_to_pcm16(codes))
        differing = [(a, b) for a, b in zip(codes, roundtripped) if a != b]
        assert differing == [(0x7F, 0xFF)]

    def test_empty_input_is_empty_output(self):
        assert ulaw_to_pcm16(b"") == b""
        assert pcm16_to_ulaw(b"") == b""
        assert alaw_to_pcm16(b"") == b""
        assert pcm16_to_alaw(b"") == b""


class TestCodecExhaustive:
    """Exhaustive cross-check against the oracle, where the oracle still exists."""

    def test_encode_matches_audioop_for_every_int16(self):
        audioop = pytest.importorskip("audioop", reason="removed in Python 3.13+")
        every = _pack(list(range(-32768, 32768)))
        assert pcm16_to_ulaw(every) == audioop.lin2ulaw(every, 2)
        assert pcm16_to_alaw(every) == audioop.lin2alaw(every, 2)


class TestInboundFraming:
    """One telephony frame in, the right number of pipeline bytes out."""

    def test_pcmu_frame_expands_fourfold_and_is_held_as_remainder(self, service, conn_id):
        _use_format(service, conn_id, AudioPCMU.model_construct(type="audio/pcmu"))
        chunks = _append(service, conn_id, b"\xff" * FRAME_BYTES)
        assert chunks == [], "640 bytes is short of the 1024-byte chunk size"
        assert len(service._state(conn_id).audio_remainder) == FRAME_PIPELINE_BYTES == 640

    def test_pcmu_two_frames_emit_one_chunk(self, service, conn_id):
        _use_format(service, conn_id, AudioPCMU.model_construct(type="audio/pcmu"))
        chunks = _append(service, conn_id, b"\xff" * (FRAME_BYTES * 2))
        assert len(chunks) == 1
        assert len(chunks[0]) == CHUNK_SIZE_BYTES == 1024
        assert len(service._state(conn_id).audio_remainder) == 256

    def test_pcma_frame_expands_fourfold(self, service, conn_id):
        _use_format(service, conn_id, AudioPCMA.model_construct(type="audio/pcma"))
        chunks = _append(service, conn_id, b"\xd5" * FRAME_BYTES)
        assert chunks == []
        assert len(service._state(conn_id).audio_remainder) == FRAME_PIPELINE_BYTES

    def test_pcmu_silence_stays_silent_through_the_pipeline(self, service, conn_id):
        """A frame of mu-law silence must not arrive as full-scale noise, which is
        exactly what reinterpreting the bytes as PCM16 produced."""
        _use_format(service, conn_id, AudioPCMU.model_construct(type="audio/pcmu"))
        _append(service, conn_id, b"\xff" * FRAME_BYTES)
        assert set(_unpack(service._state(conn_id).audio_remainder)) == {0}


class TestOutbound:
    def test_pcmu_delta_is_ulaw_at_8k(self, service, conn_id):
        _use_format(service, conn_id, AudioPCMU.model_construct(type="audio/pcmu"))
        payload = _delta_payload(service.encode_audio_chunk(conn_id, b"\x00" * FRAME_PIPELINE_BYTES))
        assert len(payload) == FRAME_BYTES, "640 pipeline bytes -> 160 mu-law bytes"
        assert payload == b"\xff" * FRAME_BYTES, "PCM16 silence encodes to the 0xFF codepoint"

    def test_pcma_delta_is_alaw_at_8k(self, service, conn_id):
        _use_format(service, conn_id, AudioPCMA.model_construct(type="audio/pcma"))
        payload = _delta_payload(service.encode_audio_chunk(conn_id, b"\x00" * FRAME_PIPELINE_BYTES))
        assert len(payload) == FRAME_BYTES


class TestPCMUnchanged:
    """The audio/pcm path must be byte-for-byte what it was before."""

    def test_pcm_16k_input_passes_through_unresampled(self, service, conn_id):
        _use_format(service, conn_id, AudioPCM.model_construct(rate=16000, type="audio/pcm"))
        payload = _pack(list(range(-256, 256)))  # 512 samples = exactly one chunk
        chunks = _append(service, conn_id, payload)
        assert chunks == [payload]

    def test_pcm_16k_output_passes_through_unresampled(self, service, conn_id):
        _use_format(service, conn_id, AudioPCM.model_construct(rate=16000, type="audio/pcm"))
        audio = _pack(list(range(-160, 160)))
        assert _delta_payload(service.encode_audio_chunk(conn_id, audio)) == audio
