"""SOP-805: a stage's end anchor is where the stage was measured to end.

A stage span is drawn as ``[ended_at_s - duration, ended_at_s]``. A stamp
taken later than the measured end (after a turn gate, or after the result
waited in a queue) leaves the duration right but moves the whole span later.
"""

from __future__ import annotations

import json
import time
from queue import Queue
from threading import Event
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import speech_to_speech.VAD.vad_handler as vad_module
from speech_to_speech.pipeline import turn_latency
from speech_to_speech.pipeline.messages import Transcription, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore
from speech_to_speech.STT import openai_compatible_handler as stt_module
from speech_to_speech.STT import streaming_handler
from speech_to_speech.tracing.voice_turns import build_stage_intervals
from tests.test_speculative_turns import _audio_bytes, _StaticVADIterator, _vad_handler_for_iterator
from tests.test_streaming_stt_handler import (
    DIALECT_CASES,
    _ack_session_update,
    _completion_event,
    _FakeSocket,
    _is_final_commit,
    _SocketFactory,
    _vad_final,
)
from tests.test_streaming_stt_handler import _handler as _streaming_handler

# How long the fake turn gate holds a finished transcription before publishing.
GATE_HOLD_S = 0.2


class _FinishedOperation:
    """An HTTP transcription that has already returned."""

    def cancel(self, reason: str = "superseded") -> None:
        return None

    def run(self, cancel_check: Any = None) -> stt_module.HttpTranscriptionResult:
        return stt_module.HttpTranscriptionResult(text="hello")


def test_openai_compatible_stt_anchor_is_the_request_end_not_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stt_module.OpenAICompatibleSTTHandler, "warmup", lambda self: None)
    handler = stt_module.OpenAICompatibleSTTHandler(
        Event(), queue_in=Queue(), queue_out=Queue(), setup_kwargs={"speculative_turns": None}
    )
    store = TurnLatencyStore()
    handler.turn_latency_store = store
    monkeypatch.setattr(handler, "_make_operation", lambda _audio: _FinishedOperation())
    gate_entered_at: list[float] = []
    gate = handler.should_emit_output

    def held_gate(output: Any) -> bool:
        # Stands in for wait_for_gate holding a final while VAD checks for resumed speech.
        gate_entered_at.append(time.perf_counter())
        time.sleep(GATE_HOLD_S)
        return gate(output)

    monkeypatch.setattr(handler, "should_emit_output", held_gate)
    started_at = time.perf_counter()
    try:
        source = VADAudio(audio=np.zeros(160, dtype=np.float32), mode="final", turn_id="turn_1", turn_revision=0)
        assert list(handler.process(source)) == []
        assert isinstance(handler.queue_out.get(timeout=2), Transcription)
    finally:
        handler.cleanup()

    tracker = store.pending_for_turn("turn_1", 0)
    assert tracker is not None and tracker.stt_s is not None
    ended_at = tracker.ended_at_s("stt_s")
    assert ended_at is not None
    assert len(gate_entered_at) == 1
    # The request ended before the gate was entered; the hold is not STT time.
    assert ended_at <= gate_entered_at[0]
    assert ended_at - tracker.stt_s >= started_at


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_streaming_stt_anchor_is_the_provider_final_not_consumption(
    monkeypatch: pytest.MonkeyPatch, handler_type: Any, dialect: str
) -> None:
    now = 100.0
    # One clock for the handler's measurement and the tracker's stamp.
    monkeypatch.setattr(streaming_handler, "perf_counter", lambda: now)
    monkeypatch.setattr(turn_latency, "perf_counter", lambda: now)
    turns = SpeculativeTurnTracker()
    turns.start_turn()

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        nonlocal now
        if _ack_session_update(event, socket, dialect):
            return
        if _is_final_commit(event, dialect):
            now += 0.125
            socket.incoming.put(json.dumps(_completion_event("received", dialect=dialect)))

    handler = _streaming_handler(_SocketFactory(on_send), handler_type=handler_type, tracker=turns)
    store = TurnLatencyStore()
    handler.turn_latency_store = store
    try:
        handler.start_turn("turn_1", 0)
        handler.append_audio(b"\x01\x00" * 512)
        handler.commit_boundary("turn_1", 0)
        pending = handler._session._pending_commits[(handler._session.generation, "turn_1", 0)][0]
        assert pending.done.wait(timeout=1)
        now += 0.6  # the provider's final waits before the STT stage consumes it
        assert [output.text for output in handler.process(_vad_final())] == ["received"]
    finally:
        handler.cleanup()

    tracker = store.pending_for_turn("turn_1", 0)
    assert tracker is not None
    assert tracker.stt_s == pytest.approx(0.125)
    assert tracker.ended_at_s("stt_s") == pytest.approx(100.125)


def test_vad_decision_anchor_is_the_decision_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """VADHandler is the only writer of vad_decision_s, and only its anchor makes a vad span.

    Same clock as test_vad_decision_reaches_terminal_response_metadata: the
    final chunk is received at 10.0 s and the decision taken at 10.04 s, with
    speech ending one 512-sample chunk (0.032 s) before receipt.
    """
    iterator = _StaticVADIterator(
        triggered=False,
        vad_output=[torch.zeros(512) for _ in range(31)],
        last_utterance_active_speech_samples=12 * 512,
    )
    iterator.last_speech_end_sample = 0
    vad = _vad_handler_for_iterator(iterator)
    store = TurnLatencyStore()
    vad.turn_latency_store = store
    clock = iter([10.0, 10.04, 10.04, 10.07])
    monkeypatch.setattr(vad_module, "time", SimpleNamespace(time=lambda: 100.0, perf_counter=lambda: next(clock)))

    assert len(list(vad.process(_audio_bytes()))) == 1

    tracker = store.pending_for_turn("turn_1", 0)
    assert tracker is not None
    assert tracker.vad_decision_s == pytest.approx(0.072)
    assert tracker.ended_at_s("vad_decision_s") == 10.04
    (stage,) = build_stage_intervals(tracker)
    assert stage.name == "vad"
    assert stage.end_perf_s == 10.04
    assert stage.duration_s == pytest.approx(0.072)
