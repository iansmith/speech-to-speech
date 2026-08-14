"""Tests for the ElevenLabs TTS handler.

No network. Every test drives a fake streaming client, because the properties
worth pinning here are all about how bytes off a socket are turned into audio
and what happens when that socket misbehaves — none of which need a real API,
and all of which would be untestable if they did.
"""

import logging
from threading import Event

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.TTS.elevenlabs_handler import ElevenLabsTTSHandler

API_KEY_ENV = "ELEVEN_LABS_TEST_KEY"


class _FakeResponse:
    def __init__(self, chunks, status_code=200, text=""):
        self._chunks = chunks
        self.status_code = status_code
        self.text = text
        self.read_called = False

    def read(self):
        self.read_called = True

    def iter_bytes(self, chunk_size=None):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeClient:
    """Records the request and replays canned chunks."""

    def __init__(self, response):
        self._response = response
        self.calls = []

    def stream(self, method, url, headers=None, params=None, json=None):
        self.calls.append({"method": method, "url": url, "headers": headers, "params": params, "json": json})
        return self._response

    def close(self):
        pass


def _handler(monkeypatch, chunks=None, status_code=200, text="", **overrides):
    """Build a handler wired to a fake client, without touching the network.

    setup() now ends in warmup(), which is a real HTTP call. It is stubbed for
    the duration of setup only, and the genuine method is re-bound as
    .real_warmup so the warmup tests below exercise the actual code rather than
    the stub -- a stub that tested itself would pass no matter what warmup did.
    """
    monkeypatch.setenv(API_KEY_ENV, "test-key-value")
    real_warmup = ElevenLabsTTSHandler.warmup
    monkeypatch.setattr(ElevenLabsTTSHandler, "warmup", lambda self: None)

    h = object.__new__(ElevenLabsTTSHandler)
    kwargs = dict(voice_id="voice-abc", api_key_env=API_KEY_ENV)
    kwargs.update(overrides)
    h.setup(Event(), **kwargs)
    h._client = _FakeClient(_FakeResponse(chunks or [], status_code=status_code, text=text))
    h.real_warmup = real_warmup.__get__(h)
    return h


def _tts_input(text="hello there", **kw):
    return TTSInput(text=text, **kw)


def _pcm(samples):
    return np.array(samples, dtype="<i2").tobytes()


# ── configuration ────────────────────────────────────────────────────────────


def test_setup_refuses_to_start_without_a_voice_id(monkeypatch):
    """A missing voice is a misconfiguration, not a per-turn failure.

    Starting anyway would mean a phone call that produces no audio with nothing
    in the log naming the cause, which costs far more than a refused start.
    """
    monkeypatch.setenv(API_KEY_ENV, "test-key-value")
    h = object.__new__(ElevenLabsTTSHandler)
    with pytest.raises(ValueError, match="elevenlabs_voice_id"):
        h.setup(Event(), voice_id=None, api_key_env=API_KEY_ENV)


def test_setup_refuses_to_start_without_an_api_key(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    h = object.__new__(ElevenLabsTTSHandler)
    with pytest.raises(ValueError, match=API_KEY_ENV):
        h.setup(Event(), voice_id="voice-abc", api_key_env=API_KEY_ENV)


def test_setup_rejects_a_non_pcm_output_format(monkeypatch):
    """The handler decodes raw PCM. mp3 would be accepted by the API and would
    arrive as bytes that np.frombuffer happily reinterprets as garbage audio,
    so this has to fail at setup rather than produce noise on a call."""
    monkeypatch.setenv(API_KEY_ENV, "test-key-value")
    h = object.__new__(ElevenLabsTTSHandler)
    with pytest.raises(ValueError, match="pcm"):
        h.setup(Event(), voice_id="voice-abc", api_key_env=API_KEY_ENV, output_format="mp3_44100_128")


def test_api_key_is_sent_as_a_header_and_never_as_a_query_param(monkeypatch):
    """Query strings land in proxy and server access logs; headers do not."""
    h = _handler(monkeypatch, chunks=[_pcm([1] * 320)])
    list(h.process(_tts_input()))

    call = h._client.calls[0]
    assert call["headers"]["xi-api-key"] == "test-key-value"
    assert "test-key-value" not in str(call["params"])
    assert "test-key-value" not in call["url"]


def test_voice_settings_are_omitted_unless_explicitly_set(monkeypatch):
    """An omitted block lets the voice's own Voice Lab settings apply. Sending
    defaults would silently override whatever was tuned there."""
    h = _handler(monkeypatch, chunks=[_pcm([1] * 320)])
    list(h.process(_tts_input()))
    assert "voice_settings" not in h._client.calls[0]["json"]


def test_voice_settings_are_sent_when_set(monkeypatch):
    h = _handler(monkeypatch, chunks=[_pcm([1] * 320)], stability=0.4, speed=1.1)
    list(h.process(_tts_input()))
    assert h._client.calls[0]["json"]["voice_settings"] == {"stability": 0.4, "speed": 1.1}


# ── decoding ─────────────────────────────────────────────────────────────────


def test_streams_int16_audio(monkeypatch):
    h = _handler(monkeypatch, chunks=[_pcm([100, -100, 300]), _pcm([7])])
    out = list(h.process(_tts_input()))

    assert [c.dtype for c in out] == [np.dtype("<i2")] * 2
    assert list(np.concatenate(out)) == [100, -100, 300, 7]


def test_a_sample_split_across_two_reads_is_not_corrupted(monkeypatch):
    """int16 is two bytes and a socket read can land mid-sample.

    Dropping the odd trailing byte would shift every following sample by one
    byte, turning the rest of the sentence into noise — audible, and the kind
    of thing that gets misread as "the voice model sounds bad".
    """
    payload = _pcm([1000, -2000, 3000])
    h = _handler(monkeypatch, chunks=[payload[:3], payload[3:]])

    out = np.concatenate(list(h.process(_tts_input())))
    assert list(out) == [1000, -2000, 3000]


def test_resamples_when_the_format_rate_differs_from_the_pipeline(monkeypatch):
    """pcm_16000 is the default precisely so this path is not taken, but a
    non-default rate must still arrive at the pipeline's rate."""
    h = _handler(monkeypatch, chunks=[_pcm([0] * 480)], output_format="pcm_24000")
    out = np.concatenate(list(h.process(_tts_input())))

    assert out.dtype == np.int16
    assert out.size == pytest.approx(320, rel=0.1)  # 480 @24k -> ~320 @16k


# ── failure is silence for one sentence, never a dead call ───────────────────


def test_a_non_200_yields_no_audio_and_does_not_raise(monkeypatch):
    h = _handler(monkeypatch, chunks=[_pcm([1] * 320)], status_code=401, text="unauthorized")
    assert list(h.process(_tts_input())) == []


def test_a_mid_stream_exception_yields_what_arrived_and_does_not_raise(monkeypatch):
    """The pipeline thread must survive a network failure. Half a sentence of
    audio then silence is recoverable; an exception out of process() is not."""

    def _explode(chunk_size=None):
        yield _pcm([5] * 320)
        raise OSError("connection reset")

    h = _handler(monkeypatch)
    h._client._response.iter_bytes = _explode

    out = list(h.process(_tts_input()))
    assert len(out) == 1 and out[0].size == 320


def test_empty_text_makes_no_request(monkeypatch):
    h = _handler(monkeypatch, chunks=[_pcm([1] * 320)])
    assert list(h.process(_tts_input(text="   "))) == []
    assert h._client.calls == []


# ── pipeline protocol ────────────────────────────────────────────────────────


def test_end_of_response_yields_the_done_sentinel(monkeypatch):
    h = _handler(monkeypatch)
    assert list(h.process(EndOfResponse())) == [AUDIO_RESPONSE_DONE]


def test_emits_the_shared_end_to_end_latency_line(monkeypatch, caplog):
    """The message text is deliberately identical to the qwen3 handler's.

    scratch/s2s-latency.sh greps for exactly this string, so a TTS backend swap
    must not silently stop reporting the number the whole experiment turns on.
    """
    from time import perf_counter

    h = _handler(monkeypatch, chunks=[_pcm([1] * 320)])
    with caplog.at_level(logging.INFO):
        list(h.process(_tts_input(speech_stopped_at_s=perf_counter())))

    assert any("Last speech detected to first speech out" in r.message for r in caplog.records)


def test_no_latency_line_when_the_turn_carries_no_speech_stop(monkeypatch, caplog):
    h = _handler(monkeypatch, chunks=[_pcm([1] * 320)])
    with caplog.at_level(logging.INFO):
        list(h.process(_tts_input()))

    assert not any("Last speech detected to first speech out" in r.message for r in caplog.records)


# ── startup warmup ───────────────────────────────────────────────────────────
#
# Added after a live call on 2026-08-14 produced a caller-facing silence whose
# only trace was a 402 in the log: "Free users cannot use library voices via the
# API." That is a CONFIGURATION error — it would have failed identically on
# every turn of every call forever — but it was discovered by dialling, because
# nothing exercised the API until a caller did. Every local TTS handler here
# warms up at setup; this one did not, so it alone could start "healthy" while
# being incapable of producing a single sample.
#
# The split below is the point. A 4xx is the operator's to fix and must refuse
# to start. A timeout or a 5xx is the provider having a bad minute, and must NOT
# stop the fleet coming up — the per-turn path already degrades those to silence
# for one sentence.


def test_warmup_refuses_to_start_on_a_4xx(monkeypatch):
    """402/401/404 are configuration, and configuration must fail at startup."""
    h = _handler(monkeypatch, chunks=[], status_code=402, text='{"detail":{"message":"paid plan required"}}')
    with pytest.raises(ValueError, match="402"):
        h.real_warmup()


def test_warmup_4xx_message_carries_the_provider_text(monkeypatch):
    """The operator needs the provider's own words. 'TTS warmup failed' sends
    someone to the wrong place; 'Free users cannot use library voices' does not."""
    h = _handler(monkeypatch, chunks=[], status_code=402, text='{"detail":{"message":"paid plan required"}}')
    with pytest.raises(ValueError, match="paid plan required"):
        h.real_warmup()


def test_warmup_tolerates_a_5xx(monkeypatch, caplog):
    """A provider blip must not stop the fleet from starting."""
    h = _handler(monkeypatch, chunks=[], status_code=503, text="upstream unavailable")
    with caplog.at_level(logging.WARNING):
        h.real_warmup()
    assert any("503" in r.message for r in caplog.records)


def test_warmup_tolerates_a_timeout(monkeypatch, caplog):
    import httpx as _httpx

    h = _handler(monkeypatch, chunks=[])

    def _timeout(*a, **kw):
        raise _httpx.TimeoutException("slow")

    h._client.stream = _timeout
    with caplog.at_level(logging.WARNING):
        h.real_warmup()
    assert any("warmup" in r.message.lower() for r in caplog.records)


def test_warmup_passes_on_audio(monkeypatch, caplog):
    h = _handler(monkeypatch, chunks=[_pcm([1] * 320)])
    with caplog.at_level(logging.INFO):
        h.real_warmup()
    assert any("warmed up" in r.message for r in caplog.records)


def test_warmup_refuses_when_the_api_returns_no_audio(monkeypatch):
    """A 200 with an empty body is still a handler that cannot speak."""
    h = _handler(monkeypatch, chunks=[])
    with pytest.raises(ValueError, match="no audio"):
        h.real_warmup()
