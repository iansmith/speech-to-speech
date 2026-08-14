"""ElevenLabs streaming TTS handler.

The only NETWORK TTS backend in this package. Two consequences follow from that
and shape everything below.

First, privacy: the assistant's reply text leaves the machine on every turn.
Nothing else in a local pipeline does that, so it is never a default — the
operator has to ask for ``--tts elevenlabs``.

Second, failure: a local model that is loaded either works or the process never
started. A network call can fail on any individual turn, forever, while the
process stays healthy. So every failure path here degrades to *silence for one
sentence* and logs, rather than raising into the pipeline thread and taking the
call down. The one exception is startup — a missing API key or voice ID refuses
to start, because that is a misconfiguration rather than a transient.

What this backend does NOT have to do, and why it is simpler than its
neighbours: no MLX lock (it holds no accelerator, so it never contends with
Parakeet STT), no model load, no warm-up, and no resampling in the default
configuration — ``pcm_16000`` is requested precisely because it is the
pipeline's own rate.
"""

from __future__ import annotations

import logging
import os
from queue import Queue
from threading import Event
from time import perf_counter
from typing import Any, Iterator, Optional

import httpx
import numpy as np

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)

# The pipeline's sample rate. qwen3_tts_handler.PIPELINE_SR and
# api.openai_realtime.service.PIPELINE_SAMPLE_RATE are the two pre-existing
# definitions of this same 16000; consolidating them is a refactor across
# modules this change has no business making, so this names the duplication
# rather than quietly adding a third unexplained copy.
PIPELINE_SR = 16000

# Sample rate carried by each supported output_format. ElevenLabs encodes the
# rate in the format name, so this is a parse rather than a configuration.
_PCM_FORMAT_RATES = {
    "pcm_8000": 8000,
    "pcm_16000": 16000,
    "pcm_22050": 22050,
    "pcm_24000": 24000,
    "pcm_44100": 44100,
}

# Bytes per streamed read. 640 bytes = 320 int16 samples = 20ms at 16kHz, which
# is one Twilio media frame — small enough that first audio is not held back
# waiting for a buffer to fill.
_READ_CHUNK_BYTES = 640


class ElevenLabsTTSHandler(BaseHandler[TTSInput, Any]):
    """Streams synthesized speech from the ElevenLabs HTTP API."""

    def setup(
        self,
        should_listen: Event,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_flash_v2_5",
        api_key_env: str = "ELEVEN_LABS_API_KEY",
        base_url: str = "https://api.elevenlabs.io/v1",
        output_format: str = "pcm_16000",
        optimize_streaming_latency: Optional[int] = None,
        timeout_s: float = 10.0,
        stability: Optional[float] = None,
        similarity_boost: Optional[float] = None,
        speed: Optional[float] = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        **_kwargs: Any,
    ) -> None:
        # Parameter names here are the POST-rename_args names: the CLI flags are
        # --elevenlabs_voice_id etc., and s2s_pipeline strips the "elevenlabs"
        # prefix before these kwargs arrive, exactly as it does for
        # responses_api_* -> api_key. Renaming a field means renaming it here too.
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns

        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.output_format = output_format
        self.optimize_streaming_latency = optimize_streaming_latency
        self.timeout_s = float(timeout_s)
        self.voice_settings = _voice_settings(
            stability, similarity_boost, speed
        )

        if self.output_format not in _PCM_FORMAT_RATES:
            raise ValueError(
                f"--elevenlabs_output_format {self.output_format!r} is not a supported PCM format. "
                f"This handler decodes raw PCM only (int16 little-endian); mp3/ulaw formats would need "
                f"a decoder. Supported: {', '.join(sorted(_PCM_FORMAT_RATES))}."
            )
        self.source_sr = _PCM_FORMAT_RATES[self.output_format]

        # Refuse to start rather than fail on the first caller. A missing key or
        # voice is a misconfiguration, and the loud version costs one restart
        # while the quiet version costs a phone call with no audio and nothing
        # in the log that names the cause.
        if not voice_id:
            raise ValueError(
                "--elevenlabs_voice_id is required when --tts elevenlabs. Voice IDs are account-scoped, "
                "so there is no sane default to fall back to: GET /v1/voices, or read it off "
                "https://elevenlabs.io/app/voice-lab."
            )
        self.voice_id = voice_id

        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise ValueError(
                f"{api_key_env} is unset or empty in this process's environment, and "
                f"--tts elevenlabs cannot work without it. Note the environment inherited here is the "
                f"one the SUPERVISOR started with, not your current shell — if the key was exported "
                f"after the supervisor launched, restarting just this server will not pick it up."
            )
        self._api_key = api_key

        self._client = httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=min(5.0, self.timeout_s)))

        logger.info(
            "ElevenLabs TTS ready: model=%s voice=%s format=%s (key from $%s)",
            self.model_id,
            self.voice_id,
            self.output_format,
            api_key_env,
        )

    # ── pipeline entry point ────────────────────────────────────────────────

    def process(self, tts_input: TTSInput) -> Iterator[Any]:
        speculative_turns = self.speculative_turns

        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id, tts_input.turn_revision
            ):
                return
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id, tts_input.turn_revision
        ):
            logger.debug(
                "Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision
            )
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        text = (tts_input.text or "").strip()
        if not text:
            return

        voice_id = self._voice_for(tts_input)
        yield from self._stream(text, voice_id, tts_input)

    # ── synthesis ───────────────────────────────────────────────────────────

    def _stream(self, text: str, voice_id: str, tts_input: TTSInput) -> Iterator[np.ndarray]:
        """Stream one sentence. Any failure yields nothing and logs."""
        gen = self.cancel_scope.generation if self.cancel_scope else None
        start = perf_counter()
        first_chunk = True
        leftover = b""
        total_samples = 0

        try:
            with self._client.stream(
                "POST",
                f"{self.base_url}/text-to-speech/{voice_id}/stream",
                headers={"xi-api-key": self._api_key, "accept": "audio/*"},
                params=self._params(),
                json=self._body(text),
            ) as resp:
                if resp.status_code != 200:
                    # read() before accessing text on a streamed response.
                    resp.read()
                    logger.error(
                        "ElevenLabs TTS %s for voice=%s: %s",
                        resp.status_code,
                        voice_id,
                        resp.text[:300],
                    )
                    return

                for raw in resp.iter_bytes(chunk_size=_READ_CHUNK_BYTES):
                    if self._is_stale(gen):
                        logger.info("ElevenLabs TTS cancelled (interruption)")
                        return
                    if not raw:
                        continue

                    # int16 is 2 bytes and a network read can split one in half,
                    # so an odd trailing byte is carried into the next chunk
                    # rather than dropped — dropping it would shift every
                    # subsequent sample by one byte and turn the rest of the
                    # sentence into noise.
                    buf = leftover + raw
                    usable = len(buf) - (len(buf) % 2)
                    leftover = buf[usable:]
                    if usable == 0:
                        continue

                    audio = np.frombuffer(buf[:usable], dtype="<i2")
                    if audio.size == 0:
                        continue

                    if first_chunk:
                        logger.info("ElevenLabs TTFA: %.2fs (%s)", perf_counter() - start, self.model_id)
                        self._log_first_audio_latency(tts_input)
                        first_chunk = False

                    audio = self._to_pipeline_sr(audio)
                    total_samples += audio.size
                    yield audio

        except httpx.TimeoutException:
            logger.error(
                "ElevenLabs TTS timed out after %.1fs — this sentence is silent, the call continues",
                self.timeout_s,
            )
            return
        except Exception:
            logger.exception("ElevenLabs TTS failed — this sentence is silent, the call continues")
            return

        if total_samples:
            logger.info(
                "ElevenLabs generated %.2fs audio in %.2fs (%s)",
                total_samples / PIPELINE_SR,
                perf_counter() - start,
                self.model_id,
            )

    # ── helpers ─────────────────────────────────────────────────────────────

    def _params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"output_format": self.output_format}
        if self.optimize_streaming_latency is not None:
            params["optimize_streaming_latency"] = self.optimize_streaming_latency
        return params

    def _body(self, text: str) -> dict[str, Any]:
        body: dict[str, Any] = {"text": text, "model_id": self.model_id}
        if self.voice_settings:
            body["voice_settings"] = self.voice_settings
        return body

    def _voice_for(self, tts_input: TTSInput) -> str:
        """Per-response voice override, mirroring the other handlers.

        Consumer-selected voice arrives either on the response or on the
        session's runtime config (AATK-83 sends it in the realtime handshake).
        Falls back to the configured voice when neither names one.
        """
        response = tts_input.response
        if response and response.audio and response.audio.output and response.audio.output.voice:
            return str(response.audio.output.voice)

        runtime_config = tts_input.runtime_config
        if runtime_config:
            audio_cfg = runtime_config.session.audio
            audio_output = audio_cfg.output if audio_cfg is not None else None
            if audio_output is not None and audio_output.voice:
                return str(audio_output.voice)

        return self.voice_id

    def _is_stale(self, gen: int | None) -> bool:
        return gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen)

    def _to_pipeline_sr(self, audio: np.ndarray) -> np.ndarray:
        if self.source_sr == PIPELINE_SR:
            return audio
        from scipy.signal import resample_poly

        gcd = np.gcd(PIPELINE_SR, self.source_sr)
        resampled = resample_poly(audio.astype(np.float32), up=PIPELINE_SR // gcd, down=self.source_sr // gcd)
        return np.clip(resampled, -32768, 32767).astype(np.int16)

    def _log_first_audio_latency(self, tts_input: TTSInput) -> None:
        """Emit the same end-to-end line the qwen3 handler emits.

        Deliberately the identical message text: it is what the latency
        tooling greps for, so a backend swap must not silently stop reporting.
        """
        if tts_input.speech_stopped_at_s is None:
            return
        latency_s = perf_counter() - tts_input.speech_stopped_at_s
        if latency_s < 0:
            return
        logger.info(
            "Last speech detected to first speech out: %.3fs (turn=%s rev=%s)",
            latency_s,
            tts_input.turn_id,
            tts_input.turn_revision,
        )

    def cleanup(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()
        logger.info("ElevenLabs TTS handler cleaned up")


def _voice_settings(
    stability: Optional[float], similarity_boost: Optional[float], speed: Optional[float]
) -> dict[str, float]:
    """Only send settings the operator actually set.

    An omitted block lets the voice's own saved settings apply; sending
    defaults would silently override whatever was tuned in Voice Lab.
    """
    settings: dict[str, float] = {}
    if stability is not None:
        settings["stability"] = float(stability)
    if similarity_boost is not None:
        settings["similarity_boost"] = float(similarity_boost)
    if speed is not None:
        settings["speed"] = float(speed)
    return settings
