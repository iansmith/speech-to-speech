from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.signal import resample_poly

# G.711 is defined only at 8 kHz, so the format identifier carries no rate field
# of its own (unlike ``audio/pcm``, which does).
G711_SAMPLE_RATE = 8000

_ULAW = "ulaw"
_ALAW = "alaw"

_FORMAT_CODECS = {"audio/pcmu": _ULAW, "audio/pcma": _ALAW}


def _build_ulaw_decode_table() -> np.ndarray:
    """ITU-T G.711 mu-law expansion, for all 256 codepoints."""
    u = (~np.arange(256, dtype=np.int32)) & 0xFF
    t = ((u & 0x0F) << 3) + 0x84
    t = t << ((u & 0x70) >> 4)
    return np.where(u & 0x80, 0x84 - t, t - 0x84).astype(np.int16)


def _build_alaw_decode_table() -> np.ndarray:
    """ITU-T G.711 A-law expansion, for all 256 codepoints."""
    a = np.arange(256, dtype=np.int32) ^ 0x55
    t = (a & 0x0F) << 4
    seg = (a & 0x70) >> 4
    t = np.where(seg == 0, t + 8, np.where(seg == 1, t + 0x108, (t + 0x108) << np.maximum(seg - 1, 0)))
    return np.where(a & 0x80, t, -t).astype(np.int16)


def _build_ulaw_encode_table() -> np.ndarray:
    """ITU-T G.711 mu-law compression, tabulated over the whole int16 domain.

    Indexed by ``sample + 32768`` so encoding is a single gather.
    """
    seg_uend = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)

    x = np.arange(-32768, 32768, dtype=np.int32) >> 2  # to 14-bit
    mask = np.where(x < 0, 0x7F, 0xFF).astype(np.int32)
    x = np.minimum(np.abs(x), 8159) + 33  # clip, then add BIAS >> 2

    seg = np.searchsorted(seg_uend, x, side="left").astype(np.int32)
    shift = np.minimum(seg + 1, 31)
    uval = (seg << 4) | (np.right_shift(x, shift) & 0x0F)
    return (np.where(seg >= 8, 0x7F, uval) ^ mask).astype(np.uint8)


def _build_alaw_encode_table() -> np.ndarray:
    """ITU-T G.711 A-law compression, tabulated over the whole int16 domain."""
    seg_aend = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF], dtype=np.int32)

    x = np.arange(-32768, 32768, dtype=np.int32) >> 3  # to 13-bit
    negative = x < 0
    mask = np.where(negative, 0x55, 0xD5).astype(np.int32)
    x = np.where(negative, -x - 1, x)

    seg = np.searchsorted(seg_aend, x, side="left").astype(np.int32)
    shift = np.where(seg < 2, 1, np.minimum(seg, 31))
    aval = (seg << 4) | (np.right_shift(x, shift) & 0x0F)
    return (np.where(seg >= 8, 0x7F, aval) ^ mask).astype(np.uint8)


_ULAW_DECODE = _build_ulaw_decode_table()
_ALAW_DECODE = _build_alaw_decode_table()
_ULAW_ENCODE = _build_ulaw_encode_table()
_ALAW_ENCODE = _build_alaw_encode_table()


def ulaw_to_pcm16(data: bytes) -> bytes:
    """Expand G.711 mu-law bytes to little-endian PCM16."""
    return _ULAW_DECODE[np.frombuffer(data, dtype=np.uint8)].tobytes()


def alaw_to_pcm16(data: bytes) -> bytes:
    """Expand G.711 A-law bytes to little-endian PCM16."""
    return _ALAW_DECODE[np.frombuffer(data, dtype=np.uint8)].tobytes()


def pcm16_to_ulaw(pcm: bytes) -> bytes:
    """Compress little-endian PCM16 to G.711 mu-law bytes."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
    return _ULAW_ENCODE[samples + 32768].tobytes()


def pcm16_to_alaw(pcm: bytes) -> bytes:
    """Compress little-endian PCM16 to G.711 A-law bytes."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
    return _ALAW_ENCODE[samples + 32768].tobytes()


def resample(audio_int16: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample int16 PCM audio between sample rates using polyphase filtering."""
    if from_rate == to_rate:
        return audio_int16
    samples = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32) / 32768.0
    gcd = np.gcd(to_rate, from_rate)
    resampled = resample_poly(samples, up=to_rate // gcd, down=from_rate // gcd)
    return np.clip(resampled * 32768, -32768, 32767).astype(np.int16).tobytes()


def resolve_format(fmt: Any, default_rate: int) -> tuple[Optional[str], int]:
    """Resolve a realtime audio format object to ``(codec, sample_rate)``.

    ``codec`` is ``None`` for linear PCM, in which case the rate comes from the
    format's own ``rate`` field (``audio/pcm`` carries one) and falls back to
    ``default_rate``. The G.711 formats carry no rate field, because the codec
    fixes it at 8 kHz -- reading ``rate`` off one yields ``None``, which is why
    callers must come through here rather than reaching for the attribute.

    An unrecognised or absent format is treated as linear PCM at
    ``default_rate``, preserving the behaviour that predates G.711 support.
    """
    # getattr off an untyped fmt yields Any | None; the codec table is keyed by
    # str, so narrow before the lookup. A non-str (or absent) type is not a known
    # codec and resolves to linear PCM below -- the same result .get(None) gave.
    fmt_type = getattr(fmt, "type", None)
    codec = _FORMAT_CODECS.get(fmt_type) if isinstance(fmt_type, str) else None
    if codec is not None:
        return codec, G711_SAMPLE_RATE
    return None, getattr(fmt, "rate", None) or default_rate


def decode_client_audio(data: bytes, fmt: Any, pipeline_rate: int) -> bytes:
    """Convert inbound client audio to PCM16 at the pipeline's sample rate."""
    codec, rate = resolve_format(fmt, pipeline_rate)
    if codec == _ULAW:
        data = ulaw_to_pcm16(data)
    elif codec == _ALAW:
        data = alaw_to_pcm16(data)
    return resample(data, rate, pipeline_rate)


def encode_client_audio(pcm: bytes, fmt: Any, pipeline_rate: int) -> bytes:
    """Convert outbound PCM16 pipeline audio to the client's declared format."""
    codec, rate = resolve_format(fmt, pipeline_rate)
    pcm = resample(pcm, pipeline_rate, rate)
    if codec == _ULAW:
        return pcm16_to_ulaw(pcm)
    if codec == _ALAW:
        return pcm16_to_alaw(pcm)
    return pcm
