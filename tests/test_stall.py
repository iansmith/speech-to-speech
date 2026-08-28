"""Holding-phrase selection and loading."""

import random

import pytest

from speech_to_speech.stall import StallClips


def _write(tmp_path, name: str, data: bytes = b"\x01\x02") -> None:
    (tmp_path / name).write_bytes(data)


def test_no_directory_is_a_silent_no_op() -> None:
    """A deployment that configures nothing behaves exactly as before."""
    clips = StallClips.load(None)
    assert len(clips) == 0
    assert clips.next_clip() is None


def test_missing_directory_does_not_raise(tmp_path) -> None:
    """A comfort feature must never be able to fail a startup."""
    clips = StallClips.load(str(tmp_path / "does-not-exist"))
    assert len(clips) == 0
    assert clips.next_clip() is None


def test_loads_only_pcm_and_skips_empty_files(tmp_path) -> None:
    _write(tmp_path, "one.pcm", b"\x01" * 100)
    _write(tmp_path, "two.pcm", b"\x02" * 100)
    _write(tmp_path, "notes.txt", b"not audio")
    _write(tmp_path, "empty.pcm", b"")

    clips = StallClips.load(str(tmp_path))
    assert len(clips) == 2


def test_never_repeats_back_to_back(tmp_path) -> None:
    """A phrase heard twice running reads as a stuck loop.

    That is the impression the whole feature exists to prevent, so it is
    asserted over many draws rather than one -- a single draw would pass by
    luck.
    """
    for i in range(4):
        _write(tmp_path, f"{i}.pcm", bytes([i]) * 50)
    clips = StallClips.load(str(tmp_path), rng=random.Random(1))

    prev = None
    for _ in range(200):
        got = clips.next_clip()
        assert got is not None
        assert got != prev, "the same holding phrase played twice in a row"
        prev = got


def test_a_lone_clip_repeats_rather_than_going_silent(tmp_path) -> None:
    """One phrase repeated is still better than silence.

    The no-repeat rule cannot be honoured with a single clip, and honouring it
    by returning nothing would defeat the purpose.
    """
    _write(tmp_path, "only.pcm", b"\x09" * 50)
    clips = StallClips.load(str(tmp_path))
    assert clips.next_clip() == b"\x09" * 50
    assert clips.next_clip() == b"\x09" * 50


def test_every_clip_is_reachable(tmp_path) -> None:
    """A selection rule that quietly ignores half the recordings is a bug.

    Someone recorded six lines; all six should be heard.
    """
    for i in range(6):
        _write(tmp_path, f"{i}.pcm", bytes([i]) * 50)
    clips = StallClips.load(str(tmp_path), rng=random.Random(7))

    seen = {clips.next_clip() for _ in range(500)}
    assert len(seen) == 6


# --- when a holding phrase is due ----------------------------------------

from speech_to_speech.stall import StallTimer  # noqa: E402


def test_not_due_before_the_threshold() -> None:
    t = StallTimer(after_s=5.0)
    t.arm(100.0)
    assert t.due(102.0) is False
    assert t.due(104.9) is False


def test_due_once_the_threshold_passes() -> None:
    t = StallTimer(after_s=5.0)
    t.arm(100.0)
    assert t.due(105.0) is True


def test_fires_at_most_once_per_wait() -> None:
    """A second phrase for the same wait would stack two apologies."""
    t = StallTimer(after_s=5.0)
    t.arm(100.0)
    assert t.due(105.0) is True
    assert t.due(106.0) is False
    assert t.due(120.0) is False


def test_disarm_means_nothing_is_owed() -> None:
    """Audio went out -- the caller is no longer waiting in silence."""
    t = StallTimer(after_s=5.0)
    t.arm(100.0)
    t.disarm()
    assert t.due(200.0) is False


def test_rearming_does_not_push_the_deadline_back() -> None:
    """Otherwise a pipeline that re-arms faster than the threshold would defer
    the phrase forever, which is exactly the silence being fixed."""
    t = StallTimer(after_s=5.0)
    t.arm(100.0)
    for now in (101.0, 102.0, 103.0, 104.0):
        t.arm(now)
    assert t.due(105.0) is True


def test_a_new_wait_after_disarm_can_fire_again() -> None:
    """Every turn gets its own holding phrase; one per call would be useless."""
    t = StallTimer(after_s=5.0)
    t.arm(100.0)
    assert t.due(105.0) is True
    t.disarm()
    t.arm(200.0)
    assert t.due(205.0) is True
