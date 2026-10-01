"""Holding phrases for the gap between a caller finishing and the model answering.

A local model answering over a phone can take fifteen seconds or more, and to
the caller that is indistinguishable from a dropped call. Measured on one
deployment: a question needing two tool round trips left 38 seconds of silence,
all of it the system working correctly, and the caller asked "are you okay?"

This plays a short recorded line -- "let me look that up" -- once the silence
passes a threshold. It does not make anything faster. It makes the waiting
legible, which is the part the caller actually suffers.

Three properties matter, and each is a way this could be worse than nothing:

* It must be RECORDED, not generated. Generating a holding phrase would cost
  the round trip it exists to cover, so the cure would arrive after the
  disease.
* It must never repeat back to back. A phrase heard twice reads as a stuck
  loop, which is the exact impression it is trying to prevent.
* It must never be the last thing said. It promises an answer; if the real
  answer never arrives, the filler has upgraded a silence into a broken
  promise.

  This module used to claim the caller-facing timeout "stays the backstop, and
  nothing here may reset it". That was wrong, and wrong in the direction that
  stops anyone checking. The phrase is emitted as a backend audio delta, and a
  consumer that resets its idle guard on backend activity -- which is the
  ordinary way to build one -- resets it on the phrase. A hung backend is
  therefore masked for one extra threshold per pending response.

  Bounded, because StallTimer fires at most once per pending response. It stops
  being bounded the moment the phrase is allowed to repeat, so a second-stall
  feature has to solve this rather than inherit it: the phrase would need a
  marker its consumer can recognise and decline to count as activity.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# Raw PCM, signed 16-bit little-endian, mono, at the pipeline's sample rate --
# the format encode_audio_chunk expects, so a clip travels the identical path
# to synthesized speech and needs no special case at the transport.
CLIP_SUFFIX = ".pcm"


class StallClips:
    """The recorded holding phrases, and which one to play next.

    Empty is a supported state and the whole class is a no-op in it: a
    deployment with no clips directory behaves exactly as it did before this
    existed. A missing clip must never be able to fail a call it was only
    meant to make more comfortable.
    """

    def __init__(self, clips: list[bytes], rng: random.Random | None = None) -> None:
        self._clips = clips
        self._rng = rng or random.Random()
        self._last: int | None = None

    @classmethod
    def load(cls, directory: str | None, rng: random.Random | None = None) -> "StallClips":
        """Read every clip in *directory*, or return an empty set.

        Every failure is logged and swallowed -- an unreadable directory, an
        unreadable file, an empty file. The caller gets a StallClips either
        way, because the alternative is a startup that dies over an
        optional comfort feature.
        """
        if not directory:
            return cls([])
        path = Path(directory).expanduser()
        try:
            files = sorted(p for p in path.iterdir() if p.suffix == CLIP_SUFFIX)
        except OSError as exc:
            logger.warning("stall: cannot read clip directory %s: %s", path, exc)
            return cls([])

        clips: list[bytes] = []
        for p in files:
            try:
                data = p.read_bytes()
            except OSError as exc:
                logger.warning("stall: cannot read %s: %s", p, exc)
                continue
            if not data:
                logger.warning("stall: %s is empty, skipping", p)
                continue
            clips.append(data)

        if clips:
            logger.info("stall: loaded %d holding phrase(s) from %s", len(clips), path)
        else:
            logger.info("stall: no holding phrases found in %s; stalls will stay silent", path)
        return cls(clips, rng)

    def __len__(self) -> int:
        return len(self._clips)

    def next_clip(self) -> bytes | None:
        """One clip, never the same one twice running.

        With a single clip that rule cannot be honoured, and honouring it by
        returning nothing would be worse: one phrase repeated is still better
        than silence. So a lone clip repeats, and the rule applies from two up.
        """
        if not self._clips:
            return None
        if len(self._clips) == 1:
            return self._clips[0]

        choices = [i for i in range(len(self._clips)) if i != self._last]
        idx = self._rng.choice(choices)
        self._last = idx
        return self._clips[idx]


class StallTimer:
    """Decides WHEN a holding phrase is due for one session.

    Kept separate from StallClips, and free of any clock of its own, so the
    rule can be tested without a live call: every method takes *now*.

    The state it tracks is "a response is expected and nothing has been heard
    yet". That begins when the caller stops speaking and the model is asked,
    and ends the moment any audio goes out -- whether that is the real answer
    or the holding phrase itself.
    """

    def __init__(self, after_s: float) -> None:
        self._after_s = after_s
        self._pending_since: float | None = None
        self._fired = False

    def arm(self, now: float) -> None:
        """A response is expected from here.

        Idempotent: re-arming an already-armed timer must not push the
        deadline back, or a chatty pipeline could defer the phrase forever by
        re-arming faster than the threshold.
        """
        if self._pending_since is None:
            self._pending_since = now
            self._fired = False

    def disarm(self) -> None:
        """Audio went out, or the response ended. Nothing is owed."""
        self._pending_since = None
        self._fired = False

    def due(self, now: float) -> bool:
        """Whether a holding phrase should play right now.

        Fires at most once per pending response. A second phrase for the same
        wait is a different feature with a different threshold, and stacking it
        on this one by accident would put two apologies back to back.
        """
        if self._pending_since is None or self._fired:
            return False
        if now - self._pending_since < self._after_s:
            return False
        self._fired = True
        return True
