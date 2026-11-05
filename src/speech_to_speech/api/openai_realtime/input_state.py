from typing import Literal, Union

from openai.types.realtime import (
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionFailedEvent,
    InputAudioBufferSpeechStoppedEvent,
)
from pydantic import BaseModel

InputTranscriptionTerminal = Union[
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionFailedEvent,
]


class SpeechToSpeechInputAudioTranscriptionSnapshotEvent(BaseModel):
    """Cumulative, replaceable STT hypothesis snapshot emitted to opted-in clients."""

    type: Literal["speech_to_speech.input_audio_transcription.snapshot"] = (
        "speech_to_speech.input_audio_transcription.snapshot"
    )
    event_id: str
    item_id: str
    content_index: int = 0
    transcript: str


class TurnEndedEvent(BaseModel):
    """Fork extension: the VAD judged a stop candidate for this item's turn.

    Sent once per VAD soft end, at the stop decision, ahead of the
    ``input_audio_buffer.speech_stopped`` that is held until the turn commits.
    It is not held and never repeated when the held terminals are released.
    Not part of the OpenAI Realtime API; generic clients ignore it.
    """

    type: Literal["turn.ended"] = "turn.ended"
    event_id: str
    item_id: str
    audio_end_ms: int


class TurnResumedEvent(BaseModel):
    """Fork extension: speech resumed in a reopened turn that keeps its item.

    Sent in place of the ``input_audio_buffer.speech_started`` that a reopen
    reusing the open item does not send. Not part of the OpenAI Realtime API.
    """

    type: Literal["turn.resumed"] = "turn.resumed"
    event_id: str
    item_id: str


class InputItemState(BaseModel):
    """Active state for one client-visible input transcription item."""

    transcript_prefix: str = ""
    latest_transcript: str = ""
    audio_duration_s: float = 0.0


class PendingInputTerminal(BaseModel):
    """One input item's terminal lifecycle, held until its turn commits.

    A speculative pause is an internal stop candidate: speech can resume and
    the same logical turn stays open. Realtime has no way to retract a
    ``speech_stopped`` or an already completed transcription, so both are kept
    here until the turn is externally committed. A revision that a later
    revision supersedes discards its held events instead of publishing them.
    """

    turn_id: str | None = None
    turn_revision: int | None = None
    speech_stopped: InputAudioBufferSpeechStoppedEvent | None = None
    transcription: InputTranscriptionTerminal | None = None
    # Set once the pipeline has finished with this item, so publishing it also
    # releases its routing and transcript state.
    input_closed: bool = False
