from dataclasses import dataclass, field
from typing import Optional

# Default voice is deliberately absent rather than guessed: voice IDs are
# account-scoped, so a hard-coded one either 404s or silently speaks in a voice
# the operator did not choose. Startup fails loudly instead — see the handler.
DEFAULT_ELEVENLABS_MODEL = "eleven_flash_v2_5"
DEFAULT_ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
DEFAULT_ELEVENLABS_API_KEY_ENV = "ELEVEN_LABS_API_KEY"


@dataclass
class ElevenLabsTTSHandlerArguments:
    """Arguments for the ``elevenlabs`` TTS backend.

    Unlike every other TTS handler here this one is a NETWORK backend: the
    assistant's reply text leaves the machine on every turn. That is a
    deliberate operator choice, not a default — ``--tts elevenlabs`` has to be
    asked for.
    """

    elevenlabs_voice_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "ElevenLabs voice ID to speak with. Required when --tts elevenlabs. Voice IDs are "
            "account-scoped; find them at https://elevenlabs.io/app/voice-lab or via GET /v1/voices."
        },
    )
    elevenlabs_model_id: str = field(
        default=DEFAULT_ELEVENLABS_MODEL,
        metadata={
            "help": "ElevenLabs model. Default 'eleven_flash_v2_5', the lowest-latency tier, which is the "
            "one that matters on a phone call. 'eleven_turbo_v2_5' trades ~150ms for quality; "
            "'eleven_multilingual_v2' is higher quality again and too slow for realtime."
        },
    )
    elevenlabs_api_key_env: str = field(
        default=DEFAULT_ELEVENLABS_API_KEY_ENV,
        metadata={
            "help": "Name of the environment variable holding the API key. The NAME is the flag, never "
            "the key itself: process arguments are world-readable via `ps` on a shared machine."
        },
    )
    elevenlabs_base_url: str = field(
        default=DEFAULT_ELEVENLABS_BASE_URL,
        metadata={"help": "API base URL. Override to point at a proxy or a test double."},
    )
    elevenlabs_output_format: str = field(
        default="pcm_16000",
        metadata={
            "help": "Audio format requested from the API. Default 'pcm_16000' matches the pipeline's own "
            "sample rate exactly, so no resampling happens on the critical path. Other pcm_* rates "
            "are accepted and resampled."
        },
    )
    elevenlabs_optimize_streaming_latency: Optional[int] = field(
        default=None,
        metadata={
            "help": "ElevenLabs' latency optimisation level, 0-4. Higher trades pronunciation quality for "
            "speed. Unset uses the API default. Note this knob is deprecated by ElevenLabs for the "
            "flash models, which are already latency-optimised."
        },
    )
    elevenlabs_timeout_s: float = field(
        default=10.0,
        metadata={
            "help": "Per-request timeout in seconds. Deliberately short: this sits between the caller "
            "finishing speaking and hearing anything, so failing fast and falling silent for one "
            "sentence beats holding the call open."
        },
    )
    elevenlabs_stability: Optional[float] = field(
        default=None,
        metadata={"help": "Voice setting 0.0-1.0. Unset uses the voice's own saved setting."},
    )
    elevenlabs_similarity_boost: Optional[float] = field(
        default=None,
        metadata={"help": "Voice setting 0.0-1.0. Unset uses the voice's own saved setting."},
    )
    elevenlabs_speed: Optional[float] = field(
        default=None,
        metadata={"help": "Speaking rate multiplier, roughly 0.7-1.2. Unset uses the voice's own setting."},
    )
