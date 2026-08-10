from dataclasses import dataclass, field


@dataclass
class LanguageModelBaseArguments:
    model_name: str = field(
        default="Qwen/Qwen3-4B-Instruct-2507",
        metadata={"help": "The pretrained language model to use."},
    )
    user_role: str = field(
        default="user",
        metadata={"help": "Role assigned to the user in the chat context. Default is 'user'."},
    )
    init_chat_role: str = field(
        default="system",
        metadata={"help": "Initial role for setting up the chat context. Default is 'system'."},
    )
    init_chat_prompt: str = field(
        default="You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words.",
        metadata={"help": "The initial chat prompt to establish context for the language model."},
    )
    chat_size: int = field(
        default=30,
        metadata={"help": "Number of interactions assistant-user to keep for the chat."},
    )
    stream_batch_sentences: int = field(
        default=3,
        metadata={
            "help": "Number of sentences to accumulate before yielding a batch during streaming. "
            "Set to 1 for sentence-by-sentence streaming. Default is 3."
        },
    )
    enable_lang_prompt: bool = field(
        default=False,
        metadata={
            "help": "When True, append a user message instructing the model to reply in the detected/selected "
            "language (e.g. 'Please reply to my message in French.'). Default is False."
        },
    )
    context_provider_url: str | None = field(
        default=None,
        metadata={
            "help": "Optional HTTP endpoint POSTed once per turn, after the transcript is final and "
            "before generation, to enrich the conversation with context the pipeline cannot know "
            "(caller identity, prior sessions, application state). Returns items to append. Unset "
            "disables the feature entirely. Failures are non-fatal: the turn generates without "
            "enrichment rather than failing."
        },
    )
    context_provider_timeout_ms: int = field(
        default=300,
        metadata={
            "help": "Timeout for context_provider_url, in milliseconds. This sits on the critical path "
            "between the caller finishing speaking and the model starting, so it is deliberately short. "
            "Default is 300."
        },
    )
    compact_history: bool = field(
        default=True,
        metadata={
            "help": "When True, summarize older turns in the background once the chat exceeds chat_size, "
            "instead of synchronously evicting them. Adds an extra LLM call per compaction. Default is True."
        },
    )
