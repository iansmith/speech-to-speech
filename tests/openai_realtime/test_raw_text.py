"""SOP-848: the model's raw output text on ``response.done`` (``s2s_raw_text``).

Expected values are literals, never derived from the spoken transcript or from
the code under test: a test that computed its expectation from the transcript
would pass if the field were sourced from it.
"""

import json

import pytest
from openai.types.realtime import (
    ConversationItemCreateEvent,
    RealtimeConversationItemFunctionCall,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
)

from speech_to_speech.api.openai_realtime.handlers import response as response_module
from speech_to_speech.LLM.chat import Chat, make_assistant_message
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    ResponseGenerationDoneEvent,
    SpeechStartedEvent,
)
from speech_to_speech.pipeline.history import ResponseHistory
from speech_to_speech.pipeline.messages import AssistantTextPart
from tests.test_provider_response_status import CASES, TEXT, _chat_chunk, _provider_payload, _run

# The model's own sentence, with markdown, curly quotes and an emoji that the
# spoken path normalises away.
MODEL_TEXT = "Done. **Booked** “ok” \U0001f642"
# Shaped like Sophie's closer append (calconfirm.Separator, one line per
# confirmation, then the finishing question), sent as ONE provider delta.
CLOSER_TAIL = (
    "\n\n"
    "Confirmed: Dentist, Thursday, October 15, 9:30 AM.\n"
    "Confirmed: Soccer, Saturday, October 17, 10 AM."
    "\n"
    "Anything else?"
)
CONFIRMATION = (
    "Done. **Booked** “ok” \U0001f642\n\n"
    "Confirmed: Dentist, Thursday, October 15, 9:30 AM.\n"
    "Confirmed: Soccer, Saturday, October 17, 10 AM.\n"
    "Anything else?"
)
# Captured from the unmodified base (origin/feat/sophie_hook 473dad9) for the
# CONFIRMATION stream: the spoken transcript and the TTS inputs must not move.
SPOKEN_TRANSCRIPT = (
    'Done. Booked "ok" \n\nConfirmed: Dentist, Thursday, October 15, 9:30 AM. '
    "Confirmed: Soccer, Saturday, October 17, 10 AM. Anything else?"
)
TTS_INPUTS = [
    'Done. Booked "ok" \n\nConfirmed: Dentist, Thursday, October 15, 9:30 AM. '
    "Confirmed: Soccer, Saturday, October 17, 10 AM.",
    "Anything else?",
]
# The wire name is a contract with Sophie's reader, so it is pinned as a literal.
RAW_TEXT_ITEM_KEY = "s2s_raw_text"
USAGE = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def _stream(*pieces, finish_reason="stop", tool_calls_at=None):
    chunks = [_chat_chunk(piece) for piece in pieces]
    if tool_calls_at is not None:
        index, deltas = tool_calls_at
        chunks.insert(index, _chat_chunk(tool_calls=deltas))
    return [*chunks, _chat_chunk(finish_reason=finish_reason), _chat_chunk(usage=USAGE)]


def _done(events) -> ResponseDoneEvent:
    (done,) = [event for event in events if event.type == "response.done"]
    return done


def _wire_output(done: ResponseDoneEvent) -> list[dict]:
    """The response.done output exactly as a client decodes it off the wire."""
    return json.loads(json.dumps(done.model_dump()))["response"]["output"]


def test_handler_key_is_the_documented_wire_name():
    assert getattr(response_module, "RAW_TEXT_ITEM_KEY", None) == RAW_TEXT_ITEM_KEY


def test_completed_audio_response_carries_raw_confirmation_text_byte_for_byte():
    assert MODEL_TEXT + CLOSER_TAIL == CONFIRMATION
    response, _, events, _, _ = _run("chat", True, "stop", "audio", _stream(MODEL_TEXT, CLOSER_TAIL))

    assert response.status == "completed"
    assert response.output[0].model_dump()[RAW_TEXT_ITEM_KEY] == CONFIRMATION
    wire = _wire_output(_done(events))
    assert wire[0][RAW_TEXT_ITEM_KEY] == CONFIRMATION
    content = wire[0]["content"][0]
    assert content["transcript"] == SPOKEN_TRANSCRIPT
    assert content["transcript"] != CONFIRMATION
    assert content.get("text") is None
    assert RAW_TEXT_ITEM_KEY not in content


def test_raw_text_separator_split_across_deltas_round_trips():
    response, _, _, _, _ = _run("chat", True, "stop", "audio", _stream("A.", "\n", "\n", "Confirmed: X."))

    assert response.output[0].model_dump()[RAW_TEXT_ITEM_KEY] == "A.\n\nConfirmed: X."


def test_tool_call_response_gives_each_message_its_own_raw_text():
    tool = [
        {
            "index": 0,
            "id": "call_lookup",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
    ]
    payload = _stream("Let me check.\n", "Found it.\n\nConfirmed: X.", tool_calls_at=(1, tool))
    response, _, events, _, _ = _run("chat", True, "stop", "audio", payload)

    assert response.status == "completed"
    assert [item.type for item in response.output] == ["message", "function_call", "message"]
    dumped = [item.model_dump() for item in response.output]
    assert dumped[0][RAW_TEXT_ITEM_KEY] == "Let me check.\n"
    assert RAW_TEXT_ITEM_KEY not in dumped[1]
    assert dumped[2][RAW_TEXT_ITEM_KEY] == "Found it.\n\nConfirmed: X."
    wire = _wire_output(_done(events))
    assert [item.get(RAW_TEXT_ITEM_KEY) for item in wire] == ["Let me check.\n", None, "Found it.\n\nConfirmed: X."]


def test_text_modality_raw_text_equals_content_text():
    response, _, _, _, _ = _run("chat", True, "stop", "text", _stream(MODEL_TEXT, CLOSER_TAIL))

    item = response.output[0].model_dump()
    assert item["content"][0]["text"] == CONFIRMATION
    assert item[RAW_TEXT_ITEM_KEY] == CONFIRMATION


@pytest.mark.parametrize("backend,signal,status,reason", CASES)
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("modality", ["text", "audio"])
def test_raw_text_present_only_on_completed_for_every_provider_ending(
    backend, signal, status, reason, stream, modality
):
    response, _, _, _, _ = _run(backend, stream, signal, modality, _provider_payload(backend, stream, signal))

    assert response.status == status
    messages = [item.model_dump() for item in response.output if item.type == "message"]
    if status == "completed":
        assert messages
        assert [message[RAW_TEXT_ITEM_KEY] for message in messages] == [TEXT] * len(messages)
    else:
        assert not any(RAW_TEXT_ITEM_KEY in item.model_dump() for item in response.output)


def _open_response_with_history(service, conn_id, text):
    st = service._state(conn_id)
    key = "response_raw"
    service.response._ensure_response(conn_id, key)
    proposal = ResponseHistory.capture(
        st.runtime_config.chat, [make_assistant_message(text)], after_item_id=None, complete=True
    )
    service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(parts=[AssistantTextPart(text=text)], response_key=key, history=proposal),
    )
    service.encode_audio_chunk(conn_id, b"\x00" * 512)
    return st, key


@pytest.mark.parametrize("ending", ["completed", "cancelled", "response_cancel", "barge_in"])
def test_cancelled_response_omits_raw_text_while_history_still_provisional(service, conn_id, ending):
    st, key = _open_response_with_history(service, conn_id, "Hello.\nWorld.")
    # The history is still provisional at this point, so only the status gate
    # keeps a cancelled response's text off response.done.
    assert st.runtime_config.chat._provisional_generations.get(key)

    if ending == "completed":
        events = service.finish_response(conn_id, status="completed")
    elif ending == "cancelled":
        events = service.finish_response(conn_id, status="cancelled", reason="turn_detected")
    elif ending == "response_cancel":
        events = service.handle_response_cancel(conn_id)
    else:
        events = service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())

    done = _done(events)
    assert done.response.status == ("completed" if ending == "completed" else "cancelled")
    raw = [item.get(RAW_TEXT_ITEM_KEY) for item in _wire_output(done)]
    assert raw == (["Hello.\nWorld."] if ending == "completed" else [None])


def test_raw_text_is_additive_events_and_tts_input_unchanged(monkeypatch):
    payload = _stream(MODEL_TEXT, CLOSER_TAIL)
    with_tts: list[str] = []
    _, _, with_raw, _, _ = _run("chat", True, "stop", "audio", payload, tts_texts=with_tts)
    monkeypatch.setattr(Chat, "assistant_text_by_item_id", lambda *args, **kwargs: {}, raising=False)
    without_tts: list[str] = []
    _, _, without_raw, _, _ = _run("chat", True, "stop", "audio", payload, tts_texts=without_tts)

    assert with_tts == without_tts == TTS_INPUTS
    assert [event.type for event in with_raw] == [event.type for event in without_raw]

    def normalised(event):
        dumped = event.model_dump()
        dumped.pop("event_id", None)
        if event.type == "response.done":
            assert RAW_TEXT_ITEM_KEY not in (dumped["response"].get("metadata") or {})
            for item in dumped["response"]["output"]:
                item.pop(RAW_TEXT_ITEM_KEY, None)
        if event.type == "response.output_item.done":
            assert RAW_TEXT_ITEM_KEY not in dumped["item"]
        return json.dumps(dumped, sort_keys=True)

    # Ids are random per run; compare with them blanked.
    def blank_ids(events):
        ids = {}
        for event in events:
            dumped = event.model_dump()
            for value in _walk_ids(dumped):
                ids.setdefault(value, f"id{len(ids)}")
        out = []
        for event in events:
            text = normalised(event)
            for value, label in ids.items():
                text = text.replace(value, label)
            out.append(text)
        return out

    assert blank_ids(with_raw) == blank_ids(without_raw)
    assert RAW_TEXT_ITEM_KEY in _wire_output(_done(with_raw))[0]


def _walk_ids(value):
    if isinstance(value, dict):
        for key, inner in value.items():
            if key in (
                "id",
                "item_id",
                "response_id",
                "event_id",
                "previous_item_id",
                "conversation_id",
            ) and isinstance(inner, str):
                yield inner
            else:
                yield from _walk_ids(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _walk_ids(inner)


def test_prefetched_tool_followup_carries_raw_text(service, conn_id, text_prompt_queue):
    st = service._state(conn_id)
    response_key = "response_origin"
    call = RealtimeConversationItemFunctionCall(
        type="function_call", id="fc_lookup", call_id="call_lookup", name="lookup", arguments="{}"
    )
    assert st.runtime_config.chat.add_provisional_generation_items(response_key, [call]) is not None
    st.in_response = True
    st.current_response_id = "resp_origin"
    st.current_response_key = response_key
    st.pending_function_calls = {0: call}
    service.dispatch_pipeline_event(
        conn_id, ResponseGenerationDoneEvent(response_key=response_key, call_ids=[call.call_id])
    )
    service.handle_conversation_item_create(
        conn_id,
        ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "call_id": call.call_id, "output": "result"},
        ),
    )
    prefetch = text_prompt_queue.get_nowait()
    service.finish_response(conn_id, response_key=response_key)
    claimed = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
    assert isinstance(claimed, ResponseCreatedEvent)
    assert st.current_response_key == prefetch.response_key

    text = "Found it.\nConfirmed: X."
    proposal = ResponseHistory.capture(
        st.runtime_config.chat, [make_assistant_message(text)], after_item_id=None, complete=True
    )
    service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(
            parts=[AssistantTextPart(text=text)], response_key=prefetch.response_key, history=proposal
        ),
    )
    events = service.finish_response(conn_id, response_key=prefetch.response_key)

    done = _done(events)
    assert done.response.status == "completed"
    assert [item.get(RAW_TEXT_ITEM_KEY) for item in _wire_output(done)] == [text]


def test_out_of_band_response_omits_raw_text():
    response, _, _, _, _ = _run("chat", True, "stop", "audio", _stream(MODEL_TEXT, CLOSER_TAIL), conversation="none")

    assert response.status == "completed"
    assert response.output
    assert not any(RAW_TEXT_ITEM_KEY in item.model_dump() for item in response.output)
