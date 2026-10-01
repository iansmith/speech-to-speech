"""Detection for a conversation left with tool results and nothing speaking them.

THE FAILURE THESE COVER. A client answered four tool calls; every
``response.create`` it sent was rejected, because the response that had emitted
those calls was itself still generating; and the conversation was left holding
four results with nothing producing speech. The rejections were ordinary error
frames and the call went silent until the carrier's idle guard dropped it two
minutes later. Measured on the demo call of 2026-09-09 11:01: tool calls at
seq 60/63/66/72, rejections at 68/69/70/74, the emitting response not done
until seq 91, and the four outputs accepted at 92-95.

The rejections are correct and are not changed here. ``response.create`` either
opens a response or fails, and the server does not open one the client did not
ask for. What is added is the ability to SEE the stuck state: context on the
refusal, and a watchdog for the shape in general.

Nothing here asserts on message wording. The error ``type`` is the contract a
client branches on and is pinned; ``message`` is prose for a person reading a
log, and a test that matched it would fail on the next rephrasing while proving
nothing.
"""

import logging

import pytest
from openai.types.realtime.conversation_item_create_event import ConversationItemCreateEvent
from openai.types.realtime.realtime_conversation_item_function_call import (
    RealtimeConversationItemFunctionCall,
)
from openai.types.realtime.realtime_conversation_item_function_call_output import (
    RealtimeConversationItemFunctionCallOutput,
)
from openai.types.realtime.response_create_event import ResponseCreateEvent

from speech_to_speech.api.openai_realtime import service as service_module
from speech_to_speech.LLM.chat import add_supported_item


def _answer_a_tool_call(service, conn_id: str, call_id: str = "call_1") -> None:
    """Record a tool call and deliver its output, the way a real turn does.

    The recorded call is not decoration. The chat REFUSES an output whose
    call_id matches nothing ("No function_call with call_id ... found"), so a
    test that only delivered the output would arm nothing and would then prove
    that a watchdog stays quiet when it was never wound -- passing against any
    implementation at all. Verified: without the call below,
    tool_output_awaiting_response_at stays None and every assertion here
    becomes vacuous.
    """
    st = service._state(conn_id)
    add_supported_item(
        st.runtime_config.chat,
        RealtimeConversationItemFunctionCall.model_construct(
            type="function_call",
            call_id=call_id,
            name="probe",
            arguments="{}",
        ),
    )
    events = service.conversation.handle_conversation_item_create(
        conn_id,
        ConversationItemCreateEvent.model_construct(
            type="conversation.item.create",
            item=RealtimeConversationItemFunctionCallOutput.model_construct(
                type="function_call_output",
                call_id=call_id,
                output='{"answer":"forty-two"}',
            ),
        ),
    )
    assert [getattr(e, "type", None) for e in events] == ["conversation.item.created"], (
        f"the tool output was not accepted into the chat: {events}"
    )
    assert st.tool_output_awaiting_response_at is not None, (
        "the output landed but armed nothing -- every assertion below would be vacuous"
    )


def _error_type(event) -> str | None:
    error = getattr(event, "error", None)
    return getattr(error, "type", None)


class TestRejectionCarriesContext:
    """B: a refusal has to say what is holding the conversation."""

    def test_type_is_unchanged_by_the_added_detail(self, service, conn_id):
        """The type is what clients branch on; detail rides in the message."""
        st = service._state(conn_id)
        st.mark_response_started()

        result = service.handle_response_create(conn_id, ResponseCreateEvent.model_construct(type="response.create"))

        assert _error_type(result) == "conversation_already_has_active_response"

    def test_message_names_the_response_that_is_holding_the_line(self, service, conn_id):
        """The active response's id reaches the client, which had nothing to wait on before.

        Asserted as a SUBSTRING of an id the test itself set, not as a phrase:
        the sentence around it is free to change.
        """
        st = service._state(conn_id)
        st.mark_response_started()
        st.current_response_id = "resp_holding_the_line"

        result = service.handle_response_create(conn_id, ResponseCreateEvent.model_construct(type="response.create"))

        assert "resp_holding_the_line" in result.error.message


class TestRejectionLogging:
    """A: the rejection that can strand a conversation is the one that WARNs."""

    def test_warns_when_tool_results_are_waiting(self, service, conn_id, caplog):
        """This is the shape that went silent for two minutes with no log line."""
        st = service._state(conn_id)
        st.note_tool_output_awaiting_response()
        st.mark_response_started()
        # mark_response_started clears the wait (a response is what it wanted),
        # so re-arm it: the measured case is outputs owed WHILE a response runs.
        st.note_tool_output_awaiting_response()

        with caplog.at_level(logging.WARNING, logger=service_module.logger.name):
            service.handle_response_create(conn_id, ResponseCreateEvent.model_construct(type="response.create"))

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "a rejection with tool results waiting must warn -- it can strand the call"

    def test_does_not_warn_on_ordinary_contention(self, service, conn_id, caplog):
        """A client racing its own turn owes nothing; warning here buries the real case."""
        st = service._state(conn_id)
        st.mark_response_started()

        with caplog.at_level(logging.WARNING, logger=service_module.logger.name):
            service.handle_response_create(conn_id, ResponseCreateEvent.model_construct(type="response.create"))

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings, f"ordinary contention warned: {[r.getMessage() for r in warnings]}"


class TestStuckConversationWatchdog:
    """C: the generic backstop, on the shape rather than on one cause."""

    def test_quiet_when_nothing_is_owed(self, service, conn_id):
        assert service.check_stuck_conversation(conn_id) is False

    def test_quiet_before_the_threshold(self, service, conn_id):
        """A tool result answered promptly is the ordinary case and must be silent."""
        _answer_a_tool_call(service, conn_id)

        assert service.check_stuck_conversation(conn_id) is False

    def test_reports_once_past_the_threshold(self, service, conn_id, caplog):
        """The measured failure, compressed: results in context, nothing generating.

        The clock is moved rather than waited on, and it is moved by more than
        STUCK_CONVERSATION_S read from the module -- so raising the threshold
        does not quietly turn this test into one that proves nothing.
        """
        _answer_a_tool_call(service, conn_id)
        st = service._state(conn_id)
        assert not st.in_response and not st.response_pending

        st.tool_output_awaiting_response_at -= service_module.STUCK_CONVERSATION_S + 1.0

        with caplog.at_level(logging.WARNING, logger=service_module.logger.name):
            first = service.check_stuck_conversation(conn_id)
            second = service.check_stuck_conversation(conn_id)

        assert first is True, "a conversation with results and no response must be reported"
        assert second is False, "the audio path ticks every 20ms; one line per episode, not fifty"
        assert st.conversation_id in caplog.text

    @pytest.mark.parametrize("field", ["in_response", "response_pending"])
    def test_quiet_while_a_response_is_running_or_queued(self, service, conn_id, field):
        """Nothing is stuck while something is on its way to speaking.

        Parameterised over both flags because either one alone would let the
        watchdog cry wolf through every ordinary tool round trip -- which is
        how a backstop gets switched off.
        """
        _answer_a_tool_call(service, conn_id)
        st = service._state(conn_id)
        st.tool_output_awaiting_response_at -= service_module.STUCK_CONVERSATION_S + 1.0
        setattr(st, field, True)

        assert service.check_stuck_conversation(conn_id) is False

    def test_a_started_response_clears_the_debt(self, service, conn_id):
        """The stuck state ends when a response opens, not when one is asked for."""
        _answer_a_tool_call(service, conn_id)
        st = service._state(conn_id)
        assert st.tool_output_awaiting_response_at is not None

        st.mark_response_started()

        assert st.tool_output_awaiting_response_at is None
        assert service.check_stuck_conversation(conn_id) is False

    def test_a_second_episode_is_reported_again(self, service, conn_id):
        """One line per episode, not one per connection.

        Without this, a call whose first stall was reported would go silent
        about every later one -- which is the failure mode of a latch nobody
        resets.
        """
        _answer_a_tool_call(service, conn_id)
        st = service._state(conn_id)
        st.tool_output_awaiting_response_at -= service_module.STUCK_CONVERSATION_S + 1.0
        assert service.check_stuck_conversation(conn_id) is True

        st.mark_response_started()
        st.mark_response_finished()
        _answer_a_tool_call(service, conn_id, "call_2")
        st.tool_output_awaiting_response_at -= service_module.STUCK_CONVERSATION_S + 1.0

        assert service.check_stuck_conversation(conn_id) is True
