"""Tests for the optional pre-generation context hook.

The property under test is almost entirely **fail-open**. The hook sits on the
critical path between a caller finishing speaking and the model starting, so every
way it can go wrong has to cost the turn its enrichment and nothing more. A test
suite for this that only covered the happy path would be testing the least
important half.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from speech_to_speech.LLM.context_provider import fetch_context_items


def _serve(handler_fn):
    """Run a one-request HTTP server; yield its URL."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's interface
            length = int(self.headers.get("Content-Length", 0))
            handler_fn(self, self.rfile.read(length))

        def log_message(self, *args):  # silence per-request stderr noise
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}/context"


def _respond(request, status: int, body: bytes) -> None:
    request.send_response(status)
    request.send_header("Content-Type", "application/json")
    request.send_header("Content-Length", str(len(body)))
    request.end_headers()
    request.wfile.write(body)


@pytest.fixture
def provider():
    servers = []

    def start(handler_fn):
        server, url = _serve(handler_fn)
        servers.append(server)
        return url

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def test_items_are_returned_and_typed_by_role(provider):
    url = provider(
        lambda req, body: _respond(
            req,
            200,
            json.dumps(
                {
                    "items": [
                        {"role": "system", "text": "The caller is Marge."},
                        {"role": "user", "text": "remember the knee"},
                        {"role": "assistant", "text": "noted"},
                    ]
                }
            ).encode(),
        )
    )

    items = fetch_context_items(url, [{"role": "user", "content": "hello"}], timeout_s=2.0)

    assert [item.role for item in items] == ["system", "user", "assistant"]
    assert items[0].content[0].text == "The caller is Marge."


def test_request_carries_the_conversation_and_turn_metadata(provider):
    seen = {}

    def handler(req, body):
        seen.update(json.loads(body))
        _respond(req, 200, b'{"items": []}')

    url = provider(handler)
    conversation = [{"role": "user", "content": "how is her knee"}]

    fetch_context_items(
        url,
        conversation,
        timeout_s=2.0,
        turn_id="turn-7",
        language_code="en",
        instructions="be brief",
    )

    assert seen["conversation"] == conversation
    assert seen["turn_id"] == "turn-7"
    assert seen["language_code"] == "en"
    assert seen["instructions"] == "be brief"


# ---- fail-open: every one of these must return [] rather than raise ----


def test_connection_refused_fails_open():
    # Port 1 on loopback: nothing listens, and connecting is refused immediately.
    assert fetch_context_items("http://127.0.0.1:1/context", [], timeout_s=1.0) == []


def test_malformed_url_fails_open():
    assert fetch_context_items("not-a-url", [], timeout_s=1.0) == []


def test_timeout_fails_open(provider):
    def slow(req, body):
        threading.Event().wait(2.0)
        _respond(req, 200, b'{"items": []}')

    url = provider(slow)
    assert fetch_context_items(url, [], timeout_s=0.15) == []


def test_server_error_fails_open(provider):
    url = provider(lambda req, body: _respond(req, 500, b"boom"))
    assert fetch_context_items(url, [], timeout_s=2.0) == []


def test_non_json_body_fails_open(provider):
    url = provider(lambda req, body: _respond(req, 200, b"<html>nope</html>"))
    assert fetch_context_items(url, [], timeout_s=2.0) == []


def test_missing_items_key_fails_open(provider):
    url = provider(lambda req, body: _respond(req, 200, b'{"something_else": []}'))
    assert fetch_context_items(url, [], timeout_s=2.0) == []


def test_items_not_a_list_fails_open(provider):
    url = provider(lambda req, body: _respond(req, 200, b'{"items": {"role": "system"}}'))
    assert fetch_context_items(url, [], timeout_s=2.0) == []


# ---- partial tolerance: one bad entry must not lose the good ones ----


def test_bad_entries_are_skipped_individually(provider):
    url = provider(
        lambda req, body: _respond(
            req,
            200,
            json.dumps(
                {
                    "items": [
                        "not an object",
                        {"role": "wizard", "text": "unknown role"},
                        {"role": "system"},
                        {"role": "system", "text": ""},
                        {"role": "system", "text": "this one is fine"},
                    ]
                }
            ).encode(),
        )
    )

    items = fetch_context_items(url, [], timeout_s=2.0)

    assert len(items) == 1, "the four unusable entries must be dropped, the good one kept"
    assert items[0].content[0].text == "this one is fine"
