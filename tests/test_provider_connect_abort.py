"""SOP-538: a stale revision's provider request must not hold up the new one.

The measured defect: the caller extends a sentence, so revision N is superseded
by revision N+1 while N's request is still waiting for the provider's response
headers.  Cancellation could only reach a response object that already existed,
and that object only exists once ``request()`` returns -- so during the header
wait there was nothing to close, the sole provider worker slot stayed held, and
N+1's request could not start until N's upstream finally answered.  15.6 s of a
24.2 s turn on the live call of 2026-09-03.

These tests drive a real socket server that accepts the request and then sends
nothing, because that is the only way to exercise what was broken.  httpx's own
``Client.close()`` does not touch a connection another thread is actively
reading, so a design verified against a fake ``create`` that blocks on a
``threading.Event`` would prove nothing about the real provider path.
"""

from __future__ import annotations

import threading
import time

import httpx
import pytest
from openai import OpenAI

from speech_to_speech.LLM.provider_connect_abort import ProviderConnectAborter
from tests.stalling_provider import STALL_S, KeepAliveProvider, StallingProvider


@pytest.fixture
def stalling_provider():
    server = StallingProvider()
    yield server
    server.close()


@pytest.fixture
def keepalive_provider():
    server = KeepAliveProvider()
    yield server
    server.close()


def _post_in_thread(client: httpx.Client, url: str, outcome: dict) -> threading.Thread:
    def run() -> None:
        outcome["thread_ident"] = threading.get_ident()
        started = time.monotonic()
        try:
            client.post(url, json={"hello": "world"})
            outcome["result"] = "returned"
        except BaseException as exc:  # noqa: BLE001 - the failure mode is the subject
            outcome["result"] = type(exc).__name__
        outcome["elapsed"] = time.monotonic() - started

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_abort_unblocks_a_request_parked_waiting_for_response_headers(stalling_provider):
    """The capability that did not exist: interrupt a thread before headers.

    ``httpx.Client.close()`` leaves a connection another thread is actively
    reading untouched, so until now the parked thread stayed parked for the
    provider's whole time to first byte.
    """
    aborter = ProviderConnectAborter()
    client = httpx.Client(timeout=httpx.Timeout(STALL_S * 3, connect=5.0))
    aborter.install(client)

    outcome: dict = {}
    thread = _post_in_thread(client, f"http://127.0.0.1:{stalling_provider.port}/v1/x", outcome)
    assert stalling_provider.request_received.wait(5.0), "server never received the request"
    time.sleep(0.2)  # let the client settle into the header read

    assert aborter.abort(outcome["thread_ident"]) is True
    thread.join(timeout=2.0)

    assert not thread.is_alive(), "abort() did not unblock the parked request"
    assert outcome["result"] != "returned"
    assert outcome["elapsed"] < 1.0


def test_abort_disconnects_the_socket_before_the_server_sends_headers(stalling_provider):
    """The stale request's socket is really closed, not merely abandoned.

    An abandoned-but-open socket leaves the provider generating -- and billing
    for -- a response nobody will ever read.
    """
    aborter = ProviderConnectAborter()
    client = httpx.Client(timeout=httpx.Timeout(STALL_S * 3, connect=5.0))
    aborter.install(client)

    outcome: dict = {}
    thread = _post_in_thread(client, f"http://127.0.0.1:{stalling_provider.port}/v1/x", outcome)
    assert stalling_provider.request_received.wait(5.0)
    time.sleep(0.2)

    aborter.abort(outcome["thread_ident"])
    thread.join(timeout=2.0)

    assert stalling_provider.client_disconnected_before_headers.wait(2.0), (
        "the server never saw the stale client hang up"
    )


def test_an_abort_armed_before_the_request_starts_still_stops_it(stalling_provider):
    """Cancellation landing in the gap between arming and the first socket read.

    The worker arms its abort *before* calling into the provider, precisely so
    that a cancellation arriving during connect is not lost; an abort that only
    worked once a read was already in flight would reopen the same hole.
    """
    aborter = ProviderConnectAborter()
    client = httpx.Client(timeout=httpx.Timeout(STALL_S * 3, connect=5.0))
    aborter.install(client)

    outcome: dict = {}
    ready = threading.Event()
    armed = threading.Event()

    def run() -> None:
        outcome["thread_ident"] = threading.get_ident()
        ready.set()
        armed.wait(5.0)
        started = time.monotonic()
        try:
            client.post(f"http://127.0.0.1:{stalling_provider.port}/v1/x", json={"a": 1})
            outcome["result"] = "returned"
        except BaseException as exc:  # noqa: BLE001
            outcome["result"] = type(exc).__name__
        outcome["elapsed"] = time.monotonic() - started

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(2.0)

    aborter.abort(outcome["thread_ident"])  # nothing in flight yet
    armed.set()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert outcome["result"] != "returned", "an armed abort must stop the request that follows it"
    assert outcome["elapsed"] < 1.0


def test_release_lets_a_worker_thread_be_used_again(stalling_provider, keepalive_provider):
    """An armed abort is scoped to one request, not to the thread forever.

    Provider workers are pooled by the runtime, so a thread that carried an
    aborted request must be usable for the next one.
    """
    aborter = ProviderConnectAborter()
    client = httpx.Client(timeout=httpx.Timeout(STALL_S * 3, connect=5.0))
    aborter.install(client)

    outcome: dict = {}
    thread = _post_in_thread(client, f"http://127.0.0.1:{stalling_provider.port}/v1/x", outcome)
    assert stalling_provider.request_received.wait(5.0)
    time.sleep(0.2)
    aborter.abort(outcome["thread_ident"])
    thread.join(timeout=2.0)

    aborter.release(outcome["thread_ident"])

    second: dict = {}
    reused = threading.Thread(
        target=lambda: second.update(
            status=client.post(f"http://127.0.0.1:{keepalive_provider.port}/v1/x", json={"a": 1}).status_code
        ),
        daemon=True,
    )
    # Run it on a *different* thread ident is not enough -- the point is that a
    # released ident works, so drive it from the main thread instead.
    reused.start()
    reused.join(timeout=5.0)
    assert second.get("status") == 200

    aborter.release(threading.get_ident())
    assert client.post(f"http://127.0.0.1:{keepalive_provider.port}/v1/x", json={"a": 1}).status_code == 200


def test_normal_requests_still_reuse_one_pooled_connection(keepalive_provider):
    """No change to the non-revised path: pooling must survive the wrapper.

    A fresh httpx client per request would also have made connect cancellable,
    at the cost of a TCP+TLS handshake on every turn -- a latency regression in
    a latency ticket.
    """
    aborter = ProviderConnectAborter()
    client = httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0))
    aborter.install(client)

    url = f"http://127.0.0.1:{keepalive_provider.port}/v1/x"
    for _ in range(3):
        assert client.post(url, json={"a": 1}).status_code == 200
    time.sleep(0.2)

    assert keepalive_provider.accepted_connections == 1, (
        f"pooling lost: {keepalive_provider.accepted_connections} TCP connections for 3 requests"
    )


def test_install_reaches_the_openai_sdk_clients_transport():
    """Canary: the SDK/httpx internals the aborter reaches through still exist.

    ``install`` goes through two private attributes. If an openai or httpx
    upgrade moves them, cancellation would silently stop working and every
    revised turn would quietly go slow again -- so the reach must fail loudly
    here rather than in production.
    """
    sdk_client = OpenAI(api_key="none", base_url="http://127.0.0.1:1/v1")
    aborter = ProviderConnectAborter()

    aborter.install(sdk_client._client)  # must not raise

    assert aborter.installed_on is sdk_client._client
