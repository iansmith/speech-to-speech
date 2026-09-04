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

import socket
import ssl
import threading
import time

import httpcore
import httpx
import pytest
from openai import OpenAI

from speech_to_speech.LLM.provider_connect_abort import (
    ProviderConnectAborter,
    ProviderRequestAborted,
    _TrackedStream,
    _TrackingBackend,
)
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


def _post_in_thread(
    client: httpx.Client,
    url: str,
    outcome: dict,
    aborter: ProviderConnectAborter,
) -> threading.Thread:
    """Issue one request the way a provider worker does: bracketed by begin/end."""

    started_flag = threading.Event()

    def run() -> None:
        token = aborter.begin()
        outcome["token"] = token
        outcome["thread_ident"] = threading.get_ident()
        started_flag.set()
        started = time.monotonic()
        try:
            client.post(url, json={"hello": "world"})
            outcome["result"] = "returned"
        except BaseException as exc:  # noqa: BLE001 - the failure mode is the subject
            outcome["result"] = type(exc).__name__
        finally:
            outcome["elapsed"] = time.monotonic() - started
            aborter.end(token)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    started_flag.wait(2.0)  # the token exists before the caller may abort it
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
    thread = _post_in_thread(client, f"http://127.0.0.1:{stalling_provider.port}/v1/x", outcome, aborter)
    assert stalling_provider.request_received.wait(5.0), "server never received the request"
    time.sleep(0.2)  # let the client settle into the header read

    assert aborter.abort(outcome["token"]).socket_torn_down is True
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
    thread = _post_in_thread(client, f"http://127.0.0.1:{stalling_provider.port}/v1/x", outcome, aborter)
    assert stalling_provider.request_received.wait(5.0)
    time.sleep(0.2)

    aborter.abort(outcome["token"])
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
        outcome["token"] = aborter.begin()  # the window opens before the request
        ready.set()
        armed.wait(5.0)
        started = time.monotonic()
        try:
            client.post(f"http://127.0.0.1:{stalling_provider.port}/v1/x", json={"a": 1})
            outcome["result"] = "returned"
        except BaseException as exc:  # noqa: BLE001
            outcome["result"] = type(exc).__name__
        finally:
            outcome["elapsed"] = time.monotonic() - started
            aborter.end(outcome["token"])

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(2.0)

    aborter.abort(outcome["token"])  # nothing in flight yet
    armed.set()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert outcome["result"] != "returned", "an armed abort must stop the request that follows it"
    assert outcome["elapsed"] < 1.0


def test_an_abort_outside_a_request_window_is_refused(stalling_provider, keepalive_provider):
    """A late abort must not poison the thread ident for whoever inherits it.

    Provider worker threads are short-lived and CPython reuses their idents. If
    a cancellation that lost the race against a finishing worker left a standing
    mark, the next worker handed that ident would fail its first socket
    operation for a request that was cancelled before it existed.
    """
    aborter = ProviderConnectAborter()
    client = httpx.Client(timeout=httpx.Timeout(STALL_S * 3, connect=5.0))
    aborter.install(client)

    outcome: dict = {}
    thread = _post_in_thread(client, f"http://127.0.0.1:{stalling_provider.port}/v1/x", outcome, aborter)
    assert stalling_provider.request_received.wait(5.0)
    time.sleep(0.2)
    aborter.abort(outcome["token"])
    thread.join(timeout=2.0)
    assert not thread.is_alive()

    # The worker has ended its window; a cancellation arriving now is refused...
    assert aborter.abort(outcome["token"]).aborted is False
    assert not aborter.is_armed(outcome["token"])

    # ...and a thread inheriting that ident still gets a working request.
    inheritor: dict = {}
    reuse = _post_in_thread(client, f"http://127.0.0.1:{keepalive_provider.port}/v1/x", inheritor, aborter)
    reuse.join(timeout=5.0)
    assert inheritor["result"] == "returned"


def test_a_late_abort_cannot_kill_the_request_that_replaced_it(stalling_provider, keepalive_provider):
    """The successor case, which is the one that actually bites.

    Provider workers are capped at one and strictly sequential, so CPython hands
    the same thread ident to consecutive workers essentially always. If aborts
    were addressed to a thread, this sequence -- a stale turn's abort callback
    firing just after its own request finished and the revision's worker took
    over the ident -- would tear down the *revision's* live request. The
    revision is not cancelled, so nothing would re-issue it.
    """
    aborter = ProviderConnectAborter()
    client = httpx.Client(timeout=httpx.Timeout(STALL_S * 3, connect=5.0))
    aborter.install(client)

    # A first request that completes normally, and whose window then closes.
    stale: dict = {}
    stale_thread = _post_in_thread(client, f"http://127.0.0.1:{keepalive_provider.port}/v1/x", stale, aborter)
    stale_thread.join(timeout=5.0)
    assert stale["result"] == "returned"

    # A second request on a token issued after the first one's window closed.
    successor: dict = {}
    successor_thread = _post_in_thread(client, f"http://127.0.0.1:{stalling_provider.port}/v1/x", successor, aborter)
    assert stalling_provider.request_received.wait(5.0)
    time.sleep(0.2)

    # Without this the test is vacuous: it only exercises the hazard if the two
    # workers really did share an ident, which is the normal case here.
    assert successor["thread_ident"] == stale["thread_ident"], (
        "thread idents were not reused, so this run never exercised the hazard"
    )

    # The stale turn's cancellation finally fires. It must find nothing.
    assert aborter.abort(stale["token"]).aborted is False
    assert not aborter.is_armed(successor["token"])
    assert successor_thread.is_alive(), "the late abort killed the request that replaced it"

    aborter.abort(successor["token"])  # tidy up: release the stalled request
    successor_thread.join(timeout=2.0)


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
    # ...and that the pooled connection is one we could actually abort. Without
    # this the test passes just as happily with install() removed, proving only
    # that httpx pools.
    pooled = list(client._transport._pool.connections)
    assert pooled, "no pooled connection to inspect"
    assert isinstance(pooled[0]._connection._network_stream, _TrackedStream)


def test_install_reaches_the_openai_sdk_clients_transport():
    """Canary: the SDK/httpx internals the aborter reaches through still exist.

    ``install`` goes through two private attributes. If an openai or httpx
    upgrade moves them, cancellation would silently stop working and every
    revised turn would quietly go slow again -- so the reach must fail loudly
    here rather than in production.
    """
    sdk_client = OpenAI(api_key="none", base_url="http://127.0.0.1:1/v1")
    aborter = ProviderConnectAborter()

    # The handler skips the install for any client without httpx underneath, so
    # this is the assertion standing between an SDK rename and every revised
    # turn quietly going slow again.
    assert isinstance(sdk_client._client, httpx.Client)

    aborter.install(sdk_client._client)  # must not raise

    assert aborter.installed_on is sdk_client._client


def test_an_abort_that_surfaces_as_a_clean_eof_is_still_terminal():
    """A torn-down socket does not always raise -- and the quiet path is the
    dangerous one.

    `read` returning b"" instead of erroring makes httpcore raise
    RemoteProtocolError, an ordinary Exception, which the OpenAI SDK treats as a
    flaky network: it sleeps and re-issues the cancelled request on the very
    provider worker thread that is supposed to be freeing its slot. Checking the
    arm only in the error path left that door open; measured at roughly one
    abort in forty.
    """

    aborter = ProviderConnectAborter()
    abort_during_read: list[int] = []

    class SilentlyClosedStream(httpcore.NetworkStream):
        """Reads clean EOF, the way a shut-down socket often does.

        The abort lands *inside* the read, which is the real ordering: another
        thread tears the socket down while this one is parked in it. Arming
        before the read would only exercise the entry check.
        """

        def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
            for token in abort_during_read:
                aborter.abort(token)
            return b""

        def write(self, buffer: bytes, timeout: float | None = None) -> None:
            return None

        def close(self) -> None:
            return None

        def get_extra_info(self, info: str) -> object | None:
            return None

    stream = _TrackedStream(SilentlyClosedStream(), aborter)

    token = aborter.begin()
    assert stream.read(10) == b"", "a healthy request must still see its EOF"

    abort_during_read.append(token)
    with pytest.raises(ProviderRequestAborted):
        stream.read(10)


def test_abort_does_not_close_the_descriptor_out_from_under_the_reader():
    """`abort` must shut the socket down and stop there.

    Closing from the aborting thread races the reader: CPython's timed `recv`
    polls a `sock_fd` that `close()` has already set to -1, and POSIX ignores a
    negative fd -- so the reader misses the wake-up entirely and sleeps out the
    full read timeout (20 s in production), which is worse than the stall this
    module removes. Measured at ~7% of aborts before this.
    """
    closed: list[str] = []

    class RecordingStream(httpcore.NetworkStream):
        def __init__(self, sock: socket.socket) -> None:
            self.sock = sock

        def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
            return b""

        def write(self, buffer: bytes, timeout: float | None = None) -> None:
            return None

        def close(self) -> None:
            closed.append("close")

        def get_extra_info(self, info: str) -> object | None:
            return self.sock if info == "socket" else None

    # A *connected* pair. On an unconnected socket shutdown() raises ENOTCONN,
    # abort()'s `except OSError` swallows it, and every statement after the
    # shutdown is skipped -- so the test would pass whatever followed it.
    ours, peer = socket.socketpair()
    try:
        _TrackedStream(RecordingStream(ours), ProviderConnectAborter()).abort()

        # The shutdown really happened, so the assertions below mean something.
        peer.settimeout(2.0)
        assert peer.recv(1) == b"", "abort() did not shut the connection down"

        assert closed == [], "abort() closed the stream; only the owning thread may"
        assert ours.fileno() != -1, (
            "abort() closed the descriptor from the aborting thread; that races the "
            "reader's poll and loses the wake-up entirely"
        )
    finally:
        ours.close()
        peer.close()


def test_repeated_aborts_all_land_promptly(stalling_provider):
    """The abort must be reliable, not merely usually reliable.

    A cancellation that silently fails to wake its reader leaves the worker slot
    held for the full read timeout while the log line claims success -- the
    original bug, invisible. One lost abort in a run of these is a real defect,
    so this fails on the first, and a correct implementation cannot fail it.
    """
    aborter = ProviderConnectAborter()
    # A short read timeout keeps a lost abort cheap to detect rather than slow.
    client = httpx.Client(timeout=httpx.Timeout(1.0, connect=5.0))
    aborter.install(client)

    url = f"http://127.0.0.1:{stalling_provider.port}/v1/x"
    for attempt in range(40):
        outcome: dict = {}
        thread = _post_in_thread(client, url, outcome, aborter)
        # request_received latches, so waiting on it would gate only the first
        # pass and let later ones abort before the read had even started.
        deadline = time.monotonic() + 5.0
        while len(stalling_provider.request_times) <= attempt and time.monotonic() < deadline:
            time.sleep(0.001)
        assert len(stalling_provider.request_times) == attempt + 1, f"the provider never received request {attempt}"
        time.sleep(0.005)
        aborter.abort(outcome["token"])
        thread.join(timeout=5.0)
        assert not thread.is_alive(), f"abort {attempt} never woke its reader"
        assert outcome["elapsed"] < 0.9, (
            f"abort {attempt} took {outcome['elapsed']:.2f}s -- it missed the reader "
            f"and the request sat until its read timeout"
        )


def test_install_covers_the_transport_a_proxied_client_actually_uses(monkeypatch):
    """A proxy must not quietly switch the whole mechanism off.

    httpx picks a transport from `_mounts` before falling back to `_transport`,
    and the SDK builds its client with trust_env, so `HTTPS_PROXY` in the
    environment routes provider traffic through a mounted transport. Patching
    only the default one left no tracked stream anywhere -- while abort() still
    returned True and the log still announced success, because the token was
    open. That is the silent degradation install() exists to prevent.
    """
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")

    sdk_client = OpenAI(api_key="none", base_url="https://provider.example/v1")
    aborter = ProviderConnectAborter()
    aborter.install(sdk_client._client)

    assert sdk_client._client._mounts, "expected the proxy env to produce mounted transports"
    request = httpx.Request("POST", "https://provider.example/v1/responses")
    chosen = sdk_client._client._transport_for_url(request.url)
    assert isinstance(chosen._pool._network_backend, _TrackingBackend), (
        "the transport this client would really use for provider traffic is untracked"
    )


def test_an_armed_request_never_reaches_the_socket_at_all():
    """The sticky arm must stop the *next* operation, not just fail the current.

    An abort can land between socket operations -- after the request is written,
    before the response read begins. Nothing is in flight to tear down then, so
    the mark is all that stops the request, and it has to be consulted on the
    way in as well as on the way out.
    """
    performed: list[str] = []

    class CountingStream(httpcore.NetworkStream):
        def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
            performed.append("read")
            return b"x"

        def write(self, buffer: bytes, timeout: float | None = None) -> None:
            performed.append("write")

        def close(self) -> None:
            return None

        def get_extra_info(self, info: str) -> object | None:
            return None

    aborter = ProviderConnectAborter()
    stream = _TrackedStream(CountingStream(), aborter)
    token = aborter.begin()

    assert stream.read(1) == b"x"
    assert performed == ["read"]

    aborter.abort(token)  # nothing bound: only the mark takes effect
    with pytest.raises(ProviderRequestAborted):
        stream.read(1)
    with pytest.raises(ProviderRequestAborted):
        stream.write(b"x")
    assert performed == ["read"], "an armed request still touched the socket"


def test_an_armed_request_is_not_allowed_to_dial_a_fresh_connection(monkeypatch):
    """A cancelled request must not open a new connection either.

    The SDK decides on a retry before this layer sees it, so without this check
    a cancelled request could still complete a TCP (and TLS) handshake it will
    never use.
    """
    dialled: list[str] = []

    def record(*_args: object, **_kwargs: object) -> httpcore.NetworkStream:
        dialled.append("connect")
        raise AssertionError("an armed request must not reach the real backend")

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", record)

    aborter = ProviderConnectAborter()
    backend = _TrackingBackend(aborter)
    token = aborter.begin()
    aborter.abort(token)

    with pytest.raises(ProviderRequestAborted):
        backend.connect_tcp("example.invalid", 443)
    assert dialled == [], "an armed request dialled anyway"


def test_the_documented_boundary_is_where_the_docstring_says_it_is():
    """`wrap_socket` detaches the socket, which is why a handshake is not abortable.

    This pins the reason, not the symptom. The module documents connection setup
    as out of reach, and it would be easy for a later reader to assume
    `start_tls` is covered simply because `_guarded` wraps it -- it is wrapped,
    the stream does get bound, and `abort()` returns True. What actually defeats
    it is that the socket object handed to `wrap_socket` is detached, so the
    handle the aborter holds is already dead.
    """
    left, right = socket.socketpair()
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    handshake = threading.Thread(
        target=lambda: _swallow(lambda: context.wrap_socket(left, server_hostname="x")),
        daemon=True,
    )
    handshake.start()
    time.sleep(0.2)

    assert left.fileno() == -1, (
        "wrap_socket no longer detaches its socket -- the TLS handshake may now be "
        "abortable, and the module docstring's account of the boundary is stale"
    )
    with pytest.raises(OSError):
        left.shutdown(socket.SHUT_RDWR)

    right.close()
    handshake.join(timeout=5.0)


def _swallow(fn):
    try:
        fn()
    except BaseException:
        pass
