"""Interrupt a provider request that is still waiting for its response headers.

The pipeline speculates: it sends a transcript upstream as soon as it has one,
and abandons that request if the caller keeps talking. Abandoning it was only
half-effective. Everything that could cancel a request -- ``CancelScope``'s
armed aborts, ``ResponsePrefetchTransaction.discard`` -- works by closing a
*response object*, and that object does not exist until ``create()`` returns,
which is to say until the provider's first byte arrives. During the wait the
worker thread is parked inside a socket read with nothing holding a handle on
it, so the abandoned request runs to completion on its own schedule and, because
provider workers are capped, the turn the caller actually finished queues behind
it. On the live call of 2026-09-03 that cost 15.6 s of a 24.2 s turn.

Closing the httpx client does not help: ``httpx.Client.close()`` closes *idle*
pooled connections and leaves alone the one another thread is actively reading.
Nor does ``socket.close()`` from a second thread -- on macOS a thread already
blocked in ``recv`` is not woken by the descriptor being closed. What does work
is ``shutdown(SHUT_RDWR)``, which tears down both directions of the TCP
connection and wakes the blocked reader immediately; following it with a close
sends the FIN that tells the provider to stop generating.

Reaching the socket means reaching the network stream httpcore opened, which
means installing a custom network backend. Two design constraints shaped this:

* **The backend wraps a shared client, not a per-request one.** Giving every
  request its own ``httpx.Client`` would also expose the socket, at the cost of
  a fresh TCP and TLS handshake on every single turn -- a latency regression
  inside a latency fix. Wrapping the shared client keeps connection reuse.
* **Aborts are addressed to a request, by token.** :meth:`begin` opens a window
  and hands back a token; everything else -- arming, finding the socket,
  closing the window -- is keyed on that token. Addressing a *thread* instead
  is the obvious shortcut and it is wrong: provider workers are capped at one
  and strictly sequential, so CPython hands the same ident to consecutive
  workers essentially always (measured: 300 sequential threads, one distinct
  ident). A cancellation that arrives just after its own worker finished would
  then land on that worker's successor and kill a live turn.

An abort that arrives when the thread is between socket operations -- in the gap
between writing the request and reading the response, say -- must not be lost,
so arming is sticky: the token is marked, and the next socket operation on it
fails at once. A token whose window has closed is refused outright, so a late
abort does nothing at all.

One case is deliberately *not* covered: a thread already parked inside
``socket.create_connection`` cannot be interrupted, because there is no socket
object yet to shut down. The backend refuses to dial on an armed token, so the
exposure is one connect timeout on a provider that accepts TCP but never
answers -- not the header wait this module exists to bound.
"""

from __future__ import annotations

import itertools
import socket
import threading
from typing import Any

import httpcore
import httpx


class ProviderRequestAborted(BaseException):
    """Raised inside a provider request whose turn has been superseded.

    Deliberately a ``BaseException`` rather than an ``Exception``. The OpenAI
    SDK turns *any* ``Exception`` escaping its transport into a retryable
    ``APIConnectionError`` -- so a torn-down socket looked to it like a flaky
    network and it slept half a second before trying the request again. That
    retry runs on the provider worker thread, which means the abandoned turn
    went on holding the sole worker slot long past the abort, and the revision
    still could not start: the bug, reintroduced one layer up. Sitting outside
    the ``Exception`` hierarchy is what makes cancellation final.
    """


class ProviderConnectAborter:
    """Aborts whatever provider request a given thread currently has in flight."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens = itertools.count(1)
        self._token_of_thread: dict[int, int] = {}
        self._stream_of_token: dict[int, _TrackedStream] = {}
        self._armed: set[int] = set()
        self._open: set[int] = set()
        self.installed_on: httpx.Client | None = None

    # ── installation ──────────────────────────────────────────────────────────

    def install(self, http_client: httpx.Client) -> None:
        """Route *http_client*'s connections through a stream we can abort.

        This reaches through three private attributes -- httpx's ``_transport``
        and ``_pool``, and httpcore's ``_network_backend`` -- which expose no
        public hook for the network backend. That is deliberate and
        deliberately loud: if an upgrade moves them, cancellation would
        otherwise stop working silently and every revised turn would quietly go
        slow again, which is exactly the failure this module exists to end. Note
        that installing on an already-used client only affects connections
        opened from here on.
        """
        transport = getattr(http_client, "_transport", None)
        pool = getattr(transport, "_pool", None)
        if pool is None or not hasattr(pool, "_network_backend"):
            raise RuntimeError(
                "Cannot make provider requests abortable: this httpx client exposes no "
                f"httpcore connection pool (transport={type(transport).__name__}). The "
                "httpx/httpcore internals ProviderConnectAborter depends on have moved."
            )
        pool._network_backend = _TrackingBackend(self)
        self.installed_on = http_client

    # ── arming ────────────────────────────────────────────────────────────────

    def begin(self) -> int:
        """Open an abort window for the request the calling thread is about to
        issue, and return the token that addresses it.

        Called before the request, so a cancellation arriving while the provider
        is still silent has something to act on.
        """
        token = next(self._tokens)
        with self._lock:
            self._token_of_thread[threading.get_ident()] = token
            self._open.add(token)
        return token

    def end(self, token: int) -> None:
        """Close *token*'s window. Aborts for it are refused from here on."""
        ident = threading.get_ident()
        with self._lock:
            self._open.discard(token)
            self._armed.discard(token)
            self._stream_of_token.pop(token, None)
            if self._token_of_thread.get(ident) == token:
                del self._token_of_thread[ident]

    def abort(self, token: int) -> bool:
        """Abort the request *token* addresses, and anything it starts next.

        Returns whether the request was aborted at all -- true both when a
        socket was torn down and when the request was between socket operations
        and only the sticky arm took effect. False means the window is closed:
        the request already finished and nothing was disturbed.
        """
        with self._lock:
            if token not in self._open:
                return False
            self._armed.add(token)
            stream = self._stream_of_token.get(token)
            if stream is not None:
                # Torn down while still holding the lock. Releasing it first
                # leaves a window in which the request can finish and return
                # its connection to the pool, and the teardown then lands on
                # whichever request picked that pooled connection up next.
                # ``shutdown``/``close`` are non-blocking syscalls, so holding
                # the lock across them cannot stall another thread for long.
                stream.abort()
        return True

    def is_armed(self, token: int) -> bool:
        with self._lock:
            return token in self._armed

    def current_thread_is_armed(self) -> bool:
        with self._lock:
            token = self._token_of_thread.get(threading.get_ident())
            return token is not None and token in self._armed

    # ── used by the tracked streams ───────────────────────────────────────────

    def _bind(self, stream: _TrackedStream) -> tuple[bool, int | None]:
        """Claim *stream* for the calling thread's in-flight request.

        Returns ``(allowed, token)``. ``allowed`` is false only when that
        request's abort is already armed, in which case the caller must not
        begin the socket operation. A thread with no open window is untracked
        -- not every user of the shared httpx client is a provider worker -- and
        is allowed through with no token.
        """
        with self._lock:
            token = self._token_of_thread.get(threading.get_ident())
            if token is None:
                return True, None
            if token in self._armed:
                return False, token
            self._stream_of_token[token] = stream
        return True, token

    def _unbind(self, token: int) -> None:
        """Release the stream, so an abort between operations arms rather than
        reaching a connection that may already be back in the pool."""
        with self._lock:
            self._stream_of_token.pop(token, None)


class _TrackedStream(httpcore.NetworkStream):
    """A network stream that remembers which thread is using it right now."""

    def __init__(self, inner: httpcore.NetworkStream, aborter: ProviderConnectAborter) -> None:
        self._inner = inner
        self._aborter = aborter

    def _guarded(self, operation: Any, *args: Any) -> Any:
        allowed, token = self._aborter._bind(self)
        if not allowed:
            self.abort()
            raise ProviderRequestAborted("provider request aborted before it reached the socket")
        try:
            return operation(*args)
        except BaseException:
            # Tearing the socket down surfaces here as an ordinary read error,
            # indistinguishable to the SDK from a flaky network -- and so
            # retryable. Re-raise as the cancellation it actually is.
            if token is not None and self._aborter.is_armed(token):
                raise ProviderRequestAborted("provider request aborted mid-flight") from None
            raise
        finally:
            if token is not None:
                self._aborter._unbind(token)

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self._guarded(self._inner.read, max_bytes, timeout)

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._guarded(self._inner.write, buffer, timeout)

    def close(self) -> None:
        self._inner.close()

    def start_tls(
        self,
        ssl_context: Any,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        inner = self._guarded(self._inner.start_tls, ssl_context, server_hostname, timeout)
        return _TrackedStream(inner, self._aborter)

    def get_extra_info(self, info: str) -> Any:
        return self._inner.get_extra_info(info)

    def abort(self) -> None:
        """Tear the connection down hard enough to wake a blocked reader.

        ``shutdown`` is what unblocks the parked thread -- closing the socket
        alone leaves it parked -- and the ``close`` that follows sends the FIN
        that tells the provider to stop generating a response nobody will read.
        """
        raw = self._inner.get_extra_info("socket")
        if raw is not None:
            try:
                raw.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # already gone; the close below still tidies up
        try:
            self._inner.close()
        except Exception:
            pass


class _TrackingBackend(httpcore.SyncBackend):
    """Hands out :class:`_TrackedStream` for every connection httpcore opens."""

    def __init__(self, aborter: ProviderConnectAborter) -> None:
        super().__init__()
        self._aborter = aborter

    def _connect(self, opener: Any, *args: Any, **kwargs: Any) -> httpcore.NetworkStream:
        # Refuse to dial at all on an armed thread, so a retry the SDK has
        # already decided on cannot open a connection that is doomed anyway.
        if self._aborter.current_thread_is_armed():
            raise ProviderRequestAborted("provider request aborted before connecting")
        return _TrackedStream(opener(*args, **kwargs), self._aborter)

    def connect_tcp(self, *args: Any, **kwargs: Any) -> httpcore.NetworkStream:
        return self._connect(super().connect_tcp, *args, **kwargs)

    def connect_unix_socket(self, *args: Any, **kwargs: Any) -> httpcore.NetworkStream:
        return self._connect(super().connect_unix_socket, *args, **kwargs)
