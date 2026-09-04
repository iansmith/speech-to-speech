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
* **Aborts are addressed to a thread, not to a request.** A provider worker is
  dedicated to one request for its lifetime, so "the socket thread T is using"
  identifies the request unambiguously, and it is knowable from outside without
  threading a handle back out of the SDK.

An abort that arrives when the thread is between socket operations -- during
connect, or in the gap between writing the request and reading the response --
must not be lost, so arming is sticky: the thread is marked, and the next socket
operation it attempts fails at once.

Sticky marks make the request boundary load-bearing. A worker brackets its
request with :meth:`ProviderConnectAborter.begin` and :meth:`end`, and an abort
for a thread that is not between those two is refused rather than remembered.
Without that, a cancellation racing a worker that had already finished would
leave a mark on a dead thread's ident -- and CPython hands those idents out
again, so the next worker to inherit it would fail its first socket operation
for a request cancelled long before it existed.
"""

from __future__ import annotations

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
        self._streams: dict[int, _TrackedStream] = {}
        self._armed: set[int] = set()
        self._active: set[int] = set()
        self.installed_on: httpx.Client | None = None

    # ── installation ──────────────────────────────────────────────────────────

    def install(self, http_client: httpx.Client) -> None:
        """Route *http_client*'s connections through a stream we can abort.

        This reaches through two private attributes of httpx and httpcore, which
        expose no public hook for the network backend. That is deliberate and
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

    def begin(self, thread_ident: int) -> None:
        """Open the window in which *thread_ident* may be aborted.

        Called by a provider worker before it issues its request, so that a
        cancellation arriving during connect has something to act on.
        """
        with self._lock:
            self._armed.discard(thread_ident)
            self._active.add(thread_ident)

    def end(self, thread_ident: int) -> None:
        """Close that window, disarming the thread for reuse."""
        with self._lock:
            self._active.discard(thread_ident)
            self._armed.discard(thread_ident)

    def abort(self, thread_ident: int) -> bool:
        """Abort *thread_ident*'s in-flight request, and any it starts next.

        Returns whether a socket was actually torn down. False can mean either
        that the thread has no request in flight (nothing to do), or that it is
        between socket operations -- in which case the arm stands and the next
        operation it attempts fails at once.
        """
        with self._lock:
            if thread_ident not in self._active:
                return False
            self._armed.add(thread_ident)
            stream = self._streams.get(thread_ident)
        if stream is None:
            return False
        stream.abort()
        return True

    def is_armed(self, thread_ident: int) -> bool:
        with self._lock:
            return thread_ident in self._armed

    # ── used by the tracked streams ───────────────────────────────────────────

    def _bind(self, stream: _TrackedStream) -> bool:
        """Register the calling thread as the owner of *stream*.

        Returns False when that thread's abort is already armed, in which case
        the caller must not begin the socket operation.
        """
        ident = threading.get_ident()
        with self._lock:
            if ident in self._armed:
                return False
            self._streams[ident] = stream
        return True

    def _unbind(self) -> None:
        with self._lock:
            self._streams.pop(threading.get_ident(), None)


class _TrackedStream(httpcore.NetworkStream):
    """A network stream that remembers which thread is using it right now."""

    def __init__(self, inner: httpcore.NetworkStream, aborter: ProviderConnectAborter) -> None:
        self._inner = inner
        self._aborter = aborter

    def _guarded(self, operation: Any, *args: Any) -> Any:
        ident = threading.get_ident()
        if not self._aborter._bind(self):
            self.abort()
            raise ProviderRequestAborted("provider request aborted before it reached the socket")
        try:
            return operation(*args)
        except BaseException:
            # Tearing the socket down surfaces here as an ordinary read error,
            # indistinguishable to the SDK from a flaky network -- and so
            # retryable. Re-raise as the cancellation it actually is.
            if self._aborter.is_armed(ident):
                raise ProviderRequestAborted("provider request aborted mid-flight") from None
            raise
        finally:
            self._aborter._unbind()

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
        if self._aborter.is_armed(threading.get_ident()):
            raise ProviderRequestAborted("provider request aborted before connecting")
        return _TrackedStream(opener(*args, **kwargs), self._aborter)

    def connect_tcp(self, *args: Any, **kwargs: Any) -> httpcore.NetworkStream:
        return self._connect(super().connect_tcp, *args, **kwargs)

    def connect_unix_socket(self, *args: Any, **kwargs: Any) -> httpcore.NetworkStream:
        return self._connect(super().connect_unix_socket, *args, **kwargs)
