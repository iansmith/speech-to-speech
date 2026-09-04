"""Local socket servers used to exercise the provider connect/abort path.

A fake ``create`` that blocks on a ``threading.Event`` cannot stand in for a
stalled provider here: the thing under test is whether a thread parked in a
real socket read can be interrupted from another thread, and that is a property
of sockets, not of the SDK surface. These servers therefore speak just enough
HTTP/1.1 to accept a request and then behave badly on purpose.
"""

from __future__ import annotations

import socket
import threading
import time

# The provider stalls for far longer than any assertion window, so a test that
# passes cannot have passed by waiting the stall out.
STALL_S = 10.0


class StallingProvider:
    """A local HTTP server that stalls, either before or after its headers.

    ``stall_after_headers`` models the shape seen on the live call of
    2026-09-04: the provider answers 200 promptly and then goes quiet
    mid-body, so the request has a response object and the worker is parked in
    a body read rather than waiting to connect.
    """

    def __init__(self, *, stall_after_headers: bool = False) -> None:
        self.stall_after_headers = stall_after_headers
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self.port: int = self._listener.getsockname()[1]

        self._lock = threading.Lock()
        self._request_times: list[float] = []
        self._conns: list[socket.socket] = []
        self.request_received = threading.Event()
        self.second_request_received = threading.Event()
        self.client_disconnected_before_headers = threading.Event()

        self._stop = threading.Event()
        threading.Thread(target=self._accept_loop, daemon=True).start()

    @property
    def request_times(self) -> list[float]:
        with self._lock:
            return list(self._request_times)

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            conns = list(self._conns)
        for conn in conns:
            try:
                conn.close()
            except OSError:
                pass
        try:
            self._listener.close()
        except OSError:
            pass

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            with self._lock:
                self._conns.append(conn)
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    @staticmethod
    def read_one_request(conn: socket.socket, buffered: bytes) -> tuple[bytes, bytes] | None:
        """Read one whole request (headers plus its Content-Length body)."""
        buf = buffered
        while b"\r\n\r\n" not in buf:
            chunk = conn.recv(65536)
            if not chunk:
                return None
            buf += chunk
        head, rest = buf.split(b"\r\n\r\n", 1)
        length = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":", 1)[1])
        while len(rest) < length:
            chunk = conn.recv(65536)
            if not chunk:
                return None
            rest += chunk
        return head, rest[length:]

    def _serve(self, conn: socket.socket) -> None:
        try:
            if self.read_one_request(conn, b"") is None:
                return
        except OSError:
            return  # closed under us by teardown
        with self._lock:
            self._request_times.append(time.monotonic())
            ordinal = len(self._request_times)
        self.request_received.set()
        if ordinal == 2:
            self.second_request_received.set()

        if self.stall_after_headers:
            # Answer, then go quiet part-way through the body. The client now
            # holds a response object -- so the existing close-based abort has
            # something to close -- and is parked in a read all the same.
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Cache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n"
                b": warming up\n\n"
            )

        # Send nothing more; just watch for the client hanging up.
        #
        # client_disconnected_before_headers is only meaningful in the default
        # mode, where nothing at all has been written to this connection, so a
        # disconnect seen here is by construction one that arrived before any
        # headers. Under stall_after_headers a 200 has already gone out, and the
        # flag would be a misnomer -- no test reads it in that mode.
        conn.settimeout(STALL_S * 3)
        try:
            if conn.recv(1) == b"":
                self.client_disconnected_before_headers.set()
        except OSError:
            self.client_disconnected_before_headers.set()  # a reset is a disconnect too
        finally:
            try:
                conn.close()
            except OSError:
                pass


class KeepAliveProvider:
    """Answers every request promptly, reusing one persistent connection."""

    def __init__(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self.port: int = self._listener.getsockname()[1]
        self.accepted_connections = 0
        self._conns: list[socket.socket] = []
        self._stop = threading.Event()
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            self.accepted_connections += 1
            self._conns.append(conn)
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        leftover = b""
        while not self._stop.is_set():
            try:
                parsed = StallingProvider.read_one_request(conn, leftover)
            except OSError:
                return  # closed under us by teardown
            if parsed is None:
                return
            _, leftover = parsed
            body = b'{"ok":true}'
            try:
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s"
                    % (len(body), body)
                )
            except OSError:
                return

    def close(self) -> None:
        self._stop.set()
        for conn in self._conns:
            try:
                conn.close()
            except OSError:
                pass
        try:
            self._listener.close()
        except OSError:
            pass
