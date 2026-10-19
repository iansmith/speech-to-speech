"""SOP-805: both serve entry points turn voice tracing on and always shut it down.

``init_tracing()`` in these two functions is the only thing that turns
tracing on in production, and ``shutdown_tracing()`` is what exports a
call's open turn roots when the process stops. Both functions live in
upstream files, so a sync that drops either line must fail a test here.
"""

from __future__ import annotations

import signal
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from speech_to_speech import configured_runtime, s2s_pipeline


class _StageFailed(Exception):
    """Raised by a stubbed stage to drive an entry point's failure path."""


def _record_tracing(monkeypatch: pytest.MonkeyPatch, module: Any, events: list[str]) -> None:
    monkeypatch.setattr(module, "init_tracing", lambda: events.append("init"))
    monkeypatch.setattr(module, "shutdown_tracing", lambda: events.append("shutdown"))


def _step(events: list[str], name: str, *, fails: bool = False) -> Callable[..., Any]:
    def run(*_args: Any, **_kwargs: Any) -> None:
        events.append(name)
        if fails:
            raise _StageFailed

    return run


@pytest.mark.parametrize(
    ("failing", "expected"),
    [
        (None, ["logger", "init", "build", "start", "wait", "stop", "shutdown"]),
        ("build", ["logger", "init", "build", "shutdown"]),
        ("stop", ["logger", "init", "build", "start", "wait", "stop", "shutdown"]),
    ],
)
def test_configured_serve_inits_before_build_and_always_shuts_down(
    monkeypatch: pytest.MonkeyPatch, failing: str | None, expected: list[str]
) -> None:
    events: list[str] = []
    _record_tracing(monkeypatch, configured_runtime, events)
    monkeypatch.setattr(s2s_pipeline, "setup_logger", _step(events, "logger"))
    monkeypatch.setattr(signal, "signal", lambda *_args: None)
    server = SimpleNamespace(
        start=_step(events, "start"),
        wait=_step(events, "wait"),
        stop=_step(events, "stop", fails=failing == "stop"),
    )

    def build(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        _step(events, "build", fails=failing == "build")()
        return server

    monkeypatch.setattr(configured_runtime, "build_configured_server", build)
    config: Any = SimpleNamespace(runtime={})

    if failing is None:
        configured_runtime.run_configured_command("serve", config)
    else:
        with pytest.raises(_StageFailed):
            configured_runtime.run_configured_command("serve", config)

    assert events == expected


@pytest.mark.parametrize(
    ("failing", "expected"),
    [
        (None, ["logger", "init", "build", "start", "wait", "shutdown"]),
        ("start", ["logger", "init", "build", "start", "shutdown"]),
    ],
)
def test_pipeline_serve_inits_before_build_and_always_shuts_down(
    monkeypatch: pytest.MonkeyPatch, failing: str | None, expected: list[str]
) -> None:
    events: list[str] = []
    handlers: dict[int, Any] = {}
    _record_tracing(monkeypatch, s2s_pipeline, events)
    args = SimpleNamespace(
        module_kwargs=SimpleNamespace(
            log_level="info", log_transcripts=False, num_pipelines=1, enable_live_transcription=False
        )
    )
    manager = SimpleNamespace(
        start=_step(events, "start", fails=failing == "start"),
        wait=_step(events, "wait"),
        stop=_step(events, "stop"),
    )
    monkeypatch.setattr(s2s_pipeline, "parse_arguments", lambda *_args, **_kwargs: args)
    monkeypatch.setattr(s2s_pipeline, "setup_logger", _step(events, "logger"))
    monkeypatch.setattr(s2s_pipeline, "prepare_all_args", lambda _args: None)
    monkeypatch.setattr(s2s_pipeline, "build_pipeline", lambda *_args: _step(events, "build")() or manager)
    monkeypatch.setattr(s2s_pipeline.signal, "signal", lambda sig, handler: handlers.__setitem__(sig, handler))

    if failing is None:
        s2s_pipeline.run_pipeline_command("serve", [])
    else:
        with pytest.raises(_StageFailed):
            s2s_pipeline.run_pipeline_command("serve", [])

    assert events == expected
    # SIGTERM stops the pipeline and exports open turn roots from the handler too.
    events.clear()
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert events == ["stop", "shutdown"]
