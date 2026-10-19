"""Suite-wide guards.

SOP-805: no test may export spans to a real Langfuse project, whatever the
shell running pytest exports. Ian's laptop shell and Sophie's enable-sophie
both set LANGFUSE_* for Sophie, and the OTel SDK reads OTEL_* by itself, so
every test starts with both families removed and with no voice tracer
installed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from speech_to_speech.tracing import voice_turns


@pytest.fixture(autouse=True)
def _no_voice_trace_export(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in list(os.environ):
        if name.startswith(("LANGFUSE_", "OTEL_")):
            monkeypatch.delenv(name, raising=False)
    voice_turns.uninstall()
    yield
    voice_turns.uninstall()
