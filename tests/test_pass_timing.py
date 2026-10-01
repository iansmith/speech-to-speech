"""Unit tests for the opt-in per-pass LLM timing log.

No GPU or live server: the module helpers are pure, and the one wiring test
drives ``_iter_stream_events`` with fake chunks shaped like mlx-serve's.

Run with pytest, or standalone:  python tests/test_pass_timing.py
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import speech_to_speech.LLM.pass_timing as pt
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler


def test_classify_pass_from_last_role():
    assert pt.classify_pass("user") == "decide"
    assert pt.classify_pass("tool") == "answer_after_tool"
    assert pt.classify_pass("assistant") == "assistant"
    assert pt.classify_pass(None) == "unknown"


def test_extract_timings_from_plain_attribute():
    chunk = SimpleNamespace(timings={"prompt_n": 10, "cached_n": 3})
    assert pt.extract_timings(chunk) == {"prompt_n": 10, "cached_n": 3}


def test_extract_timings_from_model_extra():
    # The OpenAI SDK parks unknown fields in model_extra, not as an attribute.
    chunk = SimpleNamespace(model_extra={"timings": {"prompt_n": 5}})
    chunk.timings = None
    assert pt.extract_timings(chunk) == {"prompt_n": 5}


def test_extract_timings_coerces_pydantic_like():
    class _T:
        def model_dump(self):
            return {"prompt_n": 7}

    assert pt.extract_timings(SimpleNamespace(timings=_T())) == {"prompt_n": 7}


def test_extract_timings_absent_is_none():
    assert pt.extract_timings(SimpleNamespace(choices=[])) is None


def test_build_row_computes_uncached():
    meta = {"seq": 2, "last_role": "tool", "n_messages": 9, "n_tool_messages": 1}
    timings = {
        "prompt_n": 4979,
        "cached_n": 3961,
        "prompt_ms": 11600.0,
        "prompt_per_second": 88.0,
        "predicted_n": 42,
        "predicted_per_second": 9.3,
    }
    row = pt.build_row(meta, timings, wall_ms=14400.0, ttft_ms=11800.0, prompt_tokens=4979, output_tokens=42)
    assert row["pass"] == "answer_after_tool"
    assert row["uncached_n"] == 4979 - 3961
    assert row["prefill_tok_s"] == 88.0
    assert row["decode_tok_s"] == 9.3
    assert row["ttft_ms"] == 11800.0


def test_build_row_tolerates_missing_timings():
    row = pt.build_row({"seq": 1, "last_role": "user"}, None, wall_ms=100.0, ttft_ms=None, prompt_tokens=None, output_tokens=None)
    assert row["pass"] == "decide"
    assert "uncached_n" not in row
    assert row["ttft_ms"] is None


def test_record_is_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv(pt._ENV_ENABLE, raising=False)
    monkeypatch.setenv(pt._ENV_DIR, str(tmp_path))
    pt._path = None
    pt._path_resolved = False
    pt.record({"seq": 1})
    assert list(tmp_path.iterdir()) == []


def test_iter_stream_events_writes_one_row(tmp_path, monkeypatch):
    monkeypatch.setenv(pt._ENV_ENABLE, "1")
    monkeypatch.setenv(pt._ENV_DIR, str(tmp_path))
    pt._path = None
    pt._path_resolved = False

    # Bypass __init__: _iter_stream_events only needs `_pass_meta` and the
    # stateless tool accumulator helper.
    handler = object.__new__(ChatCompletionsApiModelHandler)
    handler._pass_meta = {
        "seq": 7,
        "t_request": 1000.0,
        "last_role": "tool",
        "n_messages": 9,
        "n_tool_messages": 1,
    }

    content_chunk = SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(content="Alma's birthday is October 1st.", tool_calls=None, refusal=None))],
    )
    trailing_chunk = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=4979, completion_tokens=12),
        choices=[],
        timings={"prompt_n": 4979, "cached_n": 3961, "prompt_ms": 11600.0, "prompt_per_second": 88.0, "predicted_n": 12, "predicted_per_second": 9.3},
    )

    events = list(handler._iter_stream_events([content_chunk, trailing_chunk]))

    # The normal event stream is unchanged by the instrumentation.
    assert any(getattr(e, "text", None) == "Alma's birthday is October 1st." for e in events)

    rows = [json.loads(line) for line in (tmp_path / f"llm-timing-{__import__('os').getpid()}.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["seq"] == 7
    assert row["pass"] == "answer_after_tool"
    assert row["uncached_n"] == 1018
    assert row["prompt_n"] == 4979
    assert row["prefill_tok_s"] == 88.0
    assert row["ttft_ms"] is not None
    # meta is cleared so a later pass can't inherit it.
    assert handler._pass_meta is None


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
