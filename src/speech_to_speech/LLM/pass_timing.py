"""Opt-in per-pass LLM timing log.

A tool-calling turn is two LLM passes: the model first *decides* to call a tool
(the request whose conversation ends with the caller's ``user`` message), then
*answers from the result* (the request whose conversation ends with a ``tool``
message appended after the tool ran). mlx-serve prefills at a fixed ~90 tok/s
floor, so the whole of a turn's latency is the number of *uncached* prompt
tokens the second pass has to prefill. This module captures that number, per
pass, so a single live call answers whether the fix is protecting the prefix
(small uncached tail) or capping context growth (prefix evicted, full
re-prefill).

The signal is already in the response: mlx-serve attaches a ``timings`` block
(``prompt_n``, ``cached_n``, ``prompt_per_second``, ``predicted_per_second``)
to the usage-bearing chunk. This module reads it and writes one JSON line per
pass.

Off by default. Set ``SOPHIE_LLM_TIMING=1`` to enable; rows land in
``$SOPHIE_LLM_TIMING_DIR`` (default ``scratch/llm-timing/``), one file per
process. Every write is best-effort and never raises into the call path.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENV_ENABLE = "SOPHIE_LLM_TIMING"
_ENV_DIR = "SOPHIE_LLM_TIMING_DIR"
_DEFAULT_DIR = "scratch/llm-timing"

_seq = itertools.count(1)
_lock = threading.Lock()
_path: Path | None = None
_path_resolved = False


def enabled() -> bool:
    """True when timing capture is switched on for this process."""
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in ("1", "true", "yes", "on")


def next_seq() -> int:
    """A process-monotonic request sequence number for grouping a call's passes."""
    return next(_seq)


def _file() -> Path | None:
    """The append target, created once. None if the directory can't be opened."""
    global _path, _path_resolved
    if _path_resolved:
        return _path
    _path_resolved = True
    try:
        directory = Path(os.environ.get(_ENV_DIR) or _DEFAULT_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        _path = directory / f"llm-timing-{os.getpid()}.jsonl"
    except Exception as exc:  # a diagnostic log must never break a call
        logger.warning("pass_timing: could not open timing dir: %s", exc)
        _path = None
    return _path


def classify_pass(last_role: str | None) -> str:
    """Name the LLM pass from the conversation's last message role.

    The first pass of a turn ends with the caller's ``user`` message; the pass
    after a tool executes ends with the ``tool`` result appended to the
    conversation. That single role distinguishes 'decide' from
    'answer-after-tool' with no turn state threaded down here.
    """
    if last_role == "tool":
        return "answer_after_tool"
    if last_role == "user":
        return "decide"
    return last_role or "unknown"


def extract_timings(chunk: Any) -> dict[str, Any] | None:
    """Pull mlx-serve's non-standard ``timings`` block off a response chunk.

    The OpenAI SDK does not model ``timings``, so it keeps the field in
    ``model_extra``; a test's ``SimpleNamespace`` fake keeps it as a plain
    attribute. Handle both, and coerce a pydantic sub-model to a dict. Returns
    None when the chunk carries no timings (every chunk but the last).
    """
    timings = getattr(chunk, "timings", None)
    if timings is None:
        extra = getattr(chunk, "model_extra", None)
        if isinstance(extra, dict):
            timings = extra.get("timings")
    if timings is None:
        return None
    if isinstance(timings, dict):
        return timings
    if hasattr(timings, "model_dump"):
        try:
            return timings.model_dump()
        except Exception:
            return None
    if hasattr(timings, "__dict__"):
        return dict(vars(timings))
    return None


def build_row(
    meta: dict[str, Any],
    timings: dict[str, Any] | None,
    wall_ms: float,
    ttft_ms: float | None,
    prompt_tokens: int | None,
    output_tokens: int | None,
) -> dict[str, Any]:
    """Assemble one timing row from a pass's stashed meta and captured signals.

    ``uncached_n = prompt_n - cached_n`` is the number this whole exercise
    exists to read: at mlx-serve's fixed prefill rate it is the pass's latency.
    """
    row: dict[str, Any] = {
        "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "seq": meta.get("seq"),
        "pass": classify_pass(meta.get("last_role")),
        "last_role": meta.get("last_role"),
        "n_messages": meta.get("n_messages"),
        "n_tool_messages": meta.get("n_tool_messages"),
        "wall_ms": round(wall_ms, 1),
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }
    if timings:
        prompt_n = timings.get("prompt_n")
        cached_n = timings.get("cached_n")
        row["prompt_n"] = prompt_n
        row["cached_n"] = cached_n
        if isinstance(prompt_n, int) and isinstance(cached_n, int):
            row["uncached_n"] = prompt_n - cached_n
        row["prefill_ms"] = timings.get("prompt_ms")
        row["prefill_tok_s"] = timings.get("prompt_per_second")
        row["decode_n"] = timings.get("predicted_n")
        row["decode_tok_s"] = timings.get("predicted_per_second")
    return row


def record(row: dict[str, Any]) -> None:
    """Append one row as a JSON line. Best-effort: never raises."""
    if not enabled():
        return
    path = _file()
    if path is None:
        return
    try:
        line = json.dumps(row, ensure_ascii=False)
        with _lock, path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception as exc:  # a diagnostic log must never break a call
        logger.warning("pass_timing: failed to write row: %s", exc)
