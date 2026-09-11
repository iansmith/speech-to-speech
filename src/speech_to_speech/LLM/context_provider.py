"""Optional pre-generation context enrichment via an external HTTP service.

A consumer embedding this pipeline often knows things the pipeline cannot: who is
speaking, what they said last week, what the application has decided to tell them.
That knowledge has to reach the model *before* it generates, and the only seam
where it can is between the transcript being final and ``_generate`` being called.

This module is that seam. When ``context_provider_url`` is set, each turn POSTs its
conversation to that URL and appends whatever items come back to the turn's chat.
When it is unset the feature does not exist, exactly as ``enable_lang_prompt``
behaves when False.

**It fails open, always.** A provider that is slow, down, or returns nonsense costs
the turn its enrichment and nothing else — generation proceeds with the unmodified
chat. This is deliberate and it is the whole reason the hook is shaped as an
enrichment step rather than as a proxy in front of the model: a conversation that
loses its context is degraded, while a conversation that loses its model is over.
Nothing in this module raises.

Wire protocol, request::

    POST <context_provider_url>
    {"turn_id": "...", "language_code": "en",
     "instructions": "<the session instructions in force>",
     "conversation": [{"role": "user", "content": "..."}, ...]}

Response::

    {"items": [{"role": "system", "text": "..."}, ...]}

``role`` is one of ``system``, ``user``, ``assistant``. Unknown roles, missing text
and malformed entries are skipped individually rather than failing the batch — a
provider that gets one item wrong should still deliver the others.

Only the standard library is used, on purpose: this is meant to be upstreamable and
should not add a dependency to do one POST.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from speech_to_speech.LLM.chat import (
    SupportedItem,
    make_assistant_message,
    make_system_message,
    make_user_message,
)

logger = logging.getLogger(__name__)

_MAKERS = {
    "system": make_system_message,
    "user": make_user_message,
    "assistant": make_assistant_message,
}


def fetch_context_items(
    url: str,
    conversation: list[dict[str, Any]],
    *,
    timeout_s: float,
    turn_id: str | None = None,
    language_code: str | None = None,
    instructions: str | None = None,
) -> list[SupportedItem]:
    """Ask *url* for extra chat items for this turn.

    Returns the items to append, or an empty list if anything at all went wrong.
    Never raises — see the module docstring.
    """
    payload = json.dumps(
        {
            "turn_id": turn_id,
            "language_code": language_code,
            "instructions": instructions,
            "conversation": conversation,
        }
    ).encode("utf-8")

    try:
        # Request() itself parses the URL and raises ValueError on a malformed one,
        # so it belongs inside the guard rather than above it. Found by
        # test_malformed_url_fails_open, which failed when this was constructed
        # outside the try -- a misconfigured URL would have killed the turn, which
        # is the exact opposite of what this module promises.
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # Includes timeouts, connection refused, DNS failure and malformed URLs.
        logger.warning("context provider %s unreachable, generating without it: %s", url, exc)
        return []

    try:
        decoded = json.loads(body)
        raw_items = decoded["items"]
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning("context provider %s returned an unusable body, ignoring: %s", url, exc)
        return []

    if not isinstance(raw_items, list):
        logger.warning("context provider %s returned items of type %s, want list", url, type(raw_items).__name__)
        return []

    items: list[SupportedItem] = []
    for entry in raw_items:
        item = _build_item(entry, url)
        if item is not None:
            items.append(item)
    return items


def _build_item(entry: Any, url: str) -> SupportedItem | None:
    """Turn one wire entry into a chat item, or None if it is unusable."""
    if not isinstance(entry, dict):
        logger.warning("context provider %s: skipping non-object item %r", url, entry)
        return None

    maker = _MAKERS.get(entry.get("role"))
    if maker is None:
        logger.warning("context provider %s: skipping item with role %r", url, entry.get("role"))
        return None

    text = entry.get("text")
    if not isinstance(text, str) or not text:
        logger.warning("context provider %s: skipping %s item with empty text", url, entry.get("role"))
        return None

    return maker(text)
