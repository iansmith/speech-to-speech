"""Regenerate ``g711_golden.json`` from CPython's ``audioop`` reference codec.

``audioop`` implements the ITU-T G.711 conversions and is the oracle these
vectors are taken from. It was deprecated in Python 3.11 and **removed in
3.13**, which is precisely why the library under test cannot use it and needs
frozen vectors instead.

Run under Python 3.10-3.12:

    python tests/openai_realtime/testdata/generate_g711_golden.py
"""

from __future__ import annotations

import audioop  # noqa: F401  (absent on 3.13+; this script is generation-only)
import json
import struct
from pathlib import Path

OUT = Path(__file__).with_name("g711_golden.json")

# Deterministic sweep across the full int16 range, plus the boundary values that
# the ITU tables treat specially (zero, unit magnitude, both rails).
PROBES = sorted({-32768, -32767, -1, 0, 1, 32766, 32767} | set(range(-32768, 32768, 127)))


def _decode_table(decoder) -> list[int]:
    """Decode every one of the 256 codepoints to its linear int16 value."""
    raw = decoder(bytes(range(256)), 2)
    return [v for (v,) in struct.iter_unpack("<h", raw)]


def _encode(encoder, values: list[int]) -> list[int]:
    packed = b"".join(struct.pack("<h", v) for v in values)
    return list(encoder(packed, 2))


def main() -> None:
    golden = {
        "_source": "CPython audioop (ITU-T G.711 reference); regenerate with generate_g711_golden.py",
        "ulaw_decode": _decode_table(audioop.ulaw2lin),
        "alaw_decode": _decode_table(audioop.alaw2lin),
        "encode_probes": {
            "linear": PROBES,
            "ulaw": _encode(audioop.lin2ulaw, PROBES),
            "alaw": _encode(audioop.lin2alaw, PROBES),
        },
    }
    OUT.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(PROBES)} encode probes)")


if __name__ == "__main__":
    main()
