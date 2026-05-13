"""
Validate ``*_pc.bin`` radar point clouds in **View-of-Delft (VoD)** layout.

Per the VoD documentation (`docs/SENSORS_AND_DATA.md`), each radar ``.bin`` is a
packed ``float32`` array shaped **(N, 7)** with no header:

    [x, y, z, RCS, v_r, v_r_compensated, time]

So the file byte length must be a multiple of **28** (7 × float32).

Fallback (non-VoD recorders): if the length is not ``0 (mod 28)``, we strip a
``size % 16`` byte prefix (legacy quirk) and require the remainder to still
decode as ``N×7`` float32.

Checks:
  - non-empty
  - decodes as VoD ``N×7`` float32 after optional legacy prefix strip
  - all floats finite (unless ``--quick``)
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


def _vod_payload(blob: bytes) -> tuple[int, memoryview]:
    """Return (prefix_len, body) where body is float32 N×7."""
    n = len(blob)
    if n == 0:
        raise ValueError("empty file")
    if n % 4 != 0:
        raise ValueError(f"file size {n} is not a multiple of 4 (float32)")

    nf = n // 4

    if nf % 7 == 0:
        return 0, memoryview(blob)

    pfx = n % 16
    body = blob[pfx:]
    if len(body) % 28 != 0:
        raise ValueError(
            f"not VoD layout: {nf} floats not divisible by 7; "
            f"after {pfx}-byte prefix, {len(body)} bytes not multiple of 28"
        )
    if len(body) < 28:
        raise ValueError("payload smaller than one point (28 bytes)")
    return pfx, memoryview(body)


def payload_finite_numpy(body: memoryview) -> bool:
    arr = np.frombuffer(body, dtype=np.float32)
    return bool(np.all(np.isfinite(arr)))


def payload_finite_stdlib(body: memoryview, chunk_floats: int = 65536) -> bool:
    nbytes = len(body)
    pos = 0
    while pos < nbytes:
        take = min(chunk_floats * 4, nbytes - pos)
        if take % 4 != 0:
            return False
        nf = take // 4
        fmt = "<" + "f" * nf
        floats = struct.unpack_from(fmt, body, pos)
        if not all(math.isfinite(x) for x in floats):
            return False
        pos += take
    return True


def validate_pc_bin_file(
    path: Path,
    *,
    check_finite: bool = True,
) -> tuple[str | None, tuple[int, int] | None, int]:
    """
    Returns (error_message_or_None, (num_points, 7) on success, prefix_bytes_stripped).
    """
    path = Path(path)
    try:
        blob = path.read_bytes()
        pfx, body = _vod_payload(blob)
    except ValueError as e:
        return (str(e), None, 0)

    n_pts = len(body) // 28
    layout = (n_pts, 7)

    if check_finite:
        ok = (
            payload_finite_numpy(body)
            if np is not None
            else payload_finite_stdlib(body)
        )
        if not ok:
            return ("contains NaN or Inf float32 values", layout, pfx)

    return (None, layout, pfx)


def validate_pc_bin_dataset(
    paths: list[Path],
    *,
    check_finite: bool = True,
) -> tuple[bool, list[str], str]:
    paths = [Path(p) for p in paths]
    errors: list[str] = []
    point_counts: list[int] = []
    used_strip = 0

    do_finite = check_finite
    if check_finite and np is None:
        print(
            "warning: numpy not installed; using slower stdlib float scan",
            file=sys.stderr,
        )

    for p in paths:
        err, lay, pfx = validate_pc_bin_file(p, check_finite=do_finite)
        if err:
            errors.append(f"{p.name}: {err}")
            continue
        assert lay is not None
        point_counts.append(lay[0])
        if pfx > 0:
            used_strip += 1

    if not paths:
        summary = "no files"
    elif errors:
        summary = f"{len(paths)} file(s), {len(errors)} error(s)"
    else:
        mn, mx = min(point_counts), max(point_counts)
        summary = (
            f"{len(paths)} OK - VoD radar float32 x 7 "
            f"(x,y,z,RCS,v_r,v_r_compensated,time); "
            f"points min={mn} max={mx}"
        )
        if used_strip:
            summary += f"; {used_strip} file(s) needed legacy 0-15 byte prefix strip"

    return (len(errors) == 0, errors, summary)


def iter_pc_bins(root: Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    out = [p for p in root.iterdir() if p.is_file() and p.name.endswith("_pc.bin")]
    out.sort(key=lambda x: x.name)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "folder",
        type=Path,
        help="Directory containing *_pc.bin files",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Skip finite-float scan (only VoD size/layout)",
    )
    args = ap.parse_args(argv)

    try:
        bins = iter_pc_bins(args.folder)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2

    if not bins:
        print(f"No *_pc.bin files in {args.folder.resolve()}", file=sys.stderr)
        return 1

    ok, errors, summary = validate_pc_bin_dataset(
        bins,
        check_finite=not args.quick,
    )
    print(summary)
    for line in errors:
        print(line, file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
