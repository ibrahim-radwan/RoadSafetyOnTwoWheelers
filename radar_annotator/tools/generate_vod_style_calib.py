"""
Build VoD / KITTI-style ``calib/<stem>.txt`` files from FusionApp ``calibration_report.txt``.

Expected report sections:
  CAMERA INTRINSICS (fx, fy, cx, cy)
  ROTATION R (cam_from_radar) 3×3 in brackets
  TRANSLATION t (radar origin in camera frame)

Each output file matches the View-of-Delft layout:

  P0–P3: same 3×4 projection from intrinsics (no stereo skew).
  R0_rect: identity (rectification folded into P).
  Tr_velo_to_cam: [R | t] row-major 12 floats.
  Tr_imu_to_velo: empty body (same as VoD reference).
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path
from typing import Optional, Tuple


def _fmt_nums(vals: list[float]) -> str:
    return " ".join(f"{v:.8g}" for v in vals)


def parse_calibration_report(path: Path) -> tuple[list[float], list[list[float]], list[float]]:
    text = path.read_text(encoding="utf-8", errors="replace")

    intr = re.search(
        r"fx\s*=\s*([+-]?\d+(?:\.\d*)?(?:e[+-]?\d+)?)\s+"
        r"fy\s*=\s*([+-]?\d+(?:\.\d*)?(?:e[+-]?\d+)?)\s+"
        r"cx\s*=\s*([+-]?\d+(?:\.\d*)?(?:e[+-]?\d+)?)\s+"
        r"cy\s*=\s*([+-]?\d+(?:\.\d*)?(?:e[+-]?\d+)?)",
        text,
        re.I,
    )
    if intr is None:
        raise ValueError(f"Could not parse CAMERA INTRINSICS (fx,fy,cx,cy) in {path}")

    fx, fy, cx, cy = (float(intr.group(i)) for i in range(1, 5))

    rot_block = re.search(
        r"ROTATION\s+R[^\[]*(.*?)\s*TRANSLATION",
        text,
        re.S | re.I,
    )
    if rot_block is None:
        raise ValueError(f"Could not find ROTATION R section in {path}")

    rot_body = rot_block.group(1).split("Euler")[0]
    rows_raw = re.findall(r"\[\s*([^\]]+)\s*\]", rot_body)
    if len(rows_raw) < 3:
        raise ValueError("Expected at least three bracket rows for R matrix")
    rows_raw = rows_raw[:3]

    R: list[list[float]] = []
    for ln in rows_raw:
        nums = [float(x) for x in re.split(r"[,\s]+", ln.strip()) if x]
        if len(nums) != 3:
            raise ValueError(f"Bad R row: {ln!r}")
        R.append(nums)

    tr_line = re.search(
        r"TRANSLATION\s+t[^\n]*\n\s*x\s*=\s*([+-]?\d+(?:\.\d*)?)"
        r"\s+y\s*=\s*([+-]?\d+(?:\.\d*)?)\s+z\s*=\s*([+-]?\d+(?:\.\d*)?)",
        text,
        re.I,
    )
    if tr_line is None:
        raise ValueError(f"Could not parse TRANSLATION line (x,y,z) in {path}")
    t = [float(tr_line.group(i)) for i in range(1, 4)]

    p_flat = [
        fx,
        0.0,
        cx,
        0.0,
        0.0,
        fy,
        cy,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]

    return p_flat, R, t


def read_png_wh(path: Path) -> Tuple[int, int]:
    with Path(path).open("rb") as f:
        if f.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a PNG file: {path}")
        ln = struct.unpack(">I", f.read(4))[0]
        typ = f.read(4)
        if typ != b"IHDR" or ln < 13:
            raise ValueError(f"Missing IHDR in PNG: {path}")
        data = f.read(ln)
        w, h = struct.unpack(">II", data[:8])
        return int(w), int(h)


def build_calib_txt(
    p_flat: list[float],
    tr_flat: list[float],
    *,
    image_wh: Optional[Tuple[int, int]] = None,
) -> str:
    r0_flat = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    lines: list[str] = []
    if image_wh is not None:
        lines.append(f"# image_size {image_wh[0]} {image_wh[1]}")
    lines.extend(
        [
            f"P0: {_fmt_nums(p_flat)}",
            f"P1: {_fmt_nums(p_flat)}",
            f"P2: {_fmt_nums(p_flat)}",
            f"P3: {_fmt_nums(p_flat)}",
            f"R0_rect: {_fmt_nums(r0_flat)}",
            f"Tr_velo_to_cam: {_fmt_nums(tr_flat)}",
            "Tr_imu_to_velo: ",
            "",
        ]
    )
    return "\n".join(lines)


def stems_from_image_dir(image_dir: Path) -> list[str]:
    stems: list[str] = []
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        for p in sorted(image_dir.glob(f"*{ext}")):
            stems.append(p.stem)
    # stable unique order
    seen: set[str] = set()
    out: list[str] = []
    for s in stems:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to calibration_report.txt",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Dataset root; writes calib/ under it (or --calib-dir explicitly)",
    )
    ap.add_argument(
        "--calib-dir",
        type=Path,
        default=None,
        help="Output calib folder (default: <out-dir>/calib)",
    )
    ap.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Folder with frames (e.g. image_2); stems define output filenames",
    )
    ap.add_argument(
        "--count",
        type=int,
        default=None,
        help="If no images: emit N files named 0000.. with width from --width",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=4,
        help="Zero-padding width when using --count (default 4)",
    )
    args = ap.parse_args()

    p_flat, _R, _t = parse_calibration_report(args.report)
    tr_flat = [
        _R[0][0],
        _R[0][1],
        _R[0][2],
        _t[0],
        _R[1][0],
        _R[1][1],
        _R[1][2],
        _t[1],
        _R[2][0],
        _R[2][1],
        _R[2][2],
        _t[2],
    ]

    calib_root = args.calib_dir if args.calib_dir else args.out_dir / "calib"
    calib_root.mkdir(parents=True, exist_ok=True)

    stems: list[str] = []
    image_wh: Optional[Tuple[int, int]] = None
    if args.image_dir is not None:
        img_dir = Path(args.image_dir)
        if not img_dir.is_dir():
            print(f"image dir not found: {img_dir}", file=sys.stderr)
            return 2
        stems = stems_from_image_dir(img_dir)
        for ext in (".png", ".jpg", ".jpeg"):
            cand = sorted(img_dir.glob(f"*{ext}"))
            if cand:
                if ext == ".png":
                    try:
                        image_wh = read_png_wh(cand[0])
                    except ValueError:
                        image_wh = None
                break

    if not stems:
        if args.count is None:
            print(
                "No stems from images; pass --image-dir with frames or --count N",
                file=sys.stderr,
            )
            return 2
        w = max(1, args.width)
        stems = [f"{i:0{w}d}" for i in range(args.count)]

    body = build_calib_txt(p_flat, tr_flat, image_wh=image_wh)

    for stem in stems:
        (calib_root / f"{stem}.txt").write_text(body, encoding="utf-8")

    print(f"Wrote {len(stems)} file(s) to {calib_root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
