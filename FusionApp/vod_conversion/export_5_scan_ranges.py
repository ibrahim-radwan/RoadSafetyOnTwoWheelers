#!/usr/bin/env python
"""Export 5_scan manifest ranges into separate folders with sequences 1..N.

Preserves the VoD layout: ``image_2`` plus zero-based five-digit sample ids
shared with ``calib`` (``00000.png``, ``00000.bin``, ``00000.txt``, …).
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

DEFAULT_RANGES = [
    (1500, 1700),
    (1895, 2100),
    (2350, 2500),
    (5150, 5450),
    (5560, 5760),
    (5930, 6100),
    (6500, 6730),
    (6895, 7200),
    (7300, 7850),
    (8028, 8200),
    (8680, 8752),
    (9430, 9730),
    (10333, 10500),
    (11730, 14266),
]

FOLDERS = ("image_2", "radar", "radar_raw", "radarref", "calib")


def sample_stem(row: dict[str, str]) -> str:
    """Resolve the shared five-digit stem from sample_id or calib_source."""
    if row.get("sample_id"):
        return str(row["sample_id"])
    calib = row.get("calib_source") or ""
    if calib.endswith(".txt") and len(Path(calib).stem) == 5:
        return Path(calib).stem
    seq = int(row["sequence"])
    return f"{seq - 1:05d}"


def row_paths(src_root: Path, row: dict[str, str]) -> dict[str, Path]:
    stem = sample_stem(row)
    seq = row["sequence"]
    candidates = {
        "image_2": [
            src_root / "image_2" / f"{stem}.png",
            src_root / "image" / f"{stem}.png",
            src_root / "image_2" / f"{seq}.png",
            src_root / "image" / f"{seq}.png",
        ],
        "radar": [
            src_root / "radar" / f"{stem}.bin",
            src_root / "radar" / f"{seq}.bin",
        ],
        "radar_raw": [
            src_root / "radar_raw" / f"{stem}.bin",
            src_root / "radar_raw" / f"{seq}.bin",
        ],
        "radarref": [
            src_root / "radarref" / f"{stem}.csv",
            src_root / "radarref" / f"{seq}.csv",
        ],
        "calib": [src_root / "calib" / row["calib_source"]],
    }
    resolved: dict[str, Path] = {}
    for key, paths in candidates.items():
        for path in paths:
            if path.is_file():
                resolved[key] = path
                break
        else:
            raise FileNotFoundError(
                f"Missing {key} for sequence {seq} under {src_root}"
            )
    return resolved


def export_ranges(
    src_root: Path,
    parent: Path,
    ranges: list[tuple[int, int]],
) -> int:
    with (src_root / "manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        all_rows = list(csv.DictReader(stream))
    if not all_rows:
        raise ValueError(f"Empty manifest: {src_root / 'manifest.csv'}")

    fieldnames = list(all_rows[0].keys())
    if "sample_id" not in fieldnames:
        fieldnames = ["sequence", "sample_id"] + [
            name for name in fieldnames if name != "sequence"
        ]

    parent.mkdir(parents=True, exist_ok=True)
    grand_total = 0

    for start, end in ranges:
        dst_root = parent / f"{start}_{end}"
        if dst_root.exists():
            shutil.rmtree(dst_root)
        for folder in FOLDERS:
            (dst_root / folder).mkdir(parents=True, exist_ok=True)

        picked = [
            row for row in all_rows if start <= int(row["sequence"]) <= end
        ]
        print(f"Range {start}_{end}: {len(picked)} rows", flush=True)

        out_rows: list[dict[str, str]] = []
        for new_seq, row in enumerate(picked, start=1):
            paths = row_paths(src_root, row)
            stem = f"{new_seq - 1:05d}"
            calib_name = f"{stem}.txt"
            shutil.copy2(paths["image_2"], dst_root / "image_2" / f"{stem}.png")
            shutil.copy2(paths["radar"], dst_root / "radar" / f"{stem}.bin")
            shutil.copy2(paths["radar_raw"], dst_root / "radar_raw" / f"{stem}.bin")
            shutil.copy2(paths["radarref"], dst_root / "radarref" / f"{stem}.csv")
            shutil.copy2(paths["calib"], dst_root / "calib" / calib_name)

            out = dict(row)
            out["sequence"] = str(new_seq)
            out["sample_id"] = stem
            out["calib_source"] = calib_name
            out_rows.append(out)

        with (dst_root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)

        grand_total += len(out_rows)
        if out_rows:
            last_stem = out_rows[-1]["sample_id"]
            print(
                f"  -> wrote {dst_root.name} "
                f"(sample ids 00000..{last_stem})",
                flush=True,
            )
        else:
            print(f"  -> wrote empty {dst_root.name}", flush=True)

    print(
        f"Done: {len(ranges)} folders, {grand_total} total samples under {parent}",
        flush=True,
    )
    return grand_total


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split data/5_scan into sequence-range folders, renumbering samples "
            "to VoD five-digit stems starting at 00000."
        )
    )
    parser.add_argument(
        "recording",
        type=Path,
        help="Recording directory that contains data/5_scan/manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination root. Defaults to <recording>/data/5_scan_ranges.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    recording = args.recording.expanduser().resolve()
    src_root = recording / "data" / "5_scan"
    parent = (args.output_dir or (recording / "data" / "5_scan_ranges")).resolve()
    if not (src_root / "manifest.csv").is_file():
        raise FileNotFoundError(f"Missing manifest: {src_root / 'manifest.csv'}")
    export_ranges(src_root, parent, DEFAULT_RANGES)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
