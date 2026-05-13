"""
Keep only `.bin` + `*_pc.bin` pairs (same stem before `_pc.bin`), sorted by name.

Standalone `.bin` files with no matching `*_pc.bin` are skipped.

For each pair, copies by default:
  - the main `*.bin`
  - the matching `*_pc.bin`
  - the nearest camera image (`.png` / `.jpeg` / …), by middle `_*_` timestamp

Use --no-pc to skip `*_pc.bin`, or --no-images to skip images.

Unless ``--no-pc-validation`` is set, every paired ``*_pc.bin`` must match **VoD**
(View-of-Delft) radar bins: packed ``float32`` **(N, 7)** — ``x, y, z, RCS,
v_r, v_r_compensated, time`` (28 bytes per point), with an optional legacy
``file_size % 16`` byte prefix if needed (same decoding order as
``core.dataset.load_radar_points``). ``--pc-validation-quick`` skips the finite scan.

Sequenced layout (`--sequenced`):
  dest/<image_dir>/0000.png, 0001.png, … (same order as pairs; width auto, min 4 digits)
  dest/<pc_dir>/0000.bin, 0001.bin, … (contents of each matching `*_pc.bin`)

Defaults: image_dir=`image_2`, pc_dir=`pc_bin`. Override with `--sequenced-image-dir` /
`--sequenced-pc-dir`. In this mode only those subtrees are written (no flat copy).
"""
from __future__ import annotations

import argparse
import bisect
import shutil
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from pc_bin_validation import validate_pc_bin_dataset

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})


def iter_pairs(src: Path) -> list[tuple[Path, Path]]:
    """Return (main_bin, pc_bin) sorted by main bin name; only complete pairs."""
    if not src.is_dir():
        raise FileNotFoundError(src)

    mains: list[Path] = []
    for p in src.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".bin":
            continue
        name = p.name
        if name.endswith("_pc.bin"):
            continue
        pc = p.with_name(f"{p.stem}_pc.bin")
        if pc.is_file():
            mains.append(p)

    mains.sort(key=lambda x: x.name)
    pairs: list[tuple[Path, Path]] = []
    for main in mains:
        pc = main.with_name(f"{main.stem}_pc.bin")
        pairs.append((main, pc))
    return pairs


def _middle_underscore_int(filename: str) -> int | None:
    """Second `_`-separated field in stem, e.g. 0000000034_57031_000... -> 57031."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1], 10)
    except ValueError:
        return None


def list_images_sorted(src: Path) -> list[Path]:
    out: list[Path] = []
    for p in src.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    out.sort(key=lambda x: x.name)
    return out


def nearest_image_for_bin(bin_path: Path, images: list[Path]) -> Path | None:
    """
    Pick the image whose middle timestamp is closest to the main `.bin` stem.

    Candidates are the lexicographically adjacent image(s) around `bin_path.name`
    in the sorted image list (usually one before and/or one after the `.bin`).
    """
    if not images:
        return None
    names = [p.name for p in images]
    b = bin_path.name
    i = bisect.bisect_left(names, b)
    candidates: list[Path] = []
    if i > 0:
        candidates.append(images[i - 1])
    if i < len(images):
        candidates.append(images[i])
    if not candidates:
        return images[0]

    bin_mid = _middle_underscore_int(b)
    if bin_mid is None:
        return candidates[0]

    def dist(p: Path) -> int:
        m = _middle_underscore_int(p.name)
        if m is None:
            return 10**18
        return abs(m - bin_mid)

    return min(candidates, key=dist)


def index_field_width(n_pairs: int) -> int:
    """Zero-pad width for indices 0 .. n_pairs-1 (at least 4, e.g. 0000)."""
    if n_pairs <= 1:
        return 4
    return max(4, len(str(n_pairs - 1)))


def unique_dest_name(dest_dir: Path, preferred: str, used: set[str]) -> str:
    if preferred not in used:
        used.add(preferred)
        return preferred
    stem = Path(preferred).stem
    suf = Path(preferred).suffix
    alt = f"{stem}__dup{suf}"
    n = 2
    while alt in used or (dest_dir / alt).exists():
        alt = f"{stem}__dup{n}{suf}"
        n += 1
    used.add(alt)
    return alt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="Recording folder (flat files)")
    ap.add_argument(
        "dest",
        type=Path,
        help="Output folder (created if missing)",
    )
    ap.add_argument(
        "--no-pc",
        action="store_true",
        help="Do not copy *_pc.bin files (only main *.bin per pair)",
    )
    ap.add_argument(
        "--no-images",
        action="store_true",
        help="Do not copy camera images",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without copying",
    )
    ap.add_argument(
        "--sequenced",
        action="store_true",
        help="Write only image_2/ and pc_bin/ with matching indices (0000, …)",
    )
    ap.add_argument(
        "--sequenced-image-dir",
        default="image_2",
        help="Subfolder under dest for sequenced images (default: image_2)",
    )
    ap.add_argument(
        "--sequenced-pc-dir",
        default="pc_bin",
        help="Subfolder under dest for sequenced *_pc.bin as N.bin (default: pc_bin)",
    )
    ap.add_argument(
        "--no-pc-validation",
        action="store_true",
        help="Skip *_pc.bin structure checks (non-empty float32 N×3 or N×4)",
    )
    ap.add_argument(
        "--pc-validation-quick",
        action="store_true",
        help="PC validation without scanning floats for NaN/Inf (faster)",
    )
    args = ap.parse_args()

    if args.sequenced and (args.no_images or args.no_pc):
        print(
            "error: --sequenced needs both images and pc bins "
            "(omit --no-images and --no-pc)",
            file=sys.stderr,
        )
        return 2

    pairs = iter_pairs(args.source)
    images = [] if args.no_images else list_images_sorted(args.source)

    args.dest.mkdir(parents=True, exist_ok=True)

    print(f"Source: {args.source.resolve()}")
    print(f"Dest:   {args.dest.resolve()}")
    print(f"Pairs:  {len(pairs)} (sorted by main .bin name)")

    if (
        not args.no_pc
        and not args.no_pc_validation
        and pairs
    ):
        pc_paths = [pc for _, pc in pairs]
        ok_pc, pc_errors, pc_summary = validate_pc_bin_dataset(
            pc_paths,
            check_finite=not args.pc_validation_quick,
        )
        print(f"PC validation: {pc_summary}")
        if not ok_pc:
            for line in pc_errors:
                print(line, file=sys.stderr)
            print(
                "aborting (fix *_pc.bin files or pass --no-pc-validation)",
                file=sys.stderr,
            )
            return 1

    if args.sequenced:
        w = index_field_width(len(pairs))
        img_sub = Path(args.sequenced_image_dir)
        pc_sub = Path(args.sequenced_pc_dir)
        img_dir = args.dest / img_sub
        pc_dir = args.dest / pc_sub
        print("Layout: sequenced")
        print(f"  images -> {img_sub}/{{i:0{w}d}}<ext>")
        print(f"  pc     -> {pc_sub}/{{i:0{w}d}}.bin")
        if not args.dry_run:
            img_dir.mkdir(parents=True, exist_ok=True)
            pc_dir.mkdir(parents=True, exist_ok=True)

        for i, (main, pc) in enumerate(pairs):
            stem = f"{i:0{w}d}"
            img = nearest_image_for_bin(main, images)
            if img is None:
                print(f"  WARN no image for pair {i} ({main.name})", file=sys.stderr)
                continue
            dest_img = img_dir / f"{stem}{img.suffix.lower()}"
            dest_pc = pc_dir / f"{stem}.bin"
            if args.dry_run:
                print(f"  [{stem}] {img.name} -> {img_sub / dest_img.name}")
                print(f"  [{stem}] {pc.name} -> {pc_sub / dest_pc.name}")
            else:
                shutil.copy2(img, dest_img)
                shutil.copy2(pc, dest_pc)
                if i < 3 or i == len(pairs) - 1:
                    print(f"  [{stem}] img -> {dest_img.name}, pc -> {dest_pc.name}")
        if not args.dry_run and len(pairs) > 4:
            print(f"  ... done, {len(pairs)} pairs (log: first 3 indices and last)")
        return 0

    if args.no_pc:
        print("PC bins: skipped (--no-pc)")
    else:
        print("PC bins: copy *_pc.bin with each main *.bin")
    if args.no_images:
        print("Images: skipped (--no-images)")
    else:
        print(f"Images: {len(images)} in source; nearest per pair")

    used_names: set[str] = set()

    for main, pc in pairs:
        for label, src in (("main", main), ("pc", pc)):
            if label == "pc" and args.no_pc:
                continue
            t = args.dest / src.name
            if args.dry_run:
                print(f"  would copy {label}: {src.name}")
            else:
                shutil.copy2(src, t)
                used_names.add(src.name)
                print(f"  copied {label}: {src.name}")

        if args.no_images:
            continue

        img = nearest_image_for_bin(main, images)
        if img is None:
            print(f"    WARN no image for {main.name}", file=sys.stderr)
            continue

        dest_img_name = unique_dest_name(args.dest, img.name, used_names)
        t_img = args.dest / dest_img_name
        if dest_img_name != img.name:
            print(f"    (image renamed to avoid overwrite: {dest_img_name})")

        if args.dry_run:
            print(f"  would copy img: {img.name} -> {dest_img_name}")
        else:
            shutil.copy2(img, t_img)
            print(f"  copied img: {img.name} -> {dest_img_name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
