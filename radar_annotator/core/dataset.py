"""
Dataset manager.

Scans a dataset root, pairs radar files with image files by filename stem
(primary strategy from Section 5.2 of the spec), and exposes an ordered
frame list + navigation helpers.

Expected root layout (flexible; any of these subfolder names are accepted):
    <root>/
        radar/ or … / velodyne/ / pc_bin/ (KITTI/VOD-style radar bins)
        image/ or … / image_2/ (KITTI ``image_2`` camera folder)
        calib/ or calibration/            (optional)
        labels_internal/                  (optional, for loading existing)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np


RADAR_EXTS = {".bin", ".npy", ".pcd", ".npz", ".csv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# Per-frame calib: `<stem>.txt` (KITTI), `.json`, … must match radar/image stem.
_CALIB_STEM_SUFFIXES = (".txt", ".json", ".yaml", ".yml")


def resolve_stem_calibration_path(calib_dir: Optional[Path], stem: str) -> Optional[Path]:
    if calib_dir is None:
        return None
    base = Path(calib_dir)
    for suf in _CALIB_STEM_SUFFIXES:
        p = base / f"{stem}{suf}"
        if p.is_file():
            return p
    return None

RADAR_SUBDIRS = (
    "radar",
    "radar_pointcloud",
    "pointcloud",
    "points",
    "lidar",
    "velodyne",
    "pc_bin",
)
IMAGE_SUBDIRS = ("image", "images", "camera", "rgb", "cam", "image_2")


@dataclass
class Frame:
    index: int
    frame_id: str       # the filename stem; used as pairing key
    radar_path: Path
    image_path: Path
    calib_path: Optional[Path] = None  # calib/<stem>.txt etc., if present


class Dataset:
    """A frame-ordered, paired view of a radar + camera dataset folder."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.frames: List[Frame] = []
        self.radar_dir: Optional[Path] = None
        self.image_dir: Optional[Path] = None
        self.calib_dir: Optional[Path] = None
        self.labels_dir: Optional[Path] = None
        self._scan()

    # ------------------------------------------------------------------
    def _find_subdir(self, candidates) -> Optional[Path]:
        # Case-insensitive search, one level deep
        for child in self.root.iterdir():
            if child.is_dir() and child.name.lower() in candidates:
                return child
        return None

    def _scan(self) -> None:
        if not self.root.exists() or not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.radar_dir = self._find_subdir(RADAR_SUBDIRS)
        self.image_dir = self._find_subdir(IMAGE_SUBDIRS)
        self.calib_dir = self._find_subdir(("calib", "calibration"))
        self.labels_dir = self.root / "labels_internal"

        # Fallback: if there are no subfolders, look for files directly in root.
        if self.radar_dir is None:
            radar_files = [p for p in self.root.iterdir()
                           if p.suffix.lower() in RADAR_EXTS]
            if radar_files:
                self.radar_dir = self.root
        if self.image_dir is None:
            image_files = [p for p in self.root.iterdir()
                           if p.suffix.lower() in IMAGE_EXTS]
            if image_files:
                self.image_dir = self.root

        if self.radar_dir is None or self.image_dir is None:
            # Empty dataset is allowed; GUI will show "no frames"
            return

        radar_by_stem = {
            p.stem: p for p in sorted(self.radar_dir.iterdir())
            if p.is_file() and p.suffix.lower() in RADAR_EXTS
        }
        image_by_stem = {
            p.stem: p for p in sorted(self.image_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        }
        common = sorted(set(radar_by_stem) & set(image_by_stem))

        self.frames = [
            Frame(
                i,
                stem,
                radar_by_stem[stem],
                image_by_stem[stem],
                calib_path=resolve_stem_calibration_path(self.calib_dir, stem),
            )
            for i, stem in enumerate(common)
        ]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.frames)

    def frame(self, index: int) -> Frame:
        return self.frames[index]

    def find_by_id(self, frame_id: str) -> Optional[int]:
        for f in self.frames:
            if f.frame_id == frame_id:
                return f.index
        return None

    def all_frames_have_stem_calibration(self) -> bool:
        """True when every frame has ``calib/<frame_id>.txt`` (or .json/.yaml).

        Used to skip scanning thousands of unrelated ``*.txt`` files at dataset
        open when layout is per-frame KITTI-style calibration.
        """
        return len(self.frames) > 0 and all(
            f.calib_path is not None for f in self.frames
        )


# ----------------------------------------------------------------------
# Radar point-cloud loader
# ----------------------------------------------------------------------

def load_radar_points(path: Path) -> np.ndarray:
    """Load a radar point cloud as an (N, >=3) array of [x, y, z, ...].

    Supported formats:
        .npy  : numpy array with at least 3 columns
        .npz  : picks key 'points' or the first array
        .bin  : packed ``float32``. **View-of-Delft (VoD)** radar bins are **(N, 7)**:
                ``[x, y, z, RCS, v_r, v_r_compensated, time]`` (28 bytes per point).
                Some datasets append SNR as an 8th float. When the float count
                is not a multiple of 7 or 8, we strip a legacy ``size % 16``
                byte prefix and retry; otherwise fall back to ``(N, 4)`` or
                ``(N, 3)`` layouts.
        .csv  : comma-separated, header optional; first 3 cols are x,y,z
        .pcd  : ASCII PCD with x y z (and optional extra) fields
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        npz = np.load(path)
        key = "points" if "points" in npz.files else npz.files[0]
        arr = npz[key]
    elif suffix == ".bin":
        blob = path.read_bytes()
        raw_full = np.frombuffer(blob, dtype=np.float32)
        nf = raw_full.size
        if nf % 8 == 0 and _looks_like_8_float_radar(raw_full.reshape(-1, 8)):
            arr = raw_full.reshape(-1, 8)
        elif nf % 7 == 0:
            arr = raw_full.reshape(-1, 7)
        elif nf % 4 == 0:
            arr = raw_full.reshape(-1, 4)
        elif nf % 3 == 0:
            arr = raw_full.reshape(-1, 3)
        else:
            pfx = len(blob) % 16
            raw = np.frombuffer(memoryview(blob)[pfx:], dtype=np.float32)
            if raw.size % 8 == 0 and _looks_like_8_float_radar(raw.reshape(-1, 8)):
                arr = raw.reshape(-1, 8)
            elif raw.size % 7 == 0:
                arr = raw.reshape(-1, 7)
            elif raw.size % 4 == 0:
                arr = raw.reshape(-1, 4)
            elif raw.size % 3 == 0:
                arr = raw.reshape(-1, 3)
            else:
                raise ValueError(
                    f"Radar file {path.name}: cannot interpret binary ({len(blob)} bytes "
                    f"= {nf} float32 values before alignment strip; "
                    f"after {pfx}-byte strip, {raw.size} floats). "
                    "VoD-style radar expects length divisible by 28 "
                    "(7 floats/point). Check this path points at *_pc / VoD "
                    "radar bins, not another *.bin format."
                )
    elif suffix == ".csv":
        try:
            arr = np.loadtxt(path, delimiter=",", skiprows=1)
        except ValueError:
            arr = np.loadtxt(path, delimiter=",")
    elif suffix == ".pcd":
        arr = _load_pcd_ascii(path)
    else:
        raise ValueError(f"Unsupported radar file type: {suffix}")

    arr = np.atleast_2d(arr).astype(np.float64)
    if arr.shape[1] < 3:
        raise ValueError(
            f"Radar file {path.name} has only {arr.shape[1]} columns; "
            f"need at least x, y, z."
        )
    return arr


def _looks_like_8_float_radar(arr: np.ndarray) -> bool:
    """Avoid mistaking ordinary 4-float xyzi bins for 8-float xyzi...snr bins."""
    if arr.ndim != 2 or arr.shape[1] != 8 or arr.shape[0] == 0:
        return False
    xyz = arr[:, :3]
    if not np.all(np.isfinite(xyz)):
        return False
    # 8-float VoD-like data keeps time/SNR in the last columns. A 4-float file
    # reshaped as 8 floats puts another point's x/y/z there instead, often with
    # negative z or broad signed coordinates.
    time_col = arr[:, 6]
    snr_col = arr[:, 7]
    time_ok = np.all(np.isfinite(time_col)) and np.nanmin(time_col) >= -1e-3
    snr_ok = np.all(np.isfinite(snr_col)) and np.nanpercentile(snr_col, 95.0) > 0.0
    return bool(time_ok and snr_ok)


def _load_pcd_ascii(path: Path) -> np.ndarray:
    """Minimal ASCII PCD reader (x y z [extra...])."""
    header_done = False
    n_fields = 3
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not header_done:
                if line.startswith("FIELDS"):
                    n_fields = len(line.split()) - 1
                if line.startswith("DATA"):
                    header_done = True
                continue
            parts = line.split()
            if len(parts) >= 3:
                rows.append([float(p) for p in parts[:n_fields]])
    return np.array(rows, dtype=np.float64)
