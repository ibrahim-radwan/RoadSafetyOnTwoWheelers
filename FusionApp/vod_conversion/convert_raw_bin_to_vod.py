#!/usr/bin/env python
"""Convert recorded DCA1000 radar frames into VoD-style scan datasets.

By default this writes only ``data/{1,3,5}_scan`` packs. Each pack uses the
VoD camera folder ``image_2`` and the same zero-based five-digit sample id for
every paired file (``00000.png``, ``00000.bin``, ``00000.txt``, …).

Optional extras (``vod_pc``, previews, CSVs, range-Doppler) are off unless
explicitly requested.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import random
import shutil
import sys
import time
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# This package lives under FusionApp/vod_conversion; keep FusionApp imports working.
FUSIONAPP_ROOT = Path(__file__).resolve().parents[1]
if str(FUSIONAPP_ROOT) not in sys.path:
    sys.path.insert(0, str(FUSIONAPP_ROOT))

import numpy as np

from analysis.radar_heatmap_analyser import RadarHeatmapAnalyser
from camera.png_utils import is_valid_camera_frame, scan_png_directory
from radar.bin_utils import parse_bin_timestamp, scan_bin_directory
from recording.detections_csv import save_expected_detections_csv
from recording.sync_recording import RecordingManifest
from radar.dca1000_awr2243 import DCA1000Frame
from radar.point_cloud import write_vod_pc_bin
from render.encoders import heatmap_to_png
from sample_processing.ego_motion_compensation import (
    EgoMotionSequenceProcessor,
    align_accumulated_positions,
    cumulative_displacement_to_current,
    reconcile_accumulated_velocities,
)
from sample_processing.radar_params import ADCParams
from sample_processing.radar_tuning import default_radar_tuning
from utils import setup_logger


DEFAULT_CALIB_TEMPLATE = (
    FUSIONAPP_ROOT / "config_files" / "camera_radar_calib.txt"
)
DEFAULT_OUTDOOR_TUNING = (
    FUSIONAPP_ROOT / "config_files" / "outdoor_cfar_tuning.json"
)
SCAN_COUNTS = (1, 3, 5)
# VoD-compatible layout: camera images live in image_2; every paired modality
# shares the calib-style zero-based five-digit stem (00000, 00001, ...).
SCAN_DATA_FOLDERS = ("image_2", "radar", "radar_raw", "radarref", "calib")
SCAN_FOLDER_EXTENSIONS = {
    "image_2": ".png",
    "radar": ".bin",
    "radar_raw": ".bin",
    "radarref": ".csv",
    "calib": ".txt",
}
SCAN_MANIFEST_FIELDS = (
    "sequence",
    "sample_id",
    "scan_count",
    "image_source",
    "radar_source",
    "accumulated_radar_sources",
    "image_time",
    "camera_clock_offset_nanoseconds",
    "synchronized_image_time",
    "radar_time",
    "delta_nanoseconds",
    "point_count",
    "time_ids",
    "calib_source",
    "ego_vx_sources",
    "ego_vy_sources",
    "shift_fwd_sources",
    "shift_lat_sources",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process raw DCA1000 .bin frames with the FusionApp analyser and "
            "write VoD-style data/{1,3,5}_scan packs (image_2 + radar + calib "
            "sharing zero-based five-digit sample ids). Optional previews, "
            "vod_pc, and range-Doppler outputs are disabled by default."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A raw radar .bin frame or a recording directory containing raw frames.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Exact AWR2243 .txt profile used to record these frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output root. By default creates only data/{1,3,5}_scan under this "
            "directory. Extra artefact folders are created only when their "
            "matching --save-* flags are set. Defaults to the input directory."
        ),
    )
    parser.add_argument(
        "--feature",
        choices=("raw_power", "snr", "rcs"),
        default="raw_power",
        help="VoD column 4 feature. Default: raw peak power.",
    )
    parser.add_argument(
        "--display",
        choices=("snr", "raw_power"),
        default="snr",
        help="Preview opened by --show-preview. Both SNR and raw-power PNGs are saved.",
    )
    parser.add_argument(
        "--tuning-json",
        type=Path,
        help=(
            "Optional tuning JSON merged over the FusionApp defaults. "
            "Outdoor mode also loads config_files/outdoor_cfar_tuning.json "
            "unless this flag is set."
        ),
    )
    parser.add_argument(
        "--environment",
        choices=("indoor", "outdoor"),
        default="outdoor",
        help=(
            "CFAR detection profile for 2-TX processing. Default: outdoor "
            "(more sensitive, better for road/outdoor recordings)."
        ),
    )
    parser.add_argument(
        "--camera-clock-offset-ms",
        type=float,
        default=0.0,
        help=(
            "Optional camera timestamp offset added before matching to radar. "
            "Default: 0, matching organize_recording_data."
        ),
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help=(
            "Optional preview base path when processing one frame. Mode suffixes "
            "are added before .png. Directory conversion uses the standard "
            "pc2d_snr and pc2d_raw_power folders."
        ),
    )
    parser.add_argument(
        "--preview-frame",
        type=int,
        default=0,
        help=(
            "Zero-based frame whose saved image is opened by --show-preview. "
            "All processed frames still receive both PNGs. Default: 0."
        ),
    )
    parser.add_argument(
        "--save-previews",
        action="store_true",
        help="Also save per-frame SNR and raw-power PC-2D PNGs.",
    )
    parser.add_argument(
        "--save-detections-csv",
        action="store_true",
        help="Also save compact expected-detection CSVs for each frame.",
    )
    parser.add_argument(
        "--save-vod-pc",
        action="store_true",
        help="Also write per-frame vod_pc/*_pc.bin files (off by default).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "Numbered scan-dataset root. Defaults to <output-dir>/data and "
            "creates 1_scan, 3_scan, and 5_scan."
        ),
    )
    parser.add_argument(
        "--scan-counts",
        type=int,
        choices=SCAN_COUNTS,
        nargs="+",
        default=list(SCAN_COUNTS),
        help="Accumulation windows to export. Default: 1 3 5.",
    )
    parser.add_argument(
        "--calib-template",
        type=Path,
        default=DEFAULT_CALIB_TEMPLATE,
        help=(
            "Calibration text copied for every numbered sample. Default: "
            f"{DEFAULT_CALIB_TEMPLATE}"
        ),
    )
    parser.add_argument(
        "--no-scan-datasets",
        action="store_true",
        help="Do not create the numbered data/<N>_scan datasets.",
    )
    parser.add_argument(
        "--comparison-frame",
        type=int,
        help=(
            "Zero-based current radar frame for the 1/3/5 scan comparison "
            "grid. Default: first valid frame with five-scan history."
        ),
    )
    parser.add_argument(
        "--comparison-random-samples",
        type=int,
        default=0,
        help=(
            "Add this many deterministic, stratified-random comparison frames "
            "across the selected recording. Default: 0."
        ),
    )
    parser.add_argument(
        "--comparison-seed",
        type=int,
        default=0,
        help="Random seed used by --comparison-random-samples. Default: 0.",
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help=(
            "Generate comparison PNGs only. Processes each target frame and its "
            "four predecessors without rebuilding the full conversion."
        ),
    )
    parser.add_argument(
        "--save-scan-comparison",
        action="store_true",
        help="Also save the 1/3/5 accumulated point-cloud comparison grid.",
    )
    parser.add_argument(
        "--save-range-doppler",
        action="store_true",
        help="Also save range-Doppler matrices and RD/RGB grid PNGs.",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Open the preview selected by --display after conversion.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based first raw frame to process. Default: 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of raw frames to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing point-cloud, preview, and range-Doppler outputs.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first frame processing error.",
    )
    parser.add_argument(
        "--is-moving",
        action="store_true",
        help=(
            "Enable moving-platform ego-motion compensation. Updates "
            "velocities_comp (VoD column 5) for every point without removing "
            "detections. Raw radial velocity (doppler) is unchanged."
        ),
    )
    parser.add_argument(
        "--ego-alpha",
        type=float,
        default=0.25,
        help="EMA weight for ego-velocity smoothing when --is-moving. Default: 0.25.",
    )
    parser.add_argument(
        "--ego-ransac-tau",
        type=float,
        default=0.30,
        help="RANSAC inlier threshold in m/s when --is-moving. Default: 0.30.",
    )
    parser.add_argument(
        "--ego-static-tau",
        type=float,
        default=0.40,
        help=(
            "Static/moving diagnostic threshold in m/s when --is-moving. "
            "Default: 0.40."
        ),
    )
    parser.add_argument(
        "--ego-ransac-iterations",
        type=int,
        default=100,
        help="RANSAC iterations when --is-moving. Default: 100.",
    )
    parser.add_argument(
        "--ego-rcs-gate",
        action="store_true",
        help=(
            "Flag low-RCS points as uncertain in ego-motion diagnostics when "
            "--is-moving. Requires --feature rcs for meaningful gating."
        ),
    )
    parser.add_argument(
        "--ego-compensate-mode",
        choices=("raw", "smooth", "kalman"),
        default="kalman",
        help=(
            "Per-frame ego velocity used for velocities_comp when --is-moving. "
            "kalman (default, recommended): temporal Kalman filter on (vx, vy) "
            "with prediction-gated RANSAC for stable radar-only ego without an "
            "IMU. raw: each frame uses its own RANSAC estimate. smooth: "
            "EMA-smoothed sequence velocity."
        ),
    )
    parser.add_argument(
        "--ego-kalman-process-var",
        type=float,
        default=0.25,
        help=(
            "Kalman process variance per frame (m/s)^2 for ego velocity when "
            "--ego-compensate-mode kalman. Lower = smoother/stiffer. Default: 0.25."
        ),
    )
    parser.add_argument(
        "--ego-kalman-meas-var",
        type=float,
        default=0.40,
        help=(
            "Base Kalman measurement variance (m/s)^2 for the RANSAC ego "
            "estimate; inflated when inlier confidence is low. Default: 0.40."
        ),
    )
    parser.add_argument(
        "--ego-max-range",
        type=float,
        help=(
            "Maximum range (m) for ego-speed estimation points. Defaults to "
            "95%% of the loaded radar profile max range."
        ),
    )
    parser.add_argument(
        "--ego-min-range",
        type=float,
        default=1.0,
        help=(
            "Minimum range (m) for ego-speed RANSAC inliers when --is-moving. "
            "Default: 1.0."
        ),
    )
    parser.add_argument(
        "--no-ego-scan-reconcile",
        action="store_true",
        help=(
            "When --is-moving, do not re-reference stacked 1/3/5 scan "
            "velocities_comp to the current frame ego speed."
        ),
    )
    parser.add_argument(
        "--no-ego-scan-align",
        action="store_true",
        help=(
            "When --is-moving, do not translate stacked 1/3/5 scan x/y "
            "positions into the current frame using radar-only odometry."
        ),
    )
    return parser.parse_args(argv)


def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_tuning(
    path: Optional[Path],
    *,
    environment: str = "outdoor",
) -> Dict[str, Any]:
    tuning = default_radar_tuning()
    if environment == "outdoor":
        outdoor_path = DEFAULT_OUTDOOR_TUNING.expanduser().resolve()
        if not outdoor_path.is_file():
            raise FileNotFoundError(
                f"Outdoor tuning file not found: {outdoor_path}"
            )
        with outdoor_path.open("r", encoding="utf-8") as stream:
            outdoor_override = json.load(stream)
        if not isinstance(outdoor_override, dict):
            raise ValueError("Outdoor tuning JSON root must be an object")
        tuning = merge_dict(tuning, outdoor_override)
    if path is None:
        return tuning
    with path.open("r", encoding="utf-8") as stream:
        override = json.load(stream)
    if not isinstance(override, dict):
        raise ValueError("Tuning JSON root must be an object")
    return merge_dict(tuning, override)


def discover_raw_frames(source: Path) -> List[Path]:
    source = source.resolve()
    if source.is_file():
        if source.suffix.lower() != ".bin" or source.name.endswith("_pc.bin"):
            raise ValueError(f"Input is not a raw radar .bin frame: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {source}")
    return [Path(item[0]) for item in scan_bin_directory(str(source))]


def selected_frames(
    frames: Sequence[Path], start: int, limit: Optional[int]
) -> List[Path]:
    if start < 0:
        raise ValueError("--start must be zero or greater")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be greater than zero")
    stop = None if limit is None else start + limit
    return list(frames[start:stop])


def readable_camera_frames(
    camera_frames: Sequence[tuple[str, float, str]],
) -> List[tuple[str, float, str]]:
    """Keep the readable, nonempty frames used by organize_recording_data."""
    readable = []
    for frame in camera_frames:
        try:
            if Path(frame[0]).stat().st_size > 0:
                readable.append(frame)
        except OSError:
            continue
    return sorted(readable, key=lambda frame: (float(frame[1]), frame[2]))


def comparison_frame_indices(
    frame_count: int,
    requested_frame: Optional[int],
    random_samples: int,
    random_seed: int,
) -> List[int]:
    """Choose explicit and stratified-random frames with five-scan history."""
    if random_samples < 0:
        raise ValueError("--comparison-random-samples must be zero or greater")
    if frame_count < 5:
        return []

    targets = set()
    if requested_frame is not None:
        targets.add(requested_frame)
    elif random_samples == 0:
        targets.add(4)

    available = frame_count - 4
    sample_count = min(random_samples, available)
    if sample_count:
        rng = random.Random(random_seed)
        edges = np.linspace(4, frame_count, sample_count + 1, dtype=np.int64)
        for lower, upper in zip(edges[:-1], edges[1:]):
            start = int(lower)
            stop = max(start + 1, int(upper))
            targets.add(rng.randrange(start, min(stop, frame_count)))
    return sorted(targets)


def make_analyser(
    config_path: Path,
    tuning: Dict[str, Any],
    enable_range_azimuth: bool = False,
    *,
    is_indoor: bool = False,
) -> RadarHeatmapAnalyser:
    analyser = RadarHeatmapAnalyser(
        config_file=str(config_path),
        intensity_mode="snr",
        pc_bin_intensity_mode="raw_power",
        enable_tesseract=False,
        enable_zyx_cube=False,
        enable_range_azimuth=enable_range_azimuth,
        is_indoor=is_indoor,
    )
    analyser.logger = setup_logger("RawBinToVoD")
    analyser.adc_params = ADCParams(str(config_path))
    analyser.tuning = copy.deepcopy(tuning)
    return analyser


def expected_int16_values(adc_params: ADCParams) -> int:
    return int(
        adc_params.chirps
        * adc_params.rx
        * adc_params.tx
        * adc_params.samples
        * adc_params.IQ
    )


def read_raw_frame(path: Path, adc_params: ADCParams) -> DCA1000Frame:
    parsed = parse_bin_timestamp(path.name)
    timestamp = parsed[0] if parsed is not None else 0.0
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"raw frame cannot be read: {exc}") from exc
    if file_size == 0:
        raise ValueError("raw frame is empty")
    expected = expected_int16_values(adc_params)
    expected_bytes = expected * np.dtype(np.int16).itemsize
    if file_size != expected_bytes:
        raise ValueError(
            f"raw frame has {file_size} bytes, but config requires "
            f"{expected_bytes}; verify --config or recording integrity"
        )
    data = np.fromfile(path, dtype=np.int16)
    if data.size != expected:
        raise ValueError(
            f"raw frame has {data.size} int16 values, but config requires "
            f"{expected}; verify --config or recording integrity"
        )
    return DCA1000Frame(timestamp, data, filepath=str(path))


def _flat(pc: Dict[str, Any], key: str) -> np.ndarray:
    value = pc.get(key)
    if value is None:
        return np.array([], dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def save_vod_pc(pc: Dict[str, Any], output_path: Path, feature_mode: str) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return write_vod_pc_bin(pc, output_path, feature_mode)


def _take(values: np.ndarray, count: int, fill: float = np.nan) -> np.ndarray:
    if values.size >= count:
        return values[:count]
    return np.pad(values, (0, count - values.size), constant_values=fill)


def save_detections_csv(
    pc: Dict[str, Any],
    output_path: Path,
    *,
    camera_objects: Optional[Any] = None,
    max_radar_detections: int = 10,
    min_snr_db: float = 5.0,
    min_abs_velocity_mps: float = 0.25,
) -> int:
    """Save compact expected detections (top moving radar points, optional camera rows)."""
    return save_expected_detections_csv(
        pc,
        output_path,
        camera_objects=camera_objects,
        max_radar_detections=max_radar_detections,
        min_snr_db=min_snr_db,
        min_abs_velocity_mps=min_abs_velocity_mps,
    )


def accumulate_point_clouds(
    point_clouds: Sequence[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """Concatenate current-to-oldest scans and assign time IDs 0, -1, ..."""
    accumulated: Dict[str, List[np.ndarray]] = {
        "x": [],
        "y": [],
        "z": [],
        "doppler": [],
        "velocities_comp": [],
        "snr": [],
        "raw_intensity": [],
        "rcs": [],
        "time_ids": [],
    }
    for offset, pc in enumerate(reversed(point_clouds)):
        lateral = _flat(pc, "x")
        forward = _flat(pc, "y")
        count = int(min(lateral.size, forward.size))
        if count == 0:
            continue

        accumulated["x"].append(lateral[:count])
        accumulated["y"].append(forward[:count])
        accumulated["z"].append(_take(_flat(pc, "z"), count, 0.0))
        velocity = _take(_flat(pc, "doppler"), count)
        accumulated["doppler"].append(velocity)
        velocity_comp = _flat(pc, "velocities_comp")
        accumulated["velocities_comp"].append(
            velocity
            if velocity_comp.size == 0
            else _take(velocity_comp, count)
        )
        accumulated["snr"].append(_take(_flat(pc, "snr"), count))
        raw_power = _take(_flat(pc, "raw_intensity"), count)
        accumulated["raw_intensity"].append(raw_power)
        rcs = _flat(pc, "rcs")
        accumulated["rcs"].append(
            10.0 * np.log10(np.maximum(raw_power, 1e-6))
            if rcs.size == 0
            else _take(rcs, count)
        )
        accumulated["time_ids"].append(
            np.full(count, -offset, dtype=np.float32)
        )

    return {
        key: (
            np.concatenate(parts).astype(np.float32, copy=False)
            if parts
            else np.array([], dtype=np.float32)
        )
        for key, parts in accumulated.items()
    }


def preview_values(pc: Dict[str, Any], mode: str, n: int) -> np.ndarray:
    if mode == "snr":
        values = _flat(pc, "snr")
    elif mode == "rcs":
        values = _flat(pc, "rcs")
        if values.size < n:
            raw_power = _flat(pc, "raw_intensity")
            values = 10.0 * np.log10(np.maximum(raw_power, 1e-6))
    else:
        values = _flat(pc, "raw_intensity")
    if values.size < n:
        values = np.pad(values, (0, n - values.size), constant_values=np.nan)
    return values[:n]


def round_up_two_significant(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 10.0
    exponent = np.floor(np.log10(value))
    scale = 10.0 ** (exponent - 1.0)
    return float(np.ceil(value / scale) * scale)


def save_pc2d_preview(
    pc: Dict[str, Any],
    output_path: Path,
    display_mode: str,
    title: str,
    show: bool = False,
) -> None:
    if not show and "matplotlib.pyplot" not in sys.modules:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lateral = _flat(pc, "x")
    forward = _flat(pc, "y")
    n = int(min(lateral.size, forward.size))
    lateral = lateral[:n]
    forward = forward[:n]
    values = preview_values(pc, display_mode, n)
    finite_xy = np.isfinite(lateral) & np.isfinite(forward)
    max_range = round_up_two_significant(float(pc.get("max_range") or 10.0))

    dpi = 120
    figure, axis = plt.subplots(figsize=(640 / dpi, 640 / dpi), dpi=dpi)
    color_mask = finite_xy & np.isfinite(values)
    finite_values = values[np.isfinite(values)]
    if finite_values.size and np.any(color_mask):
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        if vmax <= vmin:
            vmax = vmin + 1.0
        bins = 6
        for bin_index in range(bins):
            lower = vmin + (bin_index / bins) * (vmax - vmin)
            upper = vmin + ((bin_index + 1) / bins) * (vmax - vmin)
            if bin_index == bins - 1:
                bin_mask = color_mask & (values >= lower) & (values <= upper)
            else:
                bin_mask = color_mask & (values >= lower) & (values < upper)
            if np.any(bin_mask):
                t = (bin_index + 0.5) / bins
                gray = int(round((1.0 - (0.86 * t + 0.08)) * 255.0))
                axis.scatter(
                    lateral[bin_mask],
                    forward[bin_mask],
                    color=(gray / 255.0, gray / 255.0, gray / 255.0),
                    s=3,
                    linewidths=0,
                )
    if np.any(finite_xy & ~np.isfinite(values)):
        fallback = finite_xy & ~np.isfinite(values)
        axis.scatter(
            lateral[fallback],
            forward[fallback],
            color=(160.0 / 255.0,) * 3,
            s=3,
            linewidths=0,
        )

    axis.set_xlim(-max_range, max_range)
    axis.set_ylim(0.0, max_range)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Lateral (m)")
    axis.set_ylabel("Forward (m)")
    axis.set_xticks(np.linspace(-max_range, max_range, 21))
    axis.set_yticks(np.linspace(0.0, max_range, 11))
    axis.grid(True, color="0.85", linewidth=0.6)
    for spine in axis.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.3)
    figure.subplots_adjust(
        left=48.0 / 640.0,
        right=1.0 - 24.0 / 640.0,
        top=1.0 - 20.0 / 640.0,
        bottom=44.0 / 640.0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    if show:
        plt.show()
    plt.close(figure)


def save_scan_comparison_grid(
    history: Sequence[Dict[str, Any]],
    output_path: Path,
    title: str,
    rgb_path: Optional[Path] = None,
) -> Dict[int, int]:
    """Render synchronized RGB plus aligned 1/3/5 scan clouds colored by SNR."""
    if "matplotlib.pyplot" not in sys.modules:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    scan_counts = (1, 3, 5)
    clouds = {
        count: accumulate_point_clouds(
            [entry["point_cloud"] for entry in history[-count:]]
        )
        for count in scan_counts
    }
    current_pc = history[-1]["point_cloud"]
    max_range = round_up_two_significant(
        float(current_pc.get("max_range") or 10.0)
    )
    color_min = 0.0
    color_center = 10.0
    color_max = 40.0
    color_norm = TwoSlopeNorm(
        vmin=color_min,
        vcenter=color_center,
        vmax=color_max,
    )

    figure = plt.figure(figsize=(15, 11), dpi=120, layout="constrained")
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1, 1, 0.045),
    )
    rgb_axis = figure.add_subplot(grid[0, 0])
    axes = [
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    ]
    colorbar_axis = figure.add_subplot(grid[:, 2])
    if rgb_path is not None and rgb_path.is_file():
        rgb_axis.imshow(plt.imread(rgb_path))
        rgb_axis.set_title(f"Corresponding RGB\n{rgb_path.name}")
    else:
        rgb_axis.text(
            0.5,
            0.5,
            "No synchronized RGB image",
            ha="center",
            va="center",
            transform=rgb_axis.transAxes,
        )
        rgb_axis.set_title("Corresponding RGB")
    rgb_axis.axis("off")

    point_counts: Dict[int, int] = {}
    scatter = None
    for axis, scan_count in zip(axes, scan_counts):
        cloud = clouds[scan_count]
        lateral = _flat(cloud, "x")
        forward = _flat(cloud, "y")
        snr = _take(
            _flat(cloud, "snr"),
            len(lateral),
        )
        count = int(min(lateral.size, forward.size))
        point_counts[scan_count] = count
        valid = (
            np.isfinite(lateral[:count])
            & np.isfinite(forward[:count])
            & np.isfinite(snr[:count])
        )
        if np.any(valid):
            scatter = axis.scatter(
                lateral[:count][valid],
                forward[:count][valid],
                c=np.clip(snr[:count][valid], color_min, color_max),
                cmap="turbo",
                norm=color_norm,
                s=4,
                linewidths=0,
            )
        axis.set_title(f"{scan_count} scan ({count} points)")
        axis.set_xlim(-max_range, max_range)
        axis.set_ylim(0.0, max_range)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Lateral (m)")
        axis.set_ylabel("Forward (m)")
        axis.grid(True, color="0.85", linewidth=0.6)

    if scatter is not None:
        colorbar = figure.colorbar(
            scatter,
            cax=colorbar_axis,
            extend="max",
        )
        colorbar.set_label(
            "SNR above local noise floor (dB)\n"
            "0-10 dB expanded; values above 40 dB clipped"
        )
    else:
        colorbar_axis.axis("off")
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    return point_counts


def range_doppler_axes(
    matrix: np.ndarray,
    range_resolution: float,
    doppler_resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    doppler_mps = (
        np.arange(matrix.shape[0], dtype=np.float32)
        - np.float32(matrix.shape[0] / 2.0)
    ) * np.float32(doppler_resolution)
    range_m = (
        np.arange(matrix.shape[1], dtype=np.float32)
        * np.float32(range_resolution)
    )
    return doppler_mps, range_m


def save_range_doppler_data(
    matrix: np.ndarray,
    output_path: Path,
    range_resolution: float,
    doppler_resolution: float,
) -> None:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("range-Doppler output is missing or is not a 2D matrix")
    doppler_mps, range_m = range_doppler_axes(
        values,
        range_resolution,
        doppler_resolution,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        range_doppler=values,
        doppler_mps=doppler_mps,
        range_m=range_m,
    )


def save_range_doppler_grid(
    matrix: np.ndarray,
    output_path: Path,
    max_range: float,
    max_speed: float,
    title: str,
    rgb_path: Optional[Path] = None,
    rgb_age_s: Optional[float] = None,
) -> None:
    if "matplotlib.pyplot" not in sys.modules:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("range-Doppler output is missing or is not a 2D matrix")

    # This is exactly the transform used by FusionRunner.get_latest_rd_png().
    app_rd = np.rot90(values, 1)
    app_png = heatmap_to_png(
        app_rd,
        extents=(-float(max_speed), float(max_speed), 0.0, float(max_range)),
        force_square=True,
        target_size=(480, 480),
    )
    if app_png is None:
        raise ValueError("FusionApp range-Doppler renderer returned no image")
    rd_image = plt.imread(io.BytesIO(app_png), format="png")

    figure, (heatmap_axis, rgb_axis) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        dpi=120,
        gridspec_kw={"width_ratios": (1.05, 1.0)},
    )
    heatmap_axis.imshow(rd_image, interpolation="nearest")
    heatmap_axis.set_title("Range-Doppler heatmap (FusionApp rendering)")
    heatmap_axis.axis("off")

    if rgb_path is not None and rgb_path.is_file():
        rgb_image = plt.imread(rgb_path)
        rgb_axis.imshow(rgb_image)
        rgb_title = f"RGB image\n{rgb_path.name}"
        if rgb_age_s is not None:
            rgb_title += f" ({rgb_age_s * 1000.0:.1f} ms before radar)"
        rgb_axis.set_title(rgb_title)
    else:
        rgb_axis.text(
            0.5,
            0.5,
            "No synchronized RGB image",
            ha="center",
            va="center",
            transform=rgb_axis.transAxes,
        )
        rgb_axis.set_title("RGB image")
    rgb_axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def output_directories(output_root: Path) -> Dict[str, Path]:
    return {
        "vod_pc": output_root / "vod_pc",
        "detections_csv": output_root / "detections_csv",
        "snr": output_root / "pc2d_snr",
        "raw_power": output_root / "pc2d_raw_power",
        "range_doppler_data": output_root / "range_doppler_data",
        "range_doppler_grid": output_root / "range_doppler_grid",
    }


def scan_dataset_directories(
    data_root: Path, scan_count: int
) -> Dict[str, Path]:
    scan_root = data_root / f"{scan_count}_scan"
    return {
        "root": scan_root,
        **{name: scan_root / name for name in SCAN_DATA_FOLDERS},
    }


def directory_has_files(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def sample_id(sequence: int) -> str:
    """Return the zero-based five-digit VoD sample id shared by all modalities.

    Sequence is 1-based in the manifest (first sample = 1). Files on disk use
    the calib-style stem: sample 1 → ``00000``, sample 2 → ``00001``, …
    """
    if sequence <= 0:
        raise ValueError("Sample sequence must be greater than zero")
    return f"{sequence - 1:05d}"


def sample_filename(sequence: int, extension: str) -> str:
    """Build ``{sample_id}{extension}`` for any scan-dataset modality."""
    suffix = extension if extension.startswith(".") else f".{extension}"
    return f"{sample_id(sequence)}{suffix}"


def calibration_filename(sequence: int) -> str:
    """Return the zero-based five-digit calibration name used by VoD data."""
    return sample_filename(sequence, ".txt")


def scan_dataset_is_complete(
    layout: Dict[str, Path],
    scan_count: int,
) -> bool:
    manifest = layout["root"] / "manifest.csv"
    if not manifest.is_file():
        return False
    try:
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return False
    for expected_sequence, row in enumerate(rows, start=1):
        try:
            if (
                int(row["sequence"]) != expected_sequence
                or int(row["scan_count"]) != scan_count
            ):
                return False
            expected_id = sample_id(expected_sequence)
            if row.get("sample_id", expected_id) != expected_id:
                return False
            image_time_ns = int(row["image_time"])
            camera_clock_offset_ns = int(
                row["camera_clock_offset_nanoseconds"]
            )
            synchronized_image_time_ns = int(
                row["synchronized_image_time"]
            )
            radar_time_ns = int(row["radar_time"])
            if (
                synchronized_image_time_ns
                != image_time_ns + camera_clock_offset_ns
                or int(row["delta_nanoseconds"])
                != radar_time_ns - synchronized_image_time_ns
            ):
                return False
        except (KeyError, TypeError, ValueError):
            return False
        for folder, extension in SCAN_FOLDER_EXTENSIONS.items():
            path = layout[folder] / sample_filename(expected_sequence, extension)
            if not path.is_file() or path.stat().st_size == 0:
                return False
        calib_name = calibration_filename(expected_sequence)
        if row.get("calib_source") != calib_name:
            return False
    return bool(rows)


def reset_managed_scan_dataset(layout: Dict[str, Path]) -> None:
    """Remove generated scan data while preserving user label folders."""
    for folder in SCAN_DATA_FOLDERS:
        path = layout[folder]
        if not path.exists():
            continue
        if path.is_dir():
            trash = path.with_name(f"{path.name}.trash_{os.getpid()}_{time.time_ns()}")
            suffix = 0
            while trash.exists():
                suffix += 1
                trash = path.with_name(
                    f"{path.name}.trash_{os.getpid()}_{suffix}"
                )
            try:
                path.rename(trash)
            except OSError:
                shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    (layout["root"] / "manifest.csv").unlink(missing_ok=True)


def reset_scan_radar_bins(layout: Dict[str, Path]) -> None:
    """Clear only compensated radar bins and manifest for a fast in-place rebuild."""
    radar_dir = layout["radar"]
    if radar_dir.is_dir():
        for bin_file in radar_dir.glob("*.bin"):
            bin_file.unlink(missing_ok=True)
    (layout["root"] / "manifest.csv").unlink(missing_ok=True)


def prepare_scan_datasets(
    data_root: Path,
    scan_counts: Sequence[int],
    overwrite: bool,
    *,
    radar_bin_only: bool = False,
) -> tuple[Dict[int, Dict[str, Path]], List[int]]:
    layouts: Dict[int, Dict[str, Path]] = {}
    build_counts: List[int] = []
    for scan_count in scan_counts:
        layout = scan_dataset_directories(data_root, scan_count)
        layouts[scan_count] = layout
        manifest = layout["root"] / "manifest.csv"
        if manifest.is_file() and not overwrite:
            if scan_dataset_is_complete(layout, scan_count):
                continue
            raise FileExistsError(
                f"Scan dataset manifest exists but required synchronized files "
                f"are missing or invalid: {layout['root']}. Use --overwrite "
                "to rebuild its managed folders."
            )
        if overwrite and radar_bin_only:
            reset_scan_radar_bins(layout)
        elif directory_has_files(layout["root"]):
            if not overwrite:
                raise FileExistsError(
                    f"Scan dataset is partially populated: {layout['root']}. "
                    "Use --overwrite to rebuild it."
                )
            reset_managed_scan_dataset(layout)
        for folder in SCAN_DATA_FOLDERS:
            layout[folder].mkdir(parents=True, exist_ok=True)
        build_counts.append(scan_count)
    return layouts, build_counts


def copy_atomic(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Copy source not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_vod_pc_atomic(
    pc: Dict[str, Any],
    destination: Path,
    feature_mode: str,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        count = save_vod_pc(pc, temporary, feature_mode)
        os.replace(temporary, destination)
        return count
    finally:
        temporary.unlink(missing_ok=True)


def save_detections_csv_atomic(
    pc: Dict[str, Any],
    destination: Path,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        count = save_detections_csv(pc, temporary)
        if not temporary.is_file():
            raise FileNotFoundError(
                f"Detection CSV was not written: {temporary}"
            )
        os.replace(temporary, destination)
        return count
    finally:
        temporary.unlink(missing_ok=True)


def filename_time_ns(path: Path) -> int:
    parts = path.name.split("_", 2)
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(f"Cannot read timestamp from filename: {path.name}")
    fraction_ns = int((parts[1] + "000000000")[:9])
    return int(parts[0]) * 1_000_000_000 + fraction_ns


def save_scan_dataset_sample(
    layout: Dict[str, Path],
    sequence: int,
    scan_count: int,
    history: Sequence[Dict[str, Any]],
    image_path: Path,
    calib_template: Path,
    feature_mode: str,
    camera_clock_offset_s: float,
    *,
    reconcile_scan_velocities: bool = False,
    align_scan_positions: bool = False,
    frame_period_s: float = 0.05,
    radar_bin_only: bool = False,
) -> Dict[str, Any]:
    current = history[-1]
    selected = list(history[-scan_count:])
    accumulated_pc = accumulate_point_clouds(
        [entry["point_cloud"] for entry in selected]
    )
    ego_vx_sources = [
        float(entry.get("ego_vx", 0.0)) for entry in selected
    ]
    ego_vy_sources = [
        float(entry.get("ego_vy", 0.0)) for entry in selected
    ]
    timestamps_ns = [filename_time_ns(entry["raw_path"]) for entry in selected]
    cum_fwd, cum_lat = cumulative_displacement_to_current(
        timestamps_ns,
        ego_vx_sources,
        ego_vy_sources,
        frame_period_s=frame_period_s,
    )
    if align_scan_positions and scan_count > 1:
        accumulated_pc = align_accumulated_positions(
            accumulated_pc,
            cum_fwd_to_current=cum_fwd,
            cum_lat_to_current=cum_lat,
        )
    if reconcile_scan_velocities and scan_count > 1:
        accumulated_pc = reconcile_accumulated_velocities(
            accumulated_pc,
            current_vx=ego_vx_sources[-1],
            current_vy=ego_vy_sources[-1],
            source_vx=ego_vx_sources,
            source_vy=ego_vy_sources,
        )
    point_count = int(_flat(accumulated_pc, "x").size)
    stem = sample_id(sequence)
    calib_name = calibration_filename(sequence)

    if not radar_bin_only:
        copy_atomic(image_path, layout["image_2"] / sample_filename(sequence, ".png"))
        copy_atomic(
            current["raw_path"],
            layout["radar_raw"] / sample_filename(sequence, ".bin"),
        )
    save_vod_pc_atomic(
        accumulated_pc,
        layout["radar"] / sample_filename(sequence, ".bin"),
        feature_mode,
    )
    if not radar_bin_only:
        save_detections_csv_atomic(
            accumulated_pc,
            layout["radarref"] / sample_filename(sequence, ".csv"),
        )
        copy_atomic(calib_template, layout["calib"] / calib_name)

    radar_time_ns = filename_time_ns(current["raw_path"])
    image_time_ns = filename_time_ns(image_path)
    camera_clock_offset_ns = int(round(camera_clock_offset_s * 1_000_000_000))
    synchronized_image_time_ns = image_time_ns + camera_clock_offset_ns
    current_to_oldest = list(reversed(selected))
    return {
        "sequence": sequence,
        "sample_id": stem,
        "scan_count": scan_count,
        "image_source": image_path.name,
        "radar_source": current["raw_path"].name,
        "accumulated_radar_sources": ";".join(
            entry["raw_path"].name for entry in current_to_oldest
        ),
        "image_time": image_time_ns,
        "camera_clock_offset_nanoseconds": camera_clock_offset_ns,
        "synchronized_image_time": synchronized_image_time_ns,
        "radar_time": radar_time_ns,
        "delta_nanoseconds": radar_time_ns - synchronized_image_time_ns,
        "point_count": point_count,
        "time_ids": ";".join(str(-index) for index in range(scan_count)),
        "calib_source": calib_name,
        "ego_vx_sources": ";".join(f"{value:.6f}" for value in ego_vx_sources),
        "ego_vy_sources": ";".join(f"{value:.6f}" for value in ego_vy_sources),
        "shift_fwd_sources": ";".join(f"{value:.6f}" for value in cum_fwd),
        "shift_lat_sources": ";".join(f"{value:.6f}" for value in cum_lat),
    }


def write_scan_manifest(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    temporary = output_path.with_name(output_path.name + ".partial")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SCAN_MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def write_ego_motion_diagnostics(
    rows: Sequence[Dict[str, Any]], output_path: Path
) -> None:
    if not rows:
        return
    fieldnames = [
        "frame",
        "n_total",
        "vx_raw",
        "vy_raw",
        "vx_apply",
        "vy_apply",
        "vx_smooth",
        "vy_smooth",
        "vx_kalman",
        "vy_kalman",
        "inlier_fraction",
        "ego_fit_inliers",
        "ego_fit_coverage",
        "ego_fit_score",
        "speed_apply",
        "speed_smooth",
        "n_static",
        "n_moving",
        "n_uncertain",
        "static_fraction",
        "use_vr_comp",
        "compensate_mode",
    ]
    temporary = output_path.with_name(output_path.name + ".partial")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def output_path_for(raw_path: Path, vod_dir: Path) -> Path:
    return vod_dir / f"{raw_path.stem}_pc.bin"


def detections_csv_path_for(raw_path: Path, csv_dir: Path) -> Path:
    return csv_dir / f"{raw_path.stem}_detections.csv"


def preview_paths(
    requested_path: Optional[Path],
    directories: Dict[str, Path],
    raw_path: Path,
) -> Dict[str, Path]:
    if requested_path is None:
        return {
            "snr": directories["snr"] / f"{raw_path.stem}_pc2d_snr.png",
            "raw_power": (
                directories["raw_power"] / f"{raw_path.stem}_pc2d_raw_power.png"
            ),
        }
    else:
        requested = requested_path.resolve()
        suffix = requested.suffix or ".png"
        base = requested.with_suffix("")
        return {
            "snr": base.with_name(f"{base.name}_snr{suffix}"),
            "raw_power": base.with_name(f"{base.name}_raw_power{suffix}"),
        }


def range_doppler_paths(
    directories: Dict[str, Path],
    raw_path: Path,
) -> Dict[str, Path]:
    return {
        "data": (
            directories["range_doppler_data"]
            / f"{raw_path.stem}_range_doppler.npz"
        ),
        "grid": (
            directories["range_doppler_grid"]
            / f"{raw_path.stem}_range_doppler_grid.png"
        ),
    }


def find_camera_file_by_pair_seq(
    recording_dir: Path, pair_seq: int
) -> Optional[Path]:
    suffix = f"_{int(pair_seq):012d}.png"
    for path in sorted(recording_dir.glob("*.png")):
        if path.name.endswith(suffix) and is_valid_camera_frame(path.name):
            return path
    return None


_MANIFEST_PAIRS_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def _cached_manifest_pairs(recording_dir: Path) -> List[Dict[str, Any]]:
    key = str(recording_dir.resolve())
    cached = _MANIFEST_PAIRS_CACHE.get(key)
    if cached is None:
        cached = list(RecordingManifest.load_pairs(key))
        _MANIFEST_PAIRS_CACHE[key] = cached
    return cached


def manifest_delta_ns_for_pair(
    recording_dir: Path, pair_seq: int
) -> Optional[int]:
    for entry in _cached_manifest_pairs(recording_dir):
        if int(entry.get("pair_seq", -1)) != int(pair_seq):
            continue
        if "delta_ns" in entry:
            return int(entry["delta_ns"])
        radar_ns = entry.get("radar_capture_mono_ns")
        camera_ns = entry.get("camera_capture_mono_ns")
        if radar_ns is not None and camera_ns is not None:
            return int(radar_ns) - int(camera_ns)
    return None


def manifest_camera_file_for_pair(
    recording_dir: Path, pair_seq: int
) -> Optional[Path]:
    for entry in _cached_manifest_pairs(recording_dir):
        if int(entry.get("pair_seq", -1)) != int(pair_seq):
            continue
        camera_file = entry.get("camera_file")
        if camera_file:
            path = recording_dir / str(camera_file)
            if path.is_file():
                return path
    return None


def matching_rgb_frame(
    camera_frames: Sequence[tuple[str, float, str]],
    camera_timestamps: Sequence[float],
    raw_path: Path,
    max_age_s: Optional[float] = None,
    allow_future_fallback: bool = False,
    camera_clock_offset_s: float = 0.0,
    recording_dir: Optional[Path] = None,
) -> tuple[Optional[Path], Optional[float]]:
    parsed = parse_bin_timestamp(raw_path.name)
    if parsed is None:
        return None, None

    radar_timestamp, pair_seq = parsed
    if recording_dir is not None:
        paired_path = manifest_camera_file_for_pair(recording_dir, pair_seq)
        if paired_path is None:
            paired_path = find_camera_file_by_pair_seq(recording_dir, pair_seq)
        if paired_path is not None:
            delta_ns = manifest_delta_ns_for_pair(recording_dir, pair_seq)
            if delta_ns is not None:
                return paired_path, float(delta_ns) / 1_000_000_000.0
            return paired_path, 0.0

    if not camera_frames:
        return None, None
    target_camera_timestamp = radar_timestamp - camera_clock_offset_s
    next_index = bisect_right(camera_timestamps, target_camera_timestamp)
    previous_index = next_index - 1
    if previous_index >= 0:
        previous = camera_frames[previous_index]
        previous_age_s = radar_timestamp - (
            float(previous[1]) + camera_clock_offset_s
        )
        if max_age_s is None or previous_age_s <= max_age_s:
            return Path(previous[0]), previous_age_s
    if allow_future_fallback and next_index < len(camera_frames):
        following = camera_frames[next_index]
        following_age_s = radar_timestamp - (
            float(following[1]) + camera_clock_offset_s
        )
        if max_age_s is None or abs(following_age_s) <= max_age_s:
            return Path(following[0]), following_age_s
    return None, None


def convert(args: argparse.Namespace) -> int:
    source = args.input.resolve()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Radar config not found: {config_path}")

    all_frames = discover_raw_frames(source)
    frames = selected_frames(all_frames, args.start, args.limit)
    if not frames:
        raise ValueError("No raw radar frames selected")
    if args.preview and len(frames) != 1:
        raise ValueError("--preview can only be used when exactly one frame is selected")
    if args.comparison_frame is not None and (
        args.comparison_frame < 4 or args.comparison_frame >= len(frames)
    ):
        raise ValueError(
            f"--comparison-frame must be between 4 and {len(frames) - 1}"
        )
    comparison_targets = comparison_frame_indices(
        len(frames),
        args.comparison_frame,
        args.comparison_random_samples,
        args.comparison_seed,
    )
    if args.comparison_only:
        # Comparison-only always writes the 1/3/5 grids.
        args.save_scan_comparison = True

    save_vod_pc_files = bool(args.save_vod_pc)
    if (
        not args.comparison_only
        and args.no_scan_datasets
        and not save_vod_pc_files
        and not args.save_previews
        and not args.save_detections_csv
        and not args.save_range_doppler
    ):
        raise ValueError(
            "Nothing to write: keep scan datasets enabled (default) or pass "
            "--save-vod-pc / --save-previews / --save-detections-csv / "
            "--save-range-doppler / --save-scan-comparison"
        )

    default_output = source if source.is_dir() else source.parent
    output_root = (args.output_dir or default_output).resolve()
    directories = output_directories(output_root)
    extra_dirs: List[Path] = []
    if save_vod_pc_files:
        extra_dirs.append(directories["vod_pc"])
    if args.save_detections_csv:
        extra_dirs.append(directories["detections_csv"])
    if args.save_previews:
        extra_dirs.extend((directories["snr"], directories["raw_power"]))
    if args.save_range_doppler:
        extra_dirs.extend(
            (directories["range_doppler_data"], directories["range_doppler_grid"])
        )
    for directory in extra_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    recording_dir = source if source.is_dir() else source.parent
    camera_frames, _ = scan_png_directory(str(recording_dir))
    camera_frames = readable_camera_frames(camera_frames)
    camera_timestamps = [float(frame[1]) for frame in camera_frames]
    camera_clock_offset_s = float(args.camera_clock_offset_ms) / 1000.0
    scan_layouts: Dict[int, Dict[str, Path]] = {}
    scan_build_counts: List[int] = []
    scan_manifest_rows: Dict[int, List[Dict[str, Any]]] = {}
    scan_sequences: Dict[int, int] = {}
    calib_template: Optional[Path] = None
    data_root: Optional[Path] = None
    comparison_enabled = (
        args.save_scan_comparison
        and len(frames) >= 5
        and bool(comparison_targets)
        and (not args.no_scan_datasets or args.comparison_only)
    )
    comparison_dir: Optional[Path] = None
    if not args.no_scan_datasets and not args.comparison_only:
        calib_template = args.calib_template.expanduser().resolve()
        if not calib_template.is_file():
            raise FileNotFoundError(
                f"Calibration template not found: {calib_template}"
            )
        requested_scan_counts = sorted(set(args.scan_counts))
        available_scan_counts = [
            count for count in requested_scan_counts if count <= len(frames)
        ]
        data_root = (args.data_dir or (output_root / "data")).resolve()
        scan_layouts, scan_build_counts = prepare_scan_datasets(
            data_root,
            available_scan_counts,
            args.overwrite,
        )
        scan_manifest_rows = {count: [] for count in scan_build_counts}
        scan_sequences = {count: 0 for count in scan_build_counts}
    if comparison_enabled:
        data_root = (args.data_dir or (output_root / "data")).resolve()
        comparison_dir = data_root / "scan_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_pending_targets = {
        index
        for index in comparison_targets
        if comparison_dir is not None
        and (
            args.overwrite
            or not (
                comparison_dir
                / f"{frames[index].stem}_1_3_5_scan_comparison.png"
            ).exists()
        )
    }
    comparison_required_indices = {
        history_index
        for target_index in comparison_pending_targets
        for history_index in range(target_index - 4, target_index + 1)
    }
    if args.save_previews and (
        args.preview_frame < 0 or args.preview_frame >= len(frames)
    ):
        raise ValueError(
            f"--preview-frame must be between 0 and {len(frames) - 1}"
        )

    tuning = load_tuning(
        args.tuning_json.resolve() if args.tuning_json else None,
        environment=args.environment,
    )
    analyser = make_analyser(
        config_path,
        tuning,
        enable_range_azimuth=False,
        is_indoor=args.environment == "indoor",
    )
    print(
        f"Config: {config_path.name} "
        f"({analyser.adc_params.tx} TX, max range "
        f"{analyser.adc_params.max_range:.2f} m)"
    )
    print(f"CFAR environment: {args.environment}")
    if args.environment == "outdoor":
        print(f"Outdoor tuning: {DEFAULT_OUTDOOR_TUNING}")
    print(f"Frames selected: {len(frames)} of {len(all_frames)}")
    print(
        "Camera timestamp correction: "
        f"{camera_clock_offset_s * 1000.0:.3f} ms "
        "(latest RGB at or before radar, as in organize_recording_data)"
    )
    if not args.comparison_only and save_vod_pc_files:
        print(f"VoD output: {directories['vod_pc']}")
    if data_root is not None:
        if scan_build_counts and not args.comparison_only:
            print(
                "Scan datasets (image_2 + shared sample ids): "
                + ", ".join(
                    str(scan_layouts[count]["root"])
                    for count in scan_build_counts
                )
            )
        elif not args.comparison_only:
            print(f"Scan datasets already complete: {data_root}")
        if comparison_dir is not None:
            print(f"Scan comparison grid: {comparison_dir}")
    if args.save_detections_csv and not args.comparison_only:
        print(
            "Detection CSVs (top 10 moving radar points): "
            f"{directories['detections_csv']}"
        )
    if args.save_previews and not args.comparison_only:
        print(f"SNR previews: {directories['snr']}")
        print(f"Raw-power previews: {directories['raw_power']}")
    if args.save_range_doppler and not args.comparison_only:
        print(f"Range-Doppler data: {directories['range_doppler_data']}")
        print(
            f"Range-Doppler/RGB grids: {directories['range_doppler_grid']} "
            f"({len(camera_frames)} camera frames available)"
        )
    if comparison_enabled:
        print(
            "Comparison target frames: "
            + ", ".join(str(index) for index in comparison_targets)
        )
    ego_processor: Optional[EgoMotionSequenceProcessor] = None
    ego_max_range = float(
        args.ego_max_range
        if args.ego_max_range is not None
        else analyser.adc_params.max_range * 0.95
    )
    ego_min_range = float(args.ego_min_range)
    if args.is_moving:
        ego_processor = EgoMotionSequenceProcessor(
            alpha=float(args.ego_alpha),
            static_tau=float(args.ego_static_tau),
            ransac_tau=float(args.ego_ransac_tau),
            ransac_iterations=int(args.ego_ransac_iterations),
            min_range=ego_min_range,
            max_range=ego_max_range,
            compensate_mode=args.ego_compensate_mode,
            kalman_process_var=float(args.ego_kalman_process_var),
            kalman_meas_var=float(args.ego_kalman_meas_var),
            use_rcs_gate=bool(args.ego_rcs_gate),
        )
        print(
            "Ego-motion compensation: enabled "
            f"(mode={args.ego_compensate_mode}, alpha={args.ego_alpha}, "
            f"ransac_tau={args.ego_ransac_tau}, static_tau={args.ego_static_tau}, "
            f"min_range={ego_min_range:.1f} m, "
            f"max_range={ego_max_range:.1f} m)"
        )
        print(
            "  Updates velocities_comp only; all points are kept. "
            "Raw doppler is unchanged."
        )
        if not args.no_ego_scan_reconcile:
            print(
                "  1/3/5 scan stacks re-reference historical velocities_comp "
                "to the current-frame ego speed."
            )
        if not args.no_ego_scan_align:
            print(
                "  1/3/5 scan stacks translate historical x/y using radar-only "
                f"odometry (frame period {analyser.adc_params.frame_periodicity / 1000.0:.3f} s)."
            )

    converted = 0
    skipped = 0
    failed = 0
    total_points = 0
    detection_csv_files = 0
    detection_csv_rows = 0
    snr_images = 0
    raw_power_images = 0
    range_doppler_files = 0
    range_doppler_images = 0
    scan_history: List[Dict[str, Any]] = []
    max_scan_count = max(
        max(scan_build_counts, default=0),
        5 if comparison_enabled else 0,
    )
    scan_samples = {count: 0 for count in scan_build_counts}
    scan_samples_without_rgb = 0
    comparison_saved_indices = set()
    comparison_point_counts: Dict[int, Dict[int, int]] = {}
    comparison_paths: List[Path] = []
    frame_period_s = float(analyser.adc_params.frame_periodicity) / 1000.0
    started = time.perf_counter()

    for index, raw_path in enumerate(frames):
        output_path = output_path_for(raw_path, directories["vod_pc"])
        detections_csv_path = detections_csv_path_for(
            raw_path, directories["detections_csv"]
        )
        paths = preview_paths(args.preview, directories, raw_path)
        rd_paths = range_doppler_paths(directories, raw_path)
        need_vod = (
            not args.comparison_only
            and save_vod_pc_files
            and (args.overwrite or not output_path.exists())
        )
        need_detections_csv = (
            not args.comparison_only
            and args.save_detections_csv
            and (args.overwrite or not detections_csv_path.exists())
        )
        need_snr = (
            not args.comparison_only
            and args.save_previews
            and (args.overwrite or not paths["snr"].exists())
        )
        need_raw_power = (
            not args.comparison_only
            and args.save_previews
            and (args.overwrite or not paths["raw_power"].exists())
        )
        need_rd_data = (
            not args.comparison_only
            and args.save_range_doppler
            and (args.overwrite or not rd_paths["data"].exists())
        )
        need_rd_grid = (
            not args.comparison_only
            and args.save_range_doppler
            and (args.overwrite or not rd_paths["grid"].exists())
        )
        need_scan_processing = bool(scan_build_counts) or (
            index in comparison_required_indices
        )
        if not (
            need_vod
            or need_detections_csv
            or need_snr
            or need_raw_power
            or need_rd_data
            or need_rd_grid
            or need_scan_processing
        ):
            skipped += 1
            continue
        try:
            result = analyser._analyse_frame(
                read_raw_frame(raw_path, analyser.adc_params)
            )
            pc = result.get("point_cloud") or {}
            ego_info: Optional[Dict[str, Any]] = None
            if ego_processor is not None:
                ego_info = ego_processor.apply(pc, frame_name=raw_path.name)
            range_doppler = result.get("range_doppler")
            if need_rd_data:
                save_range_doppler_data(
                    range_doppler,
                    rd_paths["data"],
                    analyser.adc_params.range_resolution,
                    analyser.adc_params.doppler_resolution,
                )
                range_doppler_files += 1
            if need_rd_grid:
                rgb_path, rgb_age_s = matching_rgb_frame(
                    camera_frames,
                    camera_timestamps,
                    raw_path,
                    camera_clock_offset_s=camera_clock_offset_s,
                    recording_dir=recording_dir,
                )
                save_range_doppler_grid(
                    range_doppler,
                    rd_paths["grid"],
                    analyser.adc_params.max_range,
                    analyser.adc_params.max_doppler,
                    raw_path.name,
                    rgb_path=rgb_path,
                    rgb_age_s=rgb_age_s,
                )
                range_doppler_images += 1
            if need_vod:
                count = save_vod_pc(pc, output_path, args.feature)
                converted += 1
                total_points += count
            if need_detections_csv:
                detection_csv_rows += save_detections_csv(
                    pc, detections_csv_path
                )
                detection_csv_files += 1
            show_selected = args.show_preview and index == args.preview_frame
            modes = []
            if need_snr:
                modes.append("snr")
            if need_raw_power:
                modes.append("raw_power")
            if show_selected and args.display in modes:
                modes.remove(args.display)
                modes.insert(0, args.display)
            for mode in modes:
                save_pc2d_preview(
                    pc,
                    paths[mode],
                    mode,
                    raw_path.name,
                    show=show_selected and args.display == mode,
                )
                if mode == "snr":
                    snr_images += 1
                else:
                    raw_power_images += 1
            if need_scan_processing:
                if (
                    scan_history
                    and scan_history[-1].get("frame_index") != index - 1
                ):
                    scan_history.clear()
                    if ego_processor is not None:
                        ego_processor.reset_segment()
                scan_history.append(
                    {
                        "frame_index": index,
                        "raw_path": raw_path,
                        "point_cloud": pc,
                        "ego_vx": float(
                            ego_info.get("vx_apply", 0.0) if ego_info else 0.0
                        ),
                        "ego_vy": float(
                            ego_info.get("vy_apply", 0.0) if ego_info else 0.0
                        ),
                    }
                )
                if len(scan_history) > max_scan_count:
                    scan_history = scan_history[-max_scan_count:]
                dataset_rgb_path, _ = matching_rgb_frame(
                    camera_frames,
                    camera_timestamps,
                    raw_path,
                    camera_clock_offset_s=camera_clock_offset_s,
                    recording_dir=recording_dir,
                )
                if dataset_rgb_path is None or not dataset_rgb_path.is_file():
                    scan_samples_without_rgb += 1
                else:
                    for scan_count in scan_build_counts:
                        if len(scan_history) < scan_count:
                            continue
                        scan_sequences[scan_count] += 1
                        sequence = scan_sequences[scan_count]
                        row = save_scan_dataset_sample(
                            scan_layouts[scan_count],
                            sequence,
                            scan_count,
                            scan_history,
                            dataset_rgb_path,
                            calib_template,
                            args.feature,
                            camera_clock_offset_s,
                            reconcile_scan_velocities=(
                                ego_processor is not None
                                and not args.no_ego_scan_reconcile
                            ),
                            align_scan_positions=(
                                ego_processor is not None
                                and not args.no_ego_scan_align
                            ),
                            frame_period_s=frame_period_s,
                        )
                        scan_manifest_rows[scan_count].append(row)
                        scan_samples[scan_count] += 1
                if (
                    comparison_enabled
                    and index in comparison_pending_targets
                    and len(scan_history) >= 5
                ):
                    comparison_path = (
                        comparison_dir
                        / f"{raw_path.stem}_1_3_5_scan_comparison.png"
                    )
                    point_counts = save_scan_comparison_grid(
                        scan_history,
                        comparison_path,
                        f"1/3/5 scan accumulation: {raw_path.name}",
                        rgb_path=dataset_rgb_path,
                    )
                    comparison_saved_indices.add(index)
                    comparison_point_counts[index] = point_counts
                    comparison_paths.append(comparison_path)
            done = index + 1
            if done == 1 or done % 10 == 0 or done == len(frames):
                print(
                    f"[{done}/{len(frames)}] {raw_path.name}: "
                    f"{len(_flat(pc, 'x'))} points"
                )
        except Exception as exc:
            failed += 1
            scan_history.clear()
            if ego_processor is not None:
                ego_processor.reset_segment()
            print(f"ERROR {raw_path}: {exc}", file=sys.stderr)
            if args.fail_fast:
                raise

    for scan_count in scan_build_counts:
        write_scan_manifest(
            scan_manifest_rows[scan_count],
            scan_layouts[scan_count]["root"] / "manifest.csv",
        )

    if ego_processor is not None and ego_processor.diagnostics:
        diagnostics_path = output_root / "ego_motion_diagnostics.csv"
        write_ego_motion_diagnostics(
            ego_processor.diagnostics, diagnostics_path
        )
        inspection = ego_processor.inspection_summary()
        speeds = [
            float(row.get("speed_apply", row.get("speed_smooth", 0.0)))
            for row in ego_processor.diagnostics
        ]
        static_fractions = [
            float(row["static_fraction"]) for row in ego_processor.diagnostics
        ]
        print(f"Ego-motion diagnostics: {diagnostics_path}")
        if inspection is not None:
            print(
                "  vr_comp inspection: "
                f"median |vr|={inspection['vr_median']:.3f} m/s, "
                f"median |vr_comp|={inspection['vrc_median']:.3f} m/s, "
                f"ratio={inspection['ratio']:.3f}, "
                f"use_vr_comp={inspection['use_vr_comp']}"
            )
        print(
            "  Mean applied ego speed: "
            f"{float(np.mean(speeds)):.3f} m/s "
            f"({float(np.mean(speeds)) * 3.6:.2f} km/h)"
        )
        print(
            "  Mean static yield: "
            f"{100.0 * float(np.mean(static_fractions)):.1f}%"
        )

    elapsed = time.perf_counter() - started
    print(
        f"Done in {elapsed:.2f}s: converted={converted}, skipped={skipped}, "
        f"failed={failed}, points={total_points}, snr_images={snr_images}, "
        f"raw_power_images={raw_power_images}, "
        f"detection_csv_files={detection_csv_files}, "
        f"detection_csv_rows={detection_csv_rows}, "
        f"range_doppler_files={range_doppler_files}, "
        f"range_doppler_images={range_doppler_images}, "
        f"scan_samples={scan_samples}, "
        f"scan_samples_without_rgb={scan_samples_without_rgb}, "
        f"comparison_saved={len(comparison_saved_indices)}, "
        f"comparison_points={comparison_point_counts}"
    )
    for comparison_path in comparison_paths:
        print(f"Comparison image: {comparison_path}")
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return convert(parse_args(argv))
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
