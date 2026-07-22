#!/usr/bin/env python
"""Apply ego-motion compensation to existing VoD point clouds and rebuild scan datasets.

Reads per-frame VoD files (``vod_pc/*_pc.bin`` or ``data/1_scan/radar/*.bin``),
applies the same ego-motion and multi-scan stacking logic as
``convert_raw_bin_to_vod.py --is-moving``, and writes updated scan datasets
without re-running MUSIC on raw ADC frames.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Keep FusionApp package imports working when this script is run directly.
# Script directory stays first so sibling convert_raw_bin_to_vod imports resolve.
_SCRIPT_DIR = Path(__file__).resolve().parent
_FUSIONAPP_ROOT = _SCRIPT_DIR.parent
if str(_FUSIONAPP_ROOT) not in sys.path:
    sys.path.insert(1, str(_FUSIONAPP_ROOT))

import numpy as np

from convert_raw_bin_to_vod import (
    DEFAULT_CALIB_TEMPLATE,
    SCAN_COUNTS,
    discover_raw_frames,
    matching_rgb_frame,
    output_directories,
    output_path_for,
    prepare_scan_datasets,
    readable_camera_frames,
    save_scan_dataset_sample,
    save_vod_pc_atomic,
    selected_frames,
    write_ego_motion_diagnostics,
    write_scan_manifest,
)
from camera.png_utils import scan_png_directory
from radar.point_cloud import read_vod_pc_bin, vod_to_point_cloud
from recording.sync_recording import SESSION_FILENAME
from sample_processing.ego_motion_compensation import EgoMotionSequenceProcessor
from sample_processing.radar_params import ADCParams


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild 1/3/5 scan datasets from existing VoD point clouds with "
            "ego-motion compensation, without re-processing raw radar ADC data."
        )
    )
    parser.add_argument(
        "recording_dir",
        type=Path,
        help="Recording directory containing vod_pc/ and raw .bin frames.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "AWR2243 profile for frame period and ego range limits. "
            "Defaults to recording_session.json radar_config_file when present."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output root. Defaults to the recording directory.",
    )
    parser.add_argument(
        "--vod-source",
        choices=("vod_pc", "1_scan"),
        default="vod_pc",
        help=(
            "Where to read single-frame VoD rows. vod_pc matches convert output; "
            "1_scan uses data/1_scan/radar/*.bin via manifest.csv. Default: vod_pc."
        ),
    )
    parser.add_argument(
        "--feature",
        choices=("raw_power", "snr", "rcs"),
        default="raw_power",
        help="VoD column-4 interpretation when reading and writing VoD files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Scan dataset root. Defaults to <output-dir>/data.",
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
        help=f"Calibration text copied for every numbered sample. Default: {DEFAULT_CALIB_TEMPLATE}",
    )
    parser.add_argument(
        "--camera-clock-offset-ms",
        type=float,
        default=0.0,
        help="Camera timestamp offset before matching RGB to radar. Default: 0.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based first raw frame index to process. Default: 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of frames to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing scan datasets selected by --scan-counts.",
    )
    parser.add_argument(
        "--full-scan-export",
        action="store_true",
        help=(
            "Copy images, raw radar, calibration, and radarref CSVs for every "
            "sample. Default rebuild mode updates only radar/*.bin and manifest."
        ),
    )
    parser.add_argument(
        "--no-update-vod",
        action="store_true",
        help="Do not rewrite vod_pc/*_pc.bin with compensated velocities_comp.",
    )
    parser.add_argument(
        "--no-ego",
        action="store_true",
        help=(
            "Skip per-frame ego compensation; restack using existing "
            "velocities_comp from the VoD files."
        ),
    )
    parser.add_argument(
        "--no-ego-scan-reconcile",
        action="store_true",
        help="Do not re-reference stacked velocities_comp to the current ego frame.",
    )
    parser.add_argument(
        "--no-ego-scan-align",
        action="store_true",
        help="Do not translate historical x/y into the current frame before stacking.",
    )
    parser.add_argument(
        "--ego-alpha",
        type=float,
        default=0.25,
        help="EMA weight for ego smoothing. Default: 0.25.",
    )
    parser.add_argument(
        "--ego-ransac-tau",
        type=float,
        default=0.30,
        help="RANSAC inlier threshold in m/s. Default: 0.30.",
    )
    parser.add_argument(
        "--ego-static-tau",
        type=float,
        default=0.40,
        help="Static/moving diagnostic threshold in m/s. Default: 0.40.",
    )
    parser.add_argument(
        "--ego-ransac-iterations",
        type=int,
        default=100,
        help="RANSAC iterations. Default: 100.",
    )
    parser.add_argument(
        "--ego-rcs-gate",
        action="store_true",
        help="Flag low-RCS points as uncertain in ego diagnostics.",
    )
    parser.add_argument(
        "--ego-compensate-mode",
        choices=("raw", "smooth", "kalman"),
        default="kalman",
        help="Per-frame ego velocity mode. Default: kalman.",
    )
    parser.add_argument(
        "--ego-kalman-process-var",
        type=float,
        default=0.25,
        help="Kalman process variance (m/s)^2. Default: 0.25.",
    )
    parser.add_argument(
        "--ego-kalman-meas-var",
        type=float,
        default=0.40,
        help="Kalman measurement variance (m/s)^2. Default: 0.40.",
    )
    parser.add_argument(
        "--ego-max-range",
        type=float,
        help="Maximum range (m) for ego estimation. Defaults to 95%% of profile max range.",
    )
    parser.add_argument(
        "--ego-min-range",
        type=float,
        default=1.0,
        help="Minimum range (m) for ego RANSAC inliers. Default: 1.0.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first frame error.",
    )
    return parser.parse_args(argv)


def resolve_config_path(recording_dir: Path, config_arg: Optional[Path]) -> Path:
    app_root = Path(__file__).resolve().parent
    default_config = app_root / "config_files" / "AWR2243_87m_17cm_64_3_256.txt"

    if config_arg is not None:
        config_path = config_arg.resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Radar config not found: {config_path}")
        return config_path

    candidates: List[Path] = []
    session_path = recording_dir / SESSION_FILENAME
    if session_path.is_file():
        with session_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        recorded = payload.get("radar_config_file")
        if recorded:
            recorded_path = Path(str(recorded))
            candidates.append(recorded_path)
            candidates.append(app_root / "config_files" / recorded_path.name)

    candidates.append(default_config)

    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if resolved.is_file():
            return resolved

    raise ValueError(
        "Radar config not specified and recording_session.json does not contain "
        "a valid radar_config_file. Pass --config explicitly."
    )


def load_1_scan_rows(data_root: Path) -> List[Dict[str, str]]:
    manifest = data_root / "1_scan" / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"--vod-source 1_scan requires an existing manifest: {manifest}"
        )
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"1_scan manifest is empty: {manifest}")
    rows.sort(key=lambda row: int(row["sequence"]))
    return rows


def load_point_cloud_from_vod(
    vod_path: Path,
    feature_mode: str,
) -> Dict[str, Any]:
    if not vod_path.is_file():
        raise FileNotFoundError(f"VoD point cloud not found: {vod_path}")
    cloud = read_vod_pc_bin(vod_path)
    return vod_to_point_cloud(cloud, feature_mode)


def rebuild(args: argparse.Namespace) -> int:
    recording_dir = args.recording_dir.resolve()
    if not recording_dir.is_dir():
        raise FileNotFoundError(f"Recording directory not found: {recording_dir}")

    config_path = resolve_config_path(recording_dir, args.config)
    adc_params = ADCParams(str(config_path))
    frame_period_s = float(adc_params.frame_periodicity) / 1000.0
    ego_max_range = float(
        args.ego_max_range
        if args.ego_max_range is not None
        else adc_params.max_range * 0.95
    )
    ego_min_range = float(args.ego_min_range)

    output_root = (args.output_dir or recording_dir).resolve()
    directories = output_directories(output_root)
    vod_dir = directories["vod_pc"]
    data_root = (args.data_dir or (output_root / "data")).resolve()

    raw_frames = discover_raw_frames(recording_dir)
    frames = selected_frames(raw_frames, args.start, args.limit)
    if not frames:
        raise ValueError("No raw radar frames selected")

    one_scan_rows: Optional[List[Dict[str, str]]] = None
    if args.vod_source == "1_scan":
        all_rows = load_1_scan_rows(data_root)
        stop = None if args.limit is None else args.start + args.limit
        one_scan_rows = all_rows[args.start:stop]
        if len(one_scan_rows) < len(frames):
            raise ValueError(
                f"1_scan manifest has {len(one_scan_rows)} rows for the selected "
                f"range but {len(frames)} raw frames were selected"
            )

    scan_counts = sorted(set(int(value) for value in args.scan_counts))
    radar_bin_only = not bool(args.full_scan_export)
    scan_layouts, scan_build_counts = prepare_scan_datasets(
        data_root,
        scan_counts,
        bool(args.overwrite),
        radar_bin_only=radar_bin_only,
    )
    if not scan_build_counts:
        print(f"Scan datasets already complete for counts {scan_counts}: {data_root}")
        return 0

    print(f"Processing {len(frames)} frames from {recording_dir}")
    camera_frames, _ = scan_png_directory(str(recording_dir))
    camera_frames = readable_camera_frames(camera_frames)
    camera_timestamps = [float(frame[1]) for frame in camera_frames]
    camera_clock_offset_s = float(args.camera_clock_offset_ms) / 1000.0
    calib_template = args.calib_template.resolve()

    ego_processor: Optional[EgoMotionSequenceProcessor] = None
    if not args.no_ego:
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
            f"(mode={args.ego_compensate_mode}, frame_period={frame_period_s:.3f} s)"
        )
    else:
        print("Ego-motion compensation: disabled (restacking existing velocities_comp)")

    print(f"VoD source: {args.vod_source}")
    print(
        "Scan export mode: "
        + ("radar bins + manifest only" if radar_bin_only else "full dataset copy")
    )
    print(f"Rebuilding scan counts: {scan_build_counts}")
    for count in scan_build_counts:
        print(f"  {count}_scan -> {scan_layouts[count]['root']}")
    if not args.no_update_vod and args.vod_source == "vod_pc":
        print(f"Updating compensated VoD files: {vod_dir}")

    scan_manifest_rows = {count: [] for count in scan_build_counts}
    scan_sequences = {count: 0 for count in scan_build_counts}
    scan_history: List[Dict[str, Any]] = []
    max_scan_count = max(scan_build_counts)
    scan_samples = {count: 0 for count in scan_build_counts}
    scan_samples_without_rgb = 0
    updated_vod = 0
    failed = 0
    ego_diagnostics: List[Dict[str, Any]] = []
    started = time.perf_counter()

    for index, raw_path in enumerate(frames):
        try:
            if args.vod_source == "vod_pc":
                vod_path = output_path_for(raw_path, vod_dir)
            else:
                row = one_scan_rows[index]
                sequence = int(row["sequence"])
                stem = row.get("sample_id") or f"{sequence - 1:05d}"
                vod_path = data_root / "1_scan" / "radar" / f"{stem}.bin"
                if not vod_path.is_file():
                    # Legacy 1-based filenames from older conversions.
                    legacy = data_root / "1_scan" / "radar" / f"{sequence}.bin"
                    if legacy.is_file():
                        vod_path = legacy
                manifest_raw = row.get("radar_source", "").strip()
                if manifest_raw:
                    manifest_raw_path = recording_dir / manifest_raw
                    if manifest_raw_path.is_file():
                        raw_path = manifest_raw_path

            pc = load_point_cloud_from_vod(vod_path, args.feature)
            ego_info: Dict[str, Any] = {
                "vx_apply": 0.0,
                "vy_apply": 0.0,
            }
            if ego_processor is not None:
                ego_info = ego_processor.apply(pc, frame_name=raw_path.name)
                ego_diagnostics.append(ego_info)

            if (
                not args.no_update_vod
                and args.vod_source == "vod_pc"
            ):
                destination = output_path_for(raw_path, vod_dir)
                save_vod_pc_atomic(pc, destination, args.feature)
                updated_vod += 1

            if scan_history and scan_history[-1].get("frame_index") != index - 1:
                scan_history.clear()
                if ego_processor is not None:
                    ego_processor.reset_segment()

            scan_history.append(
                {
                    "frame_index": index,
                    "raw_path": raw_path,
                    "point_cloud": pc,
                    "ego_vx": float(ego_info.get("vx_apply", 0.0)),
                    "ego_vy": float(ego_info.get("vy_apply", 0.0)),
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
                        radar_bin_only=radar_bin_only,
                    )
                    scan_manifest_rows[scan_count].append(row)
                    scan_samples[scan_count] += 1
            if index > 0 and index % 500 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  frame {index}/{len(frames)} "
                    f"updated_vod={updated_vod} scan_samples={scan_samples} "
                    f"failed={failed} elapsed={elapsed:.1f}s",
                    flush=True,
                )
        except Exception as exc:
            failed += 1
            print(f"ERROR frame {index} ({raw_path.name}): {exc}", file=sys.stderr)
            if args.fail_fast:
                break

    for scan_count in scan_build_counts:
        write_scan_manifest(
            scan_manifest_rows[scan_count],
            scan_layouts[scan_count]["root"] / "manifest.csv",
        )

    diagnostics_path = output_root / "ego_motion_diagnostics.csv"
    write_ego_motion_diagnostics(ego_diagnostics, diagnostics_path)

    elapsed = time.perf_counter() - started
    print(
        f"Done in {elapsed:.2f}s: updated_vod={updated_vod}, failed={failed}, "
        f"scan_samples={scan_samples}, "
        f"scan_samples_without_rgb={scan_samples_without_rgb}"
    )
    if ego_diagnostics:
        print(f"Ego diagnostics: {diagnostics_path}")
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return rebuild(parse_args(argv))
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
