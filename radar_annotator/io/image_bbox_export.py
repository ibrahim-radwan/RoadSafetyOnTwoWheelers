"""Comprehensive annotation export — camera + radar side, per annotated box.

Produces two files in the chosen output directory:
  annotations_export.json  — full fidelity: 3D box, 8 projected corners, all
                             radar return stats + raw points inside each box
  annotations_export.csv   — flat table (one row per box, aggregate stats only)

JSON is the primary format.  Load it back to re-render exact 3D wireframes on
the image and to recover every radar attribute the user observed.
"""
from __future__ import annotations

import csv
import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.calibration import Calibration
from ..core.geometry import Box3D, project_box_calibration, box_2d_from_projection

# VoD radar column layout (columns beyond what exists are skipped gracefully)
_COL_NAMES = [
    "x_fwd_m",
    "y_lat_m",
    "z_up_m",
    "rcs_dbsm",
    "v_r_mps",
    "v_r_comp_mps",
    "time_s",
    "snr_db",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stats(vals: np.ndarray) -> dict:
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return {}
    return {
        "min":  round(float(finite.min()),  4),
        "max":  round(float(finite.max()),  4),
        "mean": round(float(finite.mean()), 4),
        "std":  round(float(finite.std()),  4),
        "count": int(finite.size),
    }


def _radar_section(points_inside: np.ndarray) -> dict:
    """Build the 'radar' sub-dict for one annotation."""
    n = int(points_inside.shape[0])
    sec: dict = {"points_inside": n}
    if n == 0:
        return sec

    med_x = float(np.nanmedian(points_inside[:, 0]))
    med_y = float(np.nanmedian(points_inside[:, 1]))
    sec["azimuth_deg"] = round(float(np.degrees(np.arctan2(med_y, med_x))), 3)

    ncols = points_inside.shape[1]
    for ci in range(min(ncols, len(_COL_NAMES))):
        s = _stats(points_inside[:, ci])
        if s:
            sec[_COL_NAMES[ci]] = s

    # Raw points (rounded to 4 dp to keep file size manageable)
    col_labels = _COL_NAMES[:ncols]
    sec["raw_points"] = {
        "columns": col_labels,
        "data": [[round(float(v), 4) for v in row[:ncols]]
                 for row in points_inside.tolist()],
    }
    return sec


def _image_section(
    box: Box3D,
    cal: Calibration,
    image_size: Tuple[int, int],
) -> dict:
    """Build the 'image' sub-dict for one annotation."""
    corners_uv, depths = project_box_calibration(box, cal)
    if np.all(depths <= 0):
        return {"projected": False}

    bbox = box_2d_from_projection(corners_uv, depths, image_size)
    if bbox is None:
        return {"projected": False}

    x1, y1, x2, y2 = bbox
    sec = {
        "projected": True,
        "bbox_2d": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        "bbox_xywh": [round(x1, 2), round(y1, 2),
                      round(x2 - x1, 2), round(y2 - y1, 2)],
        "area_px2": round((x2 - x1) * (y2 - y1), 2),
        # 8 corners in image space, same order as Box3D.corners()
        "corners_uv": [[round(float(u), 2), round(float(v), 2)]
                       for u, v in corners_uv.tolist()],
    }
    return sec


def _cal_to_dict(cal: Calibration) -> dict:
    d: dict = {
        "K": cal.K.tolist(),
        "T_cam_from_master": cal.T.tolist(),
        "image_size": list(cal.image_size),
    }
    if cal.P_rect is not None:
        d["P_rect"] = np.asarray(cal.P_rect).tolist()
    return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_annotations(
    out_dir: Path,
    frames_payload: List[dict],
    default_calibration: Calibration,
) -> dict:
    """Export full-fidelity annotations to JSON + CSV.

    Each element of ``frames_payload`` must contain:
        frame_id:     str
        image_file:   str   (e.g. "000001.png")
        image_size:   (W, H)
        boxes:        List[Box3D]
        calibration:  Calibration | None   (falls back to default_calibration)
        radar_points: np.ndarray shape (N, >=3) | None
                      Filtered radar points for this frame — used to extract
                      per-box radar statistics and raw returns.

    Returns summary dict: {"json", "csv", "exported", "skipped_projection"}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_ann = 0
    skipped_proj = 0

    json_frames: List[dict] = []
    csv_rows: List[dict] = []

    for fr in frames_payload:
        fid        = str(fr["frame_id"])
        image_file = str(fr.get("image_file") or f"{fid}.png")
        W, H       = int(fr["image_size"][0]), int(fr["image_size"][1])
        boxes: List[Box3D]  = list(fr.get("boxes") or [])
        cal: Calibration    = fr.get("calibration") or default_calibration
        radar_pts: np.ndarray = np.asarray(
            fr.get("radar_points") if fr.get("radar_points") is not None
            else np.zeros((0, 3)),
            dtype=np.float64,
        )

        if (int(W), int(H)) != (int(cal.image_size[0]), int(cal.image_size[1])):
            cal = cal.rescaled_to_image_size((int(W), int(H)))

        if not boxes:
            continue

        frame_annotations: List[dict] = []

        for box in boxes:
            # --- Image section -------------------------------------------
            img_sec = _image_section(box, cal, (W, H))
            if not img_sec["projected"]:
                skipped_proj += 1

            # --- Radar section -------------------------------------------
            if radar_pts.size > 0:
                mask = box.points_inside(radar_pts[:, :3])
                inside = radar_pts[mask]
            else:
                inside = np.zeros((0, max(radar_pts.shape[1] if radar_pts.ndim == 2 else 3, 3)))
            rad_sec = _radar_section(inside)

            # --- 3D box attrs --------------------------------------------
            ann = {
                "object_id":  int(box.object_id),
                "track_id":   box.track_id,
                "class":      box.class_name,
                "confidence": round(float(box.confidence), 4),
                "occlusion":  int(box.occlusion),
                "truncation": round(float(box.truncation), 4),
                "notes":      box.notes,
                "box_3d": {
                    "x":      round(float(box.x),      4),
                    "y":      round(float(box.y),      4),
                    "z":      round(float(box.z),      4),
                    "length": round(float(box.length), 4),
                    "width":  round(float(box.width),  4),
                    "height": round(float(box.height), 4),
                    "yaw":    round(float(box.yaw),    6),
                    "pitch":  round(float(box.pitch),  6),
                    "roll":   round(float(box.roll),   6),
                },
                "image": img_sec,
                "radar": rad_sec,
            }
            frame_annotations.append(ann)
            total_ann += 1

            # CSV row (flat; no raw points)
            row: dict = {
                "frame_id":    fid,
                "image_file":  image_file,
                "object_id":   int(box.object_id),
                "track_id":    box.track_id if box.track_id is not None else "",
                "class":       box.class_name,
                "confidence":  round(float(box.confidence), 4),
                "occlusion":   int(box.occlusion),
                "truncation":  round(float(box.truncation), 4),
                "notes":       box.notes,
                # 3D box
                "box_x":       round(float(box.x),      4),
                "box_y":       round(float(box.y),      4),
                "box_z":       round(float(box.z),      4),
                "box_length":  round(float(box.length), 4),
                "box_width":   round(float(box.width),  4),
                "box_height":  round(float(box.height), 4),
                "box_yaw":     round(float(box.yaw),    6),
                # Image projection
                "img_projected": img_sec.get("projected", False),
                "img_x1":  img_sec.get("bbox_2d", [None]*4)[0],
                "img_y1":  img_sec.get("bbox_2d", [None]*4)[1],
                "img_x2":  img_sec.get("bbox_2d", [None]*4)[2],
                "img_y2":  img_sec.get("bbox_2d", [None]*4)[3],
                "img_width_px":  img_sec.get("bbox_xywh", [None]*4)[2],
                "img_height_px": img_sec.get("bbox_xywh", [None]*4)[3],
                "img_area_px2":  img_sec.get("area_px2"),
                # Radar aggregates
                "radar_points_inside": rad_sec.get("points_inside", 0),
                "radar_azimuth_deg":   rad_sec.get("azimuth_deg"),
            }
            # Flatten per-column stats
            for col in _COL_NAMES:
                s = rad_sec.get(col, {})
                for stat in ("min", "max", "mean", "std"):
                    row[f"{col}_{stat}"] = s.get(stat)
            csv_rows.append(row)

        json_frames.append({
            "frame_id":    fid,
            "image_file":  image_file,
            "image_size":  [W, H],
            "calibration": _cal_to_dict(cal),
            "annotations": frame_annotations,
        })

    # -----------------------------------------------------------------------
    # Write JSON
    # -----------------------------------------------------------------------
    doc = {
        "info": {
            "tool":               "radar_annotator_v3",
            "exported_at":        datetime.datetime.now().isoformat(timespec="seconds"),
            "total_frames":       len(json_frames),
            "total_annotations":  total_ann,
            "radar_columns":      _COL_NAMES,
        },
        "default_calibration": _cal_to_dict(default_calibration),
        "frames": json_frames,
    }
    json_path = out_dir / "annotations_export.json"
    json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    csv_path = out_dir / "annotations_export.csv"
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
    else:
        fieldnames = ["frame_id", "image_file", "object_id", "class"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    return {
        "json":               json_path,
        "csv":                csv_path,
        "exported":           total_ann,
        "skipped_projection": skipped_proj,
    }
