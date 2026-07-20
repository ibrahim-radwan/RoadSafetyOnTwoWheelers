"""View-of-Delft / KITTI-style object label export.

Each line follows the VoD KITTI-compatible label layout:
  type truncated occluded alpha bbox[l,t,r,b] dimensions[h,w,l] location[x,y,z] rotation

VoD differences from classic KITTI that matter here:
- ``truncated`` is kept only for KITTI compatibility, so this exporter writes
  ``0.00`` instead of inventing image-plane truncation.
- ``location`` is the bottom-face centre in camera coordinates.
- the final ``rotation`` is yaw around the LiDAR/radar negative Z axis, not
  classic KITTI camera-frame ``rotation_y``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from ..core.geometry import (
    Box3D,
    box_bottom_center_master,
    project_box_calibration,
    box_2d_from_projection,
)
from ..core.calibration import Calibration


_VOD_TYPES = frozenset({
    "Car",
    "Pedestrian",
    "Cyclist",
    "Rider",
    "Unused bicycle",
    "Bicycle rack",
    "Human depiction (e.g. statues)",
    "Moped or scooter",
    "Motor",
    "Truck",
    "Other ride",
    "Other vehicle",
    "Uncertain ride",
})


VOD_CLASS_MAP: Dict[str, str] = {
    "Car": "Car",
    "Pedestrian": "Pedestrian",
    "Cyclist": "Cyclist",
    "Rider": "Rider",
    "Bike": "Unused bicycle",
    "Bike rack": "Bicycle rack",
    "Human depiction / statue": "Human depiction (e.g. statues)",
    "E-Scooter / Moped": "Moped or scooter",
    "Motorbike": "Motor",
    "Unused bicycle": "Unused bicycle",
    "Bicycle rack": "Bicycle rack",
    "Human depiction (e.g. statues)": "Human depiction (e.g. statues)",
    "Moped or scooter": "Moped or scooter",
    "Motor": "Motor",
    "Truck": "Truck",
    "Other ride": "Other ride",
    "Other vehicle": "Other vehicle",
    "Uncertain ride": "Uncertain ride",
    "Human depiction": "Human depiction (e.g. statues)",
    "Bus": "Truck",
    "Van": "Other vehicle",
    "Motorcycle": "Motor",
    "Bicycle": "Unused bicycle",
    "Scooter": "Moped or scooter",
    "E-Scooter": "Moped or scooter",
    "Person": "Pedestrian",
    "Person_sitting": "Pedestrian",
    "Animal": "Other vehicle",
    "Traffic_Cone": "Uncertain ride",
    "Barrier": "Uncertain ride",
    "Debris": "Uncertain ride",
    "Unknown": "Uncertain ride",
    "Other": "Uncertain ride",
}


def _vod_type(class_name: str) -> str:
    typ = VOD_CLASS_MAP.get(class_name, class_name)
    if typ not in _VOD_TYPES:
        return "Uncertain ride"
    return typ


def _master_to_cam(p: np.ndarray, T: np.ndarray) -> np.ndarray:
    h = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    return (T @ h)[:3]


def _wrap_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _vod_rotation(box: Box3D) -> float:
    """VoD rotation: yaw around the LiDAR/radar negative Z axis."""
    return _wrap_pi(-float(box.yaw))


def _alpha(loc_cam: np.ndarray, rotation: float) -> float:
    """Observation angle, using the same final rotation field VoD exports."""
    return _wrap_pi(
        float(rotation) - math.atan2(float(loc_cam[0]), float(loc_cam[2]) + 1e-12)
    )


def _box_to_kitti_line(
    box: Box3D,
    cal: Calibration,
    image_size: Tuple[int, int],
) -> Optional[str]:
    W, H = int(image_size[0]), int(image_size[1])
    corners_uv, depths = project_box_calibration(box, cal)

    if np.all(depths <= 0):
        return None

    bbox_clip = box_2d_from_projection(corners_uv, depths, (W, H))
    if bbox_clip is None:
        return None
    xmin_c, ymin_c, xmax_c, ymax_c = bbox_clip

    bottom_m = box_bottom_center_master(box)
    loc_cam = _master_to_cam(bottom_m, cal.T)
    rotation = _vod_rotation(box)
    alpha = _alpha(loc_cam, rotation)

    typ = _vod_type(box.class_name)
    trunc = 0.0
    occ = int(max(0, min(2, box.occlusion)))
    h_dim, w_dim, l_dim = float(box.height), float(box.width), float(box.length)

    parts = [
        typ,
        f"{trunc:.2f}",
        str(occ),
        f"{alpha:.2f}",
        f"{xmin_c:.2f}",
        f"{ymin_c:.2f}",
        f"{xmax_c:.2f}",
        f"{ymax_c:.2f}",
        f"{h_dim:.2f}",
        f"{w_dim:.2f}",
        f"{l_dim:.2f}",
        f"{loc_cam[0]:.2f}",
        f"{loc_cam[1]:.2f}",
        f"{loc_cam[2]:.2f}",
        f"{rotation:.2f}",
    ]
    return " ".join(parts)


def export_kitti_label_dir(
    label_dir: Path,
    frames: List[dict],
    default_calibration: Calibration,
) -> None:
    """Write ``<frame_id>.txt`` under ``label_dir`` for each frame.

    Each ``frames`` element:
        frame_id: str
        boxes: List[Box3D]
        image_size: (W, H)
        calibration: optional Calibration for this frame (else default)
    """
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    for fr in frames:
        fid = str(fr["frame_id"])
        boxes: List[Box3D] = list(fr.get("boxes") or [])
        W, H = fr["image_size"]
        cal: Calibration = fr.get("calibration") or default_calibration
        if (int(W), int(H)) != (int(cal.image_size[0]), int(cal.image_size[1])):
            cal = cal.rescaled_to_image_size((int(W), int(H)))

        lines: List[str] = []
        for box in boxes:
            line = _box_to_kitti_line(box, cal, (W, H))
            if line is not None:
                lines.append(line)

        out = label_dir / f"{fid}.txt"
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
