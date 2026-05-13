"""
COCO-format exporter.

Standard COCO is 2D-only, so we follow the convention used by nuScenes-COCO,
3D-COCO, and similar datasets: we emit fully valid COCO (with the 2D bbox
derived from projecting the canonical 3D box into the image), and attach
the 3D ground truth + radar metadata as custom extension fields on each
annotation.

Output structure:
{
  "info":        { "description": ..., "date_created": ... },
  "licenses":    [],
  "categories":  [ {"id": 1, "name": "Car", "supercategory": "vehicle"}, ... ],
  "images":      [ {"id": 1, "file_name": "000001.png", "width": W,
                    "height": H, "frame_id": "000001",
                    "radar_file": "000001.bin",
                    "calibration_id": "..."} ],
  "annotations": [ {
        "id": <global>,
        "image_id": <image id>,
        "category_id": <cat id>,
        "bbox": [x, y, w, h],              # 2D COCO bbox, in pixels
        "area": w * h,
        "iscrowd": 0,
        "segmentation": [],
        # --- extension fields (non-standard, 3D sensor-fusion labels) ---
        "bbox_3d": {
            "center_master": [x, y, z],
            "size_lwh": [l, w, h],
            "rotation": {"yaw": ..., "pitch": ..., "roll": ...},
            "coordinate_frame": "master"
        },
        "track_id": int or null,
        "occlusion": int,
        "truncation": float,
        "confidence": float,
        "num_radar_points": int,
        "notes": str
  } ]
}
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..core.geometry import Box3D, project_box_calibration, box_2d_from_projection
from ..core.calibration import Calibration


def export_coco(output_path: Path,
                frames: List[dict],
                calibration: Calibration,
                description: str = "Radar-camera annotations") -> dict:
    """Export a list of annotated frames to COCO JSON.

    Each entry in `frames` must be a dict:
        {
            "frame_id": str,
            "image_file": str,          # filename (not full path)
            "radar_file": str,
            "image_size": (W, H),
            "boxes": List[Box3D],
            "radar_points": np.ndarray or None,   # for num_radar_points
        }
    """
    # Collect all unique class names -> stable category ids
    classes: List[str] = []
    for fr in frames:
        for b in fr["boxes"]:
            if b.class_name not in classes:
                classes.append(b.class_name)
    categories = [
        {"id": i + 1, "name": name, "supercategory": _supercat(name)}
        for i, name in enumerate(classes)
    ]
    cat_id_by_name = {c["name"]: c["id"] for c in categories}

    images = []
    annotations = []
    ann_counter = 1
    for img_id, fr in enumerate(frames, start=1):
        W, H = fr["image_size"]
        images.append({
            "id": img_id,
            "file_name": fr["image_file"],
            "width": int(W),
            "height": int(H),
            "frame_id": fr["frame_id"],
            "radar_file": fr["radar_file"],
            "calibration_id": calibration.id,
        })

        radar_pts = fr.get("radar_points")

        for box in fr["boxes"]:
            corners_uv, depths = project_box_calibration(box, calibration)
            bbox2d = box_2d_from_projection(corners_uv, depths, (W, H))
            # If the box doesn't project into the image, still emit the 3D
            # annotation with a zero-area 2D bbox flagged as out-of-view.
            if bbox2d is None:
                x0 = y0 = w = h = 0.0
                area = 0.0
            else:
                xmin, ymin, xmax, ymax = bbox2d
                x0, y0 = xmin, ymin
                w, h = xmax - xmin, ymax - ymin
                area = w * h

            num_pts = 0
            if radar_pts is not None and radar_pts.size > 0:
                num_pts = int(np.count_nonzero(
                    box.points_inside(radar_pts[:, :3])
                ))

            annotations.append({
                "id": ann_counter,
                "image_id": img_id,
                "category_id": cat_id_by_name[box.class_name],
                "bbox": [float(x0), float(y0), float(w), float(h)],
                "area": float(area),
                "iscrowd": 0,
                "segmentation": [],
                # -- 3D extension ---------------------------------------
                "bbox_3d": {
                    "center_master": [box.x, box.y, box.z],
                    "size_lwh": [box.length, box.width, box.height],
                    "rotation": {
                        "yaw": box.yaw,
                        "pitch": box.pitch,
                        "roll": box.roll,
                    },
                    "coordinate_frame": "master",
                },
                "track_id": box.track_id,
                "occlusion": box.occlusion,
                "truncation": box.truncation,
                "confidence": box.confidence,
                "num_radar_points": num_pts,
                "notes": box.notes,
            })
            ann_counter += 1

    coco = {
        "info": {
            "description": description,
            "version": "1.0",
            "date_created": datetime.now().isoformat(timespec="seconds"),
            "calibration_id": calibration.id,
            "notes": ("2D bbox is derived by projecting the canonical 3D box "
                      "(bbox_3d) into the image via the stored calibration. "
                      "The 3D box is the ground truth."),
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, indent=2))
    return coco


_SUPERCAT_MAP = {
    "Car": "vehicle",
    "Pedestrian": "person",
    "Cyclist": "ride",
    "Rider": "person",
    "Bike": "ride",
    "Bike rack": "static",
    "Human depiction / statue": "static",
    "E-Scooter / Moped": "ride",
    "Motorbike": "ride",
    "Unused bicycle": "ride",
    "Bicycle rack": "static",
    "Human depiction (e.g. statues)": "static",
    "Moped or scooter": "ride",
    "Motor": "ride",
    "Truck": "vehicle",
    "Other ride": "ride",
    "Other vehicle": "vehicle",
    "Uncertain ride": "ride",
    "Human depiction": "static",
    "Bus": "vehicle", "Van": "vehicle",
    "Motorcycle": "ride", "Bicycle": "ride", "Scooter": "ride",
    "E-Scooter": "ride",
    "Person": "person", "Animal": "animal",
    "Traffic_Cone": "static", "Barrier": "static", "Debris": "static",
    "Unknown": "object",
}


def _supercat(name: str) -> str:
    return _SUPERCAT_MAP.get(name, "object")
