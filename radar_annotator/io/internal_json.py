"""
Internal JSON label format (Section 11 of the design spec).

One JSON file per frame, stored under <root>/labels_internal/<frame_id>.json.
Richer than the export formats; preserves QA info and notes.

``center_master`` is the box geometric centre in the master (radar/ego) frame.
KITTI ``label_2/*.txt`` exports instead use the bottom-face centre in **camera**
coordinates — so JSON centre X and KITTI location X are not comparable directly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..core.geometry import Box3D


def save_frame_labels(path: Path,
                      frame_id: str,
                      radar_file: str,
                      image_file: str,
                      calibration_id: str,
                      boxes: List[Box3D]) -> None:
    payload = {
        "frame_id": frame_id,
        "radar_file": radar_file,
        "image_file": image_file,
        "calibration_id": calibration_id,
        "objects": [
            {
                "id": b.object_id,
                "track_id": b.track_id,
                "class_name": b.class_name,
                "center_master": [b.x, b.y, b.z],
                "size_lwh": [b.length, b.width, b.height],
                "yaw": b.yaw,
                "pitch": b.pitch,
                "roll": b.roll,
                "occlusion": b.occlusion,
                "truncation": b.truncation,
                "confidence": b.confidence,
                "notes": b.notes,
                "manual_placement": b.manual_placement,
                "uid": b.uid,
            }
            for b in boxes
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_frame_labels(path: Path) -> List[Box3D]:
    if not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text())
    boxes: List[Box3D] = []
    for obj in data.get("objects", []):
        cx, cy, cz = obj.get("center_master", [0, 0, 0])
        l, w, h = obj.get("size_lwh", [4, 2, 1.6])
        box = Box3D(
            x=cx, y=cy, z=cz,
            length=l, width=w, height=h,
            yaw=obj.get("yaw", 0.0),
            pitch=obj.get("pitch", 0.0),
            roll=obj.get("roll", 0.0),
            object_id=obj.get("id", 0),
            track_id=obj.get("track_id"),
            class_name=obj.get("class_name", "Car"),
            confidence=obj.get("confidence", 1.0),
            occlusion=obj.get("occlusion", 0),
            truncation=obj.get("truncation", 0.0),
            notes=obj.get("notes", ""),
            manual_placement=bool(obj.get("manual_placement", False)),
        )
        if "uid" in obj:
            box.uid = obj["uid"]
        boxes.append(box)
    return boxes
