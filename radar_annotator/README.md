# 3D Radar-Camera Annotation Tool

A single-screen PySide6 desktop tool for annotating paired radar + camera
recordings with 3D bounding boxes. Built to the design spec:
one canonical 3D box per object lives in the master (radar / ego) frame,
and every other view — radar BEV, image wireframe, 2D COCO bbox — is derived
from it via calibration. Edits in either view propagate immediately.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.9+, PySide6, NumPy, and OpenCV.

## Run

```bash
python -m radar_annotator
```

Then `Ctrl+O` and pick a dataset folder.

To try it right now without your own data:

```bash
python -m radar_annotator.tools.generate_sample_data ./sample_dataset
python -m radar_annotator    # then Ctrl+O -> ./sample_dataset
```

## Dataset folder layout

The tool auto-discovers the following structure (subfolder names are
case-insensitive; common aliases like `pointcloud`, `rgb`, `images` also
work):

```
dataset/
  radar/                     # .npy, .npz, .bin, .pcd, or .csv point clouds
    000001.npy
    000002.npy
    ...
  image/                     # .png, .jpg, .jpeg, .bmp, .tiff
    000001.png
    000002.png
    ...
  calib/                     # optional
    calib.json               # auto-loaded (also calib.txt / .yaml / etc.)
  labels_internal/           # created on save; re-loaded when you revisit a frame
```

Radar and image files are paired by **identical filename stem** (Section 5.2
of the design spec). An `.bin` file of float32s is treated first as
View-of-Delft radar `[x, y, z, RCS, v_r, v_r_compensated, time]` when the
layout matches 7-float radar points. The loader also accepts compatible
8-float radar layouts only when the trailing fields pass sanity checks, then
falls back to KITTI-style `[x, y, z, intensity]` or plain `[x, y, z]`.

The radar pane includes display/export filters for height, reflection mode,
automatic/custom RCS, and an adjustable XY ROI. Use **Candidates** to highlight
likely object locations in the BEV view, or **Cluster Objects** to create
editable initial boxes from density-based radar clusters. Use **Save Filtered**
to write the active filtered point cloud for the current frame as `.npz`, `.npy`,
or `.csv`; `.npz` also stores the source file, frame id, and filter settings.

RCS and Doppler are interpreted as separate fields. In VoD-style point clouds,
column 4 is RCS, column 5 is raw radial velocity, and column 6 is compensated
radial velocity. The UI uses compensated velocity for Doppler/moving filters
when available, then falls back to raw radial velocity.

### Calibration file

**Per-frame KITTI-style files** — if every paired frame has a matching
`calib/<frame_stem>.txt` (same stem as `radar/000123.bin` and `image/000123.png`),
those files are loaded automatically when you change frames. Each file should
look like the KITTI devkit format: lines `P2:` (12 floats), `R0_rect:` (9),
`Tr_velo_to_cam:` or `Tr_radar_to_cam:` (12 floats in 3×4 row-major `[R|t]`).

**Single shared calibration** — otherwise place one file under `calib/`:
preferred names `calib.json`, `calibration.json`, `calib.txt`, … then other
`*.json`, `*.txt`, `*.yaml`, `*.yml` in sorted order. Contents may be **JSON**
(schema below), **YAML** with the same keys if `PyYAML` is installed, or KITTI
plaintext as above.

```json
{
  "id": "front_radar_cam_v1",
  "image_size": [1280, 720],
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "T_cam_from_master": [[...4x4 row-major...]]
}
```

`T_cam_from_master` is the 4×4 transform that takes a point expressed in the
master (radar / ego) frame and returns it in the camera frame. You can
alternatively supply `R` (3×3) and `t` (length-3), and the tool will
assemble the 4×4 for you. Without any calibration file the tool uses a
sensible identity-default (camera 1.2m above radar, standard automotive
axis swap) so you can still see something on screen.

### Master frame convention

`x` forward, `y` left, `z` up. Yaw rotates around `+z`.

## Keyboard shortcuts

| Action | Keys |
|---|---|
| Previous / next frame | ← / → |
| Go to frame by index | Ctrl+G |
| Open dataset | Ctrl+O |
| Save frame | Ctrl+S |
| New box (then click in radar view) | N |
| Cancel 'new box' placement | Esc |
| Delete selected | Delete |
| Duplicate selected | Ctrl+D |
| Undo / redo | Ctrl+Z / Ctrl+Y |
| Translate box | W / A / S / D |
| Raise / lower | R / F |
| Rotate yaw | Q / E |
| Length − / + | Z / X |
| Width − / + | C / V |
| Snap base to ground | G |
| Fine step | Shift + any movement key |

## Mouse interactions

**Radar / BEV (top pane)**

| Interaction | Effect |
|---|---|
| Left click box | Select |
| Left drag box | Translate on ground plane |
| Right drag box | Rotate yaw |
| Shift + drag near edge | Resize along that edge |
| Wheel | Zoom around cursor |
| Middle drag | Pan |

**Camera image (bottom pane)**

| Interaction | Effect |
|---|---|
| Left click box | Select |
| Left drag | Image-constrained translate: moves the 3D box along the camera ray at its current depth, so the projection follows the cursor |

## Output: COCO format

`Export COCO` writes a standard COCO JSON whose 2D `bbox` fields are
derived by projecting each canonical 3D box into the image via the current
calibration. Because plain COCO has no 3D field, we attach the ground-truth
3D annotation as a custom extension on each annotation — the same approach
used by nuScenes-COCO, 3D-COCO, and similar datasets. Generated files pass
validation by `pycocotools`.

```json
{
  "info": { "description": "...", "calibration_id": "..." },
  "categories": [{ "id": 1, "name": "Car", "supercategory": "vehicle" }, ...],
  "images":     [{ "id": 1, "file_name": "000001.png",
                   "width": 1280, "height": 720,
                   "frame_id": "000001",
                   "radar_file": "000001.npy",
                   "calibration_id": "..." }],
  "annotations": [{
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox":   [x, y, w, h],            // standard COCO 2D, in pixels
      "area":   w * h,
      "iscrowd": 0,
      "segmentation": [],

      // --- 3D extension (the ground truth) ---
      "bbox_3d": {
          "center_master": [x, y, z],
          "size_lwh":      [l, w, h],
          "rotation":      { "yaw": ..., "pitch": ..., "roll": ... },
          "coordinate_frame": "master"
      },
      "track_id":         42,
      "occlusion":        0,
      "truncation":       0.0,
      "confidence":       1.0,
      "num_radar_points": 28,
      "notes":            ""
  }, ...]
}
```

If a 3D box projects entirely behind the camera or outside the image, the
2D `bbox` is `[0, 0, 0, 0]` with `area: 0`, and the 3D annotation is still
preserved. This lets downstream users detect and ignore out-of-view
objects without losing the underlying ground truth.

### Internal per-frame format

The editor also writes `labels_internal/<frame_id>.json` on every save,
with a richer schema (all QA fields, notes, UIDs). These files are loaded
automatically when you revisit a frame and are the source of truth the
COCO exporter walks over when you hit *Export COCO*.

## Output: VoD/KITTI labels

Use **Export VoD/KITTI** in the toolbar. Pick an output folder; the tool creates
`label_2/` with one `<frame_id>.txt` per frame (VoD KITTI-style line
format: type, truncation, occlusion, alpha, 2D box, h/w/l, camera-frame
location, rotation). Per-frame calibration files under `calib/` are used
when present. The truncation field is written as `0.00` for VoD compatibility,
and the final rotation field is yaw around radar/LiDAR `-Z`, not classic KITTI
camera `rotation_y`. Implementation: `io/kitti_export.py`.

## Architecture

```
radar_annotator/
  core/
    geometry.py            # Box3D + projection / point-in-box utilities
    calibration.py         # K + T_cam_from_master loader
    dataset.py             # folder scan, stem-pairing, radar file readers
    annotation_model.py    # Qt-signal model, undo/redo, dirty tracking
  views/
    radar_view.py          # interactive BEV canvas
    image_view.py          # projected wireframe + image-constrained editing
    object_panel.py        # right-hand editable properties
    frame_info_panel.py    # left-hand frame details + validation
  io/
    internal_json.py       # rich per-frame format
    coco_export.py         # COCO with 3D extension
    kitti_export.py        # KITTI label_2/*.txt export
  tools/
    generate_sample_data.py
  main_window_v2.py        # main UI (toolbar includes Export KITTI)
  main_window.py           # alternate / legacy assembly
  __main__.py              # `python -m radar_annotator`
```

## Known limitations / future work

- **No LiDAR / multi-camera / multi-radar yet.** The data model is designed
  for it (the master frame is shared) — adding more views is mechanical.
- **Image-side editing is translate-only.** Shift-drag edges in the BEV to
  resize. A full image-side constrained resize (pull a 2D bbox edge in
  image space) is doable but non-trivial; the BEV already covers this use
  case well.
- **K-Radar-specific packaging** is not implemented as its own export; use **Export KITTI** or **Export COCO**.
