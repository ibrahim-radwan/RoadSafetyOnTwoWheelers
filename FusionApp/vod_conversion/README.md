# VoD conversion (FusionApp)

Offline tools that turn a FusionApp recording (raw radar `.bin` + camera `.png`) into scan packs laid out like the [View-of-Delft (VoD)](https://intelligent-vehicles.org/datasets/view-of-delft/) dataset for annotation and AI training.

VoD follows a [KITTI](https://www.cvlibs.net/datasets/kitti/)-compatible folder and naming style (`image_2`, zero-based five-digit stems, shared calib ids). Official references:

- **KITTI Vision Benchmark Suite:** [https://www.cvlibs.net/datasets/kitti/](https://www.cvlibs.net/datasets/kitti/)
- **View-of-Delft (VoD):** [https://intelligent-vehicles.org/datasets/view-of-delft/](https://intelligent-vehicles.org/datasets/view-of-delft/)
- **VoD documentation / development kit:** [https://tudelft-iv.github.io/view-of-delft-dataset/](https://tudelft-iv.github.io/view-of-delft-dataset/)

These scripts reuse the same radar analyser, RGB pairing rules, and calibration files as the live FusionApp stack. You do not need the web UI or radar hardware to run them.

## Folder contents

| Item | Type | Purpose |
|------|------|---------|
| [`convert_raw_bin_to_vod.py`](./convert_raw_bin_to_vod.py) | Script | Convert raw radar + RGB into `data/{1,3,5}_scan` |
| [`rebuild_scans_from_vod.py`](./rebuild_scans_from_vod.py) | Script | Rebuild 1/3/5 packs from existing VoD clouds (e.g. after ego-motion changes) |
| [`export_5_scan_ranges.py`](./export_5_scan_ranges.py) | Script | Split `data/5_scan` into sequence-range folders |
| [`sample_data/`](./sample_data/) | Data | Small raw `.bin` + `.png` recording for a quick conversion test |
| [`figures/`](./figures/) | Images | Pipeline diagrams used below (PNG counterparts of `doc_14_july` vectors) |
| [`README.md`](./README.md) | Docs | This guide |

Run every command from the **FusionApp** root so `config_files/` and Python packages resolve.

## Quick start (data only)

By default the converter writes **only** the numbered scan datasets. Optional extras (`vod_pc`, PC-2D previews, range-Doppler, comparison grids) stay off unless you pass `--save-*`.

```powershell
cd E:\Twowheelers_18_5\RoadSafetyOnTwoWheelers-main\FusionApp

python vod_conversion\convert_raw_bin_to_vod.py `
  "E:\Twowheelers_18_5\recordings\YOUR_RECORDING_FOLDER" `
  --config "config_files\AWR2243_87m_17cm_64_3_256.txt" `
  --is-moving
```

Use the **exact** AWR2243 profile that recorded the session. Prefer `--is-moving` for road / moving-platform captures (ego-motion compensation). Add `--overwrite` if `data/` already exists.

### Try with the included sample data

[`sample_data/`](./sample_data/) is a ready-to-run FusionApp recording snippet (timestamped raw radar `.bin` frames paired with RGB `.png` images). Use it to verify VoD conversion without a full recording:

```powershell
cd E:\Twowheelers_18_5\RoadSafetyOnTwoWheelers-main\FusionApp

python vod_conversion\convert_raw_bin_to_vod.py `
  "vod_conversion\sample_data" `
  --config "config_files\AWR2243_87m_17cm_64_3_256.txt" `
  --is-moving `
  --overwrite
```

Output is written under `vod_conversion\sample_data\data\{1,3,5}_scan` with VoD/KITTI-style `image_2` and shared five-digit sample ids.

## Output layout

Default tree:

```text
YOUR_RECORDING_FOLDER/
  data/
    1_scan/
    3_scan/
    5_scan/
      image_2/00000.png
      radar/00000.bin
      radar_raw/00000.bin
      calib/00000.txt
      manifest.csv
```

| Path under each `N_scan/` | Contents |
|---------------------------|----------|
| `image_2/` | Synchronized RGB camera frame (VoD / KITTI camera folder name) |
| `radar/` | Accumulated VoD point cloud for that sample (`1`, `3`, or `5` scans) |
| `radar_raw/` | Current-frame raw DCA1000 `.bin` |
| `calib/` | Camera–radar calibration text (`00000.txt`, `00001.txt`, …) |
| `manifest.csv` | Per-sample index: timing, sources, `sample_id`, point count |

| Naming rule | Example (first sample) |
|-------------|------------------------|
| Shared five-digit stem (same as calib) | `00000` |
| Camera | `image_2/00000.png` |
| VoD cloud | `radar/00000.bin` |
| Raw radar | `radar_raw/00000.bin` |
| Calibration | `calib/00000.txt` |
| Manifest `sequence` | `1` (1-based); `sample_id` = `00000` |

`1_scan` / `3_scan` / `5_scan` share the same current RGB + calib; only the radar accumulation window changes.

## How this fits FusionApp

| Step | What happens |
|------|----------------|
| 1. Live recording | FusionApp UI / DCA1000 + RealSense stores timestamped raw radar bins and RGB frames |
| 2. Synchronization | Latest readable camera image at or before each radar timestamp (`organize_recording_data` rule) |
| 3. Offline conversion | This folder’s converter runs the FusionApp analyser and writes VoD-compatible `data/` packs |
| 4. Annotation / training | Tools consume `image_2`, `radar`, and `calib` with shared sample ids |

---

## Pipeline

How FusionApp recording becomes VoD-style `data/{1,3,5}_scan` packs. Diagrams are the PNG counterparts of the vector figures in `doc_14_july/figures`, stored under [`figures/`](./figures/) so they render on GitHub.

| Step | Diagram | Brief explanation |
|------|---------|-------------------|
| 1 | [`fig01_real_architecture.png`](./figures/fig01_real_architecture.png) | Live RGB + radar capture → sync → conversion → annotation outputs |
| 2 | [`fig06_fusionapp_synchronization.png`](./figures/fig06_fusionapp_synchronization.png) | Radar-led RGB pairing when each radar frame is saved |
| 3 | [`fig03_real_conversion_vector.png`](./figures/fig03_real_conversion_vector.png) | Raw `.bin` → analyser → `1` / `3` / `5` scan packs + manifest |
| 4 | [`fig02_real_pairing.png`](./figures/fig02_real_pairing.png) | Synchronized RGB / BEV radar and the annotation folder contract |
| 5 | [`fig04_real_velocity_compensation.png`](./figures/fig04_real_velocity_compensation.png) | Ego-motion handling for denser multi-scan clouds (`--is-moving`) |
| 6 | [`fig05_speed_compensation_mechanism.png`](./figures/fig05_speed_compensation_mechanism.png) | Bearing, ego projection, residual Doppler → `velocities_comp` |

### 1. End-to-end architecture

From live D455 RGB + AWR2243 radar capture through synchronization, offline conversion, and annotation-ready outputs.

<p align="center">
  <img src="./figures/fig01_real_architecture.png" alt="FusionApp end-to-end architecture" width="100%">
</p>

### 2. FusionApp synchronization

Radar-led pairing inside FusionApp: when a radar frame is saved, the corrected RGB buffer supplies the matching camera image for that sample.

<p align="center">
  <img src="./figures/fig06_fusionapp_synchronization.png" alt="FusionApp radar-led RGB synchronization" width="100%">
</p>

### 3. Raw radar to VoD conversion

`convert_raw_bin_to_vod.py` turns raw `.bin` frames into analyser detections and numbered `1` / `3` / `5` scan folders with a shared manifest.

<p align="center">
  <img src="./figures/fig03_real_conversion_vector.png" alt="Raw radar to VoD conversion flow (vector)" width="100%">
</p>

### 4. Pairing and annotation layout

Synchronized RGB and bird’s-eye radar view, and how calibrated footprints map onto the annotation contract (`image_2`, `radar`, `calib`).

<p align="center">
  <img src="./figures/fig02_real_pairing.png" alt="RGB and radar pairing for annotation" width="100%">
</p>

### 5. Velocity compensation on a moving platform

Why multi-scan stacks need ego-motion handling: compensated velocity and accumulation produce denser, consistent clouds when using `--is-moving`.

<p align="center">
  <img src="./figures/fig04_real_velocity_compensation.png" alt="Velocity compensation for multi-scan stacks" width="100%">
</p>

### 6. Speed-compensation mechanism

Bearing, ego-velocity projection, and residual Doppler used to update `velocities_comp` during conversion with `--is-moving`.

<p align="center">
  <img src="./figures/fig05_speed_compensation_mechanism.png" alt="Speed-compensation mechanism" width="100%">
</p>

---

## Related scripts

**Rebuild scans from existing VoD clouds** (no MUSIC re-run):

```powershell
python vod_conversion\rebuild_scans_from_vod.py `
  "E:\Twowheelers_18_5\recordings\YOUR_RECORDING_FOLDER" `
  --is-moving `
  --overwrite
```

**Export sequence ranges from `5_scan`:**

```powershell
python vod_conversion\export_5_scan_ranges.py `
  "E:\Twowheelers_18_5\recordings\YOUR_RECORDING_FOLDER"
```

| Optional flag | Effect |
|---------------|--------|
| `--save-vod-pc` | Also write per-frame `vod_pc/*_pc.bin` |
| `--save-previews` | Also write PC-2D SNR / raw-power PNGs |
| `--save-range-doppler` | Also write range-Doppler matrices and grid PNGs |
| `--save-scan-comparison` | Also write 1/3/5 scan comparison grids |
| `--overwrite` | Replace existing managed outputs |

## Calibration template

| Setting | Value |
|---------|-------|
| Default calib file | `FusionApp/config_files/camera_radar_calib.txt` |
| Override | `--calib-template <path>` |
