# Road Safety on Two Wheelers - FusionApp

![Sensing Box Overview](sensing_box.png)

## Overview
This application provides real-time object detection for road safety, specifically designed for two-wheeled vehicles. The system can detect cars, persons, and other objects to enhance rider safety.

## Sample Results

### Car Detection
| ![Car Detection 1](images/cars_1.png) | ![Car Detection 2](images/cars_2.png) | ![Car Detection 3](images/cars_3.png) |
|:---:|:---:|:---:|

### Person Detection
| ![Person Detection 1](images/persons_1.png) | ![Person Detection 2](images/persons_2.png) | ![Person Detection 3](images/persons_3.png) |
|:---:|:---:|:---:|

## Features
- Real-time object detection for cars and pedestrians
- Works in various lighting conditions
- Alert system for potential collisions
- Compact sensing box design for two-wheeled vehicles

## Installation

### Prerequisites

- **Python**: 3.10 (tested) - newer versions (3.11+) expected to work but untested
- **GPU**: NVIDIA GPU with CUDA 12.1+ support (recommended for 30fps live processing)
  - **With GPU**: Full camera+radar fusion supported
  - **Without GPU**: Radar-only mode recommended (CPU-only video analysis not supported)
  - **Note**: Other GPU models and CUDA versions are acceptable if they properly support GPU acceleration of the video processing pipeline
- **Camera**: Intel RealSense D455 (optional, for camera fusion)
- **OS**: Windows 11 (Ubuntu x64 and NVIDIA Ubuntu support coming soon)

### Installation Steps

#### 1. Update Python Package Tools

```bash
python -m pip install --upgrade pip setuptools
```

#### 2. Install PyTorch

Choose the appropriate command based on your hardware:

**For CUDA-compatible GPU** (recommended for 30fps processing):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU-only systems** (radar-only mode):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> **Note**: CPU-only video analysis is not recommended. Use radar-only mode if GPU acceleration is unavailable.

#### 3. Build and Install fpga_udp

1. Clone and build the fpga_udp module:
   ```bash
   git clone https://github.com/gaoweifan/pyRadar.git
   cd pyRadar
   # Follow the build instructions in the repository
   ```

2. Install the built module according to the repository instructions.


#### 4. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### Troubleshooting

- **UDP timeout errors**: Usually caused by installing `fpga_udp` after other packages. Reinstall `fpga_udp` first in a fresh environment.
- **Performance issues**: Ensure GPU acceleration works.

## Usage

The application provides a **web-based interface** with two main modes: **Live** processing and **Replay** analysis.

### Starting the Fusion Server

Launch the web server to access the fusion application:

```bash
python fusion_server.py --host 127.0.0.1 --port 8081
```

Then open your browser and navigate to: `http://127.0.0.1:8081`

#### Command Line Arguments

- `--host`: Server host address (default: `127.0.0.1`)
- `--port`: Server port (default: `8081`)
- `--default-replay-path`: Default directory for replay mode (optional)
- `--default-replay-cfg`: Default radar config file for replay mode (optional)

**Example with replay defaults:**
```bash
python fusion_server.py --host 127.0.0.1 --port 8081 \
    --default-replay-path C:/recordings/2025_01_15/session_001 \
    --default-replay-cfg C:/recordings/2025_01_15/AWR2243_180m_70cm_64_3_512.txt
```

---

### Live Mode

Real-time synchronization, recording, and analysis of camera and radar data.

**Features:**
- Real-time object detection and tracking (person, car, bicycle, motorcycle, bus, truck)
- Simultaneous recording of radar and camera data
- Live visualization with 2D displays (video and radar heatmap)
- Automatic data synchronization
- Radar-only mode option (toggle in web interface)
- Configurable radar settings (range, resolution)
- Process status monitoring

**Controls:**
1. Select **Live** mode in the web interface
2. Choose radar configuration (e.g., "3D - 180m")
3. Enable/disable "Radar only" mode
4. Click **Start System** to initialize hardware
5. Click **Start Recording** to capture data
6. Adjust visualization frame rates (1-30 fps)

---

### Replay Mode

Synchronized replay and analysis of previously recorded data.

**Features:**
- Synchronized playback of recorded sessions
- Interactive playback controls (play, pause, step)
- Frame-by-frame analysis
- Support for custom radar configurations
- Video and radar visualization
- Adjustable playback speed

**Controls:**
1. Select **Replay** mode in the web interface
2. Enter recording directory path
3. Enter radar config file path
4. Enable/disable "Radar only" mode
5. Click **Start System** to load data
6. Use playback controls to navigate through recorded data
7. Adjust visualization frame rates (1-30 fps)

---

### System Requirements for Optimal Performance

- **Live Mode (30fps)**: NVIDIA GPU with CUDA support required for camera analysis
- **Live Mode (Radar-Only)**: Can run on CPU-only systems
- **Replay Mode**: GPU recommended but not required (video analysis disabled without CUDA)

---

## Additional Applications

The FusionApp includes several standalone applications for offline processing and analysis:

### 1. K-Radar Frame Processor (`process_radar_kradar.py`)

Process individual radar frames through the K-Radar pipeline to generate sparse point clouds and visualizations.

**Purpose**: Batch processing of radar data with full analysis artifacts (point clouds, heatmaps, tesseract cubes).

**Usage:**
```bash
python process_radar_kradar.py --bin-file <frame.bin> --config-file <config.txt> [options]
```

**Required Arguments:**
- `--bin-file`, `-b`: Path to radar frame file (.bin)
- `--config-file`, `-c`: Path to radar configuration file (.txt)

**Optional Arguments:**
- `--output-dir`, `-o`: Output directory for artifacts (default: same as input file)
- `--pipeline-config`: YAML pipeline configuration (default: `configs/default.yaml`)
- `--set`: Override config parameters (repeatable, e.g., `--set point_cloud.roi.x=[0,50]`)
- `--az-range MIN MAX`: Override azimuth range in degrees
- `--el-range MIN MAX`: Override elevation range in degrees

**Outputs:**
- Sparse point cloud (`.npy`)
- Point cloud visualizations (`.png`: XY and XZ views)
- Heatmap visualizations (`.png`: Range-Azimuth and Range-Elevation)
- ZYX cube (`.mat`)
- Tesseract 4D tensor (`.mat`)

**Example:**
```bash
python process_radar_kradar.py \
    --bin-file data/1234567890_12345_000000000001.bin \
    --config-file config_files/AWR2243_87m_17cm_64_3_256.txt \
    --output-dir results/
```

---

### 2. K-Radar Pretrained Inference (`process_radar_kradar_pretrained.py`)

Process radar data using pretrained K-Radar deep learning models for 3D object detection.

**Purpose**: Apply trained neural networks to radar frames for automatic object detection and classification.

**Usage:**
```bash
python process_radar_kradar_pretrained.py --config <model.yml> --checkpoint <weights.pt> \
    (--bin-file <frame.bin> | --bin-path <directory>) --config-file <config.txt> [options]
```

**Required Arguments:**
- `--config`: Path to model YAML config (e.g., `cp_KRADAR/configs/cfg_RTNH_wide.yml`)
- `--checkpoint`: Path to pretrained checkpoint (`.pt`)
- `--bin-file`, `-b`: Path to single .bin radar frame file (mutually exclusive with `--bin-path`)
- `--bin-path`, `-p`: Path to directory containing multiple .bin files (mutually exclusive with `--bin-file`)
- `--config-file`, `-c`: Path to radar configuration file (.txt)

**Optional Arguments:**
- `--conf-thr`: Confidence threshold for detections (default: model-specific)
- `--pipeline-config`: YAML pipeline configuration (default: `configs/default.yaml`)
- `--set`: Override pipeline config entries (repeatable)
- `--az-range MIN MAX`: Override azimuth range in degrees
- `--el-range MIN MAX`: Override elevation range in degrees
- `--output-json`: Path to save detection results as JSON
- `--no-artifacts`: Skip saving intermediate artifacts (point clouds, heatmaps)
- `--save-mat`: Save tesseract and zyx .mat files
- `--visualize`: Generate 3D visualization of point cloud with bounding boxes
- `--interactive`: Show interactive 3D viewer (blocks until closed, requires `--visualize`)
- `--view-angle {top,side,perspective,all}`: Camera view for saved images (default: perspective)

**Outputs:**
- Detection results (console and optional JSON)
- Bounding box visualizations (`.png`)
- Sparse point clouds (`.npy`, unless `--no-artifacts`)
- Heatmap visualizations (`.png`, unless `--no-artifacts`)
- 3D visualizations (`.png`, if `--visualize` enabled)
- MAT files (`.mat`, if `--save-mat` enabled)

**Example:**
```bash
python process_radar_kradar_pretrained.py \
    --config cp_KRADAR/configs/cfg_RTNH_wide.yml \
    --checkpoint checkpoints/rtnh_wide.pt \
    --bin-path data/session_001/ \
    --config-file config_files/AWR2243_180m_70cm_64_3_512.txt \
    --visualize --interactive
```

---

### 3. Detection Video Creator (`create_detection_video.py`)

Create synchronized videos combining camera frames with radar detection visualizations.

**Purpose**: Generate side-by-side videos showing camera view and perspective radar detection overlays for presentations or analysis.

**Usage:**
```bash
python create_detection_video.py --input-dir <directory> --output <video.mp4> [options]
```

**Required Arguments:**
- `--input-dir`: Directory containing camera frames and perspective detection PNGs
- `--output`: Path for output video file (e.g., `output.mp4`)

**Optional Arguments:**
- `--fps`: Frames per second for output video (default: 10)
- `--no-labels`: Don't add text labels to frames

**File Naming Convention:**
- Camera frames: `{timestamp_int}_{timestamp_frac}_{frame_number}.png`
  - Example: `0000000412_08863_000000000000.png`
- Perspective frames: `{timestamp_int}_{timestamp_frac}_{frame_number}_detection_vis_perspective.png`
  - Example: `0000000412_08863_000000000000_detection_vis_perspective.png`

**Outputs:**
- MP4 video with side-by-side camera and detection views
- Synchronized by timestamp
- Automatic frame interpolation for missing frames

**Example:**
```bash
python create_detection_video.py \
    --input-dir recordings/2025_01_15_10_30_00/ \
    --output session_001_detections.mp4 \
    --fps 15
```

---

## Configuration Files

### Pipeline Configurations (`configs/`)

- **`default.yaml`**: Standard processing pipeline with full analysis
  - Used by: `process_radar_kradar.py`, `process_radar_kradar_pretrained.py`
  - Features: Full CFAR detection, ROI filtering, power normalization

### Radar Hardware Configurations (`config_files/`)

Various AWR2243 radar configurations for different range/resolution tradeoffs:
- `AWR2243_10m_4cm_64_3_256.txt`: Short range (10m), high resolution (4cm)
- `AWR2243_87m_17cm_64_3_256.txt`: Medium range (87m), medium resolution (17cm)
- `AWR2243_180m_70cm_64_3_512.txt`: Long range (180m), lower resolution (70cm)

Select configuration based on application requirements (urban vs. highway scenarios).
