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

The application provides two main fusion modes: **Live** processing and **Replay** analysis.

### Live Fusion (`fusion_live.py`)

Real-time synchronization, recording, and analysis of camera and radar data.

#### Full Camera + Radar Fusion
```bash
python fusion_live.py
```

#### Radar-Only Mode
```bash
python fusion_live.py --radar-only
```

**Features:**
- Real-time object detection and tracking
- Simultaneous recording of radar and camera data
- Live visualization with 2D displays
- Automatic data synchronization

---

### Replay Fusion (`fusion_replay.py`)

Synchronized replay and analysis of previously recorded data.

#### Full Camera + Radar Replay
```bash
python fusion_replay.py python fusion_replay.py --file-path /path/to/data --config-file /path/to/config.txt
```

#### Radar-Only Replay
```bash
python fusion_replay.py python fusion_replay.py --file-path /path/to/data --config-file /path/to/config.txt --radar-only
```

**Features:**
- Synchronized playback of recorded sessions
- Interactive playback controls (play, pause, seek)
- Timeline navigation
- Frame-by-frame analysis
- Support for custom radar configurations

### Command Line Arguments

#### `fusion_replay.py` Arguments
- `--file-path` (required): Path to recorded radar data directory
- `--config-file` (optional): Path to radar configuration file
- `--radar-only`: Run in radar-only mode (no camera replay)

### System Requirements for Optimal Performance

- **30fps Live Processing**: NVIDIA GPU with CUDA support
- **Radar-Only Mode**: Can run on CPU-only systems
