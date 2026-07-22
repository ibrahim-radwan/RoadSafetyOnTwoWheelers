# Road Safety on Two Wheelers

Public repository for the road safety on two-wheelers research project at the University of Canberra (2025–2027).

<table>
  <tr>
    <td align="center" valign="bottom" width="50%">
      <a href="https://github.com/ibrahim-radwan/RoadSafetyOnTwoWheelers/tree/main/FusionApp">
        <img src="FusionApp/images/about_scooter.png" alt="Instrumented two-wheeler sensing setup" width="100%" height="320"/>
      </a>
      <br/>
      <a href="https://github.com/ibrahim-radwan/RoadSafetyOnTwoWheelers/tree/main/FusionApp"><b>FusionApp</b></a>
    </td>
    <td align="center" valign="bottom" width="50%">
      <a href="https://github.com/ibrahim-radwan/RoadSafetyOnTwoWheelers/tree/main/FusionApp/vod_conversion">
        <img src="FusionApp/images/vod_architecture.png" alt="VoD conversion architecture" width="100%" height="320"/>
      </a>
      <br/>
      <a href="https://github.com/ibrahim-radwan/RoadSafetyOnTwoWheelers/tree/main/FusionApp/vod_conversion"><b>VoD conversion</b></a>
    </td>
  </tr>
  <tr>
    <td align="center" valign="bottom" width="50%">
      <a href="https://github.com/ibrahim-radwan/two-wheeler-radar-camera-annotation-tool">
        <img src="FusionApp/images/annotation_interface.png" alt="Radar–camera annotation interface" width="100%" height="320"/>
      </a>
      <br/>
      <a href="https://github.com/ibrahim-radwan/two-wheeler-radar-camera-annotation-tool"><b>Annotation tool</b></a>
    </td>
    <td align="center" valign="bottom" width="50%">
      <a href="https://github.com/ibrahim-radwan/twowheeler-safety-dashboard/tree/main">
        <img src="FusionApp/images/dashboard_overview.png" alt="Two-wheeler safety dashboard overview" width="100%" height="320"/>
      </a>
      <br/>
      <a href="https://github.com/ibrahim-radwan/twowheeler-safety-dashboard/tree/main"><b>Safety dashboard</b></a>
    </td>
  </tr>
</table>

## Components

| Component | Repository / path | Role |
|-----------|-------------------|------|
| **FusionApp** | [FusionApp](https://github.com/ibrahim-radwan/RoadSafetyOnTwoWheelers/tree/main/FusionApp) | Live and replay fusion of Intel RealSense D455 camera and TI AWR2243 / DCA1000 radar (web UI via `fusion_server.py`) |
| **VoD conversion** | [FusionApp/vod_conversion](https://github.com/ibrahim-radwan/RoadSafetyOnTwoWheelers/tree/main/FusionApp/vod_conversion) | Offline conversion of raw radar `.bin` + RGB `.png` into [View-of-Delft (VoD)](https://intelligent-vehicles.org/datasets/view-of-delft/) / [KITTI](https://www.cvlibs.net/datasets/kitti/)-style `data/{1,3,5}_scan` packs |
| **Annotation tool** | [two-wheeler-radar-camera-annotation-tool](https://github.com/ibrahim-radwan/two-wheeler-radar-camera-annotation-tool) | Labelling of synchronised radar–camera frames |
| **Safety dashboard** | [twowheeler-safety-dashboard](https://github.com/ibrahim-radwan/twowheeler-safety-dashboard/tree/main) | Web dashboard for project / dataset visualisation |

## Project overview

This project develops sensing and software for **road safety on two-wheeled vehicles**. Work in this repository centres on **FusionApp**: camera + mmWave radar capture, live fusion, and recording. Offline **VoD conversion** turns those recordings into packs compatible with the [View-of-Delft (VoD)](https://intelligent-vehicles.org/datasets/view-of-delft/) layout ([documentation / development kit](https://tudelft-iv.github.io/view-of-delft-dataset/)), which follows [KITTI](https://www.cvlibs.net/datasets/kitti/)-style folders and naming (`image_2`, five-digit stems, shared calib ids).

Labelling is done with the [radar–camera annotation tool](https://github.com/ibrahim-radwan/two-wheeler-radar-camera-annotation-tool). Results and datasets can be viewed in the [two-wheeler safety dashboard](https://github.com/ibrahim-radwan/twowheeler-safety-dashboard/tree/main).

## Features

End-to-end pipeline, from hardware to dashboard:

1. **Hardware setup** — Compact two-wheeler sensing box with Intel RealSense D455 (RGB) and TI AWR2243 radar via DCA1000; FusionApp talks to the radar through `fpga_udp` (see [FusionApp](https://github.com/ibrahim-radwan/RoadSafetyOnTwoWheelers/tree/main/FusionApp)).
2. **Live sensing & recording** — [FusionApp](https://github.com/ibrahim-radwan/RoadSafetyOnTwoWheelers/tree/main/FusionApp) web UI (`fusion_server.py`) for live camera + radar, optional GPU camera detection, and paired radar–camera recording with a shared capture clock.
3. **Replay** — Same UI replays recorded sessions for offline review without hardware.
4. **VoD conversion** — [`vod_conversion`](https://github.com/ibrahim-radwan/RoadSafetyOnTwoWheelers/tree/main/FusionApp/vod_conversion) converts raw `.bin` + `.png` into VoD / KITTI-style `1_scan` / `3_scan` / `5_scan` packs (`image_2`, `radar`, `radar_raw`, `calib`); optional `--is-moving` ego-motion compensation. Dataset references: [VoD](https://intelligent-vehicles.org/datasets/view-of-delft/), [KITTI](https://www.cvlibs.net/datasets/kitti/).
5. **Annotation** — Label synchronised radar–camera samples with the [two-wheeler radar–camera annotation tool](https://github.com/ibrahim-radwan/two-wheeler-radar-camera-annotation-tool).
6. **Dashboard** — Inspect and present project / safety outputs in the [two-wheeler safety dashboard](https://github.com/ibrahim-radwan/twowheeler-safety-dashboard/tree/main).

## Installation

See [FusionApp/README.md](FusionApp/README.md) for Python, CUDA/Jetson, `fpga_udp`, and dependency setup.

## Usage

```bash
cd FusionApp
python fusion_server.py --host 127.0.0.1 --port 8081
```

Open `http://127.0.0.1:8081`. For offline VoD packs, see [FusionApp/vod_conversion/README.md](FusionApp/vod_conversion/README.md).

## Research papers

**Publications:**
1. Hassanin, M., Alsheikh, M.A., Kuhn, C.C., Herath, D., Hoang, D.T. and Radwan, I., 2025. Towards Autonomous Riding: A Review of Perception, Planning, and Control in Intelligent Two-Wheelers. arXiv preprint arXiv:2507.11852.
   - https://arxiv.org/pdf/2507.11852
2. Deng, W., Tu, W., Radwan, I., Alsheikh, M.A., Gould, S. and Zheng, L., 2025. Confidence and Dispersity as Signals: Unsupervised Model Evaluation and Ranking. arXiv preprint arXiv:2510.02956.
   - https://arxiv.org/pdf/2510.02956

## Team members

<table>
  <tr>
    <td width="33%" valign="top">

- Ibrahim Radwan
- Javad Amiri
- Mohammed Hassanin

    </td>
    <td width="33%" valign="top">

- Mohammad Abu Alsheikh
- Carlos C. N. Kuhn
- Damith Herath

    </td>
    <td width="33%" valign="top">

- Dinh Thai Hoang
- Weijian Deng
- Wael Issa

    </td>
  </tr>
</table>
