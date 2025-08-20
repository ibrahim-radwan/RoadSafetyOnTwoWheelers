"""
Unified PyQt5-based fusion visualizer for both live and replay modes.
Supports radar visualization, camera display, and adaptive controls.
"""

import os
import time
import glob
import re
import logging
import traceback
import queue
import numpy as np

# Fix OpenCV Qt plugin conflicts before importing cv2 - only on Linux
import pyqtgraph as pg

if os.name != "nt":  # Not Windows
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    # Prefer software rendering on NVIDIA Jetson to avoid GLX/EGL issues
    if os.path.exists("/etc/nv_tegra_release"):
        os.environ.setdefault("QT_OPENGL", "software")
import cv2
from collections import deque

# Configure pyqtgraph defaults; force software to avoid GLX/EGL issues on Jetson
pg.setConfigOptions(antialias=False, imageAxisOrder="row-major", useOpenGL=False)

from typing import Optional, Callable, Dict, Any, List

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QSlider,
    QProgressBar,
    QGridLayout,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt, QRectF
from PyQt5.QtGui import QOpenGLContext, QSurfaceFormat, QOffscreenSurface

# Matplotlib removed from visualizer rendering; pyqtgraph is used for speed

from config_params import CFGS
from utils import setup_logger
from camera.d455 import D455Frame
from multiprocessing import shared_memory


class FusionVisualizer(QWidget):
    """
    Unified PyQt5-based fusion visualizer for both live and replay modes.
    Displays radar heatmaps, point clouds, and camera feeds with adaptive controls.
    """

    def __init__(
        self,
        mode: str = "live",
        stop_event=None,
        recording_dir: Optional[str] = None,
        adc_params=None,
        use_3d: bool = False,
    ):
        super().__init__()
        self.logger = setup_logger("FusionVisualizer")
        self.mode = mode
        self.stop_event = stop_event
        self.recording_dir = recording_dir
        self.adc_params = adc_params
        self.use_3d = use_3d
        # Feature gates from environment to isolate crashes
        self._disable_radar = os.environ.get("DISABLE_RADAR", "0") == "1"
        self._disable_camera = os.environ.get("DISABLE_CAMERA", "0") == "1"

        # Data access callbacks
        self.radar_data_callback: Optional[Callable] = None
        self.camera_data_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None

        # Control callbacks
        self.control_callback: Optional[Callable] = None
        self.record_callback: Optional[Callable] = None

        # Internal state
        self._fatal_error_occurred = False
        self._current_fusion_data = None
        self._current_status = None

        # PNG video support for replay mode (legacy - may not be needed with unified display)
        self._png_files = []
        self._current_png_image = None
        if mode == "replay" and recording_dir:
            self._scan_png_files()

        # Performance tracking
        self._last_log_time = time.time()
        self._plot_updates = 0

        # Visualizer profiling accumulators
        self._profile_enabled = True
        self._profile_step = 150  # reduced frequency (5x previous 30)
        self._profile_count = 0
        self._prof_sum_total = 0.0
        self._prof_sum_get_data = 0.0
        self._prof_sum_status = 0.0
        self._prof_sum_camera = 0.0
        self._prof_sum_rd = 0.0
        self._prof_sum_ra = 0.0
        self._prof_sum_pc = 0.0

        # Setup UI
        self._setup_ui()

        # Start update timer at ~50 FPS (20ms)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(20)

        self.logger.info(f"FusionVisualizer initialized in {mode} mode")

        # Track last processed timestamps to avoid redundant updates
        self._last_radar_ts = None
        self._last_camera_ts = None
        # FPS tracking over a 1-second window
        self._fps_window_s = 1.0
        self._cam_times = deque()
        self._rad_times = deque()
        self._cam_fps_text = ""
        # Attach to analyser results SHM if engine passes meta via environment (simple handoff)
        self._rd_blocks = []
        self._rd_shape = None
        self._rd_dtype = None
        self._ra_blocks = []
        self._ra_shape = None
        self._ra_dtype = None
        self._res_shm_attached = False

    def _setup_ui(self):
        """Setup the complete user interface"""
        # Set window properties
        title = (
            "Fusion Live Visualization"
            if self.mode == "live"
            else "Fusion Recording Playback"
        )
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 1400, 1000)

        # Create main layout
        main_layout = QVBoxLayout()

        # Choose rendering backend and then setup plots and widgets
        self._configure_pyqtgraph_backend()
        self._setup_plots()

        # Create 2x2 grid layout for all modes
        plots_layout = QGridLayout()
        plots_layout.addWidget(self.camera_widget, 0, 0)  # Upper-left: Camera/Video
        plots_layout.addWidget(
            self.point_cloud_widget, 0, 1
        )  # Upper-right: Point cloud
        plots_layout.addWidget(
            self.range_doppler_widget, 1, 0
        )  # Lower-left: Range-Doppler
        plots_layout.addWidget(
            self.range_azimuth_widget, 1, 1
        )  # Lower-right: Range-Azimuth

        # Set equal row and column stretch factors for 50/50 split
        plots_layout.setRowStretch(0, 1)
        plots_layout.setRowStretch(1, 1)
        plots_layout.setColumnStretch(0, 1)
        plots_layout.setColumnStretch(1, 1)

        # Add plots layout with stretch factor so it takes most of the space
        main_layout.addLayout(plots_layout, 1)

        # Setup controls
        self._setup_controls()
        main_layout.addLayout(self.controls_layout)

        # Finalize
        self.setLayout(main_layout)

    def _configure_pyqtgraph_backend(self) -> None:
        """Force software rendering; do not attempt any OpenGL context creation."""
        try:
            pg.setConfigOptions(useOpenGL=False)
            if hasattr(self, "logger"):
                self.logger.warning("pyqtgraph OpenGL not available; using software rendering")
        except Exception:
            pass

    def _setup_plots(self):
        """Setup camera display and pyqtgraph plots"""
        # Camera/Video widget - always present
        self.camera_widget = QLabel()
        self.camera_widget.setMinimumSize(400, 400)
        self.camera_widget.setStyleSheet("border: 1px solid gray;")
        self.camera_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_widget.setText("No video stream available")
        self.camera_widget.setScaledContents(True)

        # Calculate max range for point cloud axis limits
        if self.adc_params:
            try:
                self.pc_max_range = self.adc_params.max_range
                self.logger.info(
                    f"Using max range from ADC params: {self.pc_max_range} m"
                )
            except AttributeError:
                self.pc_max_range = 10.0
                self.logger.warning(
                    "ADC params don't have max_range attribute, using default: 10.0 m"
                )
        else:
            self.pc_max_range = 10.0
            self.logger.warning(
                "No ADC params provided, using default max range: 10.0 m"
            )

        # Precompute LUT similar to 'jet'
        try:
            self._lut_jet = pg.colormap.get("jet").getLookupTable(0.0, 1.0, 256)
        except Exception:
            self._lut_jet = None

        # Range-Doppler plot (pyqtgraph)
        self.range_doppler_widget = pg.GraphicsLayoutWidget()
        self.rd_plot = self.range_doppler_widget.addPlot(title="Range-Doppler Heatmap")
        self.rd_plot.setLabel("bottom", "Doppler (m/s)")
        self.rd_plot.setLabel("left", "Range (m)")
        self.rd_image = pg.ImageItem()
        self.rd_plot.addItem(self.rd_image)
        self.rd_plot.invertY(False)
        # FPS text overlay for radar (top-left)
        self._rd_fps_item = pg.TextItem(color="w", anchor=(0, 1))
        self.rd_plot.addItem(self._rd_fps_item)

        # Range-Azimuth plot (pyqtgraph)
        self.range_azimuth_widget = pg.GraphicsLayoutWidget()
        self.ra_plot = self.range_azimuth_widget.addPlot(title="Range-Azimuth Heatmap")
        self.ra_plot.setLabel("bottom", "Azimuth (degrees)")
        self.ra_plot.setLabel("left", "Range (m)")
        self.ra_image = pg.ImageItem()
        self.ra_plot.addItem(self.ra_image)
        self.ra_plot.invertY(False)

        # Point Cloud plot (2D, pyqtgraph)
        self.point_cloud_widget = pg.GraphicsLayoutWidget()
        title = "3D Point Cloud" if self.use_3d else "2D Point Cloud"
        self.pc_plot = self.point_cloud_widget.addPlot(title=title)
        self.pc_plot.setLabel("bottom", "X (m)")
        self.pc_plot.setLabel("left", "Y (m)")
        self.pc_scatter_item = pg.ScatterPlotItem(pen=None, size=4)
        self.pc_plot.addItem(self.pc_scatter_item)
        self.pc_plot.setXRange(-self.pc_max_range, self.pc_max_range, padding=0)
        self.pc_plot.setYRange(0, self.pc_max_range, padding=0)

    def _setup_controls(self):
        """Setup control buttons and status display"""
        self.controls_layout = QHBoxLayout()

        # Set a fixed height for the controls layout to prevent vertical expansion
        self.controls_layout.setSpacing(10)
        self.controls_layout.setContentsMargins(10, 5, 10, 5)

        # Playback controls
        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.setFixedHeight(35)
        self.play_pause_btn.clicked.connect(self._on_play_pause)
        self.controls_layout.addWidget(self.play_pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFixedHeight(35)
        self.stop_btn.clicked.connect(self._on_stop)
        self.controls_layout.addWidget(self.stop_btn)

        # Record control
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setFixedHeight(35)
        self.record_btn.clicked.connect(self._on_record)
        self.controls_layout.addWidget(self.record_btn)

        # Seek slider
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setFixedHeight(35)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(100)
        self.seek_slider.setValue(0)
        self.seek_slider.sliderReleased.connect(self._on_seek)
        self.controls_layout.addWidget(self.seek_slider)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(35)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.controls_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setFixedHeight(35)
        self.controls_layout.addWidget(self.status_label)

        # Configure controls based on mode
        self._configure_controls_for_mode()

        # Internal state
        self.is_playing = False
        self.is_recording = False

    def _skip_rdra(self) -> bool:
        # Allow disabling RD/RA rendering to isolate crashes
        return os.environ.get("DISABLE_RDRA", "0") == "1"

    def _safe_levels(self, arr: np.ndarray):
        try:
            finite = np.isfinite(arr)
            if not np.any(finite):
                return None
            vals = arr[finite]
            # Use robust percentiles to avoid outliers
            vmin = float(np.percentile(vals, 1.0))
            vmax = float(np.percentile(vals, 99.0))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                vmin = float(np.min(vals))
                vmax = float(np.max(vals))
                if vmin >= vmax:
                    vmax = vmin + 1.0
            return (vmin, vmax)
        except Exception:
            return None

    def _configure_controls_for_mode(self):
        """Enable/disable controls based on mode"""
        if self.mode == "live":
            # Live mode: disable playback controls, enable record
            self.play_pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.seek_slider.setEnabled(False)
            self.record_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
        else:
            # Replay mode: enable playback controls, disable record
            self.play_pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.seek_slider.setEnabled(True)
            self.record_btn.setEnabled(False)
            self.progress_bar.setVisible(True)

            # Note: Auto-start disabled - user must manually press play
            # QTimer.singleShot(2000, self._auto_start_playback)

    def _auto_start_playback(self):
        """Auto-start playback for replay mode"""
        if self.mode == "replay" and self.control_callback and not self.is_playing:
            self.logger.info("Auto-starting playback...")
            self._on_play_pause()

    # Control event handlers
    def _on_play_pause(self):
        """Handle play/pause button"""
        if self.control_callback:
            if self.is_playing:
                self.control_callback("pause")
                self.play_pause_btn.setText("Play")
                self.is_playing = False
                self.status_label.setText("Paused")
            else:
                self.control_callback("play")
                self.play_pause_btn.setText("Pause")
                self.is_playing = True
                self.status_label.setText("Playing")

    def _on_stop(self):
        """Handle stop button"""
        if self.control_callback:
            self.control_callback("stop")
            # Immediately update GUI state without waiting for status updates
            self.play_pause_btn.setText("Play")
            self.is_playing = False
            self.seek_slider.setValue(0)
            self.progress_bar.setValue(0)
            self.status_label.setText("Stopped")

            # Force an immediate GUI update
            QApplication.processEvents()

    def _on_record(self):
        """Handle record button (placeholder for now)"""
        if self.record_callback:
            if self.is_recording:
                self.record_callback("stop_recording")
                self.record_btn.setText("Start Recording")
                self.is_recording = False
                self.status_label.setText("Recording stopped")
            else:
                self.record_callback("start_recording")
                self.record_btn.setText("Stop Recording")
                self.is_recording = True
                self.status_label.setText("Recording...")
        else:
            # Placeholder behavior
            if self.is_recording:
                self.record_btn.setText("Start Recording")
                self.is_recording = False
                self.status_label.setText("Recording stopped (placeholder)")
            else:
                self.record_btn.setText("Stop Recording")
                self.is_recording = True
                self.status_label.setText("Recording... (placeholder)")

    def _on_seek(self):
        """Handle seek slider"""
        if self.control_callback:
            frame_number = self.seek_slider.value()
            self.control_callback(f"seek:{frame_number}")
            self.progress_bar.setValue(frame_number)
            self.status_label.setText(f"Seeking to frame {frame_number}")

    # Callback setters
    def set_radar_data_callback(self, callback: Callable):
        """Set callback to get radar data"""
        self.radar_data_callback = callback

    def set_camera_data_callback(self, callback: Callable):
        """Set callback to get camera data"""
        self.camera_data_callback = callback

    def set_status_callback(self, callback: Callable):
        """Set callback to get status updates"""
        self.status_callback = callback

    def set_control_callback(self, callback: Callable):
        """Set callback to send control commands"""
        self.control_callback = callback

    def set_record_callback(self, callback: Callable):
        """Set callback to handle recording commands"""
        self.record_callback = callback

    def update_display(self):
        """Main update loop - called by timer"""
        if self._fatal_error_occurred:
            return

        try:
            total_t0 = time.perf_counter()

            # Get current data
            get_t0 = time.perf_counter()
            radar_data = (self.radar_data_callback() if (self.radar_data_callback and not os.environ.get("DISABLE_RADAR", "0") == "1") else None)
            camera_data = (self.camera_data_callback() if (self.camera_data_callback and not os.environ.get("DISABLE_CAMERA", "0") == "1") else None)
            t_get_data = time.perf_counter() - get_t0

            # Set extents if needed
            rd_extents = [-5, 5, 0, 10]
            ra_extents = [-60, 60, 0, 10]
            if self.adc_params:
                try:
                    rd_extents = self.adc_params.range_doppler_extents
                    ra_extents = self.adc_params.range_azimuth_extents
                except AttributeError:
                    pass

            # Update status for replay mode
            status_dt = 0.0
            if self.mode == "replay":
                st_t0 = time.perf_counter()
                self._update_playback_status()
                status_dt = time.perf_counter() - st_t0

            # Update displays only when new frames arrive
            # Camera update
            t_camera = 0.0
            cam_ts = (
                getattr(getattr(camera_data, "frame", None), "timestamp", None)
                if camera_data is not None
                else None
            )
            if not os.environ.get("DISABLE_CAMERA", "0") == "1" and (cam_ts is None or cam_ts != self._last_camera_ts):
                cam_t0 = time.perf_counter()
                self._update_camera_display(camera_data)
                t_camera = time.perf_counter() - cam_t0
                if cam_ts is not None:
                    self._last_camera_ts = cam_ts
                # Update camera FPS window and build overlay text
                now_s = time.perf_counter()
                self._cam_times.append(now_s)
                # Drop times older than window
                while (
                    self._cam_times and now_s - self._cam_times[0] > self._fps_window_s
                ):
                    self._cam_times.popleft()
                cam_fps = len(self._cam_times) / self._fps_window_s
                self._cam_fps_text = f"{cam_fps:.1f} FPS"

            # Radar update
            rdra_times = {"rd": 0.0, "ra": 0.0, "pc": 0.0}
            radar_ts = None
            new_radar_frame = False
            if isinstance(radar_data, dict):
                radar_ts = radar_data.get("frame_timestamp")
                new_radar_frame = radar_ts is None or radar_ts != self._last_radar_ts
            if radar_data is not None and new_radar_frame and not os.environ.get("DISABLE_RADAR", "0") == "1":
                # If metadata indicates results are in SHM, attach and read
                # Handle results SHM init (names/shapes)
                if isinstance(radar_data, dict) and radar_data.get(
                    "RADAR_RES_SHM_INIT"
                ):
                    try:
                        if not self._res_shm_attached:
                            rd = radar_data.get("rd")
                            ra = radar_data.get("ra")
                            if rd:
                                self._rd_blocks = [
                                    shared_memory.SharedMemory(name=n)
                                    for n in rd.get("names", [])
                                ]
                                self._rd_shape = tuple(rd.get("shape", ()))
                                self._rd_dtype = rd.get("dtype", "float32")
                            if ra:
                                self._ra_blocks = [
                                    shared_memory.SharedMemory(name=n)
                                    for n in ra.get("names", [])
                                ]
                                self._ra_shape = tuple(ra.get("shape", ()))
                                self._ra_dtype = ra.get("dtype", "float32")
                            self._res_shm_attached = True
                            self.logger.info("Attached results SHM via init meta")
                    except Exception as e:
                        self.logger.warning(f"Failed to attach RD/RA SHM via init: {e}")
                    radar_data = None  # consume init item
                # Render from SHM frame metadata
                if isinstance(radar_data, dict) and radar_data.get(
                    "RADAR_RES_SHM_FRAME"
                ):
                    try:
                        # Lazy attach on first use via environment variables set by engine
                        if (not self._res_shm_attached) and (
                            (not self._rd_blocks)
                            or (self._rd_shape is None)
                            or (self._ra_shape is None)
                        ):
                            rd_names = os.environ.get("RADAR_RD_SHM_NAMES")
                            ra_names = os.environ.get("RADAR_RA_SHM_NAMES")
                            rd_shape = os.environ.get("RADAR_RD_SHAPE")
                            ra_shape = os.environ.get("RADAR_RA_SHAPE")
                            if rd_names and rd_shape:
                                self._rd_blocks = [
                                    shared_memory.SharedMemory(name=n)
                                    for n in rd_names.split(",")
                                ]
                                self._rd_shape = tuple(
                                    int(x) for x in rd_shape.split(",")
                                )
                                self._rd_dtype = "float32"
                            if ra_names and ra_shape:
                                self._ra_blocks = [
                                    shared_memory.SharedMemory(name=n)
                                    for n in ra_names.split(",")
                                ]
                                self._ra_shape = tuple(
                                    int(x) for x in ra_shape.split(",")
                                )
                                self._ra_dtype = "float32"
                            if self._rd_blocks or self._ra_blocks:
                                self._res_shm_attached = True
                        slot = int(radar_data.get("slot", 0)) & 1
                        rd_data = None
                        ra_data = None
                        if (
                            self._rd_blocks
                            and self._rd_shape
                            and slot < len(self._rd_blocks)
                        ):
                            try:
                                expected_elems = int(np.prod(self._rd_shape))
                                expected_bytes = expected_elems * 4
                                buf = self._rd_blocks[slot].buf
                                if len(buf) >= expected_bytes:
                                    mv = memoryview(buf)[:expected_bytes]
                                    rd_view = np.frombuffer(
                                        mv, dtype=np.float32, count=expected_elems
                                    ).reshape(self._rd_shape)
                                else:
                                    rd_view = np.array([], dtype=np.float32)
                            except Exception:
                                rd_view = np.array([], dtype=np.float32)
                            rd_data = rd_view.copy()
                        if (
                            self._ra_blocks
                            and self._ra_shape
                            and slot < len(self._ra_blocks)
                        ):
                            try:
                                expected_elems = int(np.prod(self._ra_shape))
                                expected_bytes = expected_elems * 4
                                buf = self._ra_blocks[slot].buf
                                if len(buf) >= expected_bytes:
                                    mv = memoryview(buf)[:expected_bytes]
                                    ra_view = np.frombuffer(
                                        mv, dtype=np.float32, count=expected_elems
                                    ).reshape(self._ra_shape)
                                else:
                                    ra_view = np.array([], dtype=np.float32)
                            except Exception:
                                ra_view = np.array([], dtype=np.float32)
                            ra_data = ra_view.copy()
                        payload = {"range_doppler": rd_data, "range_azimuth": ra_data}
                        # Pass through point cloud if provided in metadata
                        if (
                            isinstance(radar_data, dict)
                            and radar_data.get("point_cloud") is not None
                        ):
                            payload["point_cloud"] = radar_data.get("point_cloud")
                        rdra_times = self._update_radar_displays(
                            payload, rd_extents, ra_extents
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to read results from SHM: {e}")
                        rdra_times = self._update_radar_displays(
                            radar_data, rd_extents, ra_extents
                        )
                else:
                    rdra_times = self._update_radar_displays(
                        radar_data, rd_extents, ra_extents
                    )
                if radar_ts is not None:
                    self._last_radar_ts = radar_ts
                # Update radar FPS window
                now_s = time.perf_counter()
                self._rad_times.append(now_s)
                while (
                    self._rad_times and now_s - self._rad_times[0] > self._fps_window_s
                ):
                    self._rad_times.popleft()
                rad_fps = len(self._rad_times) / self._fps_window_s
                # Position and update FPS overlay on RD plot (top-left in data coords)
                x0, x1, y0, y1 = (
                    rd_extents[0],
                    rd_extents[1],
                    rd_extents[2],
                    rd_extents[3],
                )
                self._rd_fps_item.setText(f"{rad_fps:.1f} FPS")
                self._rd_fps_item.setPos(x0 + 0.02 * (x1 - x0), y1 - 0.02 * (y1 - y0))

            t_rd = rdra_times.get("rd", 0.0)
            t_ra = rdra_times.get("ra", 0.0)
            t_pc = rdra_times.get("pc", 0.0)

            # Performance logging
            t_total = time.perf_counter() - total_t0
            self._record_profile(
                t_total=t_total,
                t_get=t_get_data,
                t_status=status_dt,
                t_camera=t_camera,
                t_rd=t_rd,
                t_ra=t_ra,
                t_pc=t_pc,
            )
            # Latency logging (only for new radar frames) using absolute monotonic ns
            if isinstance(radar_data, dict) and new_radar_frame:
                # if CFGS.LOG_LEVEL == logging.DEBUG and isinstance(radar_data, dict) and new_radar_frame:
                now_disp_ns = time.perf_counter_ns()
                cap_ns = radar_data.get("capture_monotonic_ns")
                enq_ns = radar_data.get("enqueue_monotonic_ns")
                ana_recv_ns = radar_data.get("analyser_receive_ns")
                ana_end_ns = radar_data.get("analyser_end_ns")
                main_recv_ns = radar_data.get("main_received_ns")
                if all(
                    isinstance(v, int) and v > 0
                    for v in [cap_ns, enq_ns, ana_recv_ns, ana_end_ns, main_recv_ns]
                ):
                    e2e_ms = (now_disp_ns - cap_ns) / 1e6
                    q0_ms = (enq_ns - cap_ns) / 1e6
                    q1_ms = (ana_recv_ns - enq_ns) / 1e6
                    ana_ms = (ana_end_ns - ana_recv_ns) / 1e6
                    q2_ms = (main_recv_ns - ana_end_ns) / 1e6
                    gui_poll_ms = (now_disp_ns - main_recv_ns) / 1e6
                    # Extra analyser stats if present
                    drained = radar_data.get("drained_count")
                    dropped_total = radar_data.get("total_dropped_frames")
                    qhint = radar_data.get("input_queue_size_hint")
                    drain_ns = radar_data.get("drain_ns")
                    wait_ns = radar_data.get("first_dequeue_wait_ns")
                    # Log compact line every ~50 updates
                    if not hasattr(self, "_lat_count"):
                        self._lat_count = 0
                    self._lat_count += 1
                    # if self._lat_count % 50 == 0:
                    if True:
                        frame_ts = radar_data.get("frame_timestamp", None)
                        ts_str = (
                            f"{frame_ts:.3f}s"
                            if isinstance(frame_ts, (int, float))
                            else "NA"
                        )
                        # Absolute stage times (seconds, monotonic origin) for cross-frame comparison
                        cap_s = cap_ns / 1e9
                        enq_s = enq_ns / 1e9
                        ana_recv_s = ana_recv_ns / 1e9
                        ana_end_s = ana_end_ns / 1e9
                        main_s = main_recv_ns / 1e9
                        disp_s = now_disp_ns / 1e9
                        # Optional extras
                        extras = []
                        if isinstance(drain_ns, int) and drain_ns > 0:
                            extras.append(f"drain={drain_ns/1e6:.1f}ms")
                        if isinstance(wait_ns, int) and wait_ns > 0:
                            extras.append(f"wait={wait_ns/1e6:.1f}ms")
                        if isinstance(drained, int):
                            extras.append(f"drained={drained}")
                        if isinstance(dropped_total, int):
                            extras.append(f"dropped_total={dropped_total}")
                        if isinstance(qhint, int) and qhint >= 0:
                            extras.append(f"in_q={qhint}")
                        extras_str = (" | " + " | ".join(extras)) if extras else ""
                        # Unified pipeline line for cross-frame timing comparison
                        self.logger.debug(
                            f"PIPE ts={ts_str} | cap={cap_s:.6f} | q_start={enq_s:.6f} | ana_recv={ana_recv_s:.6f} | ana_end={ana_end_s:.6f} | main={main_s:.6f} | disp={disp_s:.6f} | "
                            f"e2e={e2e_ms:.1f}ms | cap→q={q0_ms:.1f}ms | q→ana={q1_ms:.1f}ms | ana={ana_ms:.1f}ms | ana→main={q2_ms:.1f}ms | main→GUI={gui_poll_ms:.1f}ms | render={((t_rd+t_ra+t_pc)*1000):.1f}ms{extras_str}"
                        )
            self._log_performance()

        except Exception as e:
            self.logger.error(f"Fatal error in update_display: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            self.timer.stop()
            self._fatal_error_occurred = True
            if self.stop_event:
                self.stop_event.set()

    def _update_camera_display(self, camera_data):
        """Update camera/video display"""
        try:
            # Check if we have camera data with a frame
            if getattr(camera_data, "frame", None) is not None:
                self._display_camera_frame(
                    getattr(camera_data, "frame", None),
                    getattr(camera_data, "objects", []),
                )
                # Debug log after displaying frame
                self.logger.debug(
                    f"Displayed camera frame: timestamp={getattr(camera_data, 'timestamp', None)}, detected_objects={len(getattr(camera_data, 'objects', []))}"
                )

            # If no video available, show appropriate message
            elif self.camera_widget.text() == "No video stream available":
                # Keep the default message
                pass

        except Exception as e:
            self.logger.error(f"Error updating camera display: {e}")

    def _update_radar_displays(self, radar_data, rd_extents, ra_extents):
        """Update all radar plots. Returns dict of timings for rd/ra/pc."""
        t_rd = t_ra = t_pc = 0.0
        try:
            # Update Range-Doppler plot
            if radar_data and radar_data.get("range_doppler") is not None:
                t_rd = self._update_range_doppler(
                    radar_data["range_doppler"], rd_extents
                )

            # Update Range-Azimuth plot
            if radar_data and radar_data.get("range_azimuth") is not None:
                t_ra = self._update_range_azimuth(
                    radar_data["range_azimuth"], ra_extents
                )

            # Update Point Cloud
            if radar_data and radar_data.get("point_cloud") is not None:
                t_pc = self._update_point_cloud(radar_data["point_cloud"])

        except Exception as e:
            self.logger.error(f"Error updating radar displays: {e}")
        return {"rd": t_rd, "ra": t_ra, "pc": t_pc}

    def _update_range_doppler(self, rd_data, extents):
        """Update range-doppler heatmap (pyqtgraph). Returns time spent (s)."""
        t0 = time.perf_counter()
        try:
            if self._skip_rdra():
                return time.perf_counter() - t0
            if rd_data is not None and getattr(rd_data, "size", 0) > 0:
                # Handle 3D data by summing over virtual antenna axis
                if rd_data.ndim == 3:
                    rd_data = np.sum(rd_data, axis=1)

                # Convert to dB scale and orient as (rows, cols), origin lower
                rd_db = 20.0 * np.log10(np.abs(rd_data.astype(np.float32, copy=False)) + 1e-10)
                rd_db = np.nan_to_num(rd_db, nan=0.0, posinf=0.0, neginf=0.0)
                rd_db = rd_db.T

                # Update image with LUT and levels
                levels = self._safe_levels(rd_db)
                if levels is None:
                    return time.perf_counter() - t0
                vmin, vmax = levels
                self.rd_image.setImage(
                    rd_db,
                    autoLevels=False,
                    levels=(vmin, vmax),
                    lut=self._lut_jet,
                )

                # Map image to physical extents with a rect
                x0, x1, y0, y1 = extents[0], extents[1], extents[2], extents[3]
                self.rd_image.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        except Exception as e:
            self.logger.error(f"Error updating range-doppler plot: {e}")
        return time.perf_counter() - t0

    def _update_range_azimuth(self, ra_data, extents):
        """Update range-azimuth heatmap (pyqtgraph). Returns time spent (s)."""
        t0 = time.perf_counter()
        try:
            if self._skip_rdra():
                return time.perf_counter() - t0
            if ra_data is not None and getattr(ra_data, "size", 0) > 0:
                # Handle 3D data by summing over virtual antenna axis
                if ra_data.ndim == 3:
                    ra_data = np.sum(ra_data, axis=1)

                # Convert to dB scale and orient as (rows, cols), origin lower
                ra_db = 20.0 * np.log10(np.abs(ra_data.astype(np.float32, copy=False)) + 1e-10)
                ra_db = np.nan_to_num(ra_db, nan=0.0, posinf=0.0, neginf=0.0)
                ra_db = ra_db.T

                # Update image with LUT and levels
                levels = self._safe_levels(ra_db)
                if levels is None:
                    return time.perf_counter() - t0
                vmin, vmax = levels
                self.ra_image.setImage(
                    ra_db,
                    autoLevels=False,
                    levels=(vmin, vmax),
                    lut=self._lut_jet,
                )

                # Map image to physical extents with a rect
                x0, x1, y0, y1 = extents[0], extents[1], extents[2], extents[3]
                self.ra_image.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        except Exception as e:
            self.logger.error(f"Error updating range-azimuth plot: {e}")
        return time.perf_counter() - t0

    def _update_point_cloud(self, point_cloud_data):
        """Update point cloud plot (pyqtgraph). Returns time spent (s)."""
        t0 = time.perf_counter()
        try:
            if point_cloud_data is not None and len(point_cloud_data.get("x", [])) > 0:
                x_data = np.asarray(point_cloud_data["x"], dtype=float)
                y_data = np.asarray(point_cloud_data["y"], dtype=float)

                if x_data.shape[0] != y_data.shape[0]:
                    self.logger.warning(
                        "Point cloud coordinate arrays have inconsistent lengths"
                    )
                    return time.perf_counter() - t0

                # Colors: use snr or intensity; fallback to index
                colors = point_cloud_data.get(
                    "snr", point_cloud_data.get("intensity", None)
                )
                if colors is None or len(colors) != len(x_data):
                    colors = np.arange(len(x_data), dtype=float)
                colors = np.asarray(colors, dtype=float)

                cmin = float(colors.min()) if colors.size else 0.0
                cmax = float(colors.max()) if colors.size else 1.0
                if cmax - cmin < 1e-9:
                    cmax = cmin + 1.0
                norm = (colors - cmin) / (cmax - cmin)
                idx = np.clip((norm * 255).astype(np.int32), 0, 255)

                if self._lut_jet is not None:
                    lut = self._lut_jet
                    # Build per-point brushes
                    brushes = [
                        pg.mkBrush(int(lut[i, 0]), int(lut[i, 1]), int(lut[i, 2]), 255)
                        for i in idx
                    ]
                else:
                    brushes = pg.mkBrush(0, 200, 255, 255)

                self.pc_scatter_item.setData(
                    x=x_data, y=y_data, brush=brushes, pen=None
                )

                # Keep ranges steady
                self.pc_plot.setXRange(-self.pc_max_range, self.pc_max_range, padding=0)
                self.pc_plot.setYRange(0, self.pc_max_range, padding=0)
        except Exception as e:
            self.logger.error(f"Error updating point cloud plot: {e}")
        return time.perf_counter() - t0

    def _update_playback_status(self):
        """Update playback controls based on status"""
        if self.mode == "replay" and self.status_callback:
            try:
                status = self.status_callback()
                if status:
                    # Throttle UI updates to reduce overhead on embedded devices
                    now = time.perf_counter()
                    if not hasattr(self, "_last_status_ui"):
                        self._last_status_ui = {
                            "t": 0.0,
                            "state": None,
                            "current_frame": None,
                            "total_frames": None,
                            "progress": None,
                        }

                    # Only update at most every 50 ms
                    if now - self._last_status_ui.get("t", 0.0) < 0.05:
                        return

                    total_frames = int(status.get("total_frames", 0))
                    current_frame = int(status.get("current_frame", 0))
                    progress = int(status.get("progress_percent", 0))
                    state = status.get("state", "UNKNOWN")

                    # Update slider maximum once when it changes
                    if (
                        total_frames > 0
                        and self._last_status_ui.get("total_frames") != total_frames
                    ):
                        if self.seek_slider.maximum() != total_frames:
                            self.seek_slider.setMaximum(total_frames)
                            self.progress_bar.setMaximum(total_frames)

                    # Update slider position only when not interacting and when changed
                    if not self.seek_slider.isSliderDown():
                        if self._last_status_ui.get("current_frame") != current_frame:
                            self.seek_slider.setValue(current_frame)
                            self.progress_bar.setValue(current_frame)

                    # Update status label only when changed
                    if (
                        self._last_status_ui.get("state") != state
                        or self._last_status_ui.get("current_frame") != current_frame
                        or self._last_status_ui.get("total_frames") != total_frames
                    ):
                        self.status_label.setText(
                            f"{state} - Frame {current_frame}/{total_frames}"
                        )

                    # Update button state - but don't override user actions immediately
                    if state == "PLAYING" and not self.is_playing:
                        self.play_pause_btn.setText("Pause")
                        self.is_playing = True
                    elif state in ["PAUSED", "STOPPED"] and self.is_playing:
                        self.play_pause_btn.setText("Play")
                        self.is_playing = False

                        # Reset timeline for STOPPED state
                        if state == "STOPPED":
                            self.seek_slider.setValue(0)
                            self.progress_bar.setValue(0)

                    # Record last update snapshot and timestamp
                    self._last_status_ui.update(
                        {
                            "t": now,
                            "state": state,
                            "current_frame": current_frame,
                            "total_frames": total_frames,
                            "progress": progress,
                        }
                    )

            except Exception as e:
                self.logger.debug(f"No status update available: {e}")

    def _scan_png_files(self):
        """Scan for PNG files in recording directory"""
        if not self.recording_dir or not os.path.exists(self.recording_dir):
            self.logger.warning(
                f"PNG scan: Directory does not exist: {self.recording_dir}"
            )
            return

        pattern = os.path.join(self.recording_dir, "*.png")
        png_files = glob.glob(pattern)

        if not png_files:
            self.logger.info(f"No PNG files found in directory: {self.recording_dir}")
            return

        # Parse filenames
        png_info = []
        filename_pattern = re.compile(r"(\d{10})_(\d{5})_(\d{12})\.png$")

        for filepath in png_files:
            filename = os.path.basename(filepath)
            match = filename_pattern.match(filename)

            if match:
                timestamp_int = int(match.group(1))
                timestamp_frac = int(match.group(2))
                frame_number = int(match.group(3))
                timestamp = timestamp_int + (timestamp_frac / 1e5)
                png_info.append((filepath, timestamp, frame_number))

        # Sort by timestamp
        png_info.sort(key=lambda x: x[1])
        self._png_files = png_info

        self.logger.info(f"Found {len(self._png_files)} PNG files for video playback")

    def _display_camera_frame(
        self, camera_frame, detected_objects: Optional[List[Any]] = None
    ) -> None:
        """Display camera frame with object detection boxes (works for both live and replay)"""
        if camera_frame is None:
            self.camera_widget.setText("No camera feed available")
            return

        try:
            # Extract the image from D455Frame object
            rgb_image = camera_frame.rgb_image

            # Draw object detection boxes if needed
            if detected_objects:
                cv_image = rgb_image.copy()
                cv_image_with_boxes = self._draw_object_detection_boxes(
                    cv_image, detected_objects
                )
            else:
                cv_image_with_boxes = rgb_image

            # Overlay camera FPS text (top-left)
            if self._cam_fps_text:
                try:
                    cv2.putText(
                        cv_image_with_boxes,
                        self._cam_fps_text,
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        cv_image_with_boxes,
                        self._cam_fps_text,
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                except Exception:
                    pass

            # Convert BGR to RGB for Qt
            rgb_image = cv2.cvtColor(cv_image_with_boxes, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(
                rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(q_image)

            if pixmap is not None and not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.camera_widget.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.camera_widget.setPixmap(scaled_pixmap)
            else:
                self.camera_widget.setText(
                    f"Camera\n{len(detected_objects) if detected_objects else 0} objects detected"
                )

        except Exception as e:
            self.logger.error(f"Error displaying camera frame: {e}")
            self.camera_widget.setText(
                f"Camera Error\n{len(detected_objects) if detected_objects else 0} objects detected"
            )

    def _log_performance(self):
        """Log performance statistics"""
        self._plot_updates += 1
        current_time = time.time()

        if current_time - self._last_log_time >= 5.0:
            fps = self._plot_updates / 5.0
            self.logger.debug(f"Visualization FPS: {fps:.2f}")
            self._plot_updates = 0
            self._last_log_time = current_time

    def _record_profile(
        self,
        *,
        t_total: float,
        t_get: float,
        t_status: float,
        t_camera: float,
        t_rd: float,
        t_ra: float,
        t_pc: float,
    ) -> None:
        if not self._profile_enabled:
            return
        self._profile_count += 1
        self._prof_sum_total += t_total
        self._prof_sum_get_data += t_get
        self._prof_sum_status += t_status
        self._prof_sum_camera += t_camera
        self._prof_sum_rd += t_rd
        self._prof_sum_ra += t_ra
        self._prof_sum_pc += t_pc

        if self._profile_count % self._profile_step == 0:
            n = self._profile_step
            avg_total = self._prof_sum_total / n
            fps = 1.0 / avg_total if avg_total > 0 else 0.0
            # Compact INFO line
            self.logger.info(f"AVG Runtime: {avg_total:.4f}s (frame {n})")

            # Reset sums for the next window
            self._prof_sum_total = 0.0
            self._prof_sum_get_data = 0.0
            self._prof_sum_status = 0.0
            self._prof_sum_camera = 0.0
            self._prof_sum_rd = 0.0
            self._prof_sum_ra = 0.0
            self._prof_sum_pc = 0.0

    def closeEvent(self, a0):
        """Handle window close event"""
        self.logger.info("FusionVisualizer closing...")
        self.timer.stop()
        if self.stop_event:
            self.stop_event.set()
        if a0:
            a0.accept()

    def _draw_object_detection_boxes(
        self, image: np.ndarray, detected_objects: list
    ) -> np.ndarray:
        """
        Draw bounding boxes around detected objects on the image.

        Args:
            image: Input image as numpy array (BGR format)
            detected_objects: List of detected objects with bounding box coordinates

        Returns:
            Image with bounding boxes drawn
        """
        if not detected_objects:
            return image

        # Make a copy of the image to avoid modifying the original
        image_with_boxes = image.copy()

        # Define colors for different object types (BGR format for OpenCV)
        colors = {
            "person": (0, 255, 0),  # Green for persons
            "car": (255, 0, 0),  # Blue for cars
            "bicycle": (0, 0, 255),  # Red for bicycles
            "motorcycle": (0, 255, 255),  # Yellow for motorcycles
            "default": (255, 255, 255),  # White for unknown objects
        }

        # YOLO class names mapping
        class_names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle"}

        for obj in detected_objects:
            try:
                # Handle different object formats
                if (
                    hasattr(obj, "x")
                    and hasattr(obj, "y")
                    and hasattr(obj, "width")
                    and hasattr(obj, "height")
                ):
                    # Rectangle object format from D455Analyser
                    x, y, width, height = (
                        int(obj.x),
                        int(obj.y),
                        int(obj.width),
                        int(obj.height),
                    )
                    x2, y2 = x + width, y + height

                    # Get object type and confidence from enhanced Rectangle
                    if hasattr(obj, "object_type") and hasattr(obj, "confidence"):
                        object_type = obj.object_type
                        confidence = obj.confidence
                    else:
                        object_type = "person"  # Default for old Rectangle format
                        confidence = 1.0

                elif isinstance(obj, dict):
                    # Dictionary format
                    x, y = int(obj.get("x", 0)), int(obj.get("y", 0))
                    width, height = int(obj.get("width", 0)), int(obj.get("height", 0))
                    x2, y2 = x + width, y + height
                    class_id = obj.get("class_id", 0)
                    object_type = class_names.get(class_id, "unknown")
                    confidence = obj.get("confidence", 1.0)
                else:
                    # Skip unknown formats
                    continue

                # Get color for this object type
                color = colors.get(object_type, colors["default"])

                # Draw bounding box
                cv2.rectangle(image_with_boxes, (x, y), (x2, y2), color, 2)

                # Draw label with object type and confidence
                label = f"{object_type}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                # Draw label background
                cv2.rectangle(
                    image_with_boxes,
                    (x, y - label_size[1] - 10),
                    (x + label_size[0], y),
                    color,
                    -1,
                )

                # Draw label text
                cv2.putText(
                    image_with_boxes,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )

            except Exception as e:
                self.logger.error(f"Error drawing bounding box for object {obj}: {e}")
                continue

        return image_with_boxes
