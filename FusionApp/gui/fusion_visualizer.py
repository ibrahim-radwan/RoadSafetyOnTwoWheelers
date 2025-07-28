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
if os.name != 'nt':  # Not Windows
    os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
import cv2

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
from PyQt5.QtCore import QTimer, Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from config_params import CFGS
from utils import setup_logger
from camera.d455 import D455Frame


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

        # Setup UI
        self._setup_ui()

        # Start update timer at 30 FPS (33ms)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(20)  # 30 FPS

        self.logger.info(f"FusionVisualizer initialized in {mode} mode")

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

        # Setup plots
        self._setup_plots()

        # Create 2x2 grid layout for all modes
        plots_layout = QGridLayout()
        plots_layout.addWidget(self.camera_widget, 0, 0)  # Upper-left: Camera/Video
        plots_layout.addWidget(
            self.point_cloud_canvas, 0, 1
        )  # Upper-right: Point cloud
        plots_layout.addWidget(
            self.range_doppler_canvas, 1, 0
        )  # Lower-left: Range-Doppler
        plots_layout.addWidget(
            self.range_azimuth_canvas, 1, 1
        )  # Lower-right: Range-Azimuth

        # Set equal row and column stretch factors for 50/50 split
        plots_layout.setRowStretch(0, 1)  # First row gets 50% of height
        plots_layout.setRowStretch(1, 1)  # Second row gets 50% of height
        plots_layout.setColumnStretch(0, 1)  # First column gets 50% of width
        plots_layout.setColumnStretch(1, 1)  # Second column gets 50% of width

        # Add plots layout with stretch factor so it takes most of the space
        main_layout.addLayout(plots_layout, 1)  # stretch factor of 1

        # Setup controls
        self._setup_controls()
        # Add controls layout with no stretch factor and fixed height
        main_layout.addLayout(self.controls_layout, 0)  # stretch factor of 0

        self.setLayout(main_layout)

    def _setup_plots(self):
        """Setup matplotlib figures and camera display"""
        # Camera/Video widget - always present
        self.camera_widget = QLabel()
        self.camera_widget.setMinimumSize(400, 400)
        self.camera_widget.setStyleSheet("border: 1px solid gray;")
        self.camera_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_widget.setText("No video stream available")
        self.camera_widget.setScaledContents(True)

        # Range-Doppler plot
        self.rd_figure = Figure(figsize=(6, 6))
        self.rd_ax = self.rd_figure.add_subplot(111)
        self.rd_ax.set_title("Range-Doppler Heatmap")
        self.rd_ax.set_xlabel("Doppler (m/s)")
        self.rd_ax.set_ylabel("Range (m)")
        self.range_doppler_canvas = FigureCanvas(self.rd_figure)

        # Range-Azimuth plot
        self.ra_figure = Figure(figsize=(6, 6))
        self.ra_ax = self.ra_figure.add_subplot(111)
        self.ra_ax.set_title("Range-Azimuth Heatmap")
        self.ra_ax.set_xlabel("Azimuth (degrees)")
        self.ra_ax.set_ylabel("Range (m)")
        self.range_azimuth_canvas = FigureCanvas(self.ra_figure)

        # Point Cloud plot (2D or 3D)
        self.pc_figure = Figure(figsize=(6, 6))
        if self.use_3d:
            self.pc_ax = self.pc_figure.add_subplot(111, projection="3d")
            self.pc_ax.set_title("3D Point Cloud")
            self.pc_ax.set_xlabel("X (m)")
            self.pc_ax.set_ylabel("Y (m)")
            if hasattr(self.pc_ax, "set_zlabel"):
                self.pc_ax.set_zlabel("Z (m)")  # type: ignore
        else:
            self.pc_ax = self.pc_figure.add_subplot(111)
            self.pc_ax.set_title("2D Point Cloud")
            self.pc_ax.set_xlabel("X (m)")
            self.pc_ax.set_ylabel("Y (m)")
        self.point_cloud_canvas = FigureCanvas(self.pc_figure)

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

        # Initialize plot references
        self.rd_im = None
        self.ra_im = None
        self.pc_scatter = None
        self.rd_cbar = None
        self.ra_cbar = None
        self.pc_cbar = None

        # Use tight layout
        self.rd_figure.tight_layout()
        self.ra_figure.tight_layout()
        self.pc_figure.tight_layout()

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
            # Get current data
            radar_data = (
                self.radar_data_callback() if self.radar_data_callback else None
            )
            camera_data = (
                self.camera_data_callback() if self.camera_data_callback else None
            )

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
            if self.mode == "replay":
                self._update_playback_status()

            # Update displays
            self._update_camera_display(camera_data)
            self._update_radar_displays(radar_data, rd_extents, ra_extents)

            # Performance logging
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
        """Update all radar plots"""
        try:
            # Update Range-Doppler plot
            if radar_data and radar_data.get("range_doppler") is not None:
                self._update_range_doppler(radar_data["range_doppler"], rd_extents)

            # Update Range-Azimuth plot
            if radar_data and radar_data.get("range_azimuth") is not None:
                self._update_range_azimuth(radar_data["range_azimuth"], ra_extents)

            # Update Point Cloud
            if radar_data and radar_data.get("point_cloud") is not None:
                self._update_point_cloud(radar_data["point_cloud"])

        except Exception as e:
            self.logger.error(f"Error updating radar displays: {e}")

    def _update_range_doppler(self, rd_data, extents):
        """Update range-doppler heatmap"""
        try:
            if rd_data is not None and rd_data.size > 0:
                # Handle 3D data by summing over virtual antenna axis
                if rd_data.ndim == 3:
                    rd_data = np.sum(rd_data, axis=1)

                # Convert to dB scale
                rd_db = 20 * np.log10(np.abs(rd_data) + 1e-10)
                rd_db = rd_db.T  # Transpose for correct orientation

                # Update or create image
                if self.rd_im is None:
                    self.rd_ax.set_title("Range-Doppler Heatmap")
                    self.rd_ax.set_xlabel("Doppler (m/s)")
                    self.rd_ax.set_ylabel("Range (m)")
                    self.rd_im = self.rd_ax.imshow(
                        rd_db, aspect="auto", origin="lower", cmap="jet", extent=extents
                    )
                    if self.rd_cbar is None:
                        self.rd_cbar = self.rd_figure.colorbar(
                            self.rd_im, ax=self.rd_ax
                        )
                        self.rd_cbar.set_label("Magnitude (dB)")
                else:
                    self.rd_im.set_array(rd_db)
                    self.rd_im.set_extent(extents)
                    self.rd_im.set_clim(vmin=rd_db.min(), vmax=rd_db.max())

            self.range_doppler_canvas.draw()

        except Exception as e:
            self.logger.error(f"Error updating range-doppler plot: {e}")

    def _update_range_azimuth(self, ra_data, extents):
        """Update range-azimuth heatmap"""
        try:
            if ra_data is not None and ra_data.size > 0:
                # Handle 3D data by summing over virtual antenna axis
                if ra_data.ndim == 3:
                    ra_data = np.sum(ra_data, axis=1)

                # Convert to dB scale
                ra_db = 20 * np.log10(np.abs(ra_data) + 1e-10)
                ra_db = ra_db.T  # Transpose for correct orientation

                # Update or create image
                if self.ra_im is None:
                    self.ra_ax.set_title("Range-Azimuth Heatmap")
                    self.ra_ax.set_xlabel("Azimuth (degrees)")
                    self.ra_ax.set_ylabel("Range (m)")
                    self.ra_im = self.ra_ax.imshow(
                        ra_db, aspect="auto", origin="lower", cmap="jet", extent=extents
                    )
                    if self.ra_cbar is None:
                        self.ra_cbar = self.ra_figure.colorbar(
                            self.ra_im, ax=self.ra_ax
                        )
                        self.ra_cbar.set_label("Magnitude (dB)")
                else:
                    self.ra_im.set_array(ra_db)
                    self.ra_im.set_extent(extents)
                    self.ra_im.set_clim(vmin=ra_db.min(), vmax=ra_db.max())

            self.range_azimuth_canvas.draw()

        except Exception as e:
            self.logger.error(f"Error updating range-azimuth plot: {e}")

    def _update_point_cloud(self, point_cloud_data):
        """Update point cloud plot"""
        try:
            if point_cloud_data is not None and len(point_cloud_data.get("x", [])) > 0:
                x_data = point_cloud_data["x"]
                y_data = point_cloud_data["y"]
                z_data = point_cloud_data.get("z", []) if self.use_3d else None

                # Validate data consistency
                if (
                    self.use_3d
                    and z_data is not None
                    and (len(x_data) != len(y_data) or len(x_data) != len(z_data))
                ):
                    self.logger.warning(
                        "Point cloud coordinate arrays have inconsistent lengths"
                    )
                    return
                elif not self.use_3d and len(x_data) != len(y_data):
                    self.logger.warning(
                        "Point cloud coordinate arrays have inconsistent lengths"
                    )
                    return

                # Get colors
                colors = point_cloud_data.get(
                    "snr", point_cloud_data.get("intensity", None)
                )
                if colors is None or len(colors) != len(x_data):
                    colors = np.arange(len(x_data), dtype=np.float64)

                # Update or create scatter plot
                if self.pc_scatter is not None:
                    self.pc_scatter.remove()

                if self.use_3d:
                    self.pc_scatter = self.pc_ax.scatter(
                        x_data, y_data, z_data, c=colors, cmap="jet", alpha=0.8
                    )
                    self.pc_ax.set_xlim(-self.pc_max_range, self.pc_max_range)
                    self.pc_ax.set_ylim(0, self.pc_max_range)
                    if hasattr(self.pc_ax, "set_zlim"):
                        self.pc_ax.set_zlim(-self.pc_max_range, self.pc_max_range)  # type: ignore
                else:
                    self.pc_scatter = self.pc_ax.scatter(
                        x_data, y_data, c=colors, cmap="jet", alpha=0.8
                    )
                    self.pc_ax.set_xlim(-self.pc_max_range, self.pc_max_range)
                    self.pc_ax.set_ylim(0, self.pc_max_range)

                # Create colorbar if needed
                if self.pc_cbar is None:
                    self.pc_cbar = self.pc_figure.colorbar(
                        self.pc_scatter, ax=self.pc_ax
                    )
                    self.pc_cbar.set_label("SNR (dB)")

                # Update colorbar limits
                if colors is not None:
                    self.pc_scatter.set_clim(
                        vmin=float(np.min(colors)), vmax=float(np.max(colors))
                    )

            self.point_cloud_canvas.draw()

        except Exception as e:
            self.logger.error(f"Error updating point cloud plot: {e}")

    def _update_playback_status(self):
        """Update playback controls based on status"""
        if self.mode == "replay" and self.status_callback:
            try:
                status = self.status_callback()
                if status:
                    # Update progress
                    progress = int(status.get("progress_percent", 0))
                    self.progress_bar.setValue(progress)

                    # Update slider maximum
                    total_frames = status.get("total_frames", 0)
                    if total_frames > 0 and self.seek_slider.maximum() != total_frames:
                        self.seek_slider.setMaximum(total_frames)
                        self.progress_bar.setMaximum(total_frames)

                    # Update slider position
                    if not self.seek_slider.isSliderDown():
                        current_frame = status.get("current_frame", 0)
                        self.seek_slider.setValue(current_frame)
                        self.progress_bar.setValue(current_frame)

                    # Update status label
                    state = status.get("state", "UNKNOWN")
                    current_frame = status.get("current_frame", 0)
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
                    f"Camera\n{len(detected_objects)} objects detected"
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
