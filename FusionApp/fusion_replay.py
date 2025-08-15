"""
Replay fusion application using the new polymorphic architecture.
This demonstrates radar and camera recording playback in unified replay mode.
"""

import multiprocessing
from multiprocessing import Queue, Event
import time
import signal
import sys
import os
from typing import Optional
import argparse
import threading
import queue
import logging

from mmwave import dsp

# Ensure Qt platform and GL settings are compatible on embedded/Linux (e.g., Jetson)
if os.name != "nt":
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("QT_DEBUG_PLUGINS", "0")
    # Prevent Qt xcb plugin from attempting GLX/EGL integrations on systems without them
    os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")
    # Silence XInput2 warnings on Jetson/Xorg configurations lacking XI2
    os.environ.setdefault("QT_XCB_NO_XI2", "1")

# Suppress ttyACM warnings in replay mode (no serial ports needed)
os.environ.setdefault("FUSION_SUPPRESS_TTYACM_WARNING", "1")

from PyQt5.QtWidgets import QApplication

from engine.fusion_factory import FusionFactory
from utils import setup_logger
from gui.fusion_visualizer import FusionVisualizer
from sample_processing.radar_params import ADCParams
from config_params import CFGS
from engine.sync_state import (
    create_sync_state,
    SyncStateUtils,
    PlaybackState,
    TimestampScanner,
)


class ReplayFusionApp:
    def __init__(self, recording_dir: str, config_file: Optional[str] = None):
        self.recording_dir = recording_dir
        self.config_file = config_file
        self.stop_event = Event()
        self.radar_results_queue = Queue()
        self.camera_results_queue = Queue()
        self.control_queue = Queue()
        self.status_queue = Queue()

        # Initialize synchronization state with Manager for cross-process sharing
        self.manager = multiprocessing.Manager()
        self.sync_state = create_sync_state(self.manager)

        # Set up signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Validate recording directory
        if not os.path.exists(recording_dir):
            raise ValueError(f"Recording directory does not exist: {recording_dir}")

        # Validate config file if provided
        if config_file and not os.path.exists(config_file):
            raise ValueError(f"Config file does not exist: {config_file}")

        # Set up logger
        self.logger = setup_logger("ReplayFusionApp")

        # Initialize timing information
        self._initialize_timing()

    def _initialize_timing(self):
        """Initialize timing information by scanning recording directory"""
        try:
            # Scan directory and find common start timestamp
            start_timestamp, png_files, bin_files = (
                TimestampScanner.find_common_start_timestamp(self.recording_dir)
            )

            # Set the common start timestamp in sync state
            SyncStateUtils.set_start_timestamp(self.sync_state, start_timestamp)

            # Store file information for debugging
            self.png_files = png_files
            self.bin_files = bin_files

            self.logger.info(f"Timing initialized - Start timestamp: {start_timestamp}")
            self.logger.info(
                f"Found {len(png_files)} PNG files, {len(bin_files)} BIN files"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize timing: {e}")
            # Set default values
            SyncStateUtils.set_start_timestamp(self.sync_state, 0.0)
            self.png_files = []
            self.bin_files = []

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}, shutting down...")
        self.stop_event.set()

    def run_replay_with_camera(self, use_3d: bool = False):
        """Run replay radar + camera fusion (both from recorded data) with GUI"""
        print(f"Starting Replay Radar + Camera Fusion from: {self.recording_dir}")

        # Initialize ADC parameters
        config_file_to_use = (
            self.config_file if self.config_file else CFGS.AWR2243_CONFIG_FILE
        )
        adc_params = ADCParams(config_file_to_use)

        # Create fusion engine with replay radar and camera
        fusion_engine = FusionFactory.create_replay_fusion(
            recording_dir=self.recording_dir,
            radar_config_file=self.config_file,
            sync_state=self.sync_state,
        )

        # Start fusion engine in a separate process
        fusion_process = multiprocessing.Process(
            target=fusion_engine.run,
            args=(
                self.radar_results_queue,
                self.camera_results_queue,
                self.stop_event,
                self.control_queue,
                self.status_queue,
            ),
        )

        # Current data storage
        self._current_radar_data = None
        self._current_camera_data = None
        self._current_status = None

        try:
            fusion_process.start()

            # Initialize PyQt application
            app = QApplication(sys.argv)

            # Create visualizer
            visualizer = FusionVisualizer(
                mode="replay",
                stop_event=self.stop_event,
                recording_dir=self.recording_dir,
                adc_params=adc_params,
                use_3d=use_3d,
            )

            # Set up data callbacks
            visualizer.set_radar_data_callback(lambda: self._current_radar_data)
            visualizer.set_camera_data_callback(lambda: self._current_camera_data)
            visualizer.set_status_callback(lambda: self._current_status)

            # Set up control callback that updates sync state
            def control_callback(command):
                try:
                    if command.startswith("seek:"):
                        frame_number = int(command.split(":")[1])
                        # Convert frame number to timeline position if needed
                        # For now, just pass to control queue
                        self.control_queue.put(f"seek:{frame_number}")
                    elif command == "play":
                        SyncStateUtils.set_playback_state(
                            self.sync_state, PlaybackState.PLAYING
                        )
                        self.control_queue.put(command)
                    elif command == "pause":
                        SyncStateUtils.set_playback_state(
                            self.sync_state, PlaybackState.PAUSED
                        )
                        self.control_queue.put(command)
                    elif command == "stop":
                        SyncStateUtils.set_playback_state(
                            self.sync_state, PlaybackState.STOPPED
                        )
                        self.control_queue.put(command)
                    else:
                        self.control_queue.put(command)
                except queue.Full:
                    print(f"Control queue full, dropping command: {command}")

            visualizer.set_control_callback(control_callback)

            # Show visualizer
            visualizer.show()

            # Start data processing thread
            def data_processor():
                """Process data from queues and update visualizer"""
                try:
                    while not self.stop_event.is_set():
                        # Process status updates
                        try:
                            while True:
                                status = self.status_queue.get_nowait()
                                self._current_status = status
                        except queue.Empty:
                            pass

                        # Process camera results
                        try:
                            while True:
                                camera_result = self.camera_results_queue.get_nowait()
                                self._current_camera_data = camera_result
                        except queue.Empty:
                            pass

                        # Process radar results
                        try:
                            while True:
                                radar_result = self.radar_results_queue.get_nowait()
                                self._current_radar_data = radar_result
                        except queue.Empty:
                            pass

                        time.sleep(0.01)  # Small sleep to prevent busy waiting

                except Exception as e:
                    print(f"Error in data processor: {e}")
                    self.stop_event.set()

            # Timeline controller thread
            def timeline_controller():
                """Manage the shared timeline position and coordinate synchronized start"""
                try:
                    # Wait for both feeds to be ready (scanned files and initialized)
                    self.logger.info("Waiting for feeds to be ready...")
                    if SyncStateUtils.wait_for_feeds_ready(self.sync_state, timeout=30):
                        self.logger.info(
                            "All feeds ready - signaling synchronized start"
                        )
                        SyncStateUtils.signal_start_playback(self.sync_state)
                    else:
                        self.logger.warning(
                            "Timeout waiting for feeds to be ready, proceeding anyway"
                        )
                        SyncStateUtils.signal_start_playback(self.sync_state)

                    # Main timeline management loop
                    while not self.stop_event.is_set():
                        # Update timeline position based on real-time progression
                        SyncStateUtils.update_timeline(self.sync_state)

                        # Small sleep for smooth timeline updates
                        time.sleep(0.01)  # 100 Hz update rate

                except Exception as e:
                    print(f"Error in timeline controller: {e}")
                    self.stop_event.set()

            # Start processing threads
            data_thread = threading.Thread(target=data_processor, daemon=True)
            timeline_thread = threading.Thread(target=timeline_controller, daemon=True)

            data_thread.start()
            timeline_thread.start()

            # Run GUI event loop
            try:
                app.exec_()
            finally:
                self.stop_event.set()

        except KeyboardInterrupt:
            print("Keyboard interrupt received, stopping...")
            self.stop_event.set()

        finally:
            # Clean shutdown with extended grace period before force
            print("Shutting down...")
            total_wait = 0
            while fusion_process.is_alive() and total_wait < 20:
                fusion_process.join(timeout=2)
                total_wait += 2
            if fusion_process.is_alive():
                print("Fusion process still alive, terminating...")
                fusion_process.terminate()
                fusion_process.join(timeout=5)
            if fusion_process.is_alive():
                print("Fusion process did not terminate, killing...")
                fusion_process.kill()
                fusion_process.join()

            # Clean up manager
            self.manager.shutdown()

            print("Shutdown complete.")

    def run_replay_radar_only(self, use_3d: bool = False):
        """Run replay radar-only mode with GUI (no camera)."""
        print(f"Starting Replay Radar-Only Mode from: {self.recording_dir}")

        # Initialize ADC parameters
        config_file_to_use = (
            self.config_file if self.config_file else CFGS.AWR2243_CONFIG_FILE
        )
        adc_params = ADCParams(config_file_to_use)

        # Create fusion engine with replay radar only
        fusion_engine = FusionFactory.create_replay_radar_only(
            recording_dir=self.recording_dir, radar_config_file=self.config_file
        )

        # Start fusion engine in a separate process
        fusion_process = multiprocessing.Process(
            target=fusion_engine.run,
            args=(
                self.radar_results_queue,
                None,  # No camera results queue in radar-only mode
                self.stop_event,
                self.control_queue,
                self.status_queue,
            ),
        )

        # Current data storage
        self._current_radar_data = None
        self._current_status = None

        try:
            fusion_process.start()

            # Initialize PyQt application
            app = QApplication(sys.argv)

            # Create visualizer
            visualizer = FusionVisualizer(
                mode="replay",
                stop_event=self.stop_event,
                recording_dir=self.recording_dir,
                adc_params=adc_params,
                use_3d=use_3d,
            )

            # Set up data callbacks
            visualizer.set_radar_data_callback(lambda: self._current_radar_data)
            visualizer.set_camera_data_callback(lambda: None)  # No camera data
            visualizer.set_status_callback(lambda: self._current_status)

            # Set up control callback
            def control_callback(command):
                try:
                    if command.startswith("seek:"):
                        frame_number = int(command.split(":")[1])
                        self.control_queue.put(f"seek:{frame_number}")
                    else:
                        self.control_queue.put(command)
                except queue.Full:
                    print(f"Control queue full, dropping command: {command}")

            visualizer.set_control_callback(control_callback)

            # Show visualizer
            visualizer.show()

            # Start data processing thread
            def data_processor():
                """Process radar results and update visualizer"""
                try:
                    while not self.stop_event.is_set():
                        # Process status updates
                        try:
                            while True:
                                status = self.status_queue.get_nowait()
                                self._current_status = status
                        except queue.Empty:
                            pass

                        # Process radar results
                        try:
                            while True:
                                radar_result = self.radar_results_queue.get_nowait()
                                self._current_radar_data = radar_result
                        except queue.Empty:
                            pass

                        time.sleep(0.01)

                except Exception as e:
                    print(f"Error in data processor: {e}")
                    self.stop_event.set()

            # Start data processing thread
            data_thread = threading.Thread(target=data_processor, daemon=True)
            data_thread.start()

            # Run GUI event loop
            try:
                app.exec_()
            finally:
                self.stop_event.set()

        except KeyboardInterrupt:
            print("Keyboard interrupt received, stopping...")
            self.stop_event.set()

        finally:
            # Clean shutdown with extended grace period before force
            print("Shutting down...")
            total_wait = 0
            while fusion_process.is_alive() and total_wait < 20:
                fusion_process.join(timeout=2)
                total_wait += 2
            if fusion_process.is_alive():
                print("Fusion process still alive, terminating...")
                fusion_process.terminate()
                fusion_process.join(timeout=5)
            if fusion_process.is_alive():
                print("Fusion process did not terminate, killing...")
                fusion_process.kill()
                fusion_process.join()

            # Clean up manager
            self.manager.shutdown()

            print("Shutdown complete.")


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="Replay Fusion Processing and Visualization"
    )
    parser.add_argument(
        "--file-path",
        "--file",
        dest="file_path",
        type=str,
        required=True,
        help="Path to recorded radar data directory",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to radar configuration file (optional)",
    )
    parser.add_argument(
        "--radar-only",
        action="store_true",
        help="Run in radar-only mode (no camera replay)",
    )
    parser.add_argument(
        "--3d",
        action="store_true",
        dest="use_3d",
        help="Use 3D point cloud visualization (default: 2D)",
    )

    args = parser.parse_args()

    # Validate file path
    if not os.path.exists(args.file_path):
        print(f"Error: Recording directory does not exist: {args.file_path}")
        sys.exit(1)

    # Validate config file if provided
    if args.config_file and not os.path.exists(args.config_file):
        print(f"Error: Config file does not exist: {args.config_file}")
        sys.exit(1)

    try:
        # Use spawn start method to avoid inheriting background threads that can block shutdown
        multiprocessing.set_start_method("spawn", force=True)
        app = ReplayFusionApp(args.file_path, args.config_file)

        dsp.precompile_kernels()

        if args.radar_only:
            app.run_replay_radar_only(args.use_3d)
        else:
            app.run_replay_with_camera(args.use_3d)

    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
