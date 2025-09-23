"""
Live fusion application using the new polymorphic architecture.
This is a clean, modern replacement using the refactored engine.
"""

import os
import sys
import multiprocessing
from multiprocessing import Queue, Event
import time
import signal
import threading
import queue
import argparse
import fpga_udp
from mmwave import dsp
import atexit

# Fix Qt platform plugin issues with OpenCV - only on Linux
if os.name != "nt":  # Not Windows
    # Remove the OpenCV Qt plugin path from environment
    if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ:
        del os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]

    # Force Qt to use a working platform - try xcb first, fallback to offscreen
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ["QT_DEBUG_PLUGINS"] = "0"
    # Prefer software rendering on embedded devices to avoid GLX/EGL issues
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")
    os.environ.setdefault("QT_XCB_NO_XI2", "1")

from PyQt5.QtWidgets import QApplication

from engine.fusion_factory import FusionFactory
from gui.fusion_visualizer import FusionVisualizer
from sample_processing.radar_params import ADCParams
from config_params import CFGS
from utils import setup_logger, disable_shm_resource_tracker


# Module-level logger for main function
logger = setup_logger("fusion_live")


# Ensure the radar is always powered down on interpreter exit (safety net)
def _cleanup_awr2243():
    try:
        fpga_udp.AWR2243_sensorStop()
    except Exception:
        pass
    try:
        fpga_udp.AWR2243_poweroff()
    except Exception:
        pass


atexit.register(_cleanup_awr2243)


class LiveFusionApp:
    def __init__(self):
        self.logger = setup_logger("LiveFusionApp")
        # Avoid resource_tracker noise in the main/GUI process; engine owns SHM unlink
        try:
            disable_shm_resource_tracker(self.logger)
        except Exception:
            pass
        self.stop_event = Event()
        self.radar_results_queue = Queue(maxsize=3)
        self.camera_results_queue = Queue(maxsize=2)
        # Add control queue for recording control
        self.control_queue = Queue()

        # Set up signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop_event.set()

    def run_camera_and_radar(self, use_3d: bool = False):
        """Run live camera + radar fusion with GUI"""
        self.logger.info("Starting Live Camera + Radar Fusion...")

        # Initialize ADC parameters
        adc_params = ADCParams(CFGS.AWR2243_CONFIG_FILE)

        # Create fusion engine with live camera and radar
        fusion_engine = FusionFactory.create_live_fusion()

        # Start fusion engine in a separate process
        fusion_process = multiprocessing.Process(
            target=fusion_engine.run,
            args=(
                self.radar_results_queue,
                self.camera_results_queue,
                self.stop_event,
                self.control_queue,  # Pass control queue for recording control
                None,  # status_queue not needed for live mode
            ),
        )

        # Current data storage
        self._current_radar_data = None
        self._current_camera_data = None

        try:
            fusion_process.start()

            # Initialize PyQt application
            app = QApplication(sys.argv)

            # Create visualizer
            visualizer = FusionVisualizer(
                mode="live",
                stop_event=self.stop_event,
                recording_dir=None,
                adc_params=adc_params,
                use_3d=use_3d,
            )

            # Set up data callbacks
            visualizer.set_radar_data_callback(
                lambda: self._current_radar_data)
            visualizer.set_camera_data_callback(
                lambda: self._current_camera_data)

            # Set up record callback - now actually controls recording
            def record_callback(command):
                self.logger.info(f"Record command: {command}")
                try:
                    self.control_queue.put(command)
                except Exception as e:
                    self.logger.error(f"Error sending record command: {e}")

            visualizer.set_record_callback(record_callback)

            # Show visualizer
            visualizer.show()

            # Start data processing thread
            def data_processor():
                """Process data from queues and update visualizer"""
                try:
                    while not self.stop_event.is_set():
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
                                if isinstance(radar_result, dict):
                                    radar_result["main_received_ns"] = (
                                        time.perf_counter_ns()
                                    )
                                self._current_radar_data = radar_result
                        except queue.Empty:
                            pass

                        # Faster polling to reduce backpressure
                        time.sleep(0.005)

                except Exception as e:
                    self.logger.error(f"Error in data processor: {e}")
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
            self.logger.info("Keyboard interrupt received, stopping...")
            self.stop_event.set()

        finally:
            # Clean shutdown with improved process termination
            self.logger.info("Shutting down...")
            self.stop_event.set()

            # Give processes time to clean up
            time.sleep(1)

            # Ensure AWR2243 is stopped and powered off (live mode)
            try:
                fpga_udp.AWR2243_sensorStop()
            except Exception:
                pass
            try:
                fpga_udp.AWR2243_poweroff()
            except Exception:
                pass

            # Try graceful shutdown first
            if fusion_process.is_alive():
                self.logger.info("Waiting for fusion process to terminate...")
                fusion_process.join(timeout=3)

            # Force termination if still alive
            if fusion_process.is_alive():
                self.logger.warning("Force terminating fusion process...")
                fusion_process.terminate()
                fusion_process.join(timeout=2)

            # Kill if still alive
            if fusion_process.is_alive():
                self.logger.warning("Force killing fusion process...")
                fusion_process.kill()
                fusion_process.join()

            self.logger.info("Shutdown complete.")

    def run_radar_only(self, use_3d: bool = False):
        """Run live radar-only mode with GUI"""
        self.logger.info("Starting Live Radar-Only Mode...")

        # Initialize ADC parameters
        adc_params = ADCParams(CFGS.AWR2243_CONFIG_FILE)

        # Create fusion engine with radar only
        fusion_engine = FusionFactory.create_live_radar_only()

        # Start fusion engine in a separate process
        fusion_process = multiprocessing.Process(
            target=fusion_engine.run,
            args=(
                self.radar_results_queue,
                None,  # No camera results queue
                self.stop_event,
                self.control_queue,  # Pass control queue for recording control
                None,  # status_queue not needed for live mode
            ),
        )

        # Current data storage
        self._current_radar_data = None

        try:
            fusion_process.start()

            # Initialize PyQt application
            app = QApplication(sys.argv)

            # Create visualizer
            visualizer = FusionVisualizer(
                mode="live",
                stop_event=self.stop_event,
                recording_dir=None,
                adc_params=adc_params,
                use_3d=use_3d,
            )

            # Set up data callbacks
            visualizer.set_radar_data_callback(
                lambda: self._current_radar_data)
            visualizer.set_camera_data_callback(lambda: None)  # No camera data

            # Set up record callback - now actually controls recording
            def record_callback(command):
                self.logger.info(f"Record command: {command}")
                try:
                    self.control_queue.put(command)
                except Exception as e:
                    self.logger.error(f"Error sending record command: {e}")

            visualizer.set_record_callback(record_callback)

            # Show visualizer
            visualizer.show()

            # Start data processing thread
            def data_processor():
                """Process data from queues and update visualizer"""
                try:
                    while not self.stop_event.is_set():
                        # Process radar results
                        try:
                            while True:
                                radar_result = self.radar_results_queue.get_nowait()
                                if isinstance(radar_result, dict):
                                    radar_result["main_received_ns"] = (
                                        time.perf_counter_ns()
                                    )
                                self._current_radar_data = radar_result
                        except queue.Empty:
                            pass

                        time.sleep(0.01)  # Small sleep to prevent busy waiting

                except Exception as e:
                    self.logger.error(f"Error in data processor: {e}")
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
            self.logger.info("Keyboard interrupt received, stopping...")
            self.stop_event.set()

        finally:
            # Clean shutdown with improved process termination
            self.logger.info("Shutting down...")
            self.stop_event.set()

            # Give processes time to clean up
            time.sleep(1)

            fpga_udp.AWR2243_sensorStop()
            fpga_udp.AWR2243_poweroff()

            # Try graceful shutdown first
            if fusion_process.is_alive():
                self.logger.info("Waiting for fusion process to terminate...")
                fusion_process.join(timeout=3)

            # Force termination if still alive
            if fusion_process.is_alive():
                self.logger.warning("Force terminating fusion process...")
                fusion_process.terminate()
                fusion_process.join(timeout=2)

            # Kill if still alive
            if fusion_process.is_alive():
                self.logger.warning("Force killing fusion process...")
                fusion_process.kill()
                fusion_process.join()

            self.logger.info("Shutdown complete.")


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="Live Fusion Processing and Visualization"
    )
    parser.add_argument(
        "--radar-only",
        action="store_true",
        help="Run in radar-only mode (no camera)",
    )
    parser.add_argument(
        "--3d",
        action="store_true",
        dest="use_3d",
        help="Use 3D point cloud visualization (default: 2D)",
    )

    args = parser.parse_args()

    try:
        # Disable tracker early in the main interpreter as well
        try:
            disable_shm_resource_tracker(logger)
        except Exception:
            pass
        app = LiveFusionApp()

        # initialize AWR2243 radar as it only works in main process
        ret = fpga_udp.AWR2243_init(CFGS.AWR2243_CONFIG_FILE)
        if ret != 0:
            logger.error(
                "Failed to initialize AWR2243 radar with return code: %d", ret)
            sys.exit(0)

        fpga_udp.AWR2243_setFrameCfg(0)

        ret = fpga_udp.AWR2243_sensorStart()

        if ret != 0:
            logger.error(
                "Failed to start AWR2243 sensor with return code: %d", ret)
            sys.exit(0)

        # time.sleep(1)  # Allow time for radar to start: removed as precompile takes a while
        dsp.precompile_kernels()

        if args.radar_only:
            app.run_radar_only(args.use_3d)
        else:
            app.run_camera_and_radar(args.use_3d)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Hard-exit to avoid late aborts in C extensions during interpreter teardown
        os._exit(0)
