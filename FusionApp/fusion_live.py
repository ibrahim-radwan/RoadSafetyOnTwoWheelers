"""
Live fusion application using the new polymorphic architecture.
This is a clean, modern replacement using the refactored engine.
"""

import multiprocessing
from multiprocessing import Queue, Event
import time
import signal
import sys
import threading
import queue
import argparse

from PyQt5.QtWidgets import QApplication

from engine.fusion_factory import FusionFactory
from gui.fusion_visualizer import FusionVisualizer
from sample_processing.radar_params import ADCParams
from config_params import CFGS


class LiveFusionApp:
    def __init__(self):
        self.stop_event = Event()
        self.radar_results_queue = Queue()
        self.camera_results_queue = Queue()
        # Add control queue for recording control
        self.control_queue = Queue()

        # Set up signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}, shutting down...")
        self.stop_event.set()

    def run_camera_and_radar(self, use_3d: bool = False):
        """Run live camera + radar fusion with GUI"""
        print("Starting Live Camera + Radar Fusion...")

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
            visualizer.set_radar_data_callback(lambda: self._current_radar_data)
            visualizer.set_camera_data_callback(lambda: self._current_camera_data)

            # Set up record callback - now actually controls recording
            def record_callback(command):
                print(f"Record command: {command}")
                try:
                    self.control_queue.put(command)
                except Exception as e:
                    print(f"Error sending record command: {e}")

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
                                self._current_radar_data = radar_result
                        except queue.Empty:
                            pass

                        time.sleep(0.01)  # Small sleep to prevent busy waiting

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
            # Clean shutdown
            print("Shutting down...")
            fusion_process.join(timeout=5)

            if fusion_process.is_alive():
                fusion_process.terminate()

            print("Shutdown complete.")

    def run_radar_only(self, use_3d: bool = False):
        """Run live radar-only mode with GUI"""
        print("Starting Live Radar-Only Mode...")

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
            visualizer.set_radar_data_callback(lambda: self._current_radar_data)
            visualizer.set_camera_data_callback(lambda: None)  # No camera data

            # Set up record callback - now actually controls recording
            def record_callback(command):
                print(f"Record command: {command}")
                try:
                    self.control_queue.put(command)
                except Exception as e:
                    print(f"Error sending record command: {e}")

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
                                self._current_radar_data = radar_result
                        except queue.Empty:
                            pass

                        time.sleep(0.01)  # Small sleep to prevent busy waiting

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
            # Clean shutdown
            print("Shutting down...")
            fusion_process.join(timeout=5)

            if fusion_process.is_alive():
                fusion_process.terminate()

            print("Shutdown complete.")


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
        app = LiveFusionApp()

        if args.radar_only:
            app.run_radar_only(args.use_3d)
        else:
            app.run_camera_and_radar(args.use_3d)

    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
