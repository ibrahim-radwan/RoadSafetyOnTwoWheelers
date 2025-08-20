import os
import time
import queue
from typing import Optional
import pyrealsense2 as rs
import threading
import multiprocessing
import numpy as np
import cv2

from config_params import CFGS
from engine.interfaces import CameraFeed
from utils import setup_logger


class D455Config:
    def __init__(
        self,
        dest_dir: str = CFGS.DEST_DIR,
    ):
        self.dest_dir = dest_dir


class D455Frame:
    def __init__(self, timestamp: float, image: np.ndarray):
        # self.ir_image = ir_image
        self.rgb_image = image
        self.timestamp = timestamp


class D455(CameraFeed):
    def __init__(self, d455_config: D455Config = D455Config()):
        # Store only serializable configuration
        self._config = d455_config
        self._dest_dir = d455_config.dest_dir

        # Initialize these in run() method
        self._start_time: Optional[float] = None
        self._frame_queue: Optional[queue.Queue] = None
        self._pipeline = None
        self._rs_config = None
        self._send_thread: Optional[threading.Thread] = None
        self.logger = None

        # Recording control
        self._is_recording = False
        self._control_queue: Optional[multiprocessing.Queue] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pipeline is not None:
            self._pipeline.stop()
        if self.logger is not None:
            self.logger.info("Cleaned up")

    def _check_control_commands(self):
        """Check for recording control commands"""
        if self._control_queue is None:
            return

        try:
            while True:
                command = self._control_queue.get_nowait()
                if command == "start_recording":
                    self._is_recording = True
                    if self.logger:
                        self.logger.info("Recording started")
                    else:
                        # Fallback: initialize logger if not available
                        self.logger = setup_logger("D455")
                        self.logger.info("Recording started")
                elif command == "stop_recording":
                    self._is_recording = False
                    if self.logger:
                        self.logger.info("Recording stopped")
                    else:
                        # Fallback: initialize logger if not available
                        self.logger = setup_logger("D455")
                        self.logger.info("Recording stopped")
        except queue.Empty:
            pass

    def _read_and_store_frame(self):
        assert self._pipeline is not None, "D455 camera is not initialized"
        # Read the data from the D455
        frames = self._pipeline.wait_for_frames()
        end = time.perf_counter()

        assert self._start_time is not None, "Start time is not initialized"
        timestamp = end - self._start_time

        # Only use color stream to reduce bandwidth and CPU
        rgb_frame = frames.get_color_frame()

        # ir_data = np.asanyarray(ir_frame.get_data())
        rgb_data = np.asanyarray(rgb_frame.get_data())

        # Only save frame if recording is enabled
        if self._is_recording:
            integer_part = f"{int(timestamp):010d}"
            fraction_part = f"{int((timestamp - int(timestamp)) * 1e5):05d}"
            frame_number = f"{frames.get_frame_number():012d}"
            filename = f"{integer_part}_{fraction_part}_{frame_number}.png"
            filepath = os.path.join(self._dest_dir, filename)

            # Save as numpy array
            cv2.imwrite(filepath, rgb_data)
            if self.logger:
                self.logger.debug(f"Saved data to {filepath}")

        return D455Frame(timestamp, rgb_data)

    def _send_frame(self, stream_queue: multiprocessing.Queue, stop_event):
        assert self._frame_queue is not None, "Frame queue is not initialized"
        while not stop_event.is_set():
            # Check for control commands periodically
            self._check_control_commands()

            # Wait for a frame to be available
            try:
                video_frame = self._frame_queue.get(timeout=1)
                try:
                    # Non-blocking put; drop if downstream is slow to keep latency bounded
                    stream_queue.put_nowait(video_frame)
                except Exception as e:
                    if self.logger is not None:
                        self.logger.warning(
                            f"Camera frame drop: downstream queue busy ({type(e).__name__}: {e})"
                        )
                    continue
            except queue.Empty:
                # No frame available, continue
                if self.logger is not None:
                    self.logger.debug("No frame available to send")
                continue
            except KeyboardInterrupt:
                if self.logger is not None:
                    self.logger.info("Keyboard interrupt received, stopping...")
                stop_event.set()

        if self.logger is not None:
            self.logger.info("Send frame thread stopped")

    def run(
        self,
        stream_queue: multiprocessing.Queue,
        stop_event,
        control_queue: Optional[multiprocessing.Queue] = None,
    ):
        # Initialize logger in target process
        self.logger = setup_logger("D455")
        self.logger.info("Starting...")

        # Store control queue reference
        self._control_queue = control_queue

        # Initialize components in target process
        self._start_time = time.perf_counter()
        self._frame_queue = queue.Queue(maxsize=2)

        self.logger.info("Initializing D455 camera")
        self._pipeline = rs.pipeline()
        self._rs_config = rs.config()
        # Enable only color stream to lower overhead (disable depth and IR)
        self._rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.logger.info("Starting D455 camera")
        assert self._pipeline is not None, "D455 camera is not initialized"
        self._pipeline.start(self._rs_config)

        # Create and start a thread for sending frames
        self._send_thread = threading.Thread(
            target=self._send_frame,
            name="D455SendThread",
            args=(
                stream_queue,
                stop_event,
            ),
        )
        self._send_thread.start()

        while not stop_event.is_set():
            try:
                # Check for control commands
                self._check_control_commands()

                # Update the data and check if the data is okay
                video_frame = self._read_and_store_frame()
                try:
                    self._frame_queue.put_nowait(video_frame)
                except queue.Full:
                    # Drop oldest to keep most recent frame for lower latency
                    try:
                        _ = self._frame_queue.get_nowait()
                        if self.logger is not None:
                            self.logger.warning(
                                "Camera frame drop: local queue full, dropped oldest"
                            )
                    except queue.Empty:
                        pass
                    try:
                        self._frame_queue.put_nowait(video_frame)
                    except Exception:
                        pass
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received, stopping...")
                stop_event.set()

        if self._pipeline is not None:
            self._pipeline.stop()

        self.logger.info("Stopped")
