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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pipeline is not None:
            self._pipeline.stop()
        print("***** D455 cleaned up. *****")

    def _read_and_store_frame(self):
        assert self._pipeline is not None, "D455 camera is not initialized"
        # Read the data from the D455
        frames = self._pipeline.wait_for_frames()
        end = time.perf_counter()

        assert self._start_time is not None, "Start time is not initialized"
        timestamp = end - self._start_time
        integer_part = f"{int(timestamp):010d}"
        fraction_part = f"{int((timestamp - int(timestamp)) * 1e5):05d}"
        frame_number = f"{frames.get_frame_number():012d}"
        filename = f"{integer_part}_{fraction_part}_{frame_number}.png"
        filepath = os.path.join(self._dest_dir, filename)

        # ir_frame = frames.get_infrared_frame()
        rgb_frame = frames.get_color_frame()

        # ir_data = np.asanyarray(ir_frame.get_data())
        rgb_data = np.asanyarray(rgb_frame.get_data())

        # Save as numpy array
        cv2.imwrite(filepath, rgb_data)

        # print(f"***** D455 **** Saved data to {filepath}")

        return D455Frame(timestamp, rgb_data)

    def _send_frame(self, stream_queue: multiprocessing.Queue, stop_event):
        assert self._frame_queue is not None, "Frame queue is not initialized"
        while not stop_event.is_set():
            # Wait for a frame to be available
            try:
                video_frame = self._frame_queue.get(timeout=1)
                try:
                    stream_queue.put(video_frame)
                except Exception as e:
                    print(f"***** D455 send_frame **** Error sending data: {e}")
                    break
            except queue.Empty:
                # No frame available, continue
                print("***** D455 send_frame **** Warning: No frame available to send.")
                continue
            except KeyboardInterrupt:
                print(
                    "***** D455 send_frame **** Keyboard interrupt received, stopping..."
                )
                stop_event.set()

        print("***** D455 send_frame **** Stopped... *****")

    def run(self, stream_queue: multiprocessing.Queue, stop_event):
        print("***** D455 **** Starting... *****")
        
        # Initialize components in target process
        self._start_time = time.perf_counter()
        self._frame_queue = queue.Queue()

        print("INFO: Initializing D455 camera")
        self._pipeline = rs.pipeline()
        self._rs_config = rs.config()
        self._rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self._rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self._rs_config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        print("INFO: Starting D455 camera")
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
                # Update the data and check if the data is okay
                video_frame = self._read_and_store_frame()
                self._frame_queue.put(video_frame)
            except KeyboardInterrupt:
                print("***** D455 **** Keyboard interrupt received, stopping...")
                stop_event.set()

        if self._pipeline is not None:
            self._pipeline.stop()

        print("***** D455 stopped. *****")
