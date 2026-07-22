import os
import sys
import time
import queue
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional
import pyrealsense2 as rs
import threading
import multiprocessing
import numpy as np
import cv2

from config_params import CFGS
from engine.interfaces import CameraFeed
from camera.png_utils import generate_camera_filename
from radar.bin_utils import generate_radar_filename
from recording.clock import (
    CLOCK_DOMAIN,
    capture_clock_ns,
    calibrate_realsense_offset,
    realsense_ms_to_capture_ns,
)
from recording.sync_recording import (
    RecordingManifest,
    RecordingPairState,
    parse_start_recording_command,
    relative_timestamp_s,
)
from utils import setup_logger


class D455Config:
    def __init__(
        self,
        dest_dir: Optional[str] = None,
        timestamp_origin: Optional[float] = None,
        recording_pair_meta: Optional[dict] = None,
    ):
        self.dest_dir = dest_dir or CFGS.new_recording_dir()
        self.timestamp_origin = timestamp_origin
        self.recording_pair_meta = recording_pair_meta


@dataclass
class _BufferedCameraFrame:
    capture_mono_ns: int
    rs_ms: float
    frame_number: int
    image: np.ndarray


class D455Frame:
    def __init__(self, timestamp: float, image: np.ndarray):
        # self.ir_image = ir_image
        self.rgb_image = image
        self.timestamp = timestamp
        # Optional metadata for diagnostics
        self.seq: int = 0
        self.drops_total: int = 0


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
        self._recording_pair_state: Optional[RecordingPairState] = None
        self._recording_manifest: Optional[RecordingManifest] = None
        self._recording_epoch_ns: int = 0
        self._last_pair_request_gen: int = 0
        self._frame_ring: Deque[_BufferedCameraFrame] = deque(maxlen=20)
        self._rs_mono_offset_ns: Optional[int] = None
        self._pair_lock = threading.Lock()
        self._stop_event = None
        self._frame_timeout_streak: int = 0
        # Diagnostics
        self._seq_counter: int = 0
        self._drops_total: int = 0

    def __getstate__(self):
        # Windows spawn pickles feed instances into child processes; locks and
        # other runtime handles are recreated in the child after unpickling.
        state = self.__dict__.copy()
        state.pop("_pair_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pair_lock = threading.Lock()

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
                if isinstance(command, str) and command.startswith("start_recording"):
                    start_cmd = parse_start_recording_command(command)
                    try:
                        if start_cmd.directory:
                            self._dest_dir = start_cmd.directory
                            if self.logger:
                                self.logger.info(
                                    f"Recording directory set to: {self._dest_dir}"
                                )
                    except Exception:
                        pass
                    self._is_recording = True
                    self._recording_epoch_ns = int(start_cmd.epoch_ns)
                    self._last_pair_request_gen = 0
                    with self._pair_lock:
                        self._frame_ring.clear()
                    if self._recording_pair_state is not None:
                        if self._recording_pair_state.recording_epoch_ns() <= 0:
                            self._recording_pair_state.begin_recording(
                                self._recording_epoch_ns
                            )
                    if self._dest_dir:
                        self._recording_manifest = RecordingManifest(self._dest_dir)
                    if self.logger:
                        self.logger.info(
                            "Recording started (epoch_ns=%d, paired=%s)",
                            self._recording_epoch_ns,
                            self._recording_pair_state is not None,
                        )
                    elif not self.logger:
                        self.logger = setup_logger("D455")
                        self.logger.info("Recording started")
                elif command == "stop_recording":
                    self._is_recording = False
                    self._recording_epoch_ns = 0
                    with self._pair_lock:
                        self._frame_ring.clear()
                    if self._recording_pair_state is not None:
                        if self._recording_pair_state.recording_epoch_ns() > 0:
                            self._recording_pair_state.end_recording()
                    if self.logger:
                        self.logger.info("Recording stopped")
                    else:
                        self.logger = setup_logger("D455")
                        self.logger.info("Recording stopped")
        except queue.Empty:
            pass

    def _select_buffered_frame(
        self, target_camera_mono_ns: int
    ) -> Optional[_BufferedCameraFrame]:
        if not self._frame_ring:
            return None
        return min(
            self._frame_ring,
            key=lambda frame: abs(frame.capture_mono_ns - target_camera_mono_ns),
        )

    def _save_camera_frame(
        self,
        buffered: _BufferedCameraFrame,
        pair_seq: int,
        *,
        radar_capture_mono_ns: int = 0,
        filename_mono_ns: Optional[int] = None,
    ) -> Optional[str]:
        try:
            if not os.path.exists(self._dest_dir):
                os.makedirs(self._dest_dir, exist_ok=True)
        except Exception:
            pass

        name_mono_ns = (
            int(filename_mono_ns)
            if filename_mono_ns is not None
            else int(buffered.capture_mono_ns)
        )
        timestamp = relative_timestamp_s(name_mono_ns, self._recording_epoch_ns)
        filename = generate_camera_filename(timestamp, pair_seq)
        filepath = os.path.join(self._dest_dir, filename)
        cv2.imwrite(filepath, buffered.image)
        if self.logger:
            self.logger.debug(f"Saved paired camera frame to {filepath}")
        if self._recording_manifest is not None:
            delta_ns = int(radar_capture_mono_ns) - int(buffered.capture_mono_ns)
            radar_file = generate_radar_filename(timestamp, pair_seq)
            self._recording_manifest.append(
                {
                    "pair_seq": int(pair_seq),
                    "clock_domain": CLOCK_DOMAIN,
                    "recording_epoch_mono_ns": int(self._recording_epoch_ns),
                    "radar_file": radar_file,
                    "camera_file": filename,
                    "filename_mono_ns": int(name_mono_ns),
                    "radar_capture_mono_ns": int(radar_capture_mono_ns),
                    "camera_capture_mono_ns": int(buffered.capture_mono_ns),
                    "camera_rs_ms": float(buffered.rs_ms),
                    "camera_frame_number": int(buffered.frame_number),
                    "delta_ns": int(delta_ns),
                }
            )
        return filepath

    def _process_pair_save_requests(self) -> None:
        if (
            not self._is_recording
            or self._recording_pair_state is None
            or self._recording_epoch_ns <= 0
        ):
            return

        request = self._recording_pair_state.read_pair_request()
        if request.generation <= self._last_pair_request_gen:
            return
        self._last_pair_request_gen = request.generation

        with self._pair_lock:
            buffered = self._select_buffered_frame(request.target_camera_mono_ns)
        if buffered is None:
            if self.logger:
                self.logger.warning(
                    "Paired camera save skipped: no buffered frame for seq %d",
                    request.pair_seq,
                )
            return
        # Disk I/O runs on the send thread so capture can keep polling RealSense.
        self._save_camera_frame(
            buffered,
            request.pair_seq,
            radar_capture_mono_ns=request.radar_capture_mono_ns,
            filename_mono_ns=request.radar_capture_mono_ns,
        )

    def _wait_for_color_frames(self):
        """Poll RealSense without letting disk I/O on other threads stall capture."""
        assert self._pipeline is not None, "D455 camera is not initialized"
        timeout_ms = 10000
        max_timeouts = 12
        while True:
            if self._stop_event is not None and self._stop_event.is_set():
                raise RuntimeError("Camera stop requested while waiting for frames")
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms)
                self._frame_timeout_streak = 0
                return frames
            except RuntimeError as exc:
                if "Frame didn't arrive" not in str(exc):
                    raise
                self._frame_timeout_streak += 1
                if self.logger is not None:
                    self.logger.warning(
                        "RealSense frame timeout (%d/%d): %s",
                        self._frame_timeout_streak,
                        max_timeouts,
                        exc,
                    )
                if self._frame_timeout_streak >= max_timeouts:
                    raise RuntimeError(
                        f"RealSense stopped delivering frames after "
                        f"{max_timeouts} timeouts"
                    ) from exc
                time.sleep(0.05)

    def _read_and_store_frame(self):
        frames = self._wait_for_color_frames()

        rgb_frame = frames.get_color_frame()
        if not rgb_frame:
            raise RuntimeError("RealSense frame set did not include a color frame")
        rgb_data = np.asanyarray(rgb_frame.get_data())
        rs_ms = float(rgb_frame.get_timestamp())
        frame_number = int(frames.get_frame_number())

        if self._rs_mono_offset_ns is None:
            self._rs_mono_offset_ns = calibrate_realsense_offset(rs_ms)
        capture_mono_ns = realsense_ms_to_capture_ns(rs_ms, self._rs_mono_offset_ns)

        if self._recording_pair_state is not None:
            self._recording_pair_state.publish_camera_frame(capture_mono_ns, rs_ms)

        with self._pair_lock:
            self._frame_ring.append(
                _BufferedCameraFrame(
                    capture_mono_ns=capture_mono_ns,
                    rs_ms=rs_ms,
                    frame_number=frame_number,
                    image=rgb_data.copy(),
                )
            )

        assert self._start_time is not None, "Start time is not initialized"
        timestamp = (capture_mono_ns / 1_000_000_000.0) - self._start_time

        # Legacy fallback when paired recording SHM is unavailable.
        if self._is_recording and self._recording_pair_state is None:
            rel_ts = relative_timestamp_s(
                capture_mono_ns,
                self._recording_epoch_ns or int(self._start_time * 1_000_000_000),
            )
            try:
                if not os.path.exists(self._dest_dir):
                    os.makedirs(self._dest_dir, exist_ok=True)
            except Exception:
                pass
            legacy_ts = rel_ts if self._recording_epoch_ns > 0 else timestamp
            filename = generate_camera_filename(legacy_ts, frame_number)
            filepath = os.path.join(self._dest_dir, filename)
            cv2.imwrite(filepath, rgb_data)
            if self.logger:
                self.logger.debug(f"Saved data to {filepath}")

        return D455Frame(timestamp, rgb_data.copy())

    def _send_frame(self, stream_queue: multiprocessing.Queue, stop_event):
        assert self._frame_queue is not None, "Frame queue is not initialized"
        while not stop_event.is_set():
            self._check_control_commands()
            self._process_pair_save_requests()

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
                    # Count drops due to downstream queue full/busy
                    try:
                        self._drops_total += 1
                    except Exception:
                        pass
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
        self._stop_event = stop_event

        # Avoid hang on process exit if consumer stops reading the queue.
        # This prevents waiting for the queue's feeder thread during interpreter shutdown.
        try:
            stream_queue.cancel_join_thread()
        except Exception:
            pass

        # Store control queue reference
        self._control_queue = control_queue
        if self._config.recording_pair_meta is not None:
            try:
                self._recording_pair_state = RecordingPairState.attach(
                    self._config.recording_pair_meta
                )
            except Exception as exc:
                self._recording_pair_state = None
                if self.logger:
                    self.logger.warning(
                        "Paired recording disabled; SHM attach failed: %s", exc
                    )

        # Initialize components in target process
        self._start_time = (
            float(self._config.timestamp_origin)
            if self._config.timestamp_origin is not None
            else capture_clock_ns() / 1_000_000_000.0
        )
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
                # Attach diagnostics
                try:
                    self._seq_counter += 1
                    video_frame.seq = self._seq_counter
                    video_frame.drops_total = self._drops_total
                except Exception:
                    pass
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
                        # Count drop due to local queue full
                        try:
                            self._drops_total += 1
                        except Exception:
                            pass
                    except queue.Empty:
                        pass
                    try:
                        self._frame_queue.put_nowait(video_frame)
                    except Exception:
                        pass
            except RuntimeError as exc:
                if stop_event.is_set():
                    break
                if self.logger is not None:
                    self.logger.error("Camera capture failed: %s", exc)
                stop_event.set()
                break
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received, stopping...")
                stop_event.set()

        if self._pipeline is not None:
            self._pipeline.stop()

        self.logger.info("Stopped")

        sys.exit(0)
