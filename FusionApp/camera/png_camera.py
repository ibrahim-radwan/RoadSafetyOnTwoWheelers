# PNGCamera implementation

import os
import time
import queue
import glob
from typing import Optional, List, Tuple
import threading
import multiprocessing
import os
import numpy as np
import cv2
from enum import Enum
import logging

from config_params import CFGS
from engine.interfaces import CameraFeed
from camera.d455 import D455Frame
from utils import setup_logger
from engine.sync_state import SyncStateUtils, PlaybackState as SyncPlaybackState


class PlaybackState(Enum):
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class PNGCameraConfig:
    def __init__(
        self,
        dest_dir: str = CFGS.DEST_DIR,
    ):
        self.dest_dir = dest_dir


class PNGCamera(CameraFeed):
    """
    Camera recording class that reads from previously captured PNG files.
    Implements the same interface as D455 but for replay mode.
    """

    def __init__(self, config: PNGCameraConfig = PNGCameraConfig(), sync_state=None):
        # Store only serializable configuration
        self._config = config
        self._recording_dir = config.dest_dir
        self._sync_state = sync_state

        # Initialize these in run() method
        self._frame_files: List[Tuple[str, float, str]] = (
            []
        )  # (filepath, timestamp, filename)
        self._current_frame_index = 0
        self._playback_state = PlaybackState.STOPPED
        self._frame_rate = 30.0  # Default frame rate
        self._last_frame_time = 0.0
        self.logger = None
        self._fps_divisor = 1

    def _load_frame_files(self):
        """Load all PNG files from the recording directory"""
        if not os.path.exists(self._recording_dir):
            raise FileNotFoundError(
                f"Recording directory not found: {self._recording_dir}"
            )

        # Find all PNG files
        png_files = glob.glob(os.path.join(self._recording_dir, "*.png"))

        # Parse timestamps from filenames and sort
        for filepath in png_files:
            filename = os.path.basename(filepath)
            try:
                # Parse timestamp from filename format: {integer_part}_{fraction_part}_{frame_number}.png
                parts = filename.split("_")
                if len(parts) >= 3:
                    integer_part = int(parts[0])
                    fraction_part = int(parts[1])
                    timestamp = integer_part + fraction_part / 100000.0
                    self._frame_files.append((filepath, timestamp, filename))
            except (ValueError, IndexError):
                self.logger.warning(
                    f"Could not parse timestamp from filename: {filename}"
                )
                continue

        # Sort by timestamp
        self._frame_files.sort(key=lambda x: x[1])
        self.logger.info(
            f"Loaded {len(self._frame_files)} camera frames from {self._recording_dir}"
        )

        # Signal readiness for synchronized mode
        if self._sync_state is not None:
            SyncStateUtils.signal_feed_ready(self._sync_state)
            self.logger.info("Signaled feed readiness for synchronization")

    def _read_current_frame(self) -> Optional[D455Frame]:
        """Read the current frame from file"""
        if self._current_frame_index >= len(self._frame_files):
            return None

        filepath, timestamp, filename = self._frame_files[self._current_frame_index]

        try:
            # Load image
            image = cv2.imread(filepath)
            if image is None:
                self.logger.error(f"Could not load image from {filepath}")
                return None

            self.logger.debug(
                f"Read frame from disk: timestamp={timestamp}, filename={filename}"
            )
            return D455Frame(timestamp, image)
        except Exception as e:
            self.logger.error(f"Error reading frame {filepath}: {e}")
            return None

    def _advance_frame(self):
        """Advance to the next frame"""
        if self._current_frame_index < len(self._frame_files) - self._fps_divisor:
            self._current_frame_index += self._fps_divisor
        else:
            # End of recording
            self._playback_state = PlaybackState.STOPPED

    def run(
        self,
        stream_queue: multiprocessing.Queue,
        stop_event,
        control_queue: Optional[multiprocessing.Queue] = None,
    ):
        """Main playback loop for recorded camera frames with synchronized timing"""

        # Initialize logger and load frame files in target process
        self.logger = setup_logger("PNGCamera")
        self._load_frame_files()

        # Determine if we're using synchronized mode
        use_sync = self._sync_state is not None
        if use_sync:
            self.logger.info("Using synchronized timing mode")
            # Wait for the start signal from sync state (this just means sync is ready, not to start playing)
            if not SyncStateUtils.wait_for_start_signal(self._sync_state, timeout=30):
                self.logger.warning(
                    "Timeout waiting for start signal, proceeding anyway"
                )
            self.logger.info("Synchronized mode ready - waiting for play command...")
            # In sync mode, don't auto-start - wait for user to press play
            self._playback_state = PlaybackState.STOPPED
        else:
            self.logger.info("Using frame rate-based timing mode")
            # In non-sync mode, start playing immediately
            self._playback_state = PlaybackState.PLAYING

        self._last_frame_time = time.perf_counter()
        frame_count = 0
        last_timeline_position = 0.0  # Track timeline position for seeking detection
        last_sync_playback_state = None

        # DEBUG: Log initial state
        self.logger.debug(
            f"Initial playback state: frame_count={frame_count}, _current_frame_index={self._current_frame_index}, total_frames={len(self._frame_files)}, playback_state={self._playback_state}"
        )

        while not stop_event.is_set():
            try:
                # Handle external control commands if provided
                if control_queue is not None:
                    try:
                        cmd = control_queue.get_nowait()
                        if isinstance(cmd, dict) and cmd.get("STOP"):
                            self.logger.info("Received STOP sentinel; exiting")
                            break
                        if isinstance(cmd, str):
                            # In synchronized mode, GUI updates SyncState; ignore here.
                            # In legacy mode, handle basic controls locally.
                            if not use_sync:
                                if cmd == "play":
                                    self._playback_state = PlaybackState.PLAYING
                                    self._last_frame_time = time.perf_counter()
                                elif cmd == "pause":
                                    self._playback_state = PlaybackState.PAUSED
                                elif cmd == "stop":
                                    self._playback_state = PlaybackState.STOPPED
                                    self._current_frame_index = 0
                                elif cmd.startswith("seek:"):
                                    try:
                                        frame_index = int(cmd.split(":")[1])
                                        if 0 <= frame_index < len(self._frame_files):
                                            self._current_frame_index = frame_index
                                    except Exception:
                                        pass
                    except queue.Empty:
                        pass
                # Check synchronized playback state if available
                if use_sync:
                    sync_playback_state = SyncStateUtils.get_playback_state(
                        self._sync_state
                    )
                    # Reset index when transitioning to STOPPED so restart begins at frame 0
                    if (
                        sync_playback_state == SyncPlaybackState.STOPPED
                        and last_sync_playback_state != SyncPlaybackState.STOPPED
                    ):
                        self._current_frame_index = 0
                    is_playing = sync_playback_state == SyncPlaybackState.PLAYING
                    last_sync_playback_state = sync_playback_state
                else:
                    is_playing = self._playback_state == PlaybackState.PLAYING

                # Check for seeking in synchronized mode
                if use_sync:
                    current_timeline = SyncStateUtils.get_current_timeline_position(
                        self._sync_state
                    )
                    # Detect seeking or timeline reset (backward move, large jump, or reset)
                    timeline_diff = abs(current_timeline - last_timeline_position)
                    if (
                        current_timeline < last_timeline_position - 0.05
                        or timeline_diff > 0.5
                        or current_timeline == 0.0
                    ):
                        # Timeline jumped or reset - find the correct frame to seek to
                        start_timestamp = SyncStateUtils.get_start_timestamp(
                            self._sync_state
                        )
                        target_timestamp = start_timestamp + current_timeline

                        # Find closest frame to target timestamp
                        best_index = 0
                        best_diff = float("inf")
                        for i, (_, timestamp, _) in enumerate(self._frame_files):
                            diff = abs(timestamp - target_timestamp)
                            if diff < best_diff:
                                best_diff = diff
                                best_index = i

                        self._current_frame_index = best_index
                        self.logger.debug(
                            f"Seeked to frame {best_index} (timestamp: {target_timestamp:.3f}s, timeline: {current_timeline:.3f}s)"
                        )

                    last_timeline_position = current_timeline

                # DEBUG: Log state at each loop
                self.logger.debug(
                    f"Loop: frame_count={frame_count}, _current_frame_index={self._current_frame_index}, total_frames={len(self._frame_files)}, is_playing={is_playing}"
                )

                if is_playing:
                    # Check if we have more frames to play
                    if self._current_frame_index >= len(self._frame_files):
                        self.logger.info("Reached end of recording, stopping playback")
                        if use_sync:
                            SyncStateUtils.set_playback_state(
                                self._sync_state, SyncPlaybackState.STOPPED
                            )
                        else:
                            self._playback_state = PlaybackState.STOPPED
                        break

                    # Get the current frame information
                    filepath, frame_timestamp, filename = self._frame_files[
                        self._current_frame_index
                    ]

                    send_frame = False

                    if use_sync:
                        # Synchronized timing mode
                        start_timestamp = SyncStateUtils.get_start_timestamp(
                            self._sync_state
                        )
                        relative_frame_time = frame_timestamp - start_timestamp

                        # Check if it's time to send this frame (or past time)
                        current_timeline = SyncStateUtils.get_current_timeline_position(
                            self._sync_state
                        )
                        if current_timeline >= relative_frame_time:
                            send_frame = True
                    else:
                        # Legacy frame rate-based timing mode
                        current_time = time.perf_counter()
                        time_since_last_frame = current_time - self._last_frame_time
                        if time_since_last_frame >= (1.0 / self._frame_rate):
                            send_frame = True

                    if send_frame:
                        # Read current frame
                        frame = self._read_current_frame()
                        if frame is not None:
                            try:
                                # Avoid blocking during shutdown if consumer stops
                                stream_queue.put_nowait(frame)
                                self.logger.debug(
                                    f"Sent frame {frame_count} to analyzer: {frame.timestamp}"
                                )
                                frame_count += 1

                                # Update sync state tracking if available
                                if use_sync:
                                    self._sync_state.last_camera_timestamp.value = (
                                        frame_timestamp
                                    )
                                else:
                                    self._last_frame_time = time.perf_counter()

                                # Log every 30 frames to avoid spam
                                if frame_count % 30 == 0:
                                    self.logger.debug(
                                        f"Sent {frame_count} frames to analyzer"
                                    )

                            except Exception as e:
                                # Queue full or closed; skip frame to allow clean shutdown
                                self.logger.warning(
                                    f"Could not send frame (queue busy/closed): {e}"
                                )
                                time.sleep(0.001)

                        # Advance to next frame
                        self._advance_frame()
                    else:
                        # Sleep for a short time to avoid busy waiting
                        time.sleep(0.001)

                else:
                    # Playback is paused or stopped - just wait
                    time.sleep(0.01)

            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received, stopping...")
                stop_event.set()

        # Ensure the multiprocessing queue feeder thread in this process exits
        try:
            if stream_queue is not None:
                stream_queue.close()
                stream_queue.join_thread()
        except Exception:
            pass

        self.logger.info(f"Playback stopped. Sent total {frame_count} frames")

        return

    def get_current_frame_info(self) -> Optional[Tuple[int, float, str]]:
        """Get information about the current frame"""
        if self._current_frame_index < len(self._frame_files):
            filepath, timestamp, filename = self._frame_files[self._current_frame_index]
            return (self._current_frame_index, timestamp, filename)
        return None

    def get_playback_info(self) -> dict:
        """Get comprehensive playback information"""
        current_info = self.get_current_frame_info()

        return {
            "state": self._playback_state.value,
            "current_frame": self._current_frame_index,
            "total_frames": len(self._frame_files),
            "current_timestamp": current_info[1] if current_info else None,
            "current_filename": current_info[2] if current_info else None,
            "frame_rate": self._frame_rate,
            "total_duration": (
                self._frame_files[-1][1] - self._frame_files[0][1]
                if self._frame_files
                else 0
            ),
        }

    def set_frame_rate(self, frame_rate: float):
        """Set the playback frame rate"""
        self._frame_rate = max(
            0.1, min(120.0, frame_rate)
        )  # Clamp between 0.1 and 120 FPS

    def seek_to_frame(self, frame_index: int):
        """Seek to a specific frame"""
        if 0 <= frame_index < len(self._frame_files):
            self._current_frame_index = frame_index

    def pause(self):
        """Pause playback"""
        self._playback_state = PlaybackState.PAUSED

    def resume(self):
        """Resume playback"""
        if self._playback_state == PlaybackState.PAUSED:
            self._playback_state = PlaybackState.PLAYING
            self._last_frame_time = time.perf_counter()

    def stop(self):
        """Stop playback"""
        self._playback_state = PlaybackState.STOPPED

    def reset(self):
        """Reset to beginning"""
        self._current_frame_index = 0
        self._playback_state = PlaybackState.STOPPED

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if self.logger is not None:
            self.logger.info("PNGCamera cleaned up.")
