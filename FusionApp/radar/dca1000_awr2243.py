import os
import time
import queue
import multiprocessing
import logging
import glob
import re
from enum import Enum
from mmwave.dataloader import DCA1000
import fpga_udp as radar
import threading
from typing import Optional, List, Tuple
from numpy import ndarray
import numpy as np

from config_params import CFGS
from engine.interfaces import RadarFeed
from utils import setup_logger
from engine.sync_state import SyncStateUtils, PlaybackState as SyncPlaybackState


class DCA1000Config:
    def __init__(
        self,
        cli_port: str = CFGS.AWR_CLI_PORT,
        data_port: str = CFGS.AWR_DATA_PORT,
        dca_config_file: str = CFGS.DCA_CONFIG_FILE,
        radar_config_file: str = CFGS.AWR2243_CONFIG_FILE,
        dest_dir: str = CFGS.DEST_DIR,
    ):
        self.cli_port = cli_port
        self.data_port = data_port
        self.dca_config_file = dca_config_file
        self.radar_config_file = radar_config_file
        self.dest_dir = dest_dir


class DCA1000Frame:
    def __init__(self, timestamp: float, data: ndarray):
        self.data: ndarray = data
        self.timestamp: float = timestamp


class DCA1000EVM(RadarFeed):
    def __init__(self, dca1000_config: DCA1000Config = DCA1000Config()):
        # Store only serializable configuration
        self._config = dca1000_config
        self._dest_dir = dca1000_config.dest_dir

        # Initialize these in run() method
        self._start_time: Optional[float] = None
        self._frame_queue: Optional[queue.Queue] = None
        self._dca: Optional[DCA1000] = None
        self._ADC_PARAMS_l: Optional[dict] = None
        self._last_frame_number = 0
        self.logger = None
        self._send_thread: Optional[threading.Thread] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._dca is not None:
            self._dca.fastRead_in_Cpp_thread_stop()
            self._dca.stream_stop()
            self._dca.close()
            radar.AWR2243_poweroff()
        if self.logger is not None:
            self.logger.info("***** DCA1000 cleaned up. *****")

    def _read_and_store_frame(self):
        # Read the data from the DCA1000
        start = time.perf_counter()
        assert self._dca is not None, "DCA1000 is not initialized"
        data_buf = self._dca.fastRead_in_Cpp_thread_get()
        end = time.perf_counter()

        # print(f"***** DCA1000EVM **** Read {data_buf.nbytes/1024:.3f} KBs in {end-start:.6f}")
        # print(f"***** DCA1000EVM **** Bandwidth: {data_buf.nbytes/(end-start)/1e6:.4f} MB/s")

        assert self._start_time is not None, "Start time is not initialized"
        timestamp = end - self._start_time
        integer_part = f"{int(timestamp):010d}"
        fraction_part = f"{int((timestamp - int(timestamp)) * 1e5):05d}"
        frame_number = f"{self._last_frame_number:012d}"
        filename = f"{integer_part}_{fraction_part}_{frame_number}.bin"
        filepath = os.path.join(self._dest_dir, filename)

        with open(filepath, "wb") as bin_file:
            data_buf.tofile(bin_file)

        # print(f"***** DCA1000EVM **** Saved data to {filepath}")

        return DCA1000Frame(timestamp, data_buf)

    def _send_frame(self, stream_queue: multiprocessing.Queue, stop_event):
        assert self._frame_queue is not None, "Frame queue is not initialized"
        while not stop_event.is_set():
            # Wait for a frame to be available
            try:
                dca_frame = self._frame_queue.get(timeout=1)
                # print(
                #     f"***** DCA1000EVM **** Sending frame: {dca_frame.timestamp}, messages left in queue: {self._frame_queue.qsize()}"
                # )

                # Send the frame data over the multiprocessing queue
                try:
                    stream_queue.put(dca_frame)
                except Exception as e:
                    if self.logger is not None:
                        self.logger.error(f"Error sending data: {e}")
                    break
            except queue.Empty:
                # No frame available, continue
                if self.logger is not None:
                    self.logger.warning("Warning: No frame available to send.")
                continue
            except KeyboardInterrupt:
                if self.logger is not None:
                    self.logger.info("Keyboard interrupt received, stopping...")
                stop_event.set()
                break

    def run(
        self,
        stream_queue: multiprocessing.Queue,
        stop_event,
        control_queue: Optional[multiprocessing.Queue] = None,
        status_queue: Optional[multiprocessing.Queue] = None,
    ):
        # Initialize logger and components in target process
        self.logger = setup_logger("DCA1000EVM")
        self.logger.info("Starting... *****")

        # Create destination directory if it doesn't exist
        if not os.path.exists(self._dest_dir):
            os.makedirs(self._dest_dir, exist_ok=True)
            self.logger.info(f"Created destination directory: {self._dest_dir}")
        else:
            self.logger.info(f"Using existing destination directory: {self._dest_dir}")

        self._start_time = time.perf_counter()
        self._frame_queue = queue.Queue()

        self._dca = DCA1000()

        self._dca.reset_radar()
        self._dca.reset_fpga()
        self.logger.info("Waiting 1s for radar and FPGA reset...")
        time.sleep(1)

        radar.AWR2243_init(self._config.radar_config_file)

        radar.AWR2243_setFrameCfg(0)

        (
            LVDSDataSizePerChirp_l,
            maxSendBytesPerChirp_l,
            self._ADC_PARAMS_l,
            CFG_PARAMS_l,
        ) = self._dca.AWR2243_read_config(self._config.radar_config_file)
        self._dca.refresh_parameter()

        self.logger.info(
            "LVDSDataSizePerChirp:%d must <= maxSendBytesPerChirp:%d"
            % (LVDSDataSizePerChirp_l, maxSendBytesPerChirp_l)
        )

        self.logger.info("System connection check: %s", self._dca.sys_alive_check())
        self.logger.info(self._dca.read_fpga_version())
        self.logger.info(
            "Config fpga: %s", self._dca.config_fpga(self._config.dca_config_file)
        )
        self.logger.info(
            "Config record packet delay: %s",
            self._dca.config_record(self._config.dca_config_file),
        )

        # Pass ADC parameters to the stream_queue before streaming data
        stream_queue.put({"ADC_PARAMS": self._ADC_PARAMS_l})

        self._dca.stream_start()
        self._dca.fastRead_in_Cpp_thread_start()

        radar.AWR2243_sensorStart()

        # Create and start a thread for sending frames
        self._send_thread = threading.Thread(
            target=self._send_frame,
            name="DCA1000SendThread",
            args=(
                stream_queue,
                stop_event,
            ),
        )
        self._send_thread.start()

        while not stop_event.is_set():
            try:
                # Update the data and check if the data is okay
                radar_frame = self._read_and_store_frame()

                self._frame_queue.put(radar_frame)

                self._last_frame_number += 1
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received, stopping...")
                stop_event.set()
                break

        if self._dca is not None:
            self._dca.fastRead_in_Cpp_thread_stop()
            self._dca.stream_stop()
            self._dca.close()
            radar.AWR2243_poweroff()

        self.logger.info("***** DCA1000 stopped. *****")


class PlaybackState(Enum):
    """Playback state enumeration"""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class DCA1000Recording(RadarFeed):
    """
    DCA1000 Recording playback class that reads recorded .bin files from a directory
    and plays them back with timing control, navigation, and play/pause functionality.
    """

    def __init__(
        self, dca1000_config: DCA1000Config = DCA1000Config(), sync_state=None
    ):
        # Store only serializable configuration
        self._config = dca1000_config
        self._dest_dir = dca1000_config.dest_dir
        self._sync_state = sync_state

        # Initialize these in run() method
        self._frame_queue: Optional[queue.Queue] = None
        self._send_thread: Optional[threading.Thread] = None
        self.logger = None

        # Playback control
        self._playback_state = PlaybackState.STOPPED
        self._current_frame_index = 0
        self._frame_rate = 10.0  # Default frame rate, will be updated from config
        self._frame_files: List[Tuple[str, float, int]] = (
            []
        )  # (filepath, timestamp, frame_number)

        # ADC parameters
        self._ADC_PARAMS_l: Optional[dict] = None

    def _initialize(self):
        """Initialize the recording playback by scanning files and loading config"""
        if self.logger is not None:
            self.logger.info("Initializing DCA1000 Recording playback...")
        self._scan_recording_files()
        self._load_radar_config()
        if self.logger is not None:
            self.logger.info(f"Found {len(self._frame_files)} frame files for playback")

        # Signal readiness for synchronized mode
        if self._sync_state is not None:
            SyncStateUtils.signal_feed_ready(self._sync_state)
            self.logger.info("Signaled feed readiness for synchronization")

    def _scan_recording_files(self):
        """Scan the destination directory for .bin files matching the naming pattern"""
        if not os.path.exists(self._dest_dir):
            raise FileNotFoundError(
                f"Recording directory does not exist: {self._dest_dir}"
            )

        # Pattern: {timestamp_int}_{timestamp_frac}_{frame_number}.bin
        pattern = os.path.join(self._dest_dir, "*.bin")
        bin_files = glob.glob(pattern)

        if not bin_files:
            raise FileNotFoundError(
                f"No .bin files found in directory: {self._dest_dir}"
            )

        # Parse filenames and extract timing information
        frame_info = []
        filename_pattern = re.compile(r"(\d{10})_(\d{5})_(\d{12})\.bin$")

        for filepath in bin_files:
            filename = os.path.basename(filepath)
            match = filename_pattern.match(filename)

            if match:
                timestamp_int = int(match.group(1))
                timestamp_frac = int(match.group(2))
                frame_number = int(match.group(3))

                # Reconstruct timestamp
                timestamp = timestamp_int + (timestamp_frac / 1e5)

                frame_info.append((filepath, timestamp, frame_number))
            else:
                self.logger.warning(
                    f"Skipping file with invalid naming pattern: {filename}"
                )

        if not frame_info:
            raise ValueError(
                f"No valid .bin files found with correct naming pattern in: {self._dest_dir}"
            )

        # Sort by timestamp to ensure proper playback order
        frame_info.sort(key=lambda x: x[1])  # Sort by timestamp
        self._frame_files = frame_info

        self.logger.info(f"Scanned {len(self._frame_files)} valid frame files")

    def _load_radar_config(self):
        """Load radar configuration to extract frame rate and ADC parameters"""
        (
            LVDSDataSizePerChirp_l,
            maxSendBytesPerChirp_l,
            self._ADC_PARAMS_l,
            CFG_PARAMS_l,
        ) = DCA1000.AWR2243_read_config(self._config.radar_config_file)

        # Extract frame rate from configuration
        if "frame_periodicity" in CFG_PARAMS_l:
            self._frame_rate = CFG_PARAMS_l["frame_periodicity"] / 5

            self.logger.info(f"Extracted frame rate: {self._frame_rate:.2f} Hz")
        else:
            self.logger.warning(
                "frameCfg not found in config, using default frame rate"
            )

    def _read_frame_from_file(self, filepath: str) -> DCA1000Frame:
        """Read a frame from a .bin file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Frame file not found: {filepath}")

        # Extract timestamp from filepath
        filename = os.path.basename(filepath)
        filename_pattern = re.compile(r"(\d{10})_(\d{5})_(\d{12})\.bin$")
        match = filename_pattern.match(filename)

        if not match:
            raise ValueError(f"Invalid filename pattern: {filename}")

        timestamp_int = int(match.group(1))
        timestamp_frac = int(match.group(2))
        timestamp = timestamp_int + (timestamp_frac / 1e5)

        # Read binary data
        data_buf = np.fromfile(filepath, dtype=np.int16)

        if data_buf.size == 0:
            raise ValueError(f"Empty or corrupted frame file: {filepath}")

        return DCA1000Frame(timestamp, data_buf)

    def _send_frame(self, stream_queue: multiprocessing.Queue, stop_event):
        """Send frames to the radar stream queue with synchronized timing control"""
        self.logger.info("Starting frame sender thread...")

        # Determine if we're using synchronized mode
        use_sync = self._sync_state is not None
        last_timeline_position = 0.0  # Track timeline position for seeking detection

        if use_sync:
            self.logger.info("Using synchronized timing mode")
            # Wait for the start signal from sync state
            if not SyncStateUtils.wait_for_start_signal(self._sync_state, timeout=30):
                self.logger.warning(
                    "Timeout waiting for start signal, proceeding anyway"
                )
        else:
            self.logger.info("Using frame rate-based timing mode")

        while not stop_event.is_set():
            try:
                # Check synchronized playback state if available
                if use_sync:
                    sync_playback_state = SyncStateUtils.get_playback_state(
                        self._sync_state
                    )
                    is_playing = sync_playback_state == SyncPlaybackState.PLAYING

                    # Check for seeking (timeline position jumped significantly)
                    current_timeline = SyncStateUtils.get_current_timeline_position(
                        self._sync_state
                    )
                    timeline_diff = abs(current_timeline - last_timeline_position)

                    # Detect seeking (timeline position jumped significantly) or reset to beginning
                    if timeline_diff > 1.0 or (
                        current_timeline == 0.0 and last_timeline_position > 1.0
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
                        self.logger.info(
                            f"Seeked to frame {best_index} (timestamp: {target_timestamp:.3f}s, timeline: {current_timeline:.3f}s)"
                        )

                    last_timeline_position = current_timeline
                else:
                    is_playing = self._playback_state == PlaybackState.PLAYING

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
                        continue

                    # Get the current frame information
                    filepath, frame_timestamp, _ = self._frame_files[
                        self._current_frame_index
                    ]

                    if use_sync:
                        # Synchronized timing mode
                        start_timestamp = SyncStateUtils.get_start_timestamp(
                            self._sync_state
                        )
                        relative_frame_time = frame_timestamp - start_timestamp

                        # Wait until the shared timeline reaches this frame's time
                        while not stop_event.is_set():
                            current_timeline = (
                                SyncStateUtils.get_current_timeline_position(
                                    self._sync_state
                                )
                            )

                            # Check if it's time to send this frame (or past time)
                            if current_timeline >= relative_frame_time:
                                break

                            # Check if playback was paused while waiting
                            if (
                                SyncStateUtils.get_playback_state(self._sync_state)
                                != SyncPlaybackState.PLAYING
                            ):
                                break

                            # Sleep briefly to avoid busy waiting
                            time.sleep(0.001)

                        # Check if we should still send the frame (playback might have been paused/stopped)
                        if (
                            stop_event.is_set()
                            or SyncStateUtils.get_playback_state(self._sync_state)
                            != SyncPlaybackState.PLAYING
                        ):
                            continue
                    else:
                        # Legacy frame rate-based timing mode
                        # Wait for stream_queue to be empty (or nearly empty)
                        while not stop_event.is_set() and stream_queue.qsize() > 1:
                            time.sleep(0.001)  # Small delay to avoid busy waiting

                        if stop_event.is_set():
                            break

                    # Read and send the current frame
                    try:
                        frame = self._read_frame_from_file(filepath)
                        stream_queue.put(frame)

                        self.logger.debug(
                            f"Sent frame {self._current_frame_index}: {os.path.basename(filepath)} "
                            f"(timestamp: {frame_timestamp:.3f})"
                        )

                        # Update sync state tracking if available
                        if use_sync:
                            self._sync_state.last_radar_timestamp.value = (
                                frame_timestamp
                            )

                        # Advance to next frame
                        self._current_frame_index += 1

                        # Send status update every 5 frames for smoother progress
                        if self._current_frame_index % 5 == 0:
                            self._send_status_update()

                        # Legacy timing: Wait for frame period before sending next frame
                        if not use_sync:
                            frame_period = 1.0 / self._frame_rate
                            time.sleep(frame_period)

                    except Exception as e:
                        self.logger.error(
                            f"Failed to send frame {self._current_frame_index}: {e}"
                        )
                        # Skip this frame and continue
                        self._current_frame_index += 1

                else:
                    # Playback is paused or stopped - just wait
                    time.sleep(0.01)

            except Exception as e:
                self.logger.error(f"Error in frame sender thread: {e}")
                break

        self.logger.info("Frame sender thread stopped")

    def play(self):
        """Start or resume playback"""
        if self._current_frame_index >= len(self._frame_files):
            self.logger.warning("Cannot play: at end of recording")
            return

        self._playback_state = PlaybackState.PLAYING
        self.logger.info(f"Playback started from frame {self._current_frame_index}")
        self._send_status_update()

    def pause(self):
        """Pause playback"""
        self._playback_state = PlaybackState.PAUSED
        self.logger.info("Playback paused")
        self._send_status_update()

    def stop(self):
        """Stop playback and reset to beginning"""
        self._playback_state = PlaybackState.STOPPED
        self._current_frame_index = 0
        self.logger.info("Playback stopped and reset to beginning")
        self._send_status_update()

    def seek_to_frame(self, frame_index: int):
        """Seek to a specific frame index"""
        if frame_index < 0 or frame_index >= len(self._frame_files):
            raise ValueError(
                f"Frame index {frame_index} out of range [0, {len(self._frame_files)-1}]"
            )

        self._current_frame_index = frame_index
        self.logger.info(f"Seeked to frame {frame_index}")

    def seek_to_time(self, timestamp: float):
        """Seek to a specific timestamp"""
        # Find the frame closest to the requested timestamp
        best_index = 0
        best_diff = float("inf")

        for i, (_, ts, _) in enumerate(self._frame_files):
            diff = abs(ts - timestamp)
            if diff < best_diff:
                best_diff = diff
                best_index = i

        self.seek_to_frame(best_index)
        self.logger.info(f"Seeked to timestamp {timestamp:.3f}s (frame {best_index})")

    def seek_to_percent(self, percent: float):
        """Seek to a specific percentage of the recording (0-100)"""
        if percent < 0 or percent > 100:
            raise ValueError(f"Percent {percent} out of range [0, 100]")

        if not self._frame_files:
            self.logger.warning("No frame files available for seeking")
            return

        # Calculate target frame index based on percentage
        max_index = len(self._frame_files) - 1
        target_index = int((percent / 100.0) * max_index)

        # Clamp to valid range
        target_index = max(0, min(target_index, max_index))

        self.seek_to_frame(target_index)
        self.logger.info(f"Seeked to {percent:.1f}% (frame {target_index}/{max_index})")

    def get_current_frame_info(self) -> Optional[Tuple[int, float, str]]:
        """Get information about current frame: (index, timestamp, filename)"""
        if self._current_frame_index >= len(self._frame_files):
            return None

        filepath, timestamp, _ = self._frame_files[self._current_frame_index]
        filename = os.path.basename(filepath)
        return (self._current_frame_index, timestamp, filename)

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

    def run(
        self,
        stream_queue: multiprocessing.Queue,
        stop_event,
        control_queue: Optional[multiprocessing.Queue] = None,
        status_queue: Optional[multiprocessing.Queue] = None,
    ):
        """Main run method similar to DCA1000EVM with playback control support"""
        # Initialize logger and scanner in target process
        self.logger = setup_logger("DCA1000Recording")
        self._initialize()

        self.logger.info("Starting DCA1000 Recording playback...")

        # Send ADC parameters first (similar to live DCA1000EVM)
        if self._ADC_PARAMS_l is None:
            raise RuntimeError("ADC parameters not loaded")

        stream_queue.put({"ADC_PARAMS": self._ADC_PARAMS_l})
        self.logger.info("Sent ADC parameters to processing queue")

        # Create and start the frame sender thread
        self._status_queue = status_queue  # Store reference for status updates
        self._send_thread = threading.Thread(
            target=self._send_frame,
            name="DCA1000RecordingSendThread",
            args=(stream_queue, stop_event),
        )
        self._send_thread.start()

        # Note: Playback will be controlled by sync_state or control commands

        # Main loop - handle control commands and wait for stop event
        while not stop_event.is_set():
            try:
                # Check for control commands
                if control_queue is not None:
                    try:
                        command = control_queue.get_nowait()
                        self._handle_control_command(command)
                    except queue.Empty:
                        pass

                time.sleep(0.1)
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received, stopping playback...")
                stop_event.set()
                break

        # Wait for sender thread to finish
        if self._send_thread and self._send_thread.is_alive():
            self._send_thread.join(timeout=5.0)

        self.logger.info("DCA1000 Recording playback stopped")

    def _handle_control_command(self, command):
        """Handle playback control commands from UI"""
        self.logger.info(f"Received control command: {command}")

        # Check if command is a string (expected) or something else
        if not isinstance(command, str):
            self.logger.warning(
                f"Expected string command, got {type(command)}: {command}"
            )
            return

        # In synchronized mode, controls should update sync_state; otherwise use internal state
        use_sync = self._sync_state is not None

        if command == "play":
            if use_sync:
                SyncStateUtils.set_playback_state(
                    self._sync_state, SyncPlaybackState.PLAYING
                )
            else:
                self.play()
            self._send_status_update()
            self.logger.info("Playback started")
        elif command == "pause":
            if use_sync:
                SyncStateUtils.set_playback_state(
                    self._sync_state, SyncPlaybackState.PAUSED
                )
            else:
                self.pause()
            self._send_status_update()
            self.logger.info("Playback paused")
        elif command == "stop":
            if use_sync:
                SyncStateUtils.set_playback_state(
                    self._sync_state, SyncPlaybackState.STOPPED
                )
                # Reset timeline to beginning
                SyncStateUtils.seek_to_time(self._sync_state, 0.0)
            else:
                self.stop()
            self._send_status_update()
            self.logger.info("Playback stopped")
        elif command.startswith("seek:"):
            position_str = command.split(":")[1]
            try:
                position = int(position_str)
                if use_sync:
                    # In synchronized mode, convert frame to timeline position
                    if 0 <= position < len(self._frame_files):
                        _, target_timestamp, _ = self._frame_files[position]
                        start_timestamp = SyncStateUtils.get_start_timestamp(
                            self._sync_state
                        )
                        relative_time = target_timestamp - start_timestamp
                        SyncStateUtils.seek_to_time(self._sync_state, relative_time)
                        self.logger.info(
                            f"Seeked to timeline position {relative_time:.3f}s (frame {position})"
                        )
                    else:
                        self.logger.error(
                            f"Frame index {position} out of range [0, {len(self._frame_files)-1}]"
                        )
                else:
                    # Legacy mode - seek to frame directly
                    self.seek_to_frame(position)
                    self.logger.info(f"Seeked to frame {position}")
                self._send_status_update()
            except ValueError:
                self.logger.error(f"Invalid seek position: {position_str}")
        else:
            self.logger.warning(f"Unknown control command: {command}")

    def _send_status_update(self):
        """Send current playback status to the status queue"""
        if self._status_queue is not None:
            try:
                total_frames = len(self._frame_files)
                current_frame = self._current_frame_index
                progress_percent = (
                    (current_frame / total_frames * 100) if total_frames > 0 else 0
                )

                # Get state from sync_state if available, otherwise use internal state
                if self._sync_state is not None:
                    # Use sync state
                    sync_playback_state = SyncStateUtils.get_playback_state(
                        self._sync_state
                    )
                    if sync_playback_state == SyncPlaybackState.PLAYING:
                        state_name = "PLAYING"
                    elif sync_playback_state == SyncPlaybackState.PAUSED:
                        state_name = "PAUSED"
                    else:
                        state_name = "STOPPED"
                else:
                    # Use internal state
                    state_name = self._playback_state.name

                status = {
                    "state": state_name,
                    "current_frame": current_frame,
                    "total_frames": total_frames,
                    "progress_percent": progress_percent,
                }

                self._status_queue.put_nowait(status)
            except queue.Full:
                # Don't block on status updates
                pass
            except Exception as e:
                self.logger.warning(f"Failed to send status update: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.logger.info("DCA1000 Recording cleaned up")
