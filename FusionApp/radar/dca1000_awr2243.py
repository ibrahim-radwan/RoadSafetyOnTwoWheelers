import os
import time
import queue
import multiprocessing
from multiprocessing import shared_memory
from enum import Enum
from mmwave.dataloader import DCA1000
import threading
from typing import Optional, List, Tuple
from numpy import ndarray
import numpy as np

from config_params import CFGS
from engine.interfaces import RadarFeed
from radar.bin_utils import (
    generate_radar_filename,
    scan_bin_directory,
    parse_bin_timestamp,
)
from utils import setup_logger, disable_shm_resource_tracker
from engine.sync_state import SyncStateUtils, PlaybackState as SyncPlaybackState
from recording.clock import capture_clock_ns
from recording.sync_recording import (
    RecordingPairState,
    parse_start_recording_command,
    relative_timestamp_s,
)


class DCA1000Config:
    def __init__(
        self,
        cli_port: str = CFGS.AWR_CLI_PORT,
        data_port: str = CFGS.AWR_DATA_PORT,
        dca_config_file: str = CFGS.DCA_CONFIG_FILE,
        radar_config_file: str = CFGS.AWR2243_CONFIG_FILE,
        dest_dir: str = CFGS.DEST_DIR,
        timestamp_origin: Optional[float] = None,
    ):
        self.cli_port = cli_port
        self.data_port = data_port
        self.dca_config_file = dca_config_file
        self.radar_config_file = radar_config_file
        self.dest_dir = dest_dir
        self.timestamp_origin = timestamp_origin


class DCA1000Frame:
    def __init__(
        self,
        timestamp: float,
        data: ndarray,
        *,
        capture_monotonic_ns: int = None,
        capture_wall_ns: int = None,
        enqueue_monotonic_ns: int = None,
        filepath: Optional[str] = None,
    ):
        self.data: ndarray = data
        # Legacy relative timestamp in seconds (kept for compatibility)
        self.timestamp: float = timestamp
        # Absolute capture times (monotonic for latency, wall for human logs)
        self.capture_monotonic_ns: int = (
            capture_monotonic_ns if capture_monotonic_ns is not None else 0
        )
        self.capture_wall_ns: int = (
            capture_wall_ns if capture_wall_ns is not None else 0
        )
        # Time when queued to inter-process queue (monotonic ns)
        self.enqueue_monotonic_ns: int = (
            enqueue_monotonic_ns if enqueue_monotonic_ns is not None else 0
        )
        # Optional source filepath (used in replay to map to .bin basename)
        self.filepath: Optional[str] = filepath


class DCA1000EVM(RadarFeed):
    """Interface for the TI DCA1000 radar capture card in live-acquisition mode."""

    def __init__(
        self,
        dca1000_config: DCA1000Config = DCA1000Config(),
        *,
        prealloc_shm_meta: Optional[dict] = None,
        recording_pair_meta: Optional[dict] = None,
    ):
        """Capture settings and shared-memory handles are provided by the fusion engine."""
        # Store only serializable configuration
        self._config = dca1000_config
        self._dest_dir = dca1000_config.dest_dir

        # Initialize these in run() method
        self._start_time: Optional[float] = None
        self._dca: Optional[DCA1000] = None
        self._ADC_PARAMS_l: Optional[dict] = None
        self._last_frame_number = 0
        self.logger = None
        # Shared memory for zero-copy IPC
        self._shm_blocks: Optional[List[shared_memory.SharedMemory]] = None
        self._shm_names: Optional[List[str]] = None
        self._shm_nbytes: Optional[int] = None
        self._shm_dtype: Optional[str] = None
        self._shm_shape: Optional[Tuple[int, ...]] = None
        self._shm_seq: int = 0
        self._shm_inited: bool = False
        self._prealloc_shm_meta: Optional[dict] = prealloc_shm_meta
        self._recording_pair_meta: Optional[dict] = recording_pair_meta

        # Enforce that SHM is always preallocated by the parent (engine)
        if self._prealloc_shm_meta is None:
            raise ValueError(
                "prealloc_shm_meta is required; engine must preallocate shared memory"
            )

        # Recording control
        self._is_recording = False
        self._control_queue: Optional[multiprocessing.Queue] = None
        self._recording_pair_state: Optional[RecordingPairState] = None
        self._recording_epoch_ns: int = 0

    def __enter__(self):
        """Support context-manager usage so higher layers can manage setup/teardown."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the physical radar connection and release shared memory blocks."""
        if self._dca is not None:
            self._dca.fastRead_in_Cpp_thread_stop()
            self._dca.stream_stop()
            self._dca.close()
        # Cleanup shared memory blocks if created
        if self._shm_blocks:
            for shm in self._shm_blocks:
                try:
                    shm.close()
                except Exception as e:
                    if self.logger is not None:
                        self.logger.error(
                            f"SHM close failed for raw block ({getattr(shm,'name','?')}): {e}"
                        )
        if self.logger is not None:
            self.logger.info("Cleaned up")

    def _check_control_commands(self):
        """Check for recording control commands"""
        if self._control_queue is None:
            return

        try:
            while True:
                command = self._control_queue.get_nowait()
                # Support dynamic recording dir: start_recording[:<path>]
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
                    self._last_frame_number = 0
                    if self._recording_pair_state is not None:
                        if self._recording_pair_state.recording_epoch_ns() <= 0:
                            self._recording_pair_state.begin_recording(
                                self._recording_epoch_ns
                            )
                    if self.logger:
                        self.logger.info(
                            "Recording started (epoch_ns=%d, paired=%s)",
                            self._recording_epoch_ns,
                            self._recording_pair_state is not None,
                        )
                    else:
                        # Fallback: initialize logger if not available
                        self.logger = setup_logger("DCA1000EVM")
                        self.logger.info("Recording started")
                elif command == "stop_recording":
                    self._is_recording = False
                    self._recording_epoch_ns = 0
                    if self._recording_pair_state is not None:
                        if self._recording_pair_state.recording_epoch_ns() > 0:
                            self._recording_pair_state.end_recording()
                    if self.logger:
                        self.logger.info("Recording stopped")
                    else:
                        # Fallback: initialize logger if not available
                        self.logger = setup_logger("DCA1000EVM")
                        self.logger.info("Recording stopped")
        except queue.Empty:
            pass

    def _read_and_store_frame(self) -> Optional[DCA1000Frame]:
        """Fetch a single radar frame from the card and optionally write it to disk."""
        capture_mono_ns = capture_clock_ns()
        read_start = capture_mono_ns / 1_000_000_000.0
        assert self._dca is not None, "DCA1000 is not initialized"
        data_buf = self._dca.fastRead_in_Cpp_thread_get()
        read_end = time.perf_counter()

        actual_nbytes = int(getattr(data_buf, "nbytes", 0))
        expected_nbytes = int(
            (self._prealloc_shm_meta or {}).get("nbytes") or 0
        )
        if actual_nbytes == 0 or (
            expected_nbytes > 0 and actual_nbytes != expected_nbytes
        ):
            if self._is_recording and self._recording_pair_state is None:
                self._last_frame_number += 1
            if self.logger:
                self.logger.warning(
                    "Dropping incomplete radar frame: received %d bytes, expected %d",
                    actual_nbytes,
                    expected_nbytes,
                )
            return None

        if self.logger:
            self.logger.debug(
                f"Read {actual_nbytes/1024:.3f} KBs in {read_end-read_start:.6f}"
            )
            self.logger.debug(
                f"Bandwidth: {actual_nbytes/(read_end-read_start)/1e6:.4f} MB/s"
            )

        assert self._start_time is not None, "Start time is not initialized"
        timestamp = (capture_mono_ns / 1_000_000_000.0) - self._start_time
        capture_wall_ns = time.time_ns()

        filepath = None
        if self._is_recording:
            try:
                if not os.path.exists(self._dest_dir):
                    os.makedirs(self._dest_dir, exist_ok=True)
            except Exception:
                pass

            if self._recording_pair_state is not None and self._recording_epoch_ns > 0:
                pair_seq, target_camera_mono_ns = (
                    self._recording_pair_state.request_pair_save(capture_mono_ns)
                )
                rel_ts = relative_timestamp_s(
                    capture_mono_ns, self._recording_epoch_ns
                )
                filename = generate_radar_filename(rel_ts, pair_seq)
                filepath = os.path.join(self._dest_dir, filename)
                with open(filepath, "wb") as bin_file:
                    data_buf.tofile(bin_file)
                self._last_frame_number = pair_seq
            else:
                self._last_frame_number += 1
                filename = generate_radar_filename(timestamp, self._last_frame_number)
                filepath = os.path.join(self._dest_dir, filename)
                with open(filepath, "wb") as bin_file:
                    data_buf.tofile(bin_file)

            if self.logger:
                self.logger.debug(f"Saved data to {filepath}")

        return DCA1000Frame(
            timestamp,
            data_buf,
            capture_monotonic_ns=capture_mono_ns,
            capture_wall_ns=capture_wall_ns,
            filepath=filepath,
        )

    def _init_shm_if_needed(self, dca_frame) -> bool:
        """Initialize or attach to shared memory blocks once."""
        if self._shm_inited:
            return True
        try:
            # Attach to preallocated SHM if provided by engine/app
            if self._prealloc_shm_meta:
                names = self._prealloc_shm_meta.get("names")
                nbytes = int(self._prealloc_shm_meta.get("nbytes"))
                dtype_str = str(self._prealloc_shm_meta.get("dtype"))
                shape_tuple = tuple(self._prealloc_shm_meta.get("shape"))
                self._shm_blocks = [
                    shared_memory.SharedMemory(name=name) for name in names
                ]
                self._shm_names = names
                self._shm_nbytes = nbytes
                self._shm_dtype = dtype_str
                self._shm_shape = shape_tuple
                self._shm_inited = True
                if self.logger is not None:
                    self.logger.info(
                        f"Attached preallocated radar SHM: names={names}, bytes={nbytes}, dtype={dtype_str}, shape={shape_tuple}"
                    )
                return True
            # No engine-provided SHM: fail as per policy
            raise RuntimeError("Missing preallocated radar SHM metadata from engine")
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to initialize/attach radar SHM: {e}")
            return False

    def _send_frame(self, stream_queue: multiprocessing.Queue, stop_event):
        """Loop that copies captured frames into shared memory and notifies the engine."""
        # Producer-consumer thread removed; send directly from capture loop via a small local buffer
        while not stop_event.is_set():
            self._check_control_commands()
            try:
                # Read next frame
                dca_frame = self._read_and_store_frame()
                if dca_frame is None:
                    continue
                # Initialize or attach shared memory on first frame
                if not self._shm_inited:
                    if not self._init_shm_if_needed(dca_frame):
                        continue
                    # With engine-owned SHM, no need to send SHM_INIT meta
                    try:
                        stream_queue.put_nowait({"ADC_PARAMS": self._ADC_PARAMS_l})
                    except Exception as e2:
                        if self.logger is not None:
                            self.logger.warning(
                                f"ADC_PARAMS meta drop: {type(e2).__name__}: {e2}"
                            )

                # Copy into alternating SHM slot and send compact metadata
                try:
                    slot = self._shm_seq & 1
                    mv = memoryview(self._shm_blocks[slot].buf)[: self._shm_nbytes]
                    mv[:] = dca_frame.data.tobytes()
                    try:
                        mv.release()
                    except Exception as e:
                        if self.logger is not None:
                            self.logger.error(
                                f"Radar SHM memoryview release failed: {e}"
                            )
                    seq = self._shm_seq
                    self._shm_seq += 1
                    meta = {
                        "RADAR_SHM_FRAME": True,
                        "slot": slot,
                        "seq": seq,
                        "enqueue_monotonic_ns": time.perf_counter_ns(),
                        "capture_monotonic_ns": getattr(
                            dca_frame, "capture_monotonic_ns", 0
                        ),
                        "frame_timestamp": dca_frame.timestamp,
                        "src_filepath": getattr(dca_frame, "filepath", None),
                    }
                    try:
                        full_analysis = os.environ.get("FULL_ANALYSIS", "0") in (
                            "1",
                            "true",
                            "True",
                        )
                        if self._is_recording or full_analysis:
                            # Block until consumer is ready, to preserve recorded frames
                            stream_queue.put(meta)
                        else:
                            stream_queue.put_nowait(meta)
                    except queue.Full as e:
                        if self.logger is not None:
                            if self._is_recording or full_analysis:
                                self.logger.error(
                                    f"Unexpected full queue while recording/full-analysis (should have blocked): {type(e).__name__}: {e}"
                                )
                            else:
                                self.logger.warning(
                                    f"Radar SHM frame meta drop: {type(e).__name__}: {e}"
                                )
                except Exception as e:
                    if self.logger is not None:
                        self.logger.error(f"Radar SHM copy/send failed: {e}")
                    continue
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                if self.logger is not None:
                    self.logger.info("Keyboard interrupt received, stopping...")
                stop_event.set()

        if self.logger is not None:
            self.logger.info("Sender loop stopped")

    def run(
        self,
        stream_queue: multiprocessing.Queue,
        stop_event,
        control_queue: Optional[multiprocessing.Queue] = None,
        status_queue: Optional[multiprocessing.Queue] = None,
        ack_queue: Optional[multiprocessing.Queue] = None,
    ):
        """Main process entry: boot hardware, stream frames, and respond to stop signals."""
        # Initialize logger in target process
        self.logger = setup_logger("DCA1000EVM")
        try:
            disable_shm_resource_tracker(self.logger)
        except Exception:
            pass

        # Prevent queue feeder hang on producer exit
        try:
            if stream_queue is not None:
                stream_queue.cancel_join_thread()
        except Exception as e:
            if self.logger is not None:
                self.logger.error(
                    f"cancel_join_thread unavailable or failed for stream_queue: {e}"
                )

        self._control_queue = control_queue
        if self._recording_pair_meta is not None:
            try:
                self._recording_pair_state = RecordingPairState.attach(
                    self._recording_pair_meta
                )
            except Exception as exc:
                self._recording_pair_state = None
                if self.logger:
                    self.logger.warning(
                        "Paired recording disabled; SHM attach failed: %s", exc
                    )
        self.logger.info("Starting live DCA1000 acquisition...")

        if not os.path.exists(self._dest_dir):
            os.makedirs(self._dest_dir, exist_ok=True)
            self.logger.info(f"Created destination directory: {self._dest_dir}")
        else:
            self.logger.info(f"Using existing destination directory: {self._dest_dir}")

        self._start_time = (
            float(self._config.timestamp_origin)
            if self._config.timestamp_origin is not None
            else capture_clock_ns() / 1_000_000_000.0
        )

        # Initialize board
        try:
            self._dca = DCA1000()
        except Exception as e:
            self.logger.error(f"Failed to create DCA1000 instance: {e}")
            return

        try:
            self._dca.reset_fpga()
            (
                LVDSDataSizePerChirp_l,
                maxSendBytesPerChirp_l,
                self._ADC_PARAMS_l,
                CFG_PARAMS_l,
            ) = self._dca.AWR2243_read_config(self._config.radar_config_file)
            self._dca.refresh_parameter()
            self.logger.info(
                "LVDSDataSizePerChirp:%d must <= maxSendBytesPerChirp:%d",
                LVDSDataSizePerChirp_l,
                maxSendBytesPerChirp_l,
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
        except Exception as e:
            self.logger.error(f"Failed during DCA1000 configuration: {e}")
            return

        # Pass ADC params
        try:
            stream_queue.put({"ADC_PARAMS": self._ADC_PARAMS_l})
        except Exception:
            pass

        # Start streaming
        try:
            self._dca.stream_start()
            self._dca.fastRead_in_Cpp_thread_start()
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            return

        # Acquisition loop
        self._send_frame(stream_queue, stop_event)

        # Cleanup
        try:
            if self._dca is not None:
                self._dca.fastRead_in_Cpp_thread_stop()
                self._dca.stream_stop()
                self._dca.close()
        except Exception:
            pass

        if self._shm_blocks:
            for shm in self._shm_blocks:
                try:
                    shm.close()
                except Exception as e:
                    if self.logger is not None:
                        self.logger.error(
                            f"SHM close failed for raw block ({getattr(shm,'name','?')}): {e}"
                        )
            self._shm_blocks = None
            self._shm_names = None
            self._shm_inited = False

        self.logger.info("Live DCA1000 acquisition stopped")
        try:
            time.sleep(0.2)
        except Exception:
            pass
        try:
            if stream_queue is not None:
                stream_queue.close()
        except Exception:
            pass


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
            self.logger.info("Initializing playback...")
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
        try:
            # Use shared utility to scan directory
            # Returns: List[Tuple[str, float, int, str]] (filepath, timestamp, frame_number, filename)
            radar_files = scan_bin_directory(self._dest_dir)

            # Convert to format expected by playback: (filepath, timestamp, frame_number)
            self._frame_files = [(fp, ts, fn) for fp, ts, fn, _ in radar_files]

            self.logger.info(f"Scanned {len(self._frame_files)} valid frame files")

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Recording directory does not exist: {self._dest_dir}"
            ) from e

        if not self._frame_files:
            raise ValueError(
                f"No valid .bin files found with correct naming pattern in: {self._dest_dir}"
            )

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

        # Use shared utility to parse timestamp from filepath
        filename = os.path.basename(filepath)
        result = parse_bin_timestamp(filename)

        if result is None:
            raise ValueError(f"Invalid filename pattern: {filename}")

        timestamp, _ = result  # We don't need frame_number here

        # Read binary data
        data_buf = np.fromfile(filepath, dtype=np.int16)

        if data_buf.size == 0:
            raise ValueError(f"Empty or corrupted frame file: {filepath}")

        return DCA1000Frame(timestamp, data_buf, filepath=filepath)

    def _send_frame(self, stream_queue: multiprocessing.Queue, stop_event):
        """Send frames to the radar stream queue with optional synchronized timing & ACK pacing"""
        self.logger.info("Starting frame sender thread...")
        use_sync = self._sync_state is not None
        ack_queue = getattr(self, "_ack_queue", None)
        full_analysis = os.environ.get("FULL_ANALYSIS", "0") in ("1", "true", "True")
        if full_analysis and ack_queue is not None:
            self.logger.info(
                "FULL_ANALYSIS replay: disabling timeline-based seeking; enforcing sequential ACK-paced playback"
            )

        # Synchronized start signal if applicable
        if use_sync:
            self.logger.info(
                "Using synchronized timing mode%s",
                " with ACK pacing" if ack_queue else "",
            )
            if not SyncStateUtils.wait_for_start_signal(self._sync_state, timeout=30):
                self.logger.warning(
                    "Timeout waiting for start signal, proceeding anyway"
                )
        else:
            self.logger.info(
                "Using frame rate-based timing mode%s",
                " with ACK pacing" if ack_queue else "",
            )

        # Legacy variables for original timing code
        use_sync = self._sync_state is not None
        last_timeline_position = 0.0  # Track timeline position for seeking detection
        last_sent_index = -1  # For sequence gap detection in full-analysis

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
                    if not (full_analysis and ack_queue is not None):
                        # Only apply timeline-based seeking outside full-analysis deterministic mode
                        current_timeline = SyncStateUtils.get_current_timeline_position(
                            self._sync_state
                        )
                        timeline_diff = abs(current_timeline - last_timeline_position)
                        if (
                            current_timeline < last_timeline_position - 0.05
                            or timeline_diff > 0.5
                            or current_timeline == 0.0
                        ):
                            start_timestamp = SyncStateUtils.get_start_timestamp(
                                self._sync_state
                            )
                            target_timestamp = start_timestamp + current_timeline
                            best_index = 0
                            best_diff = float("inf")
                            for i, (_, timestamp, _) in enumerate(self._frame_files):
                                diff = abs(timestamp - target_timestamp)
                                if diff < best_diff:
                                    best_diff = diff
                                    best_index = i
                            if best_index != self._current_frame_index:
                                self._current_frame_index = best_index
                                self.logger.debug(
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

                    # Get the current frame information (sequential in full-analysis)
                    # Capture frame index at start of iteration for seek detection
                    frame_to_send = self._current_frame_index
                    filepath, frame_timestamp, _ = self._frame_files[frame_to_send]
                    if (
                        full_analysis
                        and ack_queue is not None
                        and last_sent_index != -1
                        and frame_to_send != last_sent_index + 1
                    ):
                        self.logger.warning(
                            "REPLAY_SEQ_GAP expected=%d got=%d (seek?)",
                            last_sent_index + 1,
                            frame_to_send,
                        )

                    if use_sync:
                        # In full-analysis mode, do not throttle by real-time timeline;
                        # publish radar-driven window and proceed immediately.
                        if os.environ.get("FULL_ANALYSIS", "0") not in (
                            "1",
                            "true",
                            "True",
                        ):
                            # Synchronized timing mode (real-time replay)
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
                        # Publish the radar-driven processing window
                        try:
                            # Determine next frame timestamp for window end
                            if frame_to_send + 1 < len(self._frame_files):
                                _, ts_next, _ = self._frame_files[frame_to_send + 1]
                            else:
                                # Last frame: allow camera to flush remaining frames
                                ts_next = frame_timestamp + 1e9
                            # Publish window [frame_timestamp, ts_next)
                            try:
                                SyncStateUtils.set_radar_window(
                                    self._sync_state,
                                    frame_timestamp,
                                    ts_next,
                                    frame_to_send,
                                )
                            except Exception:
                                pass
                        except Exception:
                            pass
                    else:
                        # Legacy frame rate-based timing mode
                        # Wait for stream_queue to be empty (or nearly empty)
                        while not stop_event.is_set() and stream_queue.qsize() > 1:
                            # Small delay to avoid busy waiting
                            time.sleep(0.001)

                        if stop_event.is_set():
                            break

                    # Read and send the current frame
                    try:
                        frame = self._read_frame_from_file(filepath)
                        # Put frame (blocking in full-analysis to align with analyser readiness)
                        send_start = time.perf_counter()
                        try:
                            frame_payload = {
                                "RADAR_REPLAY_FILEPATH": filepath,
                                "FRAME": frame,
                                "REPLAY_FRAME_INDEX": frame_to_send,
                            }
                            if os.environ.get("FULL_ANALYSIS", "0") in (
                                "1",
                                "true",
                                "True",
                            ):
                                stream_queue.put(frame_payload)
                            else:
                                stream_queue.put_nowait(frame_payload)
                        except Exception as e:
                            if os.environ.get("FULL_ANALYSIS", "0") in (
                                "1",
                                "true",
                                "True",
                            ):
                                self.logger.error(
                                    f"Replay frame queue failure in full-analysis: {e}"
                                )
                                stop_event.set()
                                time.sleep(0.001)
                                continue
                            else:
                                self.logger.warning(
                                    f"Queue busy/closed, dropping frame: {e}"
                                )
                                time.sleep(0.001)

                        self.logger.debug(
                            f"Sent frame {frame_to_send}: {os.path.basename(filepath)} "
                            f"(timestamp: {frame_timestamp:.3f})"
                        )
                        if ack_queue is not None and full_analysis:
                            self.logger.info(
                                "REPLAY_SEND frame_index=%d send_queue_time_ms=%.2f",
                                frame_to_send,
                                (time.perf_counter() - send_start) * 1000.0,
                            )

                        # Update sync state tracking if available
                        if use_sync:
                            self._sync_state.last_radar_timestamp.value = (
                                frame_timestamp
                            )

                        # ACK pacing: wait for analyser acknowledgement before advancing
                        if ack_queue is not None and full_analysis:
                            try:
                                ack_wait_start = time.perf_counter()
                                ack = ack_queue.get(timeout=3600)  # very conservative
                                if not (
                                    isinstance(ack, dict)
                                    and ack.get("RADAR_FRAME_PROCESSED")
                                    == frame_to_send
                                ):
                                    self.logger.warning(
                                        f"Unexpected ACK payload: {ack} (expected frame {frame_to_send})"
                                    )
                                else:
                                    self.logger.info(
                                        "REPLAY_ACK  frame_index=%d wait_ms=%.2f",
                                        frame_to_send,
                                        (time.perf_counter() - ack_wait_start) * 1000.0,
                                    )
                            except queue.Empty:
                                self.logger.error("ACK timeout; stopping playback")
                                break

                        # Advance to next frame only after ACK (or immediately if no ACK)
                        # In Full-Analysis mode, check if frame index was externally modified by seek
                        if full_analysis and ack_queue is not None:
                            # We just sent frame_to_send
                            # Check if _current_frame_index was modified during send (seek command)
                            if self._current_frame_index == frame_to_send:
                                # Normal case: no seek during send, advance sequentially
                                self._current_frame_index = frame_to_send + 1
                                last_sent_index = frame_to_send
                            else:
                                # Seek detected: index was changed during send
                                self.logger.info(
                                    f"REPLAY_SEEK in Full-Analysis: sent frame {frame_to_send}, jumping to {self._current_frame_index}"
                                )
                                # Update last_sent to what we actually sent
                                last_sent_index = frame_to_send
                                # Don't modify _current_frame_index - seek already set it to target
                        else:
                            # Non-Full-Analysis mode: normal increment
                            self._current_frame_index += 1
                            last_sent_index += 1

                        # In full-analysis mode, let the camera catch up to window end before advancing further
                        try:
                            if use_sync and os.environ.get("FULL_ANALYSIS", "0") in (
                                "1",
                                "true",
                                "True",
                            ):
                                # Compute the end timestamp of the just-published window
                                try:
                                    if self._current_frame_index < len(
                                        self._frame_files
                                    ):
                                        _, next_ts, _ = self._frame_files[
                                            self._current_frame_index
                                        ]
                                    else:
                                        # Last frame already sent; no need to wait
                                        next_ts = frame_timestamp + 1e-6
                                except Exception:
                                    next_ts = frame_timestamp + 1e-6
                                # Wait until camera processed up to next_ts (or until paused/stopped/timeout)
                                start_wait = time.perf_counter()
                                max_wait = 2.0  # seconds safety to prevent deadlock if camera stalls
                                while not stop_event.is_set():
                                    # Break if playback paused or stopped
                                    if (
                                        SyncStateUtils.get_playback_state(
                                            self._sync_state
                                        )
                                        != SyncPlaybackState.PLAYING
                                    ):
                                        break
                                    try:
                                        cam_ts = float(
                                            self._sync_state.last_camera_timestamp.value
                                        )
                                    except Exception:
                                        cam_ts = 0.0
                                    if cam_ts >= next_ts - 1e-9:
                                        break
                                    if time.perf_counter() - start_wait > max_wait:
                                        # Timeout: proceed to avoid permanent stall
                                        break
                                    time.sleep(0.002)
                        except Exception:
                            pass

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
        self.logger.debug(f"Seeked to frame {frame_index}")

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
        ack_queue: Optional[multiprocessing.Queue] = None,
    ):
        """Main run method similar to DCA1000EVM with playback control support"""
        # Initialize logger and scanner in target process
        self.logger = setup_logger("DCA1000Recording")
        self._initialize()

        self.logger.info("Starting playback...")

        # Send ADC parameters first (similar to live DCA1000EVM)
        if self._ADC_PARAMS_l is None:
            raise RuntimeError("ADC parameters not loaded")

        stream_queue.put({"ADC_PARAMS": self._ADC_PARAMS_l})
        self.logger.info("Sent ADC parameters to processing queue")

        # Create and start the frame sender thread
        self._status_queue = status_queue  # Store reference for status updates
        # Store ack queue if provided (full-analysis replay pacing)
        if ack_queue is not None:
            self._ack_queue = ack_queue
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

        # Ensure stream_queue feeder exits from this process
        try:
            if stream_queue is not None:
                stream_queue.close()
                stream_queue.join_thread()
        except Exception:
            pass

        return

    def _handle_control_command(self, command):
        """Handle playback control commands from UI"""
        self.logger.info(f"Received control command: {command}")

        # Check if command is a string (expected) or something else
        if not isinstance(command, str):
            self.logger.warning(
                f"Expected string command, got {type(command)}: {command}"
            )
            return

        # Accept tuning commands (they are meant for analyser; feed just acknowledges)
        if command.startswith("TUNING:"):
            try:
                import json as _json

                payload = command.split(":", 1)[1]
                self._tuning = _json.loads(payload)  # stored for potential future use
                self.logger.debug(
                    "TUNING command stored in feed (forwarded independently to analyser)"
                )
            except Exception:
                self.logger.warning("Failed to parse TUNING command payload")
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
                # Check if we're in Full-Analysis ACK-paced mode
                full_analysis = os.environ.get("FULL_ANALYSIS", "0") in (
                    "1",
                    "true",
                    "True",
                )
                if use_sync:
                    # In synchronized mode, convert frame to timeline position
                    if 0 <= position < len(self._frame_files):
                        _, target_timestamp, _ = self._frame_files[position]
                        start_timestamp = SyncStateUtils.get_start_timestamp(
                            self._sync_state
                        )
                        relative_time = target_timestamp - start_timestamp

                        # In Full-Analysis mode, directly set frame index (timeline seeking is disabled)
                        if (
                            full_analysis
                            and hasattr(self, "_ack_queue")
                            and self._ack_queue is not None
                        ):
                            self._current_frame_index = position
                            self.logger.info(
                                f"REPLAY_SEEK_DIRECT: Set frame index to {position} (Full-Analysis mode)"
                            )
                        else:
                            # Regular replay mode: use timeline-based seeking
                            SyncStateUtils.seek_to_time(self._sync_state, relative_time)
                            self.logger.debug(
                                f"Seeked to timeline position {relative_time:.3f}s (frame {position})"
                            )
                    else:
                        self.logger.error(
                            f"Frame index {position} out of range [0, {len(self._frame_files)-1}]"
                        )
                else:
                    # Legacy mode - seek to frame directly
                    self.seek_to_frame(position)
                    self.logger.debug(f"Seeked to frame {position}")
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
        self.logger.info("Cleaned up")
