"""
Synchronization state management for multi-process fusion replay.
Provides shared state for timeline synchronization between radar and camera feeds.
"""

import multiprocessing
import time
import glob
import os
import re
from typing import List, Tuple, Optional
from utils import setup_logger


# Playback state constants
class PlaybackState:
    STOPPED = 0
    PLAYING = 1
    PAUSED = 2


def create_sync_state(manager):
    """
    Create a shared synchronization state using a Manager.
    Returns a Manager.Namespace() object that can be safely shared between processes.
    """
    sync_state = manager.Namespace()

    # Shared timeline state (controlled by GUI)
    sync_state.playback_time = manager.Value(
        "d", 0.0
    )  # Current timeline position (seconds)
    sync_state.playback_speed = manager.Value(
        "d", 1.0
    )  # Speed multiplier (1.0 = real-time)
    sync_state.playback_state = manager.Value(
        "i", PlaybackState.STOPPED
    )  # Current state

    # Synchronization points
    sync_state.start_timestamp = manager.Value(
        "d", 0.0
    )  # Common start timestamp for both feeds
    sync_state.playback_start_time = manager.Value(
        "d", 0.0
    )  # Real-time when playback started

    # Process synchronization
    sync_state.feeds_ready = manager.Event()  # Both feeds scanned and ready
    sync_state.sync_start = manager.Event()  # Start signal for synchronized playback

    # Statistics and debugging
    sync_state.last_camera_timestamp = manager.Value("d", 0.0)
    sync_state.last_radar_timestamp = manager.Value("d", 0.0)

    return sync_state


class SyncStateUtils:
    """
    Utility class for working with shared synchronization state.
    Contains methods that operate on the shared state namespace.
    """

    @staticmethod
    def reset(sync_state):
        """Reset all state for new playback session"""
        sync_state.playback_time.value = 0.0
        sync_state.playback_speed.value = 1.0
        sync_state.playback_state.value = PlaybackState.STOPPED
        sync_state.start_timestamp.value = 0.0
        sync_state.playback_start_time.value = 0.0

        sync_state.feeds_ready.clear()
        sync_state.sync_start.clear()

    @staticmethod
    def set_start_timestamp(sync_state, timestamp: float):
        """Set the common start timestamp for synchronization"""
        sync_state.start_timestamp.value = timestamp
        logger = setup_logger("SyncStateUtils")
        logger.info(f"Set start timestamp: {timestamp}")

    @staticmethod
    def get_start_timestamp(sync_state) -> float:
        """Get the common start timestamp"""
        return sync_state.start_timestamp.value

    @staticmethod
    def set_playback_state(sync_state, state: int):
        """Set playback state (PLAYING/PAUSED/STOPPED)"""
        old_state = sync_state.playback_state.value
        sync_state.playback_state.value = state

        if state == PlaybackState.PLAYING and old_state != PlaybackState.PLAYING:
            # Starting playback - record real start time
            sync_state.playback_start_time.value = time.perf_counter()

        logger = setup_logger("SyncStateUtils")
        logger.debug(f"Playback state changed: {old_state} -> {state}")

    @staticmethod
    def get_playback_state(sync_state) -> int:
        """Get current playback state"""
        return sync_state.playback_state.value

    @staticmethod
    def set_playback_speed(sync_state, speed: float):
        """Set playback speed multiplier"""
        sync_state.playback_speed.value = speed
        logger = setup_logger("SyncStateUtils")
        logger.debug(f"Playback speed set to: {speed}x")

    @staticmethod
    def get_playback_speed(sync_state) -> float:
        """Get current playback speed"""
        return sync_state.playback_speed.value

    @staticmethod
    def update_timeline(sync_state):
        """Update timeline position based on elapsed real time and speed"""
        if SyncStateUtils.get_playback_state(sync_state) == PlaybackState.PLAYING:
            current_real_time = time.perf_counter()
            start_time = sync_state.playback_start_time.value
            speed = sync_state.playback_speed.value

            if start_time > 0:
                elapsed_real = current_real_time - start_time
                timeline_advance = elapsed_real * speed

                sync_state.playback_time.value += timeline_advance
                sync_state.playback_start_time.value = current_real_time

    @staticmethod
    def get_current_timeline_position(sync_state) -> float:
        """Get current timeline position"""
        return sync_state.playback_time.value

    @staticmethod
    def seek_to_time(sync_state, timeline_position: float):
        """Seek to specific timeline position"""
        sync_state.playback_time.value = timeline_position
        sync_state.playback_start_time.value = time.perf_counter()
        logger = setup_logger("SyncStateUtils")
        logger.debug(f"Seeked to timeline position: {timeline_position}")

    @staticmethod
    def signal_feed_ready(sync_state):
        """Signal that a feed has completed initialization and is ready"""
        sync_state.feeds_ready.set()

    @staticmethod
    def wait_for_feeds_ready(sync_state, timeout: Optional[float] = None) -> bool:
        """Wait for all feeds to be ready"""
        return sync_state.feeds_ready.wait(timeout)

    @staticmethod
    def signal_start_playback(sync_state):
        """Signal all feeds to start synchronized playback"""
        sync_state.sync_start.set()
        logger = setup_logger("SyncStateUtils")
        logger.info("Signaled synchronized playback start")

    @staticmethod
    def wait_for_start_signal(sync_state, timeout: Optional[float] = None) -> bool:
        """Wait for the synchronized start signal"""
        return sync_state.sync_start.wait(timeout)


class TimestampScanner:
    """
    Utility class for scanning and parsing timestamps from recording files.
    Supports both PNG and BIN file naming conventions.
    """

    @staticmethod
    def parse_timestamp_from_filename(filename: str) -> Optional[float]:
        """
        Parse timestamp from filename using the standard naming convention.
        Format: {timestamp_int}_{timestamp_frac}_{frame_number}.{ext}

        Returns:
            Timestamp as float, or None if parsing fails
        """
        # Pattern: {timestamp_int}_{timestamp_frac}_{frame_number}.{ext}
        pattern = re.compile(r"(\d{10})_(\d{5})_(\d{12})\.(png|bin)$")
        match = pattern.match(filename)

        if match:
            timestamp_int = int(match.group(1))
            timestamp_frac = int(match.group(2))
            timestamp = timestamp_int + (timestamp_frac / 1e5)
            return timestamp

        return None

    @staticmethod
    def scan_directory_timestamps(
        directory: str, file_extension: str
    ) -> List[Tuple[str, float]]:
        """
        Scan directory for files with given extension and extract timestamps.

        Args:
            directory: Directory to scan
            file_extension: File extension to look for (e.g., 'png', 'bin')

        Returns:
            List of (filepath, timestamp) tuples sorted by timestamp
        """
        logger = setup_logger("TimestampScanner")

        if not os.path.exists(directory):
            logger.warning(f"Directory does not exist: {directory}")
            return []

        pattern = os.path.join(directory, f"*.{file_extension}")
        files = glob.glob(pattern)

        if not files:
            logger.info(f"No {file_extension} files found in directory: {directory}")
            return []

        # Parse timestamps
        timestamped_files = []
        for filepath in files:
            filename = os.path.basename(filepath)
            timestamp = TimestampScanner.parse_timestamp_from_filename(filename)

            if timestamp is not None:
                timestamped_files.append((filepath, timestamp))
            else:
                logger.debug(f"Skipping file with invalid naming pattern: {filename}")

        # Sort by timestamp
        timestamped_files.sort(key=lambda x: x[1])

        logger.info(f"Found {len(timestamped_files)} valid {file_extension} files")
        return timestamped_files

    @staticmethod
    def find_common_start_timestamp(
        recording_dir: str,
    ) -> Tuple[float, List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Find the optimal start timestamp where both PNG and BIN files are available.

        Args:
            recording_dir: Directory containing both PNG and BIN files

        Returns:
            Tuple of (start_timestamp, png_files, bin_files)
        """
        logger = setup_logger("TimestampScanner")

        # Scan both file types
        png_files = TimestampScanner.scan_directory_timestamps(recording_dir, "png")
        bin_files = TimestampScanner.scan_directory_timestamps(recording_dir, "bin")

        if not png_files or not bin_files:
            logger.warning("Missing PNG or BIN files for synchronization")
            # Fall back to earliest available
            all_timestamps = []
            if png_files:
                all_timestamps.extend([t for _, t in png_files])
            if bin_files:
                all_timestamps.extend([t for _, t in bin_files])

            start_timestamp = min(all_timestamps) if all_timestamps else 0.0
            logger.warning(f"Using fallback start timestamp: {start_timestamp}")
            return start_timestamp, png_files, bin_files

        # Find earliest timestamp where both feeds have data
        png_timestamps = set(timestamp for _, timestamp in png_files)
        bin_timestamps = set(timestamp for _, timestamp in bin_files)

        # Start at the latest of the two earliest timestamps
        png_start = min(png_timestamps)
        bin_start = min(bin_timestamps)
        start_timestamp = max(png_start, bin_start)

        logger.info(f"Synchronization start timestamp: {start_timestamp}")
        logger.info(f"PNG files: {len(png_files)} (start: {png_start})")
        logger.info(f"BIN files: {len(bin_files)} (start: {bin_start})")

        return start_timestamp, png_files, bin_files
