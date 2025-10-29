"""
Utility functions for working with binary radar frame files.

These utilities handle timestamp parsing, file scanning, and filename generation
for .bin files used in both live recording and replay modes.
"""

import os
import glob
import re
from typing import Optional, List, Tuple
from pathlib import Path


def encode_timestamp_parts(timestamp: float) -> Tuple[str, str]:
    """
    Encode timestamp into integer and fractional string parts for filename.

    Args:
        timestamp: Timestamp in seconds (float)

    Returns:
        Tuple of (integer_part, fraction_part) as zero-padded strings
        - integer_part: 10 digits (seconds)
        - fraction_part: 5 digits (fractional seconds * 100000)

    Example:
        >>> encode_timestamp_parts(412.06917)
        ('0000000412', '06917')
    """
    integer_part = f"{int(timestamp):010d}"
    fraction_part = f"{int((timestamp - int(timestamp)) * 1e5):05d}"
    return integer_part, fraction_part


def generate_radar_filename(timestamp: float, frame_number: int) -> str:
    """
    Generate standardized radar frame filename.

    Format: {timestamp_int}_{timestamp_frac}_{frame_number}.bin
    Example: 0000000412_06917_000000000001.bin

    Args:
        timestamp: Timestamp in seconds (float)
        frame_number: Frame sequence number

    Returns:
        Filename string with .bin extension
    """
    integer_part, fraction_part = encode_timestamp_parts(timestamp)
    frame_num_str = f"{frame_number:012d}"
    return f"{integer_part}_{fraction_part}_{frame_num_str}.bin"


def parse_bin_timestamp(filename: str) -> Optional[Tuple[float, int]]:
    """
    Parse timestamp and frame number from .bin filename.

    Expected format: {timestamp_int}_{timestamp_frac}_{frame_number}.bin
    Example: 0000000412_06917_000000000001.bin

    Args:
        filename: Name of the .bin file (with or without path)

    Returns:
        Tuple of (timestamp, frame_number) or None if pattern doesn't match
    """
    # Extract just the filename if full path is provided
    basename = os.path.basename(filename)

    # Pattern: 10 digits, 5 digits, 12 digits, .bin
    pattern = re.compile(r"(\d{10})_(\d{5})_(\d{12})\.bin$")
    match = pattern.match(basename)

    if not match:
        return None

    try:
        timestamp_int = int(match.group(1))
        timestamp_frac = int(match.group(2))
        frame_number = int(match.group(3))

        # Reconstruct timestamp (same as PNG: /100000)
        timestamp = timestamp_int + (timestamp_frac / 1e5)

        return timestamp, frame_number
    except (ValueError, IndexError):
        return None


def is_valid_radar_frame(filename: str) -> bool:
    """
    Check if filename is a valid radar frame .bin file.

    Valid radar frames match the pattern: {10digits}_{5digits}_{12digits}.bin

    Args:
        filename: Name of the file to check

    Returns:
        True if valid radar frame, False otherwise
    """
    basename = os.path.basename(filename)
    pattern = re.compile(r"(\d{10})_(\d{5})_(\d{12})\.bin$")
    return pattern.match(basename) is not None


def scan_bin_directory(
    directory: str,
) -> List[Tuple[str, float, int, str]]:
    """
    Scan directory for radar .bin frame files.

    Args:
        directory: Path to directory containing .bin files

    Returns:
        List of (filepath, timestamp, frame_number, filename) tuples sorted by timestamp

    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all .bin files
    bin_pattern = str(directory / "*.bin")
    all_bins = glob.glob(bin_pattern)

    radar_files = []

    for filepath in all_bins:
        filename = os.path.basename(filepath)

        # Check for valid radar frame
        if is_valid_radar_frame(filename):
            result = parse_bin_timestamp(filename)
            if result is not None:
                timestamp, frame_number = result
                radar_files.append((filepath, timestamp, frame_number, filename))

    # Sort by timestamp
    radar_files.sort(key=lambda x: x[1])

    return radar_files


__all__ = [
    "encode_timestamp_parts",
    "generate_radar_filename",
    "parse_bin_timestamp",
    "is_valid_radar_frame",
    "scan_bin_directory",
]
