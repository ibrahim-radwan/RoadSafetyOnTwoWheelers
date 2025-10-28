"""
Utility functions for working with PNG camera frames.

These utilities handle timestamp parsing and file scanning for PNG files
used in both live recording and replay modes.
"""

import os
import glob
import re
from typing import Optional, List, Tuple
from pathlib import Path


def parse_png_timestamp(filename: str) -> Optional[float]:
    """
    Parse timestamp from PNG filename.

    Expected format: {timestamp_int}_{timestamp_frac}_{frame_number}.png
    Example: 0000000412_06917_000000012314.png

    Args:
        filename: Name of the PNG file (with or without path)

    Returns:
        Timestamp as float (seconds), or None if pattern doesn't match
    """
    # Extract just the filename if full path is provided
    basename = os.path.basename(filename)

    # Remove .png extension if present
    name_without_ext = basename[:-4] if basename.endswith(".png") else basename

    # Validate format: only digits and underscores, ending with a digit
    if not name_without_ext or not name_without_ext[-1].isdigit():
        return None

    if not all(c.isdigit() or c == "_" for c in name_without_ext):
        return None

    # Parse using split method (handles variable-length parts)
    try:
        parts = name_without_ext.split("_")
        if len(parts) >= 3:
            integer_part = int(parts[0])
            fraction_part = int(parts[1])
            # Combine: integer seconds + fractional part (5 digits = /100000)
            timestamp = integer_part + fraction_part / 100000.0
            return timestamp
    except (ValueError, IndexError):
        return None

    return None


def is_valid_camera_frame(filename: str) -> bool:
    """
    Check if filename is a valid camera frame PNG.

    Valid camera frames contain only numbers and underscores,
    and end with a number (e.g., 0000000412_06917_000000012314.png).

    Excludes visualization files like:
    - _xz.png, _xy.png (point cloud views)
    - frame_001.png (generic frames)
    - heatmap_xz.png (radar heatmaps)
    - *_detection_vis_perspective.png (detection overlays)

    Args:
        filename: Name of the file to check

    Returns:
        True if valid camera frame, False otherwise
    """
    # Quick exclusion patterns
    if "_detection_vis_perspective.png" in filename:
        return False
    if filename.startswith("heatmap_") or filename.startswith("frame_"):
        return False

    # Extract basename and check format
    basename = os.path.basename(filename)
    name_without_ext = basename[:-4] if basename.endswith(".png") else basename

    # Must end with a digit and contain only digits and underscores
    if not name_without_ext or not name_without_ext[-1].isdigit():
        return False

    return all(c.isdigit() or c == "_" for c in name_without_ext)


def scan_png_directory(
    directory: str, include_detection_files: bool = False
) -> Tuple[List[Tuple[str, float, str]], Optional[List[Tuple[str, float, str]]]]:
    """
    Scan directory for PNG camera frames and optionally detection files.

    Args:
        directory: Path to directory containing PNG files
        include_detection_files: If True, also scan for detection visualization files

    Returns:
        Tuple of (camera_files, detection_files) where each is a list of
        (filepath, timestamp, filename) tuples sorted by timestamp.
        If include_detection_files is False, detection_files will be None.

    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all PNG files
    png_pattern = str(directory / "*.png")
    all_pngs = glob.glob(png_pattern)

    camera_files = []
    detection_files = [] if include_detection_files else None

    for filepath in all_pngs:
        filename = os.path.basename(filepath)

        # Check for detection visualization files
        if "_detection_vis_perspective.png" in filename:
            if include_detection_files:
                timestamp = parse_png_timestamp(filename)
                if timestamp is not None:
                    detection_files.append((filepath, timestamp, filename))
            continue

        # Check for valid camera frame
        if is_valid_camera_frame(filename):
            timestamp = parse_png_timestamp(filename)
            if timestamp is not None:
                camera_files.append((filepath, timestamp, filename))

    # Sort by timestamp
    camera_files.sort(key=lambda x: x[1])
    if detection_files is not None:
        detection_files.sort(key=lambda x: x[1])

    return camera_files, detection_files


def find_closest_frame_before(
    frame_list: List[Tuple[str, float, str]], target_timestamp: float
) -> Optional[Tuple[str, float, str]]:
    """
    Find the frame with timestamp immediately before or equal to target timestamp.

    Args:
        frame_list: List of (filepath, timestamp, filename) tuples (must be sorted)
        target_timestamp: Target timestamp to find closest frame before

    Returns:
        Tuple of (filepath, timestamp, filename) for the matching frame,
        or None if no frame found before target timestamp
    """
    matching_frame = None

    for frame_tuple in frame_list:
        filepath, timestamp, filename = frame_tuple
        if timestamp <= target_timestamp:
            matching_frame = frame_tuple
        else:
            # Since list is sorted, break once we exceed the target
            break

    return matching_frame


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


def generate_camera_filename(timestamp: float, frame_number: int) -> str:
    """
    Generate standardized camera frame filename.

    Format: {timestamp_int}_{timestamp_frac}_{frame_number}.png
    Example: 0000000412_06917_000000012314.png

    Args:
        timestamp: Timestamp in seconds (float)
        frame_number: Frame sequence number

    Returns:
        Filename string with .png extension
    """
    integer_part, fraction_part = encode_timestamp_parts(timestamp)
    frame_num_str = f"{frame_number:012d}"
    return f"{integer_part}_{fraction_part}_{frame_num_str}.png"


__all__ = [
    "parse_png_timestamp",
    "is_valid_camera_frame",
    "scan_png_directory",
    "find_closest_frame_before",
    "encode_timestamp_parts",
    "generate_camera_filename",
]
