#!/usr/bin/env python3
"""
Create a video from perspective detection PNGs and camera frames.

This script combines camera frames and radar detection visualizations into a side-by-side
video, synchronized by timestamp.
"""

import argparse
import glob
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np


def parse_timestamp_from_filename(filename: str) -> Optional[float]:
    """
    Extract timestamp from filename pattern: {timestamp_int}_{timestamp_frac}_{frame_number}.png

    Args:
        filename: Name of the file (without path)

    Returns:
        Timestamp as float, or None if pattern doesn't match
    """
    # Pattern: timestamp_int_timestamp_frac_frame_number[_detection_vis_perspective].png
    # Camera frames: 0000000412_06917_000000012314.png
    # Detection frames: 0000000425_13524_000000000000_detection_vis_perspective.png
    pattern = r"(\d{10})_(\d{5})_\d+"
    match = re.match(pattern, filename)

    if match:
        timestamp_int = int(match.group(1))
        timestamp_frac = int(match.group(2))
        # Reconstruct timestamp
        timestamp = timestamp_int + (timestamp_frac / 1e5)
        return timestamp

    return None


def scan_files(
    directory: str,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Scan directory for camera frames and perspective detection files.

    Args:
        directory: Path to directory containing the files

    Returns:
        Tuple of (camera_files, perspective_files) where each is a list of (filepath, timestamp) tuples
    """
    directory = Path(directory)

    # Find all PNG files
    camera_pattern = str(directory / "*.png")
    all_pngs = glob.glob(camera_pattern)

    camera_files = []
    perspective_files = []

    for filepath in all_pngs:
        filename = os.path.basename(filepath)

        if "_detection_vis_perspective.png" in filename:
            # This is a perspective detection file
            timestamp = parse_timestamp_from_filename(filename)
            if timestamp is not None:
                perspective_files.append((filepath, timestamp))
        elif re.match(r"\d{10}_\d{5}_\d+\.png$", filename):
            # This is a camera frame (only numbers, underscores, and .png)
            timestamp = parse_timestamp_from_filename(filename)
            if timestamp is not None:
                camera_files.append((filepath, timestamp))

    # Sort by timestamp
    camera_files.sort(key=lambda x: x[1])
    perspective_files.sort(key=lambda x: x[1])

    return camera_files, perspective_files


def find_matching_camera_frame(
    camera_files: List[Tuple[str, float]], perspective_timestamp: float
) -> Optional[str]:
    """
    Find the camera frame with timestamp immediately before the perspective timestamp.

    Args:
        camera_files: List of (filepath, timestamp) tuples for camera frames
        perspective_timestamp: Timestamp of the perspective frame

    Returns:
        Path to the matching camera frame, or None if not found
    """
    # Find the camera frame with the largest timestamp that is <= perspective_timestamp
    matching_frame = None

    for filepath, timestamp in camera_files:
        if timestamp <= perspective_timestamp:
            matching_frame = filepath
        else:
            # Since list is sorted, we can break once we exceed the timestamp
            break

    return matching_frame


def resize_to_match_height(
    img1: np.ndarray, img2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize images to have the same height while maintaining aspect ratio.
    Uses the larger height to maintain quality.

    Args:
        img1: First image
        img2: Second image

    Returns:
        Tuple of (resized_img1, resized_img2) with matching heights
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Use the larger height as target to maintain quality
    target_height = max(h1, h2)

    # Resize first image if needed
    if h1 != target_height:
        aspect_ratio = w1 / h1
        new_width = int(target_height * aspect_ratio)
        img1 = cv2.resize(
            img1, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4
        )

    # Resize second image if needed
    if h2 != target_height:
        aspect_ratio = w2 / h2
        new_width = int(target_height * aspect_ratio)
        img2 = cv2.resize(
            img2, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4
        )

    return img1, img2


def create_side_by_side_frame(
    camera_img: np.ndarray, perspective_img: np.ndarray, add_labels: bool = True
) -> np.ndarray:
    """
    Create a side-by-side frame combining camera and perspective images.

    Args:
        camera_img: Camera frame image
        perspective_img: Perspective detection image
        add_labels: Whether to add text labels

    Returns:
        Combined image with both frames side by side
    """
    # Resize to match heights
    camera_img, perspective_img = resize_to_match_height(camera_img, perspective_img)

    # Concatenate horizontally
    combined = np.hstack([camera_img, perspective_img])

    # Add labels if requested
    if add_labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (255, 255, 255)  # White
        bg_color = (0, 0, 0)  # Black background

        # Add label for camera frame
        text = "Camera"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = 10
        text_y = 40
        cv2.rectangle(
            combined,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            bg_color,
            -1,
        )
        cv2.putText(
            combined, text, (text_x, text_y), font, font_scale, color, thickness
        )

        # Add label for perspective frame
        text = "Radar Detection"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = camera_img.shape[1] + 10
        cv2.rectangle(
            combined,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            bg_color,
            -1,
        )
        cv2.putText(
            combined, text, (text_x, text_y), font, font_scale, color, thickness
        )

    return combined


def create_video(
    input_dir: str, output_path: str, fps: float = 10.0, add_labels: bool = True
) -> None:
    """
    Create a video from camera frames and perspective detection visualizations.
    Tries multiple codecs with fallback for compatibility.

    Args:
        input_dir: Directory containing the PNG files
        output_path: Path for the output video file
        fps: Frames per second for the output video
        add_labels: Whether to add text labels to frames
    """
    print(f"Scanning directory: {input_dir}")
    camera_files, perspective_files = scan_files(input_dir)

    print(f"Found {len(camera_files)} camera frames")
    print(f"Found {len(perspective_files)} perspective detection frames")

    if not perspective_files:
        print("Error: No perspective detection files found!")
        return

    if not camera_files:
        print("Error: No camera frames found!")
        return

    # Initialize video writer (will be set after reading first frame)
    video_writer = None
    frames_written = 0
    target_width = None
    target_height = None

    # List of codecs to try in order of preference
    # Each codec is (fourcc_string, description)
    codecs_to_try = [
        ("mp4v", "MPEG-4"),  # Most compatible
        ("XVID", "Xvid"),  # Good compatibility
        ("MJPG", "Motion JPEG"),  # Widely supported
        ("X264", "H.264 (x264)"),  # May work with ffmpeg
        ("avc1", "H.264 (avc1)"),  # Try H.264 variants
        ("H264", "H.264"),
    ]

    print(f"\nCreating video: {output_path}")
    print(f"FPS: {fps}")

    for perspective_path, perspective_timestamp in perspective_files:
        # Find matching camera frame
        camera_path = find_matching_camera_frame(camera_files, perspective_timestamp)

        if camera_path is None:
            print(
                f"Warning: No matching camera frame for perspective at {perspective_timestamp:.5f}s"
            )
            continue

        # Read images
        camera_img = cv2.imread(camera_path)
        perspective_img = cv2.imread(perspective_path)

        if camera_img is None:
            print(f"Warning: Failed to read camera frame: {camera_path}")
            continue

        if perspective_img is None:
            print(f"Warning: Failed to read perspective frame: {perspective_path}")
            continue

        # Create combined frame
        combined_frame = create_side_by_side_frame(
            camera_img, perspective_img, add_labels
        )

        # Initialize video writer on first frame
        if video_writer is None:
            height, width = combined_frame.shape[:2]

            # Ensure dimensions are even (required by many codecs)
            if width % 2 != 0:
                width -= 1
                combined_frame = combined_frame[:, :width]
            if height % 2 != 0:
                height -= 1
                combined_frame = combined_frame[:height, :]

            # Try each codec until one works
            codec_used = None
            for codec_fourcc, codec_desc in codecs_to_try:
                print(f"Trying codec: {codec_desc} ({codec_fourcc})...")
                fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
                video_writer = cv2.VideoWriter(
                    output_path, fourcc, fps, (width, height)
                )

                if video_writer.isOpened():
                    codec_used = codec_desc
                    print(f"✓ Successfully initialized with {codec_desc} codec")
                    break
                else:
                    video_writer.release()
                    video_writer = None

            if video_writer is None or not video_writer.isOpened():
                print(f"Error: Failed to open video writer with any codec")
                print(f"Codecs tried: {', '.join([c[1] for c in codecs_to_try])}")
                return

            print(f"Video resolution: {width}x{height}")
            print(f"Using codec: {codec_used}")

            # Store target dimensions for consistency check
            target_width = width
            target_height = height
        else:
            # Ensure this frame matches the target dimensions
            current_height, current_width = combined_frame.shape[:2]
            if current_width != target_width or current_height != target_height:
                combined_frame = cv2.resize(
                    combined_frame,
                    (target_width, target_height),
                    interpolation=cv2.INTER_LANCZOS4,
                )

        # Write frame
        video_writer.write(combined_frame)
        frames_written += 1

        if frames_written % 10 == 0:
            print(f"Processed {frames_written}/{len(perspective_files)} frames...")

    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo created successfully!")
        print(f"Total frames written: {frames_written}")
        print(f"Duration: {frames_written / fps:.2f} seconds")
    else:
        print("Error: Failed to create video (no frames written)")


def main():
    parser = argparse.ArgumentParser(
        description="Create a video from camera frames and perspective detection visualizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create video from frames in a directory
  python create_detection_video.py \\
      --input-dir /path/to/frames \\
      --output output_video.mp4 \\
      --fps 10

  # Create video without labels at 30 fps
  python create_detection_video.py \\
      --input-dir /path/to/frames \\
      --output output_video.mp4 \\
      --fps 30 \\
      --no-labels

File naming convention:
  - Camera frames: {timestamp_int}_{timestamp_frac}_{frame_number}.png
  - Perspective frames: {timestamp_int}_{timestamp_frac}_{frame_number}_detection_vis_perspective.png
  
  Example:
    - Camera: 0000000412_08863_000000000000.png
    - Perspective: 0000000412_08863_000000000000_detection_vis_perspective.png
""",
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing camera frames and perspective detection PNGs",
    )

    parser.add_argument(
        "--output", required=True, help="Path for output video file (e.g., output.mp4)"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second for output video (default: 10)",
    )

    parser.add_argument(
        "--no-labels", action="store_true", help="Don't add text labels to frames"
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Create video
    try:
        create_video(
            input_dir=args.input_dir,
            output_path=args.output,
            fps=args.fps,
            add_labels=not args.no_labels,
        )
        return 0
    except Exception as e:
        print(f"Error creating video: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
