import argparse
import json
import os
import glob
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import torch
import numpy as np

from kradar.sparse_radar_pc_inference import SparseRadarPCInference
from sample_processing.radar_params import ADCParams
from sample_processing.radar_proc_kradar import process_3d_radar_frame_kradar
from sample_processing.config import (
    RadarPipelineConfig,
    load_radar_config,
    parse_override_entries,
)
from utils import setup_logger

# Import artifact saving functions from process_radar_kradar
# We'll use them directly to maintain consistency
from process_radar_kradar import (
    load_bin_frame,
    preprocess_frame,
    apply_roi,
    save_point_cloud_npy,
    save_point_cloud_pngs,
    save_heatmap_pngs,
    save_tesseract_mat,
    save_arr_zyx_mat,
)

# Import visualization utilities
try:
    from kradar.visualization import visualize_detections, create_detection_summary_text

    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    print(
        f"Warning: Visualization not available (missing open3d). Install with: pip install open3d"
    )


def generate_sparse_point_cloud(
    bin_filepath: str,
    config_filepath: str,
    pipeline_config_path: Optional[str],
    pipeline_overrides: Optional[Dict[str, Any]],
    az_range: Optional[Tuple[float, float]],
    el_range: Optional[Tuple[float, float]],
    logger,
) -> Tuple[np.ndarray, Dict[str, Any], ADCParams, Any]:
    """
    Generate sparse point cloud from .bin radar frame.

    Returns:
        Tuple of (point_cloud_array, result_dict, adc_params, pipeline_cfg)
        point_cloud_array: Nx4 array [x, y, z, power] in K-Radar format
        result_dict: Full processing result with all artifacts
        adc_params: ADC parameters for visualization
        pipeline_cfg: Radar pipeline configuration object
    """
    logger.info(f"Processing radar frame: {bin_filepath}")
    logger.info(f"Using config file: {config_filepath}")

    # Load ADC parameters from config
    logger.info("Loading ADC parameters from config...")
    adc_params = ADCParams(config_filepath)

    if int(getattr(adc_params, "tx", 0)) != 3:
        raise ValueError(
            f"K-Radar pipeline requires 3 TX antennas, but config has {adc_params.tx} TX"
        )

    logger.info(
        f"ADC params: TX={adc_params.tx}, RX={adc_params.rx}, "
        f"chirps={adc_params.chirps}, samples={adc_params.samples}"
    )

    # Load raw data from .bin file
    logger.info("Loading .bin frame...")
    raw_data = load_bin_frame(bin_filepath)
    logger.info(f"Loaded {raw_data.size} int16 samples")

    # Preprocess to complex frame
    logger.info("Preprocessing frame...")
    complex_frame = preprocess_frame(raw_data, adc_params)
    logger.info(f"Preprocessed frame shape: {complex_frame.shape}")

    # Load pipeline configuration (YAML)
    pipeline_cfg: RadarPipelineConfig = load_radar_config(pipeline_config_path)

    # Apply overrides
    overrides: Dict[str, Any] = {}
    if az_range is not None:
        overrides["angle.azimuth_range"] = [float(az_range[0]), float(az_range[1])]
    if el_range is not None:
        overrides["angle.elevation_range"] = [float(el_range[0]), float(el_range[1])]
    if pipeline_overrides:
        overrides.update(pipeline_overrides)
    if overrides:
        pipeline_cfg = pipeline_cfg.overridden(overrides)

    # Process with K-Radar pipeline
    logger.info("Processing with K-Radar pipeline...")
    result = process_3d_radar_frame_kradar(
        complex_frame,
        adc_params,
        config=pipeline_cfg,
    )

    # Extract sparse point cloud with ROI filtering
    x = np.asarray(result.get("x_pos", []), dtype=float)
    y = np.asarray(result.get("y_pos", []), dtype=float)
    z = np.asarray(result.get("z_pos", []), dtype=float)
    snr = np.asarray(result.get("snrs", []), dtype=float)

    # Apply ROI filtering
    x, y, z, snr = apply_roi(x, y, z, snr, logger)

    # Build point cloud array in K-Radar format: [x, y, z, power]
    n = min(x.shape[0], y.shape[0], z.shape[0], snr.shape[0])
    point_cloud = np.stack((x[:n], y[:n], z[:n], snr[:n]), axis=1)

    logger.info(f"Generated sparse point cloud with {n} detections")

    return point_cloud, result, adc_params, pipeline_cfg


@torch.no_grad()
def run_single_inference(
    cfg_path: str,
    checkpoint_path: str,
    bin_filepath: str,
    config_filepath: str,
    conf_thr: float,
    device: torch.device,
    pipeline_config_path: Optional[str] = None,
    pipeline_overrides: Optional[Dict[str, Any]] = None,
    az_range: Optional[Tuple[float, float]] = None,
    el_range: Optional[Tuple[float, float]] = None,
    save_artifacts: bool = True,
    visualize: bool = False,
    interactive: bool = False,
    view_angle: str = "perspective",
) -> Dict:
    """
    Run inference on a radar frame by first generating sparse point cloud.

    Args:
        cfg_path: Path to model YAML config
        checkpoint_path: Path to model checkpoint
        bin_filepath: Path to .bin radar frame
        config_filepath: Path to radar config .txt file
        conf_thr: Confidence threshold for detections
        device: Torch device
        pipeline_config_path: Optional pipeline YAML config
        pipeline_overrides: Optional pipeline config overrides
        az_range: Optional azimuth range (min, max) in degrees
        el_range: Optional elevation range (min, max) in degrees
        save_artifacts: Whether to save intermediate artifacts
        visualize: Whether to generate 3D visualizations
        interactive: Whether to show interactive 3D viewer (requires visualize=True)
        view_angle: View angle for saved images: 'top', 'side', 'perspective', or 'all'

    Returns:
        Dictionary with detection results and metadata
    """
    logger = setup_logger("process_radar_kradar_pretrained")

    # Generate sparse point cloud
    point_cloud, result, adc_params, pipeline_cfg = generate_sparse_point_cloud(
        bin_filepath=bin_filepath,
        config_filepath=config_filepath,
        pipeline_config_path=pipeline_config_path,
        pipeline_overrides=pipeline_overrides,
        az_range=az_range,
        el_range=el_range,
        logger=logger,
    )

    # Save artifacts if requested — always to the directory containing the bin file
    if save_artifacts:
        output_dir_local = os.path.dirname(os.path.abspath(bin_filepath))
        os.makedirs(output_dir_local, exist_ok=True)

        bin_basename = os.path.basename(bin_filepath)
        output_stem = os.path.join(output_dir_local, os.path.splitext(bin_basename)[0])

        logger.info(f"Saving artifacts to: {output_dir_local}")

        # Save point cloud and visualizations
        save_point_cloud_npy(output_stem, result, logger)
        save_point_cloud_pngs(output_stem, result, adc_params, logger)
        save_heatmap_pngs(output_stem, result, adc_params, logger)

        # Save tesseract and arr_zyx if available
        tesseract = result.get("tesseract")
        if isinstance(tesseract, np.ndarray) and tesseract.size > 0:
            save_tesseract_mat(output_stem, tesseract, logger)

            az_grid = result.get("tesseract_az_grid_deg")
            el_grid = result.get("tesseract_el_grid_deg")
            if isinstance(az_grid, np.ndarray) and isinstance(el_grid, np.ndarray):
                save_arr_zyx_mat(
                    output_stem,
                    result,
                    tesseract,
                    az_grid,
                    el_grid,
                    adc_params,
                    logger,
                )

    # Run inference on the point cloud
    logger.info("Running K-Radar model inference...")
    runner = SparseRadarPCInference(cfg_path, checkpoint_path, device)
    frame_id = Path(bin_filepath).name
    inference_result = runner.run_on_points(point_cloud, frame_id, conf_thr)

    # Add metadata
    inference_result["num_input_points"] = len(point_cloud)
    inference_result["bin_file"] = bin_filepath
    inference_result["config_file"] = config_filepath

    logger.info(f"Inference complete. Detections: {len(inference_result['boxes'])}")

    # Always generate 3D visualization (save PNG files regardless of detections)
    if VISUALIZATION_AVAILABLE:
        output_dir_local = os.path.dirname(os.path.abspath(bin_filepath))
        bin_basename = os.path.basename(bin_filepath)
        output_stem = os.path.join(output_dir_local, os.path.splitext(bin_basename)[0])

        logger.info("Generating 3D visualization...")

        # Print detection summary
        summary = create_detection_summary_text(
            inference_result, conf_threshold=conf_thr
        )
        logger.info("\n" + summary)

        # Extract grid limits from pipeline config
        grid_limits = None
        if hasattr(pipeline_cfg, "point_cloud"):
            pc_cfg = pipeline_cfg.point_cloud
            # Use ROI if enabled, otherwise fall back to x/y/z_limits
            if hasattr(pc_cfg, "roi") and getattr(pc_cfg.roi, "enabled", False):
                roi = pc_cfg.roi
                grid_limits = {
                    "x": getattr(roi, "x", [0.0, 100.0]),
                    "y": getattr(roi, "y", [-6.2, 6.2]),
                    "z": getattr(roi, "z", [-1.8, 5.8]),
                }
            else:
                grid_limits = {
                    "x": getattr(pc_cfg, "x_limits", [0.0, 99.6]),
                    "y": getattr(pc_cfg, "y_limits", [-80.0, 79.6]),
                    "z": getattr(pc_cfg, "z_limits", [-30.0, 29.6]),
                }

        # Create visualization (interactive only if visualize flag is set)
        visualize_detections(
            pc_array=point_cloud,
            detections=inference_result,
            save_path=f"{output_stem}_detection_vis",
            interactive=(
                interactive and visualize
            ),  # Only show interactive if explicitly requested
            conf_threshold=conf_thr,
            view_angle=view_angle,
            grid_limits=grid_limits,
        )

        logger.info(f"Visualization saved to: {output_stem}_detection_vis_*.png")

        # Save JSON output alongside visualization
        json_output_path = f"{output_stem}_detection.json"
        with open(json_output_path, "w") as f:
            json.dump(inference_result, f, indent=2)
        logger.info(f"Detection results saved to: {json_output_path}")
    elif not VISUALIZATION_AVAILABLE:
        logger.warning("Open3d is not available. Skipping visualization.")

    return inference_result


def prepare_inference(
    cfg_path: str, checkpoint_path: str, device: torch.device
) -> SparseRadarPCInference:
    """Factory helper to create a reusable inference runner."""
    return SparseRadarPCInference(cfg_path, checkpoint_path, device)


def scan_bin_files(bin_path: str, logger) -> List[Tuple[str, float, int]]:
    """
    Scan directory for .bin files and extract timestamps from filenames.

    Args:
        bin_path: Directory containing .bin files
        logger: Logger instance

    Returns:
        List of (filepath, timestamp, frame_number) tuples sorted by timestamp
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Directory does not exist: {bin_path}")

    # Pattern: {timestamp_int}_{timestamp_frac}_{frame_number}.bin
    pattern = os.path.join(bin_path, "*.bin")
    bin_files = glob.glob(pattern)

    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in directory: {bin_path}")

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
            logger.warning(f"Skipping file with invalid naming pattern: {filename}")

    if not frame_info:
        raise ValueError(
            f"No valid .bin files found with correct naming pattern in: {bin_path}"
        )

    # Sort by timestamp to ensure proper playback order
    frame_info.sort(key=lambda x: x[1])

    logger.info(f"Found {len(frame_info)} valid frame files")
    return frame_info


@torch.no_grad()
def run_batch_inference(
    cfg_path: str,
    checkpoint_path: str,
    bin_path: str,
    config_filepath: str,
    conf_thr: float,
    device: torch.device,
    pipeline_config_path: Optional[str] = None,
    pipeline_overrides: Optional[Dict[str, Any]] = None,
    az_range: Optional[Tuple[float, float]] = None,
    el_range: Optional[Tuple[float, float]] = None,
    save_artifacts: bool = True,
    visualize: bool = False,
    interactive: bool = False,
    view_angle: str = "perspective",
) -> List[Dict]:
    """
    Run inference on all .bin files in a directory, processing them in timestamp order.

    Args:
        cfg_path: Path to model YAML config
        checkpoint_path: Path to model checkpoint
        bin_path: Directory containing .bin radar frame files
        config_filepath: Path to radar config .txt file
        conf_thr: Confidence threshold for detections
        device: Torch device
        pipeline_config_path: Optional pipeline YAML config
        pipeline_overrides: Optional pipeline config overrides
        az_range: Optional azimuth range (min, max) in degrees
        el_range: Optional elevation range (min, max) in degrees
        save_artifacts: Whether to save intermediate artifacts
        visualize: Whether to generate 3D visualizations
        interactive: Whether to show interactive 3D viewer (updated in real-time)
        view_angle: View angle for saved images

    Returns:
        List of detection result dictionaries, one per frame
    """
    logger = setup_logger("process_radar_kradar_pretrained_batch")

    # Scan and sort .bin files by timestamp
    logger.info(f"Scanning directory: {bin_path}")
    frame_files = scan_bin_files(bin_path, logger)
    logger.info(f"Processing {len(frame_files)} frames in timestamp order")

    # Create inference runner (reuse for all frames)
    logger.info("Initializing K-Radar model...")
    runner = SparseRadarPCInference(cfg_path, checkpoint_path, device)

    # Initialize visualization window if interactive mode is enabled
    vis = None
    geometries_cache = []
    if interactive and visualize and VISUALIZATION_AVAILABLE:
        try:
            import open3d as o3d

            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name="K-Radar Batch Processing - Real-time Updates",
                width=1280,
                height=720,
            )
            # Set render options
            render_option = vis.get_render_option()
            render_option.point_size = 2.0 / 3
            render_option.line_width = 6.0
            logger.info("Created interactive visualization window")
        except Exception as e:
            logger.warning(f"Failed to create interactive window: {e}")
            vis = None

    results = []

    # Process each frame
    for idx, (filepath, timestamp, frame_number) in enumerate(frame_files):
        logger.info(f"\n{'='*80}")
        logger.info(
            f"Processing frame {idx+1}/{len(frame_files)}: {os.path.basename(filepath)}"
        )
        logger.info(f"Timestamp: {timestamp:.5f}s, Frame number: {frame_number}")
        logger.info(f"{'='*80}")

        try:
            # Generate sparse point cloud
            point_cloud, result, adc_params, pipeline_cfg = generate_sparse_point_cloud(
                bin_filepath=filepath,
                config_filepath=config_filepath,
                pipeline_config_path=pipeline_config_path,
                pipeline_overrides=pipeline_overrides,
                az_range=az_range,
                el_range=el_range,
                logger=logger,
            )

            # Save artifacts if requested
            if save_artifacts:
                output_dir_local = os.path.dirname(os.path.abspath(filepath))
                bin_basename = os.path.basename(filepath)
                output_stem = os.path.join(
                    output_dir_local, os.path.splitext(bin_basename)[0]
                )

                save_point_cloud_npy(output_stem, result, logger)
                save_point_cloud_pngs(output_stem, result, adc_params, logger)
                save_heatmap_pngs(output_stem, result, adc_params, logger)

                tesseract = result.get("tesseract")
                if isinstance(tesseract, np.ndarray) and tesseract.size > 0:
                    save_tesseract_mat(output_stem, tesseract, logger)
                    az_grid = result.get("tesseract_az_grid_deg")
                    el_grid = result.get("tesseract_el_grid_deg")
                    if isinstance(az_grid, np.ndarray) and isinstance(
                        el_grid, np.ndarray
                    ):
                        save_arr_zyx_mat(
                            output_stem,
                            result,
                            tesseract,
                            az_grid,
                            el_grid,
                            adc_params,
                            logger,
                        )

            # Run inference
            logger.info("Running K-Radar model inference...")
            frame_id = Path(filepath).name
            inference_result = runner.run_on_points(point_cloud, frame_id, conf_thr)

            # Add metadata
            inference_result["num_input_points"] = len(point_cloud)
            inference_result["bin_file"] = filepath
            inference_result["config_file"] = config_filepath
            inference_result["timestamp"] = timestamp
            inference_result["frame_number"] = frame_number
            inference_result["frame_index"] = idx

            results.append(inference_result)

            logger.info(f"Detections: {len(inference_result['boxes'])}")

            # Always save visualization (even if no detections)
            if VISUALIZATION_AVAILABLE:
                output_dir_local = os.path.dirname(os.path.abspath(filepath))
                bin_basename = os.path.basename(filepath)
                output_stem = os.path.join(
                    output_dir_local, os.path.splitext(bin_basename)[0]
                )

                # Save static visualization images
                from kradar.visualization import (
                    visualize_detections,
                    create_detection_summary_text,
                )

                summary = create_detection_summary_text(
                    inference_result, conf_threshold=conf_thr
                )
                logger.info("\n" + summary)

                # Extract grid limits from pipeline config
                grid_limits = None
                if hasattr(pipeline_cfg, "point_cloud"):
                    pc_cfg = pipeline_cfg.point_cloud
                    # Use ROI if enabled, otherwise fall back to x/y/z_limits
                    if hasattr(pc_cfg, "roi") and getattr(pc_cfg.roi, "enabled", False):
                        roi = pc_cfg.roi
                        grid_limits = {
                            "x": getattr(roi, "x", [0.0, 100.0]),
                            "y": getattr(roi, "y", [-6.2, 6.2]),
                            "z": getattr(roi, "z", [-1.8, 5.8]),
                        }
                    else:
                        grid_limits = {
                            "x": getattr(pc_cfg, "x_limits", [0.0, 99.6]),
                            "y": getattr(pc_cfg, "y_limits", [-80.0, 79.6]),
                            "z": getattr(pc_cfg, "z_limits", [-30.0, 29.6]),
                        }

                # Always save visualization (controlled by visualize flag for interactive updates only)
                visualize_detections(
                    pc_array=point_cloud,
                    detections=inference_result,
                    save_path=f"{output_stem}_detection_vis",
                    interactive=False,  # Don't block for each frame
                    conf_threshold=conf_thr,
                    view_angle=view_angle,
                    grid_limits=grid_limits,
                )

                # Save JSON output alongside visualization
                json_output_path = f"{output_stem}_detection.json"
                with open(json_output_path, "w") as f:
                    json.dump(inference_result, f, indent=2)
                logger.info(f"Detection results saved to: {json_output_path}")

                # Update interactive window if available
                if vis is not None:
                    try:
                        # Clear previous geometries
                        for geom in geometries_cache:
                            vis.remove_geometry(geom, reset_bounding_box=False)
                        geometries_cache.clear()

                        # Create new geometries
                        from kradar.util_geometry import (
                            get_pc_for_vis,
                            get_bbox_for_vis,
                        )
                        import open3d as o3d

                        # Point cloud with power-based coloring
                        pcd = get_pc_for_vis(point_cloud, color="power")
                        geometries_cache.append(pcd)
                        vis.add_geometry(pcd, reset_bounding_box=(idx == 0))

                        # Bounding boxes
                        boxes = np.array(inference_result.get("boxes", []))
                        scores = np.array(inference_result.get("scores", []))
                        labels = np.array(inference_result.get("labels", []))
                        class_names_list = inference_result.get("class_names", [])

                        if len(boxes) > 0 and len(scores) > 0:
                            mask = scores >= conf_thr
                            boxes = boxes[mask]
                            labels = labels[mask]
                            # Get class names with safety check
                            filtered_class_names = []
                            for label in labels:
                                label_idx = int(label)
                                if label_idx < len(class_names_list):
                                    filtered_class_names.append(
                                        class_names_list[label_idx]
                                    )
                                else:
                                    filtered_class_names.append(
                                        f"Unknown(label={label_idx})"
                                    )

                            line_sets_bbox = get_bbox_for_vis(
                                boxes, class_names=filtered_class_names
                            )
                            for line_set in line_sets_bbox:
                                geometries_cache.append(line_set)
                                vis.add_geometry(line_set, reset_bounding_box=False)

                        # Add coordinate frame
                        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=5.0, origin=[0, 0, 0]
                        )
                        geometries_cache.append(coord_frame)
                        vis.add_geometry(coord_frame, reset_bounding_box=False)

                        # Update window
                        vis.poll_events()
                        vis.update_renderer()

                    except Exception as e:
                        logger.warning(
                            f"Failed to update interactive visualization: {e}"
                        )

        except Exception as e:
            logger.error(f"Failed to process frame {filepath}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Close interactive window if it was created
    if vis is not None:
        try:
            logger.info("Closing interactive visualization window...")
            vis.destroy_window()
        except Exception as e:
            logger.warning(f"Failed to close visualization window: {e}")

    logger.info(f"\n{'='*80}")
    logger.info(
        f"Batch processing complete: {len(results)}/{len(frame_files)} frames processed successfully"
    )
    logger.info(f"{'='*80}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process radar .bin file(s) and run K-Radar inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single frame
  python process_radar_kradar_pretrained.py \\
      --config kradar/configs/cfg_RTNH_wide.yml \\
      --checkpoint checkpoints/rtnh_wide.pt \\
      --bin-file data/frame.bin \\
      --config-file config_files/AWR2243_87m_17cm_64_3_256.txt
  
  # Process all frames in a directory (sorted by timestamp)
  python process_radar_kradar_pretrained.py \\
      --config kradar/configs/cfg_RTNH_wide.yml \\
      --checkpoint checkpoints/rtnh_wide.pt \\
      --bin-path data/recording/ \\
      --config-file config_files/AWR2243_87m_17cm_64_3_256.txt \\
      --visualize --interactive
        """,
    )

    # Model configuration
    parser.add_argument(
        "--config",
        required=True,
        help="Path to model YAML config (e.g., cfg_RTNH_wide.yml)",
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to pretrained checkpoint (.pt)"
    )
    parser.add_argument(
        "--conf-thr",
        type=float,
        default=0.4,
        help="Confidence threshold for detections",
    )

    # Radar data inputs
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--bin-file", "-b", help="Path to a single .bin radar frame file"
    )
    input_group.add_argument(
        "--bin-path",
        "-p",
        help="Path to directory containing multiple .bin files (processes all in timestamp order)",
    )
    parser.add_argument(
        "--config-file", "-c", required=True, help="Path to radar config .txt file"
    )

    # Pipeline configuration
    parser.add_argument(
        "--pipeline-config",
        help="Path to pipeline YAML configuration (defaults to configs/default.yaml)",
        default=None,
    )
    parser.add_argument(
        "--set",
        dest="pipeline_overrides",
        metavar="KEY=VALUE",
        action="append",
        default=None,
        help="Override pipeline config entries (repeatable)",
    )
    parser.add_argument(
        "--az-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Optional azimuth range override in degrees",
    )
    parser.add_argument(
        "--el-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Optional elevation range override in degrees",
    )

    parser.add_argument(
        "--output-json", help="Optional path to save detection results as JSON"
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Skip saving intermediate artifacts (point clouds, heatmaps, etc.)",
    )

    # Visualization options
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate 3D visualization of point cloud with detected bounding boxes",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show interactive 3D viewer (requires --visualize, blocks execution until closed)",
    )
    parser.add_argument(
        "--view-angle",
        choices=["top", "side", "perspective", "all"],
        default="perspective",
        help="Camera view angle for saved visualization images (default: perspective)",
    )

    args = parser.parse_args()

    # Parse pipeline overrides
    pipeline_overrides = parse_override_entries(args.pipeline_overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check visualization requirements
    if (args.visualize or args.interactive) and not VISUALIZATION_AVAILABLE:
        print("Warning: Visualization requested but open3d is not installed.")
        print("Install with: pip install open3d")
        print("Proceeding without visualization...")

    try:
        # Determine if batch or single file processing
        if args.bin_path:
            # Batch processing mode
            results = run_batch_inference(
                cfg_path=args.config,
                checkpoint_path=args.checkpoint,
                bin_path=args.bin_path,
                config_filepath=args.config_file,
                conf_thr=args.conf_thr,
                device=device,
                pipeline_config_path=args.pipeline_config,
                pipeline_overrides=pipeline_overrides or None,
                az_range=tuple(args.az_range) if args.az_range else None,
                el_range=tuple(args.el_range) if args.el_range else None,
                save_artifacts=not args.no_artifacts,
                visualize=args.visualize,
                interactive=args.interactive,
                view_angle=args.view_angle,
            )

            # Print summary
            print("\n" + "=" * 80)
            print(f"BATCH PROCESSING SUMMARY: {len(results)} frames processed")
            print("=" * 80)
            for idx, result in enumerate(results):
                num_detections = len(result.get("boxes", []))
                print(
                    f"Frame {idx+1}: {os.path.basename(result['bin_file'])} - {num_detections} detections"
                )

            # Save batch results if requested
            if args.output_json:
                output_path = Path(args.output_json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # Save as list of results
                batch_output = {"num_frames": len(results), "frames": results}
                output_path.write_text(json.dumps(batch_output, indent=2))
                print(f"\nSaved batch results to {args.output_json}")

        else:
            # Single file processing mode
            result = run_single_inference(
                cfg_path=args.config,
                checkpoint_path=args.checkpoint,
                bin_filepath=args.bin_file,
                config_filepath=args.config_file,
                conf_thr=args.conf_thr,
                device=device,
                pipeline_config_path=args.pipeline_config,
                pipeline_overrides=pipeline_overrides or None,
                az_range=tuple(args.az_range) if args.az_range else None,
                el_range=tuple(args.el_range) if args.el_range else None,
                save_artifacts=not args.no_artifacts,
                visualize=args.visualize,
                interactive=args.interactive,
                view_angle=args.view_angle,
            )

            print("\n" + "=" * 80)
            print("DETECTION RESULTS")
            print("=" * 80)
            print(json.dumps(result, indent=2))

            if args.output_json:
                output_path = Path(args.output_json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(result, indent=2))
                print(f"\nSaved detection results to {args.output_json}")

    except Exception as e:
        print(f"Error: {e}", file=__import__("sys").stderr)
        import traceback

        traceback.print_exc()
        __import__("sys").exit(1)


if __name__ == "__main__":
    main()
