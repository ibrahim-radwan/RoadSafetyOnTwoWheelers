import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
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
) -> Tuple[np.ndarray, Dict[str, Any], ADCParams]:
    """
    Generate sparse point cloud from .bin radar frame.

    Returns:
        Tuple of (point_cloud_array, result_dict, adc_params)
        point_cloud_array: Nx4 array [x, y, z, power] in K-Radar format
        result_dict: Full processing result with all artifacts
        adc_params: ADC parameters for visualization
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

    return point_cloud, result, adc_params


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
    point_cloud, result, adc_params = generate_sparse_point_cloud(
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

    # Generate 3D visualization if requested
    if visualize and VISUALIZATION_AVAILABLE:
        output_dir_local = os.path.dirname(os.path.abspath(bin_filepath))
        bin_basename = os.path.basename(bin_filepath)
        output_stem = os.path.join(output_dir_local, os.path.splitext(bin_basename)[0])

        logger.info("Generating 3D visualization...")

        # Print detection summary
        summary = create_detection_summary_text(
            inference_result, conf_threshold=conf_thr
        )
        logger.info("\n" + summary)

        # Create visualization
        visualize_detections(
            pc_array=point_cloud,
            detections=inference_result,
            save_path=f"{output_stem}_detection_vis",
            interactive=interactive,
            conf_threshold=conf_thr,
            view_angle=view_angle,
        )

        logger.info(f"Visualization saved to: {output_stem}_detection_vis_*.png")
    elif visualize and not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization requested but open3d is not available. Skipping.")

    return inference_result


def prepare_inference(
    cfg_path: str, checkpoint_path: str, device: torch.device
) -> SparseRadarPCInference:
    """Factory helper to create a reusable inference runner."""
    return SparseRadarPCInference(cfg_path, checkpoint_path, device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process radar .bin file and run K-Radar inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_radar_kradar_pretrained.py \\
      --config kradar/configs/cfg_RTNH_wide.yml \\
      --checkpoint checkpoints/rtnh_wide.pt \\
                --bin-file data/frame.bin \\
            --config-file config_files/AWR2243_87m_17cm_64_3_256.txt
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
        default=0.3,
        help="Confidence threshold for detections",
    )

    # Radar data inputs (replacing --npy)
    parser.add_argument(
        "--bin-file", "-b", required=True, help="Path to .bin radar frame file"
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
