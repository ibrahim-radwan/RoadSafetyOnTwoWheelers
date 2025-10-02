#!/usr/bin/env python3
"""
Standalone script to process a single radar frame (.bin) and produce K-Radar artifacts.

This script takes a .bin radar frame file and a radar config file as inputs,
processes the frame through the K-Radar processing pipeline, and outputs:
- Sparse point cloud (.npy)
- arr_zyx cube (.mat)
- Tesseract 4D tensor (.mat)

Usage:
    python process_radar_kradar.py <frame.bin> <config_file.txt> [--output-dir OUTPUT_DIR]

Example:
    python process_radar_kradar.py data/1234567890_12345_000000000001.bin config_files/AWR2243_87m_17cm_64_3_256.txt
"""

import os
import sys
import argparse
import numpy as np
from typing import Optional

from sample_processing.radar_params import ADCParams
from sample_processing.radar_proc_kradar import process_3d_radar_frame_kradar
from utils import setup_logger

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_bin_frame(filepath: str) -> np.ndarray:
    """
    Load a radar frame from .bin file.

    Args:
        filepath: Path to .bin file

    Returns:
        Raw data array (dtype=int16)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Frame file not found: {filepath}")

    data_buf = np.fromfile(filepath, dtype=np.int16)

    if data_buf.size == 0:
        raise ValueError(f"Empty or corrupted frame file: {filepath}")

    return data_buf


def preprocess_frame(raw_data: np.ndarray, adc_params: ADCParams) -> np.ndarray:
    """
    Preprocess raw radar data to complex frame format.

    This follows the same preprocessing logic as used in the analyser:
    - Reshape according to ADC parameters (chirps, tx, samples, IQ, rx)
    - Transpose to (chirps, tx, rx, samples, IQ)
    - Convert to complex64 format

    Args:
        raw_data: Raw int16 data from .bin file
        adc_params: ADC parameters from config file

    Returns:
        Complex-valued radar frame with shape (chirps, tx, rx, samples)
    """
    frame = np.reshape(
        raw_data,
        (
            adc_params.chirps,
            adc_params.tx,
            adc_params.samples,
            adc_params.IQ,
            adc_params.rx,
        ),
    )

    frame = np.transpose(frame, (0, 1, 4, 2, 3))
    # Shape is now (chirps, tx, rx, samples, IQ)

    complex_frame = (1j * frame[..., 1] + frame[..., 0]).astype(np.complex64)  # I first

    expected_shape = (
        adc_params.chirps,
        adc_params.tx,
        adc_params.rx,
        adc_params.samples,
    )

    if complex_frame.shape != expected_shape:
        raise ValueError(
            f"complex_frame shape mismatch! Expected: {expected_shape}, Actual: {complex_frame.shape}"
        )

    return complex_frame


def save_point_cloud_npy(output_stem: str, result: dict, logger) -> None:
    """
    Save sparse point cloud to .npy file.

    Format: Nx4 array [y, x, z, power] matching K-Radar convention

    Args:
        output_stem: Output file stem (without extension)
        result: Processing result dictionary from kradar pipeline
        logger: Logger instance
    """
    try:
        x = np.asarray(result.get("x_pos", []), dtype=float)
        y = np.asarray(result.get("y_pos", []), dtype=float)
        z = np.asarray(result.get("z_pos", []), dtype=float)
        snr = np.asarray(result.get("snrs", []), dtype=float)

        n = min(x.shape[0], y.shape[0], z.shape[0], snr.shape[0])

        # K-Radar convention: [y, x, z, power]
        arr = np.stack((y[:n], x[:n], z[:n], snr[:n]), axis=1)

        output_path = output_stem + ".npy"
        np.save(output_path, arr)
        logger.info(f"Saved sparse point cloud: {output_path} (shape={arr.shape})")
    except Exception as e:
        logger.error(f"Failed to save point cloud .npy: {e}")


def save_point_cloud_pngs(output_stem: str, result: dict, adc_params, logger) -> None:
    """
    Save sparse point cloud visualizations as PNG images.

    Creates two views:
    - XY view (forward-lateral): x (forward, always positive) vs y (lateral)
    - XZ view (forward-vertical): x (forward, always positive) vs z (vertical)

    Args:
        output_stem: Output file stem (without extension)
        result: Processing result dictionary from kradar pipeline
        adc_params: ADC parameters for max_range
        logger: Logger instance
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, skipping PNG visualizations")
        return

    try:
        x_pos = np.asarray(result.get("x_pos", []), dtype=float)
        y_pos = np.asarray(result.get("y_pos", []), dtype=float)
        z_pos = np.asarray(result.get("z_pos", []), dtype=float)
        snr = np.asarray(result.get("snrs", []), dtype=float)

        # Get max_range from adc_params for axis limits
        max_range = getattr(adc_params, "max_range", 100.0)

        # Normalize SNR for colormap (only if we have detections)
        if len(x_pos) > 0:
            if np.max(snr) > np.min(snr):
                # Use logarithmic scaling to enhance contrast
                snr_log = np.log10(snr + 1e-10)  # Add small value to avoid log(0)

                # Use percentile-based normalization for better color distribution
                # Map bottom 20% to light gray, top 20% to black, rest distributed in between
                p20 = np.percentile(snr_log, 20)
                p80 = np.percentile(snr_log, 80)

                # Clip and normalize to [0, 1]
                snr_norm = np.clip((snr_log - p20) / (p80 - p20), 0.0, 1.0)

                # Log the SNR distribution for debugging
                logger.info(
                    f"SNR stats: min={np.min(snr):.2e}, max={np.max(snr):.2e}, "
                    f"mean={np.mean(snr):.2e}, std={np.std(snr):.2e}"
                )
                logger.info(
                    f"SNR log range: [{np.min(snr_log):.2f}, {np.max(snr_log):.2f}], "
                    f"p20={p20:.2f}, p80={p80:.2f}"
                )
                logger.info(
                    f"Normalized SNR: min={np.min(snr_norm):.3f}, max={np.max(snr_norm):.3f}, "
                    f"median={np.median(snr_norm):.3f}"
                )
            else:
                snr_norm = np.ones_like(snr) * 0.5
        else:
            logger.warning("No detections to visualize, creating empty plots")
            snr_norm = np.array([])

        # Create custom colormap: light gray to black (excluding white)
        # Start from 0.7 gray (light gray) to 0.0 (black)
        colors = [(0.7, 0.7, 0.7), (0.0, 0.0, 0.0)]  # Light gray to black
        n_bins = 100
        cmap_custom = LinearSegmentedColormap.from_list("gray_custom", colors, N=n_bins)

        # --- XY View (Forward-Lateral) ---
        fig, ax = plt.subplots(figsize=(10, 10))
        if len(x_pos) > 0:
            scatter = ax.scatter(
                y_pos,
                x_pos,  # y is lateral (x-axis), x is forward (y-axis)
                c=snr_norm,
                s=0.5,  # 1/5 of original size (30/5 = 6)
                cmap=cmap_custom,  # Custom: light gray (low) to black (high)
                vmin=0.0,
                vmax=1.0,
                alpha=1.0,
            )
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(scatter, cax=cax)
            cbar.set_label("Normalized Power", fontsize=10)

        ax.set_xlabel("y (m)", fontsize=12, fontweight="bold")
        ax.set_ylabel("x (m)", fontsize=12, fontweight="bold")
        ax.set_title("Point Cloud - XY View (Top-Down)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="blue", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color="blue", linestyle="--", linewidth=0.8, alpha=0.5)

        # Set limits to max_range
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([0, max_range])

        plt.tight_layout()
        xy_path = output_stem + "_xy.png"
        plt.savefig(xy_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Saved XY view: {xy_path}")

        # --- XZ View (Forward-Vertical) ---
        fig, ax = plt.subplots(figsize=(10, 10))
        if len(x_pos) > 0:
            scatter = ax.scatter(
                z_pos,
                x_pos,  # z is vertical (x-axis), x is forward (y-axis)
                c=snr_norm,
                s=0.5,  # 1/5 of original size (30/5 = 6)
                cmap=cmap_custom,  # Custom: light gray (low) to black (high)
                vmin=0.0,
                vmax=1.0,
                alpha=1.0,
            )
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(scatter, cax=cax)
            cbar.set_label("Normalized Power", fontsize=10)

        ax.set_xlabel("z (m)", fontsize=12, fontweight="bold")
        ax.set_ylabel("x (m)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Point Cloud - XZ View (Side View)", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="blue", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color="blue", linestyle="--", linewidth=0.8, alpha=0.5)

        # Set limits: z symmetric around 0, x from 0 to max_range
        z_max = max_range * 0.3  # 30% of max_range for vertical
        ax.set_xlim([-z_max, z_max])
        ax.set_ylim([0, max_range])

        plt.tight_layout()
        xz_path = output_stem + "_xz.png"
        plt.savefig(xz_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Saved XZ view: {xz_path}")

    except Exception as e:
        logger.error(f"Failed to save point cloud PNGs: {e}")
        import traceback

        logger.error(traceback.format_exc())


def save_tesseract_mat(output_stem: str, tesseract: np.ndarray, logger) -> None:
    """
    Save 4D tesseract (DREA tensor) to .mat file.

    Args:
        output_stem: Output file stem (without extension)
        tesseract: 4D array with shape (Doppler, Range, Elevation, Azimuth)
        logger: Logger instance
    """
    try:
        from scipy.io import savemat

        if tesseract is None:
            tesseract = np.zeros((0, 0, 0, 0), dtype=np.float32)

        output_path = output_stem + "_tesseract.mat"
        savemat(output_path, {"arrDREA": tesseract})
        logger.info(f"Saved tesseract: {output_path} (shape={tesseract.shape})")
    except Exception as e:
        logger.error(f"Failed to save tesseract .mat: {e}")


def save_arr_zyx_mat(
    output_stem: str,
    tesseract: np.ndarray,
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
    adc_params: ADCParams,
    logger,
) -> None:
    """
    Generate and save K-Radar style arr_zyx cube from tesseract.

    The arr_zyx cube is derived from the tesseract via:
    1. Aggregate across Doppler dimension (mean)
    2. Interpolate from polar (Range, Elevation, Azimuth) to Cartesian (Z, Y, X)
    3. Transpose to K-Radar ordering (Z, X, Y)

    Args:
        output_stem: Output file stem (without extension)
        tesseract: 4D array (Doppler, Range, Elevation, Azimuth)
        az_grid_deg: Azimuth grid in degrees
        el_grid_deg: Elevation grid in degrees
        adc_params: ADC parameters for resolution and range
        logger: Logger instance
    """
    try:
        from scipy.io import savemat

        if (
            tesseract is None
            or not isinstance(tesseract, np.ndarray)
            or tesseract.size == 0
        ):
            output_path = output_stem + "_arr_zyx.mat"
            savemat(output_path, {"arr_zyx": np.zeros((0, 0, 0), dtype=np.float32)})
            logger.warning(f"Saved empty arr_zyx: {output_path}")
            return

        # Average across Doppler to get Range/Elevation/Azimuth cube
        rea = np.mean(tesseract, axis=0)  # (R, E, A)

        range_res = float(getattr(adc_params, "range_resolution", 1.0))
        max_range = getattr(adc_params, "max_range", None)
        if max_range is None:
            max_range = range_res * float(getattr(adc_params, "samples", rea.shape[0]))
        max_range = float(max_range)

        # Ensure az/el grids present
        if az_grid_deg is None:
            az_grid_deg = np.linspace(-90.0, 90.0, rea.shape[2], dtype=np.float32)
        if el_grid_deg is None:
            el_grid_deg = np.linspace(-30.0, 30.0, rea.shape[1], dtype=np.float32)

        az_r = np.deg2rad(az_grid_deg).astype(np.float32)  # (A,)
        el_r = np.deg2rad(el_grid_deg).astype(np.float32)  # (E,)
        r_bins = np.arange(rea.shape[0], dtype=np.float32) * range_res

        dr = range_res
        x_max = max_range
        y_max = max_range
        z_max = 0.3 * max_range

        xs = np.arange(-x_max, x_max + 1e-9, dr, dtype=np.float32)  # lateral
        ys = np.arange(0.0, y_max + 1e-9, dr, dtype=np.float32)  # forward
        zs = np.arange(-z_max, z_max + 1e-9, dr, dtype=np.float32)  # vertical

        # Internal build uses (Z, Y, X); initialize with -1 sentinel
        zyx_cube_internal = np.full((zs.size, ys.size, xs.size), -1.0, dtype=np.float32)

        # Precompute 2D lateral/forward grids (Ny, Nx)
        y_grid, x_grid = np.meshgrid(ys, xs, indexing="ij")
        x2 = x_grid * x_grid
        y2 = y_grid * y_grid
        horiz_sq = x2 + y2
        horiz = np.sqrt(horiz_sq, dtype=np.float32)

        # Helper to obtain interpolation indices & fractional part
        def _interval_indices(values: np.ndarray, grid: np.ndarray):
            idx = np.searchsorted(grid, values, side="right") - 1
            valid = (idx >= 0) & (idx < grid.size - 1)
            idx_clipped = np.clip(idx, 0, grid.size - 2)
            g0 = grid[idx_clipped]
            g1 = grid[idx_clipped + 1]
            denom = g1 - g0
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(
                    valid, (values - g0) / np.where(denom == 0, 1.0, denom), 0.0
                )
            np.clip(t, 0.0, 1.0, out=t)
            return idx_clipped, t.astype(np.float32), valid

        Ny, Nx = y_grid.shape
        # Iterate only along Z to keep memory bounded; vectorize X/Y plane
        report_every = max(1, zs.size // 10)
        for iz, z in enumerate(zs):
            z2 = float(z * z)
            # Range for this slice
            r = np.sqrt(horiz_sq + z2, dtype=np.float32)  # (Ny, Nx)
            # Masks for valid geometry
            in_range = r <= max_range
            # Elevation: arctan2(z, horiz)
            with np.errstate(divide="ignore", invalid="ignore"):
                el = np.arctan2(z, np.where(horiz > 0.0, horiz, 1.0)).astype(np.float32)
            # Replace where horiz == 0 explicitly
            if z > 0:
                el = np.where(horiz == 0.0, 0.5 * np.pi, el)
            elif z < 0:
                el = np.where(horiz == 0.0, -0.5 * np.pi, el)
            else:
                el = np.where(horiz == 0.0, 0.0, el)
            # Azimuth: arctan2(x, y)
            az = np.arctan2(x_grid, y_grid).astype(np.float32)

            ir, tr, valid_r = _interval_indices(r, r_bins)
            ie, te, valid_e = _interval_indices(el, el_r)
            ia, ta, valid_a = _interval_indices(az, az_r)
            valid = in_range & valid_r & valid_e & valid_a
            if not np.any(valid):
                continue

            # Prepare corner indices
            ir1 = ir + 1
            ie1 = ie + 1
            ia1 = ia + 1

            # Advanced indexing for corners (broadcast over Ny, Nx)
            c000 = rea[ir, ie, ia]
            c001 = rea[ir, ie, ia1]
            c010 = rea[ir, ie1, ia]
            c011 = rea[ir, ie1, ia1]
            c100 = rea[ir1, ie, ia]
            c101 = rea[ir1, ie, ia1]
            c110 = rea[ir1, ie1, ia]
            c111 = rea[ir1, ie1, ia1]

            # Interpolate along azimuth (ta)
            c00 = c000 * (1.0 - ta) + c001 * ta
            c01 = c010 * (1.0 - ta) + c011 * ta
            c10 = c100 * (1.0 - ta) + c101 * ta
            c11 = c110 * (1.0 - ta) + c111 * ta
            # Interpolate along elevation (te)
            c0 = c00 * (1.0 - te) + c01 * te
            c1 = c10 * (1.0 - te) + c11 * te
            # Interpolate along range (tr)
            vals = c0 * (1.0 - tr) + c1 * tr
            # Assign only valid voxels (others remain -1)
            slice_view = zyx_cube_internal[iz]
            slice_view[valid] = vals[valid]

            if iz % report_every == 0 or iz == zs.size - 1:
                logger.debug(
                    f"arr_zyx build progress: {iz + 1}/{zs.size} slices ({100.0 * (iz + 1) / zs.size:.1f}%)"
                )

        # Convert to K-Radar ordering (z, x, y) from internal (z, y, x)
        arr_zyx = np.transpose(zyx_cube_internal, (0, 2, 1))

        output_path = output_stem + "_arr_zyx.mat"
        savemat(output_path, {"arr_zyx": arr_zyx})
        logger.info(f"Saved arr_zyx: {output_path} (shape={arr_zyx.shape})")
    except Exception as e:
        logger.error(f"Failed to save arr_zyx .mat: {e}")


def process_single_frame(
    bin_filepath: str,
    config_filepath: str,
    output_dir: Optional[str] = None,
    az_range: tuple = (-53, 53),
    el_range: tuple = (-18, 18),
) -> None:
    """
    Process a single radar frame and produce K-Radar artifacts.

    Args:
        bin_filepath: Path to .bin radar frame file
        config_filepath: Path to radar config .txt file
        output_dir: Optional output directory (defaults to same directory as .bin file)
        az_range: Azimuth range in degrees (min, max)
        el_range: Elevation range in degrees (min, max)
    """
    logger = setup_logger("process_radar_kradar")

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
    logger.info(
        f"Range resolution: {adc_params.range_resolution:.4f} m, "
        f"Doppler resolution: {adc_params.doppler_resolution:.4f} m/s"
    )

    # Load raw data from .bin file
    logger.info("Loading .bin frame...")
    raw_data = load_bin_frame(bin_filepath)
    logger.info(f"Loaded {raw_data.size} int16 samples")

    # Preprocess to complex frame
    logger.info("Preprocessing frame...")
    complex_frame = preprocess_frame(raw_data, adc_params)
    logger.info(f"Preprocessed frame shape: {complex_frame.shape}")

    # Process through K-Radar pipeline
    logger.info(
        f"Processing with K-Radar pipeline (az_range={az_range}, el_range={el_range})..."
    )
    result = process_3d_radar_frame_kradar(
        complex_frame,
        adc_params,
        tuning=None,
        az_range=az_range,
        el_range=el_range,
    )

    num_detections = len(result.get("x_pos", []))
    logger.info(f"Processing complete. Detections: {num_detections}")

    # Determine output file stem
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(bin_filepath))
    else:
        os.makedirs(output_dir, exist_ok=True)

    bin_basename = os.path.basename(bin_filepath)
    output_stem = os.path.join(output_dir, os.path.splitext(bin_basename)[0])

    logger.info(f"Saving artifacts to: {output_dir}")

    # Save artifacts
    save_point_cloud_npy(output_stem, result, logger)
    save_point_cloud_pngs(output_stem, result, adc_params, logger)

    tesseract = result.get("tesseract")
    if tesseract is not None:
        save_tesseract_mat(output_stem, tesseract, logger)

        az_grid = result.get("tesseract_az_grid_deg")
        el_grid = result.get("tesseract_el_grid_deg")
        save_arr_zyx_mat(output_stem, tesseract, az_grid, el_grid, adc_params, logger)
    else:
        logger.warning(
            "Tesseract not present in result, skipping tesseract and arr_zyx artifacts"
        )

    logger.info("Processing complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Process a single radar frame and produce K-Radar artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using positional arguments
  python process_radar_kradar.py data/1234567890_12345_000000000001.bin \\
      config_files/AWR2243_87m_17cm_64_3_256.txt --output-dir processed/

  # Using named arguments
  python process_radar_kradar.py --bin-file data/frame.bin \\
      --config-file config_files/AWR2243_87m_17cm_64_3_256.txt

This will create:
  - processed/1234567890_12345_000000000001.npy (sparse point cloud)
  - processed/1234567890_12345_000000000001_xy.png (top-down view)
  - processed/1234567890_12345_000000000001_xz.png (side view)
  - processed/1234567890_12345_000000000001_tesseract.mat (4D DREA tensor)
  - processed/1234567890_12345_000000000001_arr_zyx.mat (3D Cartesian cube)
        """,
    )

    parser.add_argument(
        "--bin-file",
        "-b",
        dest="bin_file_flag",
        help="Path to .bin radar frame file",
    )

    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file_flag",
        help="Path to radar config .txt file",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory for artifacts (default: same directory as .bin file)",
        default=None,
    )

    parser.add_argument(
        "--az-range",
        nargs=2,
        type=int,
        default=[-53, 53],
        metavar=("MIN", "MAX"),
        help="Azimuth range in degrees (default: -53 53)",
    )

    parser.add_argument(
        "--el-range",
        nargs=2,
        type=int,
        default=[-18, 18],
        metavar=("MIN", "MAX"),
        help="Elevation range in degrees (default: -18 18)",
    )

    args = parser.parse_args()

    # Determine bin_file: use flag if provided, otherwise positional
    bin_file = args.bin_file_flag if args.bin_file_flag else args.bin_file
    if not bin_file:
        parser.error(
            "bin_file is required (either as positional argument or --bin-file)"
        )

    # Determine config_file: use flag if provided, otherwise positional
    config_file = args.config_file_flag if args.config_file_flag else args.config_file
    if not config_file:
        parser.error(
            "config_file is required (either as positional argument or --config-file)"
        )

    try:
        process_single_frame(
            bin_file,
            config_file,
            output_dir=args.output_dir,
            az_range=tuple(args.az_range),
            el_range=tuple(args.el_range),
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
