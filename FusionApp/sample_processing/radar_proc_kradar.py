"""FFT-based 3D radar processing pipeline producing a Doppler-Range-Elevation-Azimuth tensor."""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple, cast

import numpy as np
from mmwave import dsp
from scipy import ndimage, signal

from sample_processing.radar_proc import logger

# Constants for polar quantile detection (K-Radar approach)
# Adjusted for more detections: 0.985 = top 1.5% (~15k points)
# K-Radar defaults: 0.99 (1%), 0.999 (0.1%)
POLAR_POWER_QUANTILE = 0.985  # Keep top 1.5% of power values

# Constants for ZYX cube generation (kept for backward compatibility if needed)
ZYX_X_MAX_MULTIPLIER = 1.0  # multiplier for max_range to get x_max
ZYX_Y_MAX_MULTIPLIER = 1.0  # multiplier for max_range to get y_max
ZYX_Z_MAX_MULTIPLIER = 0.3  # multiplier for max_range to get z_max

# Constants for 3D CFAR (ZYX) - kept for backward compatibility
# NOTE: Smaller guard/train cells = more sensitive detection but more false alarms
# If missing detections, try reducing train cells or increasing FA_RATE
CFAR_GUARD_CELL_ZYX = [1, 1, 1]  # Z, Y, X guard cells (number of cells)
CFAR_TRAIN_CELL_ZYX = [1, 1, 1]  # Z, Y, X training cells (number of cells)


def _compute_fft_drea(
    aoa_input: np.ndarray,
    adc_params,
    az_range: Tuple[int, int],
    el_range: Tuple[int, int],
    az_fft_size: int,
    el_fft_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 4D Doppler-Range-Elevation-Azimuth tensor via cascaded FFTs."""

    num_range_bins, num_vrx, num_doppler_bins = aoa_input.shape
    num_tx = int(getattr(adc_params, "tx", 0))
    num_rx = int(getattr(adc_params, "rx", 0))
    expected_vrx = num_tx * num_rx
    if num_vrx != expected_vrx:
        raise ValueError(
            f"aoa_input expects {expected_vrx} virtual receivers (tx*rx) but got {num_vrx}"
        )

    if num_tx < 3:
        raise ValueError("FFT DREA pipeline requires at least 3 TX antennas")

    aoa_cube = aoa_input.reshape(num_range_bins, num_tx, num_rx, num_doppler_bins)
    aoa_cube = np.transpose(aoa_cube, (3, 0, 1, 2)).astype(np.complex64, copy=False)

    az_virtual = aoa_cube[:, :, :2, :].reshape(num_doppler_bins, num_range_bins, -1)
    el_virtual = aoa_cube[:, :, 2:, :].reshape(num_doppler_bins, num_range_bins, -1)

    if az_virtual.shape[2] == 0 or el_virtual.shape[2] == 0:
        raise ValueError(
            "Insufficient virtual elements to perform azimuth/elevation FFTs"
        )

    az_bins = int(max(int(az_fft_size), az_virtual.shape[2]))
    el_bins = int(max(int(el_fft_size), el_virtual.shape[2]))

    az_fft = np.fft.fft(az_virtual, n=az_bins, axis=2)
    el_fft = np.fft.fft(el_virtual, n=el_bins, axis=2)

    az_spectrum = np.abs(np.fft.fftshift(az_fft, axes=2)) ** 2
    el_spectrum = np.abs(np.fft.fftshift(el_fft, axes=2)) ** 2

    tesseract = el_spectrum[:, :, :, None] * az_spectrum[:, :, None, :]
    tesseract = tesseract.astype(np.float32, copy=False)

    azimuth_grid_deg = np.linspace(az_range[0], az_range[1], az_bins, dtype=np.float32)
    elevation_grid_deg = np.linspace(
        el_range[0], el_range[1], el_bins, dtype=np.float32
    )

    try:
        print(
            "[KRadar] _compute_fft_drea outputs -> tesseract: {} az_grid: {} el_grid: {}".format(
                getattr(tesseract, "shape", None),
                getattr(azimuth_grid_deg, "shape", None),
                getattr(elevation_grid_deg, "shape", None),
            )
        )
    except Exception:
        pass

    return tesseract, azimuth_grid_deg, elevation_grid_deg


def _apply_polar_quantile_detection(
    tesseract: np.ndarray,
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
    adc_params,
    quantile: float = POLAR_POWER_QUANTILE,
) -> np.ndarray:
    """
    Apply quantile-based detection directly on polar (REA) cube.
    Matches K-Radar's tools/radar_film/get_pw_dist.py approach.

    This avoids interpolation loss by detecting in polar domain,
    then converting only the detected points to Cartesian.

    Args:
        tesseract: 4D array (D, R, E, A) - Doppler, Range, Elevation, Azimuth
        az_grid_deg: Azimuth angles in degrees
        el_grid_deg: Elevation angles in degrees
        adc_params: ADC parameters for range/doppler resolution
        quantile: Power quantile threshold (default 0.999 = top 0.1%)

    Returns:
        point_cloud: Nx5 array [x, y, z, power, velocity]
    """
    # Aggregate doppler: take max and track which doppler bin
    # This preserves moving target signal strength

    # CRITICAL FIX: Exclude zero-doppler (static) bins to avoid detecting clutter
    # K-Radar uses this approach to focus on moving targets
    doppler_center = tesseract.shape[0] // 2
    doppler_zero_width = 2  # Exclude ±2 bins around zero velocity

    # Create a mask that zeros out the static bins
    tesseract_moving = tesseract.copy()
    tesseract_moving[
        doppler_center - doppler_zero_width : doppler_center + doppler_zero_width + 1,
        :,
        :,
        :,
    ] = 0

    rea_cube = np.max(tesseract_moving, axis=0).astype(np.float32)  # (R, E, A)
    doppler_idx_max = np.argmax(
        tesseract_moving, axis=0
    )  # (R, E, A) - which doppler bin had max

    # Apply quantile threshold on polar cube (only moving targets now)
    # Filter out zeros from the quantile calculation
    rea_cube_nonzero = rea_cube[rea_cube > 0]
    if rea_cube_nonzero.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    power_threshold = np.quantile(rea_cube_nonzero, quantile)

    # Extract indices where power exceeds threshold
    r_idx, e_idx, a_idx = np.where(rea_cube > power_threshold)

    if len(r_idx) == 0:
        return np.empty((0, 5), dtype=np.float32)

    # Get power values
    powers = rea_cube[r_idx, e_idx, a_idx].astype(np.float32)

    # Get doppler indices for velocity calculation
    d_idx = doppler_idx_max[r_idx, e_idx, a_idx]

    # Convert indices to physical coordinates
    range_res = float(getattr(adc_params, "range_resolution", 1.0))
    doppler_res = float(getattr(adc_params, "doppler_resolution", 0.1))

    # Range (in meters)
    r = r_idx.astype(np.float32) * range_res

    # Azimuth and Elevation (convert to radians)
    az = np.deg2rad(az_grid_deg[a_idx]).astype(np.float32)
    el = np.deg2rad(el_grid_deg[e_idx]).astype(np.float32)

    # K-Radar coordinate convention: flip azimuth and elevation
    # (matches tools/radar_film/get_pw_dist.py lines 254-256)
    az = -az
    el = -el

    # Convert polar to Cartesian
    # (matches tools/radar_film/get_pw_dist.py lines 258-261)
    cos_el = np.cos(el)
    cos_az = np.cos(az)
    sin_az = np.sin(az)
    sin_el = np.sin(el)

    x = r * cos_el * cos_az
    y = r * cos_el * sin_az
    z = r * sin_el

    # Calculate velocity from doppler
    # tesseract is fftshifted, so center is at shape[0]//2
    doppler_center = tesseract.shape[0] // 2
    doppler_offsets = (d_idx - doppler_center).astype(np.float32)
    velocities = doppler_offsets * doppler_res

    # Stack into point cloud: [x, y, z, power, velocity]
    point_cloud = np.column_stack([x, y, z, powers, velocities])

    return point_cloud


def _compute_zyx_cube(
    tesseract: np.ndarray,
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
    adc_params,
    doppler_aggregation: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert DREA tesseract to ZYX cube via trilinear interpolation.

    Follows the MATLAB gen_3_get_zyx_cube.m logic:
    1. Aggregate across doppler to get REA (Range-Elevation-Azimuth) cube
    2. Create Cartesian grid (z, y, x)
    3. For each (z, y, x), convert to polar (r, e, a) and interpolate from REA

    Args:
        doppler_aggregation: Method to aggregate doppler dimension
            - "max": Take maximum (best for moving targets, recommended)
            - "mean": Average (smooths noise but dilutes moving targets)
            - "percentile_90": 90th percentile (balance between max and mean)

    Returns:
        arr_zyx: 3D array with shape (Z, Y, X), initialized with -1 for invalid voxels
        arr_z: Z-axis grid (vertical, in meters)
        arr_y: Y-axis grid (forward, in meters)
        arr_x: X-axis grid (lateral, in meters)
    """
    # Aggregate across Doppler dimension
    if doppler_aggregation == "max":
        rea = np.max(tesseract, axis=0)  # (R, E, A)
    elif doppler_aggregation == "percentile_90":
        rea = np.percentile(tesseract, 90, axis=0)  # (R, E, A)
    elif doppler_aggregation == "mean":
        rea = np.mean(tesseract, axis=0)  # (R, E, A)
    else:
        raise ValueError(f"Unknown doppler_aggregation: {doppler_aggregation}")

    # Get range parameters
    range_res = float(getattr(adc_params, "range_resolution", 1.0))
    max_range = getattr(adc_params, "max_range", None)
    if max_range is None:
        max_range = range_res * float(getattr(adc_params, "samples", rea.shape[0]))
    max_range = float(max_range)

    # Convert angle grids to radians
    az_r = np.deg2rad(az_grid_deg).astype(np.float32)  # (A,)
    el_r = np.deg2rad(el_grid_deg).astype(np.float32)  # (E,)

    # Range bins
    r_bins = np.arange(rea.shape[0], dtype=np.float32) * range_res

    # Define Cartesian grid bounds
    dr = range_res
    x_max = max_range * ZYX_X_MAX_MULTIPLIER
    y_max = max_range * ZYX_Y_MAX_MULTIPLIER
    z_max = max_range * ZYX_Z_MAX_MULTIPLIER

    arr_x = np.arange(-x_max, x_max + 1e-9, dr, dtype=np.float32)  # lateral
    arr_y = np.arange(0.0, y_max + 1e-9, dr, dtype=np.float32)  # forward
    arr_z = np.arange(-z_max, z_max + 1e-9, dr, dtype=np.float32)  # vertical

    # Initialize ZYX cube with -1 (sentinel for invalid voxels)
    arr_zyx = np.full((arr_z.size, arr_y.size, arr_x.size), -1.0, dtype=np.float32)

    # Precompute 2D grids for X-Y plane
    y_grid, x_grid = np.meshgrid(arr_y, arr_x, indexing="ij")  # (Ny, Nx)
    x2 = x_grid * x_grid
    y2 = y_grid * y_grid
    horiz_sq = x2 + y2
    horiz = np.sqrt(horiz_sq, dtype=np.float32)

    # Range bounds for validity check
    r_min = r_bins[0]
    r_max = r_bins[-1]
    a_min = az_r[0]
    a_max = az_r[-1]
    e_min = el_r[0]
    e_max = el_r[-1]

    # Helper function to find interpolation indices
    def _interval_indices(values: np.ndarray, grid: np.ndarray):
        """Find bracketing indices and fractional position for interpolation."""
        idx = np.searchsorted(grid, values, side="right") - 1
        valid = (idx >= 0) & (idx < grid.size - 1)
        idx_clipped = np.clip(idx, 0, grid.size - 2)
        g0 = grid[idx_clipped]
        g1 = grid[idx_clipped + 1]
        denom = g1 - g0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(valid, (values - g0) / np.where(denom == 0, 1.0, denom), 0.0)
        np.clip(t, 0.0, 1.0, out=t)
        return idx_clipped, t.astype(np.float32), valid

    # Precompute azimuth once (doesn't depend on z)
    # MATLAB uses: a = atan(-y/x), which is atan2(-y, x)
    # This matches their coordinate convention where positive y points LEFT
    az = np.arctan2(-y_grid, x_grid).astype(np.float32)

    # Process each Z slice
    for iz, z in enumerate(arr_z):
        z2 = z * z

        # Compute range for this Z slice
        r = np.sqrt(horiz_sq + z2, dtype=np.float32)  # (Ny, Nx)

        # Compute elevation: arctan(z / horiz) with vectorized conditionals
        with np.errstate(divide="ignore", invalid="ignore"):
            el = np.arctan2(z, horiz).astype(np.float32)
        # Handle horiz == 0 cases more efficiently
        horiz_zero = horiz == 0.0
        if np.any(horiz_zero):
            if z > 0:
                el[horiz_zero] = 0.5 * np.pi
            elif z < 0:
                el[horiz_zero] = -0.5 * np.pi
            # else: el[horiz_zero] remains 0.0 from arctan2

        # Check bounds
        in_bounds = (
            (r >= r_min)
            & (r <= r_max)
            & (az >= a_min)
            & (az <= a_max)
            & (el >= e_min)
            & (el <= e_max)
        )

        # Find interpolation indices
        ir, tr, valid_r = _interval_indices(r, r_bins)
        ie, te, valid_e = _interval_indices(el, el_r)
        ia, ta, valid_a = _interval_indices(az, az_r)

        valid = in_bounds & valid_r & valid_e & valid_a
        if not np.any(valid):
            continue

        # Trilinear interpolation: get 8 corners and compute in one expression chain
        # This reduces intermediate array allocations
        ir1 = ir + 1
        ie1 = ie + 1
        ia1 = ia + 1

        # Precompute interpolation weights
        ta_inv = 1.0 - ta
        te_inv = 1.0 - te
        tr_inv = 1.0 - tr

        # Vectorized trilinear interpolation (reduces memory allocations)
        vals = (
            rea[ir, ie, ia] * tr_inv * te_inv * ta_inv
            + rea[ir, ie, ia1] * tr_inv * te_inv * ta
            + rea[ir, ie1, ia] * tr_inv * te * ta_inv
            + rea[ir, ie1, ia1] * tr_inv * te * ta
            + rea[ir1, ie, ia] * tr * te_inv * ta_inv
            + rea[ir1, ie, ia1] * tr * te_inv * ta
            + rea[ir1, ie1, ia] * tr * te * ta_inv
            + rea[ir1, ie1, ia1] * tr * te * ta
        )

        # Assign to valid voxels
        arr_zyx[iz][valid] = vals[valid]

    return arr_zyx, arr_z, arr_y, arr_x


def _apply_3d_cfar(
    arr_zyx: np.ndarray,
    arr_z: np.ndarray,
    arr_y: np.ndarray,
    arr_x: np.ndarray,
    guard_cell_zyx: Tuple[int, int, int] = None,
    train_cell_zyx: Tuple[int, int, int] = None,
    fa_rate: float = None,
) -> np.ndarray:
    """
    Apply 3D CA-CFAR to the ZYX cube to extract sparse point cloud.

    Follows the approach from util_cfar.py ca_cfar method but adapted for 3D without doppler.

    Args:
        arr_zyx: 3D cube (Z, Y, X) with power values, -1 for invalid voxels
        arr_z, arr_y, arr_x: coordinate grids
        guard_cell_zyx: (nz, ny, nx) half-guard cells for each dimension
        train_cell_zyx: (nz, ny, nx) half-train cells for each dimension
        fa_rate: false alarm rate

    Returns:
        point cloud array with columns [x, y, z, power]
    """
    if guard_cell_zyx is None:
        guard_cell_zyx = tuple(CFAR_GUARD_CELL_ZYX)
    if train_cell_zyx is None:
        train_cell_zyx = tuple(CFAR_TRAIN_CELL_ZYX)
    if fa_rate is None:
        fa_rate = CFAR_FA_RATE

    # Mark invalid voxels
    invalid_mask = arr_zyx == -1.0

    # Normalize cube (use float32 to reduce memory bandwidth)
    cube_norm = arr_zyx.astype(np.float32, copy=True)
    cube_norm[invalid_mask] = 0.0
    cube_norm *= 1.0 / 1e13  # in-place normalization
    # Set invalid to mean to suppress detections at boundaries
    mean_val = np.mean(cube_norm[~invalid_mask]) if np.any(~invalid_mask) else 0.0
    cube_norm[invalid_mask] = mean_val

    # Generate 3D mask for CA-CFAR
    nh_g_z, nh_g_y, nh_g_x = guard_cell_zyx
    nh_t_z, nh_t_y, nh_t_x = train_cell_zyx

    mask_size = (
        2 * (nh_g_z + nh_t_z) + 1,
        2 * (nh_g_y + nh_t_y) + 1,
        2 * (nh_g_x + nh_t_x) + 1,
    )
    mask = np.ones(mask_size, dtype=np.float32)

    # Set guard cells to 0 (exclude from training)
    mask[
        nh_t_z : nh_t_z + 2 * nh_g_z + 1,
        nh_t_y : nh_t_y + 2 * nh_g_y + 1,
        nh_t_x : nh_t_x + 2 * nh_g_x + 1,
    ] = 0.0

    num_total_train_cells = np.count_nonzero(mask)
    mask = mask / num_total_train_cells  # normalize for average

    # Calculate alpha (CFAR threshold multiplier)
    alpha = num_total_train_cells * (fa_rate ** (-1.0 / num_total_train_cells) - 1.0)

    # Convolve to get local average of training cells
    # Use FFT-based convolution for large kernels (much faster)
    conv_out = signal.fftconvolve(cube_norm, mask, mode="same")
    # Handle boundary effects by using the mirror mode approximation
    # (fftconvolve uses zero-padding, so we clip to valid range after)
    conv_out = alpha * conv_out

    # Apply threshold and filter invalid voxels in one step
    detections = (cube_norm > conv_out) & (~invalid_mask)
    pc_idx = np.where(detections)

    # Extract corresponding power values (unnormalized)
    correp_power = arr_zyx[pc_idx]

    # Convert indices to Cartesian coordinates
    z_ind, y_ind, x_ind = pc_idx

    # Get grid resolution
    grid_size = float(arr_x[1] - arr_x[0]) if len(arr_x) > 1 else 1.0

    # Compute coordinates (center of voxel) - use vectorized operations
    z_min, y_min, x_min = float(arr_z[0]), float(arr_y[0]), float(arr_x[0])
    z_pc_coord = z_min + z_ind.astype(np.float32) * grid_size
    y_pc_coord = y_min + y_ind.astype(np.float32) * grid_size
    x_pc_coord = x_min + x_ind.astype(np.float32) * grid_size

    # Stack into point cloud: [x, y, z, power] - more efficient stacking
    total_values = np.column_stack(
        [x_pc_coord, y_pc_coord, z_pc_coord, correp_power]
    ).astype(np.float32)

    return total_values


def process_3d_radar_frame_kradar(
    frame: np.ndarray,
    adc_params,
    tuning: Optional[Dict] = None,
    az_range: Tuple[int, int] = (-60, 60),
    el_range: Tuple[int, int] = (-30, 30),
    *,
    az_fft_size: int = 64,
    el_fft_size: int = 32,
    range_window=dsp.utils.Window.HAMMING,
    doppler_window=dsp.utils.Window.HAMMING,
) -> Dict[str, np.ndarray | None]:
    """FFT-only 3D processing pipeline yielding a DREA tensor and sparse detections."""

    function_start = time.perf_counter()
    assert (
        int(getattr(adc_params, "tx", 0)) == 3
    ), "KRadar pipeline requires 3 TX antennas"

    step_start = time.perf_counter()
    frame = frame.reshape(
        adc_params.chirps * adc_params.tx, adc_params.rx, adc_params.samples
    )
    radar_cube = dsp.range_processing(frame, window_type_1d=range_window)
    radar_cube = radar_cube.reshape(
        adc_params.chirps, adc_params.tx * adc_params.rx, adc_params.samples
    )
    t_range = time.perf_counter() - step_start

    step_start = time.perf_counter()
    det_matrix, aoa_input = dsp.doppler_processing(
        radar_cube,
        num_tx_antennas=adc_params.tx,
        clutter_removal_enabled=False,
        interleaved=False,
        window_type_2d=doppler_window,
    )
    det_matrix = np.fft.fftshift(det_matrix, axes=1)
    aoa_input = np.fft.fftshift(aoa_input, axes=2)
    t_doppler = time.perf_counter() - step_start

    step_start = time.perf_counter()
    az_span = float(az_range[1] - az_range[0])
    el_span = float(el_range[1] - el_range[0])
    az_bins_target = max(2, int(np.ceil(abs(az_span))) + 1)
    el_bins_target = max(2, int(np.ceil(abs(el_span))) + 1)
    resolved_az_fft_size = max(az_bins_target, int(az_fft_size))
    resolved_el_fft_size = max(el_bins_target, int(el_fft_size))

    tesseract, azimuth_grid_deg, elevation_grid_deg = _compute_fft_drea(
        aoa_input,
        adc_params,
        az_range,
        el_range,
        resolved_az_fft_size,
        resolved_el_fft_size,
    )
    t_aoa = time.perf_counter() - step_start

    # Compute Range-Azimuth map by taking MAX over Doppler (axis 0) and Elevation (axis 2)
    # Tesseract shape: (D, R, E, A) -> RA shape: (R, A)
    # Use max to preserve peak signal strength (sum would wash out targets with noise)
    # Exclude static clutter (zero-doppler bins) before aggregation
    doppler_center = tesseract.shape[0] // 2
    doppler_zero_width = 2  # Exclude ±2 bins around zero velocity (same as detection)
    tesseract_for_ra = tesseract.copy()
    tesseract_for_ra[
        doppler_center - doppler_zero_width : doppler_center + doppler_zero_width + 1,
        :,
        :,
        :,
    ] = 0
    range_azimuth_map = np.max(
        tesseract_for_ra, axis=0
    )  # First max over doppler -> (R, E, A)
    range_azimuth_map = np.max(
        range_azimuth_map, axis=1
    )  # Then max over elevation -> (R, A)

    # Apply log scale to range-azimuth for better visualization
    range_azimuth_map = 20.0 * np.log10(np.abs(range_azimuth_map) + 1e-10)

    # K-RADAR PIPELINE: Apply polar quantile detection (matches their dataset generation)
    step_start = time.perf_counter()
    detection_params = (
        (tuning or {}).get("polar_detection", {}) if isinstance(tuning, dict) else {}
    )
    quantile = detection_params.get("power_quantile", POLAR_POWER_QUANTILE)

    point_cloud = _apply_polar_quantile_detection(
        tesseract, azimuth_grid_deg, elevation_grid_deg, adc_params, quantile=quantile
    )
    t_detection = time.perf_counter() - step_start

    num_det = point_cloud.shape[0]

    # Diagnostic logging for polar detection results
    try:
        if num_det > 0:
            logger.info(
                "[KRadar-Polar-DEBUG] Detections: N=%d, "
                "x: [%.2f, %.2f], y: [%.2f, %.2f], z: [%.2f, %.2f], "
                "power: min=%.2e, max=%.2e, mean=%.2e, "
                "velocity: [%.2f, %.2f] m/s",
                num_det,
                np.min(point_cloud[:, 0]),
                np.max(point_cloud[:, 0]),
                np.min(point_cloud[:, 1]),
                np.max(point_cloud[:, 1]),
                np.min(point_cloud[:, 2]),
                np.max(point_cloud[:, 2]),
                np.min(point_cloud[:, 3]),
                np.max(point_cloud[:, 3]),
                np.mean(point_cloud[:, 3]),
                np.min(point_cloud[:, 4]),
                np.max(point_cloud[:, 4]),
            )
    except Exception as e:
        logger.warning(f"[KRadar-Polar-DEBUG] Failed to log detection stats: {e}")

    if num_det == 0:
        total_time = time.perf_counter() - function_start
        try:
            logger.info(
                "[KRadar-Polar] total=%.3fs | range=%.3fs, doppler=%.3fs, aoa=%.3fs, detect=%.3fs, detN=0",
                total_time,
                t_range,
                t_doppler,
                t_aoa,
                t_detection,
            )
        except Exception:
            pass

        return {
            "range_doppler": det_matrix.T,
            "range_azimuth": range_azimuth_map.T,
            "x_pos": np.array([]),
            "y_pos": np.array([]),
            "z_pos": np.array([]),
            "velocities": np.array([]),
            "snrs": np.array([]),
            "azimuth_deg": np.array([]),
            "elevation_deg": np.array([]),
            "cluster_labels": np.array([]),
            "tesseract": tesseract,
            "tesseract_az_grid_deg": azimuth_grid_deg,
            "tesseract_el_grid_deg": elevation_grid_deg,
        }

    # Extract from point cloud: [x, y, z, power, velocity]
    xs = point_cloud[:, 0]
    ys = point_cloud[:, 1]
    zs = point_cloud[:, 2]
    powers = point_cloud[:, 3]
    velocities = point_cloud[:, 4]

    # Calculate azimuth and elevation from Cartesian coordinates
    horiz = np.sqrt(xs**2 + ys**2)
    azimuths_deg = np.rad2deg(np.arctan2(xs, ys)).astype(np.float32)
    elevations_deg = np.rad2deg(np.arctan2(zs, horiz)).astype(np.float32)

    # Use power as SNR proxy
    snrs = powers.astype(np.float32)

    total_time = time.perf_counter() - function_start
    try:
        logger.info(
            "[KRadar-Polar] total=%.3fs | range=%.3fs, doppler=%.3fs, aoa=%.3fs, detect=%.3fs, detN=%d",
            total_time,
            t_range,
            t_doppler,
            t_aoa,
            t_detection,
            num_det,
        )
    except Exception:
        pass

    return {
        "range_doppler": det_matrix.T,
        "range_azimuth": range_azimuth_map.T,
        "x_pos": xs,
        "y_pos": ys,
        "z_pos": zs,
        "velocities": velocities,
        "snrs": snrs,
        "azimuth_deg": azimuths_deg,
        "elevation_deg": elevations_deg,
        "cluster_labels": np.array([]),
        "tesseract": tesseract,
        "tesseract_az_grid_deg": azimuth_grid_deg,
        "tesseract_el_grid_deg": elevation_grid_deg,
    }
