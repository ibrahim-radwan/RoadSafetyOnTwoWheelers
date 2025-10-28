"""FFT-based 3D radar processing pipeline producing a Doppler-Range-Elevation-Azimuth tensor."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from mmwave import dsp
from scipy import signal

from sample_processing.radar_proc import logger
from sample_processing.config import (
    CalibrationConfig,
    PolarDetectionConfig,
    RadarPipelineConfig,
    SpatialWindowConfig,
    ROIConfig,
)

# Constants for polar quantile detection (K-Radar approach)
# Adjusted for more detections: 0.985 = top 1.5% (~15k points)
# K-Radar defaults: 0.99 (1%), 0.999 (0.1%)
POLAR_POWER_QUANTILE = 0.985  # Keep top 1.5% of power values
CFAR_DEFAULT_PFA = 1e-4
CFAR_DEFAULT_GUARD = (1, 1, 2)  # range, elevation, azimuth guard cells
CFAR_DEFAULT_TRAIN = (4, 3, 6)  # range, elevation, azimuth training cells
CFAR_DEFAULT_OS_RANK = 0.75  # fraction (0-1) of sorted samples
CFAR_DEFAULT_OS_ALPHA = None  # Use CA-style alpha if None
CFAR_FA_RATE = 1e-4
CFAR_METHOD_ALIASES = {
    "ca_cfar": "ca",
    "cfar": "ca",
    "ca": "ca",
    "go": "go",
    "cago": "go",
    "so": "so",
    "caso": "so",
    "gos": "gos",
    "os": "os",
    "ordered": "os",
}
POLAR_METHOD_ALIASES = {**CFAR_METHOD_ALIASES, "quant": "quantile", "q": "quantile"}

# Constants for 3D CFAR (ZYX) - kept for backward compatibility
CFAR_GUARD_CELL_ZYX = [1, 1, 1]  # Z, Y, X guard cells (number of cells)
CFAR_TRAIN_CELL_ZYX = [1, 1, 1]  # Z, Y, X training cells (number of cells)

# TI AWR2243BOOST virtual array layout (3 TX x 4 RX)
AWR2243_AZ_TX_INDICES = (0, 2)
AWR2243_EL_TX_INDICES = (1,)
AWR2243_SPATIAL_GRID_X_WL = np.arange(8, dtype=np.float32) * 0.5
AWR2243_SPATIAL_GRID_Y_WL = np.array([0.0, 0.5], dtype=np.float32)
AWR2243_ELEVATION_COLUMN_OFFSET = (
    2  # Elevation row (TX1) begins at x = 1.0λ (column index 2)
)
AWR2243_SPATIAL_VALID_MASK = np.array(
    [
        [True, True, True, True, True, True, True, True],
        [False, False, True, True, True, True, False, False],
    ],
    dtype=bool,
)


def _resolve_dsp_window(name: str) -> dsp.utils.Window:
    """Map string aliases onto ``mmwave.dsp`` window enums."""

    lookup = {
        "hamming": dsp.utils.Window.HAMMING,
        "hann": dsp.utils.Window.HANNING,
        "hanning": dsp.utils.Window.HANNING,
        "blackman": dsp.utils.Window.BLACKMAN,
        "bartlett": dsp.utils.Window.BARTLETT,
    }
    key = str(name).strip().lower()
    if key not in lookup:
        raise ValueError(f"Unsupported FFT window '{name}'.")
    return cast(dsp.utils.Window, lookup[key])


def _resolve_spatial_window_config(
    window: Optional[Union[str, SpatialWindowConfig]],
) -> Optional[Union[str, Dict[str, str]]]:
    """Normalize spatial window configuration for angle FFTs."""

    if window is None:
        return None
    if isinstance(window, SpatialWindowConfig):
        resolved: Dict[str, str] = {}
        if window.azimuth:
            resolved["azimuth"] = str(window.azimuth)
        if window.elevation:
            resolved["elevation"] = str(window.elevation)
        return resolved
    return str(window)


def apply_radar_calibrations(
    frame: np.ndarray, adc_params, calibration: CalibrationConfig
) -> np.ndarray:
    """Apply DC offset removal and channel equalisation as configured."""

    calibrated = frame.astype(np.complex64, copy=True)

    if calibration.dc_offset.enabled:
        method = calibration.dc_offset.method.lower()
        if method in {"per_channel", "per_virtual", "per_vrx"}:
            offsets = np.mean(calibrated, axis=-1, keepdims=True)
            calibrated -= offsets
        else:
            offset = np.mean(calibrated, keepdims=True)
            calibrated -= offset

    if calibration.channel_equalization.enabled:
        method = calibration.channel_equalization.method.lower()
        if method in {"rms", "rms_normalized"}:
            # Equalize each virtual receiver to have similar RMS power.
            power = np.sqrt(
                np.mean(np.abs(calibrated) ** 2, axis=(0, 2), keepdims=True)
            )
            power[power == 0] = 1.0
            calibrated /= power
            # Restore global gain to preserve overall energy distribution.
            global_rms = np.sqrt(np.mean(np.abs(calibrated) ** 2))
            if global_rms > 0:
                calibrated *= global_rms

    return calibrated


def _compute_fft_drea(
    aoa_input: np.ndarray,
    adc_params,
    az_range: Tuple[int, int],
    el_range: Tuple[int, int],
    az_fft_size: int,
    el_fft_size: int,
    *,
    angle_mode: str = "1d_fft",
    spatial_window: Optional[Dict[str, str] | str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 4D Doppler-Range-Elevation-Azimuth tensor via spatial FFTs."""

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

    angle_mode_norm = str(angle_mode).strip().lower()

    if angle_mode_norm in {"1d", "1d_fft", "legacy"}:
        missing_tx = [
            idx
            for idx in AWR2243_AZ_TX_INDICES + AWR2243_EL_TX_INDICES
            if idx >= num_tx
        ]
        if missing_tx:
            raise ValueError(
                f"AWR2243 virtual array expects TX indices {AWR2243_AZ_TX_INDICES + AWR2243_EL_TX_INDICES} but got only {num_tx} TX"
            )

        az_virtual = aoa_cube[:, :, AWR2243_AZ_TX_INDICES, :].reshape(
            num_doppler_bins, num_range_bins, -1
        )
        el_virtual = aoa_cube[:, :, AWR2243_EL_TX_INDICES, :].reshape(
            num_doppler_bins, num_range_bins, -1
        )

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

    elif angle_mode_norm in {"2d", "2d_fft"}:
        if not (num_tx == 3 and num_rx == 4):
            raise ValueError(
                "2D FFT angle mode currently supports 3 TX x 4 RX (AWR2243BOOST) geometry"
            )

        ny = AWR2243_SPATIAL_GRID_Y_WL.size
        nx = AWR2243_SPATIAL_GRID_X_WL.size
        spatial_cube = np.zeros(
            (num_doppler_bins, num_range_bins, ny, nx), dtype=np.complex64
        )

        spatial_cube[:, :, 0, 0:num_rx] = aoa_cube[:, :, AWR2243_AZ_TX_INDICES[0], :]
        spatial_cube[:, :, 0, num_rx : num_rx * 2] = aoa_cube[
            :, :, AWR2243_AZ_TX_INDICES[1], :
        ]
        spatial_cube[
            :,
            :,
            1,
            AWR2243_ELEVATION_COLUMN_OFFSET : AWR2243_ELEVATION_COLUMN_OFFSET + num_rx,
        ] = aoa_cube[:, :, AWR2243_EL_TX_INDICES[0], :]

        spatial_cube *= AWR2243_SPATIAL_VALID_MASK.astype(np.complex64, copy=False)

        def _resolve_window(kind: str, length: int) -> np.ndarray:
            kind_norm = kind.strip().lower()
            if kind_norm in {"rect", "rectangular", "none", "box"}:
                return np.ones(length, dtype=np.float32)
            if kind_norm in {"hann", "hanning"}:
                return np.hanning(length).astype(np.float32)
            if kind_norm == "hamming":
                return np.hamming(length).astype(np.float32)
            if kind_norm == "blackman":
                return np.blackman(length).astype(np.float32)
            if kind_norm == "bartlett":
                return np.bartlett(length).astype(np.float32)
            raise ValueError(f"Unknown spatial window kind: {kind}")

        window_cfg = spatial_window
        if window_cfg is None:
            window_cfg = "hamming"

        if isinstance(window_cfg, dict):
            az_kind = str(window_cfg.get("az", window_cfg.get("azimuth", "hann")))
            el_kind = str(window_cfg.get("el", window_cfg.get("elevation", az_kind)))
        else:
            az_kind = str(window_cfg)
            el_kind = str(window_cfg)

        az_weights = _resolve_window(az_kind, nx)
        el_weights = _resolve_window(el_kind, ny)

        # Normalize spatial window energy so DREA power matches rectangular-window legacy scale
        az_scale = float(np.sum(az_weights) / max(nx, 1)) or 1.0
        el_scale = float(np.sum(el_weights) / max(ny, 1)) or 1.0
        az_weights = az_weights / az_scale
        el_weights = el_weights / el_scale

        spatial_weights = el_weights[:, None] * az_weights[None, :]
        valid_mask = AWR2243_SPATIAL_VALID_MASK.astype(np.float32, copy=False)
        valid_sum = float(np.sum(spatial_weights * valid_mask))
        desired_sum = float(np.count_nonzero(AWR2243_SPATIAL_VALID_MASK))
        if valid_sum > 0.0:
            spatial_weights *= desired_sum / valid_sum

        spatial_cube *= spatial_weights.astype(np.complex64, copy=False)

        az_bins = int(max(int(az_fft_size), nx))
        el_bins = int(max(int(el_fft_size), ny))

        fft2d = np.fft.fft2(spatial_cube, s=(el_bins, az_bins), axes=(-2, -1))
        fft2d_shift = np.fft.fftshift(fft2d, axes=(-2, -1))
        tesseract = np.abs(fft2d_shift) ** 2
        effective_virtual = float(np.count_nonzero(AWR2243_SPATIAL_VALID_MASK))
        nominal_virtual = float(
            max(
                len(AWR2243_AZ_TX_INDICES)
                * num_rx
                * len(AWR2243_EL_TX_INDICES)
                * num_rx,
                1,
            )
        )
        power_scale = (nominal_virtual / max(effective_virtual, 1.0)) ** 2
        tesseract *= power_scale
        tesseract = tesseract.astype(np.float32, copy=False)

    else:
        raise ValueError(
            f"Unknown angle_mode '{angle_mode}'. Use '1d_fft' or '2d_fft'."
        )

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


def _normalize_triplet(values: Sequence[Union[int, float]]) -> Tuple[int, int, int]:
    nums = [int(round(float(v))) for v in values]
    if not nums:
        nums = [0, 0, 0]
    while len(nums) < 3:
        nums.append(nums[-1])
    if len(nums) > 3:
        nums = nums[:3]
    return nums[0], nums[1], nums[2]


def _coerce_triplet(
    value: Optional[
        Union[
            int,
            float,
            Sequence[Union[int, float]],
            Iterable[Union[int, float]],
            Dict[str, Union[int, float]],
        ]
    ],
    default: Sequence[Union[int, float]],
) -> Tuple[int, int, int]:
    """Normalize configuration values (range, elevation, azimuth) to an int triplet."""

    default_triplet = _normalize_triplet(default)
    if value is None:
        return default_triplet
    if isinstance(value, (int, float)):
        val = int(round(float(value)))
        return (val, val, val)
    if isinstance(value, dict):
        value_dict = cast(Dict[str, Union[int, float]], value)
        return (
            int(round(float(value_dict.get("range", default_triplet[0])))),
            int(round(float(value_dict.get("elevation", default_triplet[1])))),
            int(round(float(value_dict.get("azimuth", default_triplet[2])))),
        )
    seq = cast(Sequence[Union[int, float]], value)
    return _normalize_triplet(seq)


def _cfar_alpha(num_train_cells: int, pfa: float) -> float:
    """Compute CFAR scaling factor for a desired false alarm rate."""

    num = max(int(num_train_cells), 1)
    pfa = float(np.clip(pfa, 1e-9, 1.0 - 1e-6))
    return num * (pfa ** (-1.0 / num) - 1.0)


def _polar_cfar_detect(
    rea_cube: np.ndarray,
    *,
    method: str,
    guard: Tuple[int, int, int],
    train: Tuple[int, int, int],
    pfa: float,
    os_rank: Optional[Union[int, float]] = None,
    os_alpha: Optional[float] = None,
) -> np.ndarray:
    """Run CFAR along the range dimension for every (elevation, azimuth) slice."""

    method_norm = CFAR_METHOD_ALIASES.get(method.lower(), method.lower())

    if method_norm not in {"ca", "go", "so", "gos", "os"}:
        raise ValueError(
            f"Unsupported CFAR method '{method}'. Use one of 'ca', 'go', 'so', 'gos', 'os'."
        )

    range_guard, _, _ = guard
    range_train, _, _ = train
    num_train_cells = max(2 * range_train, 1)
    alpha = _cfar_alpha(num_train_cells, pfa)

    detections = np.zeros_like(rea_cube, dtype=bool)

    # Current implementation applies CFAR along range only, replicating K-Radar tooling.
    # Angle guard/train parameters remain available for future multi-dimensional CFAR.

    for el_idx in range(rea_cube.shape[1]):
        for az_idx in range(rea_cube.shape[2]):
            cut = rea_cube[:, el_idx, az_idx]
            if method_norm == "ca":
                _, noise_floor = dsp.cfar.ca_(
                    cut,
                    guard_len=range_guard,
                    noise_len=range_train,
                    mode="constant",
                    l_bound=0,
                )
                threshold = noise_floor * alpha
            elif method_norm == "go":
                _, noise_floor = dsp.cfar.cago_(
                    cut,
                    guard_len=range_guard,
                    noise_len=range_train,
                    mode="constant",
                    l_bound=0,
                )
                threshold = noise_floor * alpha
            elif method_norm == "so":
                _, noise_floor = dsp.cfar.caso_(
                    cut,
                    guard_len=range_guard,
                    noise_len=range_train,
                    mode="constant",
                    l_bound=0,
                )
                threshold = noise_floor * alpha
            elif method_norm == "gos":
                _, noise_go = dsp.cfar.cago_(
                    cut,
                    guard_len=range_guard,
                    noise_len=range_train,
                    mode="constant",
                    l_bound=0,
                )
                _, noise_so = dsp.cfar.caso_(
                    cut,
                    guard_len=range_guard,
                    noise_len=range_train,
                    mode="constant",
                    l_bound=0,
                )
                threshold = np.minimum(noise_go, noise_so) * alpha
            else:  # OS-CFAR
                num_train_cells = max(2 * range_train, 1)
                if num_train_cells <= 0:
                    detections[:, el_idx, az_idx] = False
                    continue
                if os_rank is None:
                    rank_fraction = CFAR_DEFAULT_OS_RANK
                    k = int(
                        np.clip(
                            round(rank_fraction * (num_train_cells - 1)),
                            0,
                            num_train_cells - 1,
                        )
                    )
                elif isinstance(os_rank, float):
                    k = int(
                        np.clip(
                            round(os_rank * (num_train_cells - 1)),
                            0,
                            num_train_cells - 1,
                        )
                    )
                else:
                    k = int(np.clip(os_rank, 0, num_train_cells - 1))
                scale = (
                    os_alpha
                    if os_alpha is not None
                    else _cfar_alpha(num_train_cells, pfa)
                )
                threshold, _ = dsp.cfar.os_(
                    cut,
                    guard_len=range_guard,
                    noise_len=range_train,
                    k=k,
                    scale=scale,
                )

            detections[:, el_idx, az_idx] = cut > threshold

    return detections


def _apply_polar_detection(
    tesseract: np.ndarray,
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
    adc_params,
    detection_config: PolarDetectionConfig,
    roi_config: Optional[ROIConfig] = None,
) -> np.ndarray:
    """Apply configurable detection directly on the polar (REA) cube."""

    method = POLAR_METHOD_ALIASES.get(
        detection_config.method.lower(), detection_config.method.lower()
    )

    doppler_center = tesseract.shape[0] // 2
    doppler_zero_width = int(max(detection_config.doppler_guard_bins, 0))

    tesseract_moving = tesseract.copy()
    tesseract_moving[
        doppler_center - doppler_zero_width : doppler_center + doppler_zero_width + 1,
        :,
        :,
        :,
    ] = 0

    rea_cube = np.max(tesseract_moving, axis=0).astype(np.float32)
    doppler_idx_max = np.argmax(tesseract_moving, axis=0)

    if method in {"quantile", "quant", "q"}:
        quantile = float(detection_config.power_quantile)
        rea_flat = rea_cube.reshape(-1)

        if rea_flat.size == 0:
            return np.empty((0, 5), dtype=np.float32)

        candidate_idx = np.arange(rea_flat.size)
        if roi_config is not None and roi_config.enabled:
            range_res = float(getattr(adc_params, "range_resolution", 1.0))
            range_bins = np.arange(rea_cube.shape[0], dtype=np.float32) * range_res
            azimuth_rad = -np.deg2rad(az_grid_deg).astype(np.float32)
            elevation_rad = -np.deg2rad(el_grid_deg).astype(np.float32)

            r_grid = range_bins[:, None, None]
            cos_el = np.cos(elevation_rad)[None, :, None].astype(np.float32, copy=False)
            sin_el = np.sin(elevation_rad)[None, :, None].astype(np.float32, copy=False)
            cos_az = np.cos(azimuth_rad)[None, None, :].astype(np.float32, copy=False)
            sin_az = np.sin(azimuth_rad)[None, None, :].astype(np.float32, copy=False)

            x_grid = r_grid * cos_el * cos_az
            y_grid = r_grid * cos_el * sin_az
            z_grid = r_grid * sin_el

            roi_mask = (
                (x_grid >= float(roi_config.x[0]))
                & (x_grid <= float(roi_config.x[1]))
                & (y_grid >= float(roi_config.y[0]))
                & (y_grid <= float(roi_config.y[1]))
                & (z_grid >= float(roi_config.z[0]))
                & (z_grid <= float(roi_config.z[1]))
            )

            candidate_idx = np.flatnonzero(roi_mask.reshape(-1))

        if candidate_idx.size == 0:
            return np.empty((0, 5), dtype=np.float32)

        values = rea_flat[candidate_idx]
        if not np.any(values > 0.0):
            return np.empty((0, 5), dtype=np.float32)

        tail_fraction = float(1.0 - quantile)
        total_candidates = int(candidate_idx.size)
        if tail_fraction <= 0.0:
            k = total_candidates
        elif tail_fraction >= 1.0:
            k = 1
        else:
            k = int(np.ceil(tail_fraction * total_candidates))
            k = max(min(k, total_candidates), 1)

        logger.info(
            "[KRadar-Polar] quantile_select total=%d candidates=%d quantile=%.6f tail=%.6f topK=%d",
            rea_flat.size,
            total_candidates,
            quantile,
            tail_fraction,
            k,
        )

        tail_idx_local = np.argpartition(values, -k)[-k:]
        tail_idx = candidate_idx[tail_idx_local]

        detection_mask = np.zeros_like(rea_cube, dtype=bool)
        detection_mask.reshape(-1)[tail_idx] = True

        mask_count = int(np.count_nonzero(detection_mask))
        logger.info("[KRadar-Polar] quantile_mask_count=%d", mask_count)
    else:
        guard = _coerce_triplet(detection_config.guard, CFAR_DEFAULT_GUARD)
        train = _coerce_triplet(detection_config.train, CFAR_DEFAULT_TRAIN)
        pfa = float(detection_config.pfa)
        os_rank = detection_config.os_rank
        os_alpha = detection_config.os_alpha
        detection_mask = _polar_cfar_detect(
            rea_cube,
            method=method,
            guard=guard,
            train=train,
            pfa=pfa,
            os_rank=os_rank,
            os_alpha=os_alpha,
        )

    r_idx, e_idx, a_idx = np.where(detection_mask)

    logger.info(
        "[KRadar-Polar] quantile_indices count=%d",
        r_idx.size,
    )

    if len(r_idx) == 0:
        return np.empty((0, 5), dtype=np.float32)

    powers = rea_cube[r_idx, e_idx, a_idx].astype(np.float32)
    d_idx = doppler_idx_max[r_idx, e_idx, a_idx]

    range_res = float(getattr(adc_params, "range_resolution", 1.0))
    doppler_res = float(getattr(adc_params, "doppler_resolution", 0.1))

    r = r_idx.astype(np.float32) * range_res
    az = np.deg2rad(az_grid_deg[a_idx]).astype(np.float32)
    el = np.deg2rad(el_grid_deg[e_idx]).astype(np.float32)

    az = -az
    el = -el

    cos_el = np.cos(el)
    cos_az = np.cos(az)
    sin_az = np.sin(az)
    sin_el = np.sin(el)

    x = r * cos_el * cos_az
    y = r * cos_el * sin_az
    z = r * sin_el

    doppler_offsets = (d_idx - doppler_center).astype(np.float32)
    velocities = doppler_offsets * doppler_res

    point_cloud = np.column_stack([x, y, z, powers, velocities])

    logger.info(
        "[KRadar-Polar] quantile_point_cloud count=%d",
        point_cloud.shape[0],
    )

    return point_cloud


def _compute_zyx_cube(
    tesseract: np.ndarray,
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
    adc_params,
    doppler_aggregation: str = "mean",
    grid_size: Optional[float] = None,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    z_limits: Optional[Tuple[float, float]] = None,
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
        grid_size: Optional voxel edge length in meters when constructing the Cartesian grid.
        x_limits: Inclusive [min, max] bounds for the lateral axis in meters.
        y_limits: Inclusive [min, max] bounds for the forward axis in meters.
        z_limits: Inclusive [min, max] bounds for the vertical axis in meters.

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
    if grid_size is not None and grid_size > 0.0:
        dr = float(grid_size)
    else:
        dr = range_res

    if x_limits is not None:
        x_min, x_max = float(x_limits[0]), float(x_limits[1])
    else:
        x_extent = float(max_range)
        x_min, x_max = -x_extent, x_extent

    if y_limits is not None:
        y_min, y_max = float(y_limits[0]), float(y_limits[1])
    else:
        y_min, y_max = 0.0, float(max_range)

    if z_limits is not None:
        z_min, z_max = float(z_limits[0]), float(z_limits[1])
    else:
        z_extent = 0.3 * float(max_range)
        z_min, z_max = -z_extent, z_extent

    arr_x = np.arange(x_min, x_max + 1e-9, dr, dtype=np.float32)  # lateral
    arr_y = np.arange(y_min, y_max + 1e-9, dr, dtype=np.float32)  # forward
    arr_z = np.arange(z_min, z_max + 1e-9, dr, dtype=np.float32)  # vertical

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


def _generate_cartesian_quantile_point_cloud(
    arr_zyx: np.ndarray,
    arr_z: np.ndarray,
    arr_y: np.ndarray,
    arr_x: np.ndarray,
    *,
    quantile_rate: float,
    normalization_value: Optional[float],
    roi: Optional[ROIConfig],
    add_half_grid_offset: bool,
    offset_type: str,
) -> np.ndarray:
    """Select strongest Cartesian voxels via quantile, mimicking KRadar sparse cube."""

    if arr_zyx.size == 0:
        return np.empty((0, 5), dtype=np.float32)

    valid_mask = arr_zyx > 0.0

    if roi is not None and roi.enabled:
        x_mask = (arr_x >= float(roi.x[0])) & (arr_x <= float(roi.x[1]))
        y_mask = (arr_y >= float(roi.y[0])) & (arr_y <= float(roi.y[1]))
        z_mask = (arr_z >= float(roi.z[0])) & (arr_z <= float(roi.z[1]))
        roi_mask = z_mask[:, None, None]
        roi_mask = roi_mask & y_mask[None, :, None]
        roi_mask = roi_mask & x_mask[None, None, :]
        valid_mask &= roi_mask

    if not np.any(valid_mask):
        return np.empty((0, 5), dtype=np.float32)

    values = arr_zyx[valid_mask]
    total_candidates = values.size
    tail_rate = float(np.clip(quantile_rate, 0.0, 1.0))

    if tail_rate <= 0.0:
        top_k = 1
    elif tail_rate >= 1.0:
        top_k = total_candidates
    else:
        top_k = int(np.ceil(tail_rate * total_candidates))
        top_k = max(min(top_k, total_candidates), 1)

    threshold = np.partition(values, total_candidates - top_k)[total_candidates - top_k]
    detection_mask = valid_mask & (arr_zyx >= threshold)

    z_idx, y_idx, x_idx = np.where(detection_mask)
    if z_idx.size == 0:
        return np.empty((0, 5), dtype=np.float32)

    powers = arr_zyx[z_idx, y_idx, x_idx].astype(np.float32, copy=False)
    if normalization_value is not None and normalization_value > 0.0:
        powers = powers / float(normalization_value)

    x_coords = arr_x[x_idx].astype(np.float32, copy=False)
    y_coords = arr_y[y_idx].astype(np.float32, copy=False)
    z_coords = arr_z[z_idx].astype(np.float32, copy=False)

    if add_half_grid_offset:
        x_step = float(arr_x[1] - arr_x[0]) if arr_x.size > 1 else 0.0
        y_step = float(arr_y[1] - arr_y[0]) if arr_y.size > 1 else 0.0
        z_step = float(arr_z[1] - arr_z[0]) if arr_z.size > 1 else 0.0
        offset_sign = -1.0 if offset_type.strip().lower() == "minus" else 1.0
        half_x = 0.5 * x_step
        half_y = 0.5 * y_step
        half_z = 0.5 * z_step
        if half_x:
            x_coords = x_coords + offset_sign * half_x
        if half_y:
            y_coords = y_coords + offset_sign * half_y
        if half_z:
            z_coords = z_coords + offset_sign * half_z

    logger.info(
        "[KRadar-Cartesian] quantile_select total=%d candidates=%d pick_rate=%.4f topK=%d",
        arr_zyx.size,
        total_candidates,
        tail_rate,
        top_k,
    )
    logger.info(
        "[KRadar-Cartesian] quantile_mask_count=%d",
        int(z_idx.size),
    )

    velocities = np.zeros_like(powers, dtype=np.float32)
    point_cloud = np.column_stack([x_coords, y_coords, z_coords, powers, velocities])

    logger.info(
        "[KRadar-Cartesian] quantile_point_cloud count=%d",
        point_cloud.shape[0],
    )

    return point_cloud.astype(np.float32, copy=False)


def process_3d_radar_frame_kradar(
    frame: np.ndarray,
    adc_params,
    config: Optional[RadarPipelineConfig] = None,
) -> Dict[str, Any]:
    """Process a raw radar frame into sparse detections using configured pipeline."""

    cfg = config or RadarPipelineConfig()
    function_start = time.perf_counter()

    if int(getattr(adc_params, "tx", 0)) != 3:
        raise ValueError("KRadar pipeline requires 3 TX antennas")

    frame = frame.reshape(
        adc_params.chirps * adc_params.tx, adc_params.rx, adc_params.samples
    )

    step_start = time.perf_counter()
    calibrated = apply_radar_calibrations(frame, adc_params, cfg.calibration)

    range_window = _resolve_dsp_window(cfg.windows.range)
    radar_cube = dsp.range_processing(calibrated, window_type_1d=range_window)
    radar_cube = radar_cube.reshape(
        adc_params.chirps, adc_params.tx * adc_params.rx, adc_params.samples
    )
    t_range = time.perf_counter() - step_start

    step_start = time.perf_counter()
    doppler_window = _resolve_dsp_window(cfg.windows.doppler)
    clutter_enabled = bool(cfg.calibration.clutter_removal.enabled)

    det_matrix, aoa_input = dsp.doppler_processing(
        radar_cube,
        num_tx_antennas=adc_params.tx,
        clutter_removal_enabled=clutter_enabled,
        interleaved=False,
        window_type_2d=doppler_window,
    )

    if clutter_enabled:
        logger.info("[Calibration] Clutter removal applied (static target suppression)")

    det_matrix = np.fft.fftshift(det_matrix, axes=1)
    aoa_input = np.fft.fftshift(aoa_input, axes=2)
    t_doppler = time.perf_counter() - step_start

    angle_cfg = cfg.angle
    az_range_cfg = (
        float(angle_cfg.azimuth_range[0]),
        float(angle_cfg.azimuth_range[1]),
    )
    el_range_cfg = (
        float(angle_cfg.elevation_range[0]),
        float(angle_cfg.elevation_range[1]),
    )
    az_span = float(az_range_cfg[1] - az_range_cfg[0])
    el_span = float(el_range_cfg[1] - el_range_cfg[0])
    az_bins_target = max(2, int(np.ceil(abs(az_span))) + 1)
    el_bins_target = max(2, int(np.ceil(abs(el_span))) + 1)
    resolved_az_fft_size = max(az_bins_target, int(angle_cfg.azimuth_fft_size))
    resolved_el_fft_size = max(el_bins_target, int(angle_cfg.elevation_fft_size))
    spatial_window = _resolve_spatial_window_config(angle_cfg.spatial_window)

    az_range = (
        int(np.floor(az_range_cfg[0])),
        int(np.ceil(az_range_cfg[1])),
    )
    el_range = (
        int(np.floor(el_range_cfg[0])),
        int(np.ceil(el_range_cfg[1])),
    )

    step_start = time.perf_counter()
    tesseract, azimuth_grid_deg, elevation_grid_deg = _compute_fft_drea(
        aoa_input,
        adc_params,
        az_range,
        el_range,
        resolved_az_fft_size,
        resolved_el_fft_size,
        angle_mode=angle_cfg.mode,
        spatial_window=spatial_window,
    )
    t_aoa = time.perf_counter() - step_start

    doppler_center = tesseract.shape[0] // 2
    doppler_guard = int(max(cfg.polar_detection.doppler_guard_bins, 0))
    tesseract_for_ra = tesseract.copy()
    if doppler_guard >= 0:
        start = max(doppler_center - doppler_guard, 0)
        stop = min(doppler_center + doppler_guard + 1, tesseract.shape[0])
        tesseract_for_ra[start:stop, :, :, :] = 0

    range_azimuth_map = np.max(tesseract_for_ra, axis=0)
    range_azimuth_map = np.max(range_azimuth_map, axis=1)
    range_azimuth_map = 20.0 * np.log10(np.abs(range_azimuth_map) + 1e-10)

    pc_cfg = cfg.point_cloud
    generation_mode = pc_cfg.generation_mode.strip().lower()
    cartesian_aliases = {"cartesian_quantile", "zyx_quantile", "cartesian"}
    polar_aliases = {"polar_quantile", "polar", "polar_cfar", "polar_cfar_quantile"}

    arr_zyx_cube: Optional[np.ndarray] = None
    arr_z_grid: Optional[np.ndarray] = None
    arr_y_grid: Optional[np.ndarray] = None
    arr_x_grid: Optional[np.ndarray] = None

    detection_method = generation_mode
    log_tag = "[KRadar-Polar]"

    if generation_mode in polar_aliases:
        step_start = time.perf_counter()
        point_cloud = _apply_polar_detection(
            tesseract,
            azimuth_grid_deg,
            elevation_grid_deg,
            adc_params,
            detection_config=cfg.polar_detection,
            roi_config=pc_cfg.roi,
        )
        detection_method = POLAR_METHOD_ALIASES.get(
            cfg.polar_detection.method.lower(), cfg.polar_detection.method.lower()
        )
        t_detection = time.perf_counter() - step_start
    elif generation_mode in cartesian_aliases:
        log_tag = "[KRadar-Cartesian]"
        cart_cfg = pc_cfg.cartesian_quantile

        step_start = time.perf_counter()
        arr_zyx_cube, arr_z_grid, arr_y_grid, arr_x_grid = _compute_zyx_cube(
            tesseract,
            azimuth_grid_deg,
            elevation_grid_deg,
            adc_params,
            doppler_aggregation=cart_cfg.doppler_aggregation,
            grid_size=cart_cfg.grid_size,
            x_limits=cart_cfg.x_limits,
            y_limits=cart_cfg.y_limits,
            z_limits=cart_cfg.z_limits,
        )
        t_zyx = time.perf_counter() - step_start

        step_start = time.perf_counter()
        roi_for_cart = pc_cfg.roi if pc_cfg.roi.enabled else None
        point_cloud = _generate_cartesian_quantile_point_cloud(
            arr_zyx_cube,
            arr_z_grid,
            arr_y_grid,
            arr_x_grid,
            quantile_rate=cart_cfg.quantile_rate,
            normalization_value=cart_cfg.normalization_value,
            roi=roi_for_cart,
            add_half_grid_offset=cart_cfg.add_half_grid_offset,
            offset_type=cart_cfg.offset_type,
        )
        t_detection = t_zyx + (time.perf_counter() - step_start)
        detection_method = "cartesian_quantile"
    else:
        raise ValueError(
            f"Unsupported point_cloud.generation_mode '{generation_mode}'. Use 'polar_quantile' or 'cartesian_quantile'."
        )

    pn_cfg = pc_cfg.power_normalization
    divide_by = float(getattr(pn_cfg, "divide_by", 1.0))
    range_based_dividers = getattr(pn_cfg, "range_based_divide_by", None)
    range_divider_segments: Optional[List[Tuple[float, float, float]]] = (
        list(range_based_dividers) if range_based_dividers else None
    )
    range_default_divider = getattr(pn_cfg, "range_based_default_divide_by", None)
    fallback_divider = float(divide_by)
    if range_default_divider is not None and range_default_divider > 0.0:
        fallback_divider = float(range_default_divider)
    elif fallback_divider <= 0.0:
        fallback_divider = 1.0

    range_divider_used: Optional[np.ndarray] = None
    range_bin_edges: Optional[np.ndarray] = None
    range_bin_indices: Optional[np.ndarray] = None
    effective_clip_max: Optional[float] = None

    if point_cloud.shape[0] > 0 and pc_cfg.roi.enabled:
        roi = pc_cfg.roi
        roi_mask = (
            (point_cloud[:, 0] >= roi.x[0])
            & (point_cloud[:, 0] <= roi.x[1])
            & (point_cloud[:, 1] >= roi.y[0])
            & (point_cloud[:, 1] <= roi.y[1])
            & (point_cloud[:, 2] >= roi.z[0])
            & (point_cloud[:, 2] <= roi.z[1])
        )
        point_cloud = point_cloud[roi_mask]

    raw_max_observed = 0.0

    if point_cloud.shape[0] > 0:
        if pc_cfg.range_scale != 1.0:
            point_cloud[:, :3] *= float(pc_cfg.range_scale)

        logger.info(
            "%s quantile_point_cloud pre_norm=%d",
            log_tag,
            point_cloud.shape[0],
        )

        raw_power = point_cloud[:, 3].astype(np.float32, copy=False)

        raw_max = float(np.max(raw_power)) if raw_power.size > 0 else 0.0
        raw_max_observed = raw_max
        clip_input_max_cfg = pn_cfg.clip_input_max
        power_to_scale = raw_power

        if clip_input_max_cfg is not None:
            clip_input_max = float(clip_input_max_cfg)
            if clip_input_max > 0.0:
                effective_clip_max = clip_input_max
                power_to_scale = np.clip(raw_power, 0.0, clip_input_max)

        normalized: np.ndarray = power_to_scale.copy()
        if range_divider_segments:
            roi_x_min, roi_x_max = float(pc_cfg.roi.x[0]), float(pc_cfg.roi.x[1])
            if pc_cfg.range_scale != 1.0:
                scale = float(pc_cfg.range_scale)
                roi_x_min *= scale
                roi_x_max *= scale

            if roi_x_max <= roi_x_min:
                logger.warning(
                    "%s Invalid ROI X bounds (min %.3f >= max %.3f); falling back to default divider",
                    log_tag,
                    roi_x_min,
                    roi_x_max,
                )
            else:
                segments_sorted = sorted(
                    range_divider_segments,
                    key=lambda seg: float(seg[0]),
                )
                num_segments = len(segments_sorted)
                values = np.array(
                    [float(seg[2]) for seg in segments_sorted], dtype=np.float32
                )
                starts = np.array(
                    [float(seg[0]) for seg in segments_sorted], dtype=np.float32
                )
                ends = np.array(
                    [float(seg[1]) for seg in segments_sorted], dtype=np.float32
                )

                dividers_per_point = np.full(
                    power_to_scale.shape, fallback_divider, dtype=np.float32
                )
                bin_indices = np.full(power_to_scale.shape, -1, dtype=np.int32)
                x_coords = point_cloud[:, 0]

                for seg_idx, (start, end, value) in enumerate(segments_sorted):
                    value_f = float(value)
                    if value_f <= 0.0:
                        logger.warning(
                            "%s Range divider %.3e for segment %d is non-positive; skipping",
                            log_tag,
                            value_f,
                            seg_idx,
                        )
                        continue

                    start_f = float(start)
                    end_f = float(end)
                    if seg_idx == num_segments - 1:
                        mask = (x_coords >= start_f) & (x_coords <= end_f)
                    else:
                        mask = (x_coords >= start_f) & (x_coords < end_f)
                    if not np.any(mask):
                        continue
                    dividers_per_point[mask] = value_f
                    bin_indices[mask] = seg_idx

                outside_mask = bin_indices < 0
                if np.any(outside_mask):
                    logger.warning(
                        "%s Range dividers left %d detections outside specified ranges; applying fallback divider=%.3e",
                        log_tag,
                        int(np.count_nonzero(outside_mask)),
                        fallback_divider,
                    )

                normalized = power_to_scale / np.where(
                    dividers_per_point > 0.0, dividers_per_point, fallback_divider
                )
                range_divider_used = values
                range_bin_edges = np.column_stack((starts, ends))
                range_bin_indices = bin_indices

        if range_divider_used is None:
            normalized = power_to_scale / fallback_divider

        point_cloud[:, 3] = normalized.astype(np.float32, copy=False)

        logger.debug(
            "%s Power normalization raw_max=%.3e clip=%.3e divide_strategy=%s",
            log_tag,
            raw_max,
            effective_clip_max if effective_clip_max is not None else 0.0,
            (
                "range"
                if range_divider_used is not None
                else f"scalar({fallback_divider:.3e})"
            ),
        )
        if (
            range_divider_used is not None
            and range_bin_edges is not None
            and range_bin_indices is not None
            and range_divider_used.size > 0
        ):
            raw_max_per_bin = np.zeros(range_divider_used.size, dtype=np.float32)
            norm_max_per_bin = np.zeros(range_divider_used.size, dtype=np.float32)
            valid_bin_mask = range_bin_indices >= 0
            if np.any(valid_bin_mask):
                np.maximum.at(
                    raw_max_per_bin,
                    range_bin_indices[valid_bin_mask],
                    power_to_scale[valid_bin_mask],
                )
                np.maximum.at(
                    norm_max_per_bin,
                    range_bin_indices[valid_bin_mask],
                    normalized[valid_bin_mask],
                )
            for seg_idx in range(range_divider_used.size):
                start_edge, end_edge = range_bin_edges[seg_idx]
                logger.info(
                    "%s Range bin %d x=[%.2f, %.2f] divider=%.3e raw_max=%.3e norm_max=%.3e",
                    log_tag,
                    seg_idx,
                    float(start_edge),
                    float(end_edge),
                    float(range_divider_used[seg_idx]),
                    float(raw_max_per_bin[seg_idx]),
                    float(norm_max_per_bin[seg_idx]),
                )

    num_det = point_cloud.shape[0]

    logger.info("%s quantile_point_cloud final=%d", log_tag, num_det)

    power_norm_info: Dict[str, Any] = {
        "divide_by": float(divide_by),
        "clip_input_max": (
            float(effective_clip_max) if effective_clip_max is not None else 0.0
        ),
        "raw_max_observed": raw_max_observed,
    }
    if range_default_divider is not None:
        power_norm_info["range_based_default_divide_by"] = float(range_default_divider)
    if range_divider_segments:
        power_norm_info["range_based_divide_by"] = [
            {
                "range": [float(start), float(end)],
                "divide_by": float(value),
            }
            for start, end, value in range_divider_segments
        ]

    try:
        if num_det > 0:
            logger.info(
                "%s-DEBUG method=%s Detections: N=%d, "
                "x: [%.2f, %.2f], y: [%.2f, %.2f], z: [%.2f, %.2f], "
                "power: min=%.2e, max=%.2e, mean=%.2e, "
                "velocity: [%.2f, %.2f] m/s",
                log_tag,
                detection_method,
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
    except Exception as exc:
        logger.warning(f"[KRadar-Polar-DEBUG] Failed to log detection stats: {exc}")

    if num_det == 0:
        total_time = time.perf_counter() - function_start
        try:
            logger.info(
                "%s method=%s total=%.3fs | range=%.3fs, doppler=%.3fs, aoa=%.3fs, detect=%.3fs, detN=0",
                log_tag,
                detection_method,
                total_time,
                t_range,
                t_doppler,
                t_aoa,
                t_detection,
            )
        except Exception:
            pass

        empty = np.array([])
        return {
            "range_doppler": det_matrix.T,
            "range_azimuth": range_azimuth_map.T,
            "x_pos": empty,
            "y_pos": empty,
            "z_pos": empty,
            "velocities": empty,
            "snrs": empty,
            "azimuth_deg": empty,
            "elevation_deg": empty,
            "cluster_labels": empty,
            "tesseract": tesseract,
            "tesseract_az_grid_deg": azimuth_grid_deg,
            "tesseract_el_grid_deg": elevation_grid_deg,
            "power_normalization": power_norm_info,
            "arr_zyx": arr_zyx_cube,
            "arr_z": arr_z_grid,
            "arr_y": arr_y_grid,
            "arr_x": arr_x_grid,
        }

    xs = point_cloud[:, 0].astype(np.float32, copy=False)
    ys = point_cloud[:, 1].astype(np.float32, copy=False)
    zs = point_cloud[:, 2].astype(np.float32, copy=False)
    powers = point_cloud[:, 3].astype(np.float32, copy=False)
    velocities = point_cloud[:, 4].astype(np.float32, copy=False)

    horiz = np.sqrt(xs**2 + ys**2)
    azimuths_deg = np.rad2deg(np.arctan2(xs, ys)).astype(np.float32)
    elevations_deg = np.rad2deg(np.arctan2(zs, horiz)).astype(np.float32)
    snrs = powers.astype(np.float32)

    total_time = time.perf_counter() - function_start
    try:
        logger.info(
            "%s method=%s total=%.3fs | range=%.3fs, doppler=%.3fs, aoa=%.3fs, detect=%.3fs, detN=%d",
            log_tag,
            detection_method,
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
        "power_normalization": power_norm_info,
        "arr_zyx": arr_zyx_cube,
        "arr_z": arr_z_grid,
        "arr_y": arr_y_grid,
        "arr_x": arr_x_grid,
    }
