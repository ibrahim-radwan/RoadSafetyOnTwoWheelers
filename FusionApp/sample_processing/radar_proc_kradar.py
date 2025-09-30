"""FFT-based 3D radar processing pipeline producing a Doppler-Range-Elevation-Azimuth tensor."""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple, cast

import numpy as np
from mmwave import dsp

from sample_processing.radar_proc import logger


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

    step_start = time.perf_counter()
    fft2d_sum = det_matrix.astype(np.int64)
    t3d = (tuning or {}).get("cfar_3d", {}) if isinstance(tuning, dict) else {}
    t3d_d = t3d.get("doppler", {})
    t3d_r = t3d.get("range", {})
    lb_d = float(t3d_d.get("l_bound", 1.5))
    gl_d = int(t3d_d.get("guard_len", 4))
    nl_d = int(t3d_d.get("noise_len", 16))
    lb_r = float(t3d_r.get("l_bound", 2.5))
    gl_r = int(t3d_r.get("guard_len", 4))
    nl_r = int(t3d_r.get("noise_len", 16))

    thrD, _ = np.apply_along_axis(
        dsp.ca_,
        axis=0,
        arr=fft2d_sum.T,
        l_bound=cast(int, lb_d),
        guard_len=gl_d,
        noise_len=nl_d,
    )
    thrR, noiseR = np.apply_along_axis(
        dsp.ca_,
        axis=0,
        arr=fft2d_sum,
        l_bound=cast(int, lb_r),
        guard_len=gl_r,
        noise_len=nl_r,
    )
    thrD = thrD.T
    full_mask = (det_matrix > thrD) & (det_matrix > thrR)
    det_peaks_indices = np.argwhere(full_mask)
    peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
    snr = peakVals - noiseR[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
    t_cfar = time.perf_counter() - step_start

    step_start = time.perf_counter()
    dtype_location = f"({adc_params.tx},)<f4"
    dtype_detObj2D = np.dtype(
        {
            "names": ["rangeIdx", "dopplerIdx", "peakVal", "location", "SNR"],
            "formats": ["<i4", "<i4", "<f4", dtype_location, "<f4"],
        }
    )
    detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
    detObj2DRaw["rangeIdx"] = det_peaks_indices[:, 0].squeeze()
    detObj2DRaw["dopplerIdx"] = det_peaks_indices[:, 1].squeeze()
    detObj2DRaw["peakVal"] = peakVals.flatten()
    detObj2DRaw["SNR"] = snr.flatten()
    t_struct = time.perf_counter() - step_start

    step_start = time.perf_counter()
    detObj2D = dsp.prune_to_peaks(
        detObj2DRaw, det_matrix, adc_params.chirps, reserve_neighbor=True
    )
    detObj2D = dsp.peak_grouping_along_doppler(detObj2D, det_matrix, adc_params.chirps)
    t_group = time.perf_counter() - step_start

    step_start = time.perf_counter()
    th3d = (tuning or {}).get("thresholds_3d", {}) if isinstance(tuning, dict) else {}
    SNRThresholds2 = np.array(
        th3d.get("snr_table", [[2, 10.5], [10, 7.5], [35, 5.0]]), dtype=np.float32
    )
    peakValThresholds2 = np.array(
        th3d.get("peak_table", [[4, 100], [1, 400], [500, 0]]), dtype=np.float32
    )
    detObj2D = dsp.range_based_pruning(
        detObj2D,
        SNRThresholds2,
        peakValThresholds2,
        adc_params.samples,
        0.5,
        adc_params.range_resolution,
    )
    t_prune = time.perf_counter() - step_start

    num_det = (
        len(detObj2D["rangeIdx"])
        if isinstance(detObj2D, (np.void, dict))
        else detObj2D.shape[0]
    )

    if num_det == 0:
        total_time = time.perf_counter() - function_start
        try:
            logger.info(
                "[KRadar] total=%.3fs | range=%.3fs, doppler=%.3fs, aoa=%.3fs, cfar=%.3fs, struct=%.3fs, group=%.3fs, prune=%.3fs",
                total_time,
                t_range,
                t_doppler,
                t_aoa,
                t_cfar,
                t_struct,
                t_group,
                t_prune,
            )
        except Exception:
            pass

        return {
            "range_doppler": det_matrix.T,
            "range_azimuth": None,
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

    xs = np.zeros(num_det, dtype=np.float32)
    ys = np.zeros(num_det, dtype=np.float32)
    zs = np.zeros(num_det, dtype=np.float32)
    azimuths_deg = np.zeros(num_det, dtype=np.float32)
    elevations_deg = np.zeros(num_det, dtype=np.float32)

    r_idx = detObj2D["rangeIdx"].astype(int)
    k_idx = detObj2D["dopplerIdx"].astype(int)
    el_bins = elevation_grid_deg.size
    az_bins = azimuth_grid_deg.size

    for i in range(num_det):
        r = int(r_idx[i])
        k = int(k_idx[i])
        slice_drea = tesseract[k, r]
        flat_idx = int(np.argmax(slice_drea))
        el_idx, az_idx = np.unravel_index(flat_idx, (el_bins, az_bins))
        az_deg = float(azimuth_grid_deg[az_idx])
        el_deg = float(elevation_grid_deg[el_idx])
        azimuths_deg[i] = az_deg
        elevations_deg[i] = el_deg

        azr = np.deg2rad(az_deg)
        elr = np.deg2rad(el_deg)
        ux = np.cos(elr) * np.sin(azr)
        uy = np.sin(elr)
        uz = np.cos(elr) * np.cos(azr)
        rng_m = adc_params.range_resolution * float(r)
        xs[i] = ux * rng_m
        ys[i] = uz * rng_m
        zs[i] = uy * rng_m

    velocities = detObj2D["dopplerIdx"] * adc_params.doppler_resolution
    snrs = detObj2D["SNR"]

    total_time = time.perf_counter() - function_start
    try:
        logger.info(
            "[KRadar] total=%.3fs | range=%.3fs, doppler=%.3fs, aoa=%.3fs, cfar=%.3fs, struct=%.3fs, group=%.3fs, prune=%.3fs, detN=%d",
            total_time,
            t_range,
            t_doppler,
            t_aoa,
            t_cfar,
            t_struct,
            t_group,
            t_prune,
            num_det,
        )
    except Exception:
        pass

    return {
        "range_doppler": det_matrix.T,
        "range_azimuth": None,
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
