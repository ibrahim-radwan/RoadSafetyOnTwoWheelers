import time
from typing import Optional, Tuple

import numpy as np
from mmwave import dsp

from sample_processing.radar_proc import logger  # reuse existing logger


def _virtual_array_positions_1843_in_wavelengths(
    num_tx: int, num_rx: int
) -> np.ndarray:
    """Return virtual antenna positions in units of wavelength for TI xWR1843/2243.

    Assumptions (commonly used for 3TXx4RX TDM-MIMO):
    - Azimuth row: 8 elements spaced at 0.5 λ along x-axis: x = 0..7 * 0.5, y = 0
    - Elevation row: 4 elements centered above, spaced at 0.5 λ along x: x = 2..5 * 0.5, y = 0.5
    - z = 0 for a planar array

    Returns positions as array of shape (M, 3) with columns (x, y, z) in wavelengths, where M = num_tx*num_rx.
    """
    assert (
        num_tx == 3 and num_rx == 4
    ), "This helper assumes 3TX x 4RX (12 VRx) geometry"
    # 12 virtual antennas
    xs_az = np.arange(8, dtype=np.float32) * 0.5
    ys_az = np.zeros_like(xs_az)
    xs_el = (np.arange(4, dtype=np.float32) + 2.0) * 0.5
    ys_el = np.full_like(xs_el, 0.5, dtype=np.float32)
    xs = np.concatenate([xs_az, xs_el])
    ys = np.concatenate([ys_az, ys_el])
    zs = np.zeros_like(xs)
    positions = np.stack([xs, ys, zs], axis=1)
    return positions.astype(np.float32)


def _gen_steering_matrix_2d(
    positions_wl: np.ndarray, az_grid: np.ndarray, el_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate 2D steering matrix for a planar array for azimuth/elevation grids.

    positions_wl: (M, 3) positions in wavelengths
    az_grid: (Na,) degrees in [-90, 90]
    el_grid: (Ne,) degrees (e.g., [-30, 30])

    Returns (A, az_grid, el_grid) where A has shape (M, Na*Ne) with column-major order over (az, el).
    """
    M = positions_wl.shape[0]
    az_r = np.deg2rad(az_grid).astype(np.float32)
    el_r = np.deg2rad(el_grid).astype(np.float32)
    # Unit propagation vectors u = [ux, uy, uz]
    # Choose mapping so that:
    # - x (lateral) spans negative/positive with az
    # - y (vertical) is >= 0 when elevation >= 0
    # u = [cos(el) * sin(az), sin(el), cos(el) * cos(az)]
    ux = np.cos(el_r)[:, None] * np.sin(az_r)[None, :]
    uy = np.sin(el_r)[:, None] * np.ones_like(az_r)[None, :]
    uz = np.cos(el_r)[:, None] * np.cos(az_r)[None, :]
    # Flatten grid in column-major order over az fast then el (or el then az). We choose az-major inside el loop.
    ux_flat = (ux.reshape(-1, 1, order="C").T).reshape(-1)
    uy_flat = (uy.reshape(-1, 1, order="C").T).reshape(-1)
    uz_flat = (uz.reshape(-1, 1, order="C").T).reshape(-1)

    # Dot with positions (in wavelengths): phase = 2π * (ux*x + uy*y + uz*z)
    kxyz = np.stack([ux_flat, uy_flat, uz_flat], axis=1).astype(
        np.float32
    )  # (Na*Ne, 3)
    phase = 2.0 * np.pi * (positions_wl.astype(np.float32) @ kxyz.T)  # (M, Na*Ne)
    A = np.exp(1j * phase).astype(np.complex64)
    return A, az_grid, el_grid


def _music_2d_peak(
    Rxx: np.ndarray,
    positions_wl: np.ndarray,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    num_sources: int = 1,
) -> Tuple[float, float]:
    """Compute 2D MUSIC peak (az, el) in degrees.

    Rxx: (M, M) covariance (Hermitian)
    positions_wl: (M, 3) positions in wavelengths
    Returns peak az, el.
    """
    # Eigen-decomposition (Hermitian)
    w, v = np.linalg.eigh(Rxx.astype(np.complex64))
    # Noise subspace = eigenvectors associated with smallest M - d eigenvalues
    M = Rxx.shape[0]
    d = max(1, min(num_sources, M - 1))
    En = v[:, : (M - d)]  # ascending order from eigh

    # Precompute steering matrix for grids
    A, azg, elg = _gen_steering_matrix_2d(positions_wl, az_grid, el_grid)
    # P(θ, φ) = 1 / ||En^H a||^2
    vprod = En.conj().T @ A  # ((M-d), Na*Ne)
    denom = np.sum(np.abs(vprod) ** 2, axis=0).real + 1e-12
    P = 1.0 / denom
    # Argmax
    idx = int(np.argmax(P))
    Na = az_grid.size
    Ne = el_grid.size
    # Our flattening followed el as outer? We built ux,uy with el as first dim then az; so flatten order matches (el, az)
    el_idx = idx // Na
    az_idx = idx % Na
    return float(az_grid[az_idx]), float(el_grid[el_idx])


def process_3D_radar_frame_music_2d(
    frame,
    adc_params,
    tuning: Optional[dict] = None,
    az_range: Tuple[int, int] = (-90, 90),
    az_step: int = 1,
    el_range: Tuple[int, int] = (-30, 30),
    el_step: int = 2,
    doppler_halfspan: int = 2,
):
    """3D processing using 2D MUSIC for azimuth/elevation per detection.

    Returns the same dict keys as process_3D_radar_frame for compatibility.
    """
    function_start = time.perf_counter()
    assert int(getattr(adc_params, "tx", 0)) == 3, "MUSIC-2D 3D pipeline requires 3 TX"

    # 1) Range FFT
    step_start = time.perf_counter()
    frame = frame.reshape(
        adc_params.chirps * adc_params.tx, adc_params.rx, adc_params.samples
    )
    radar_cube = dsp.range_processing(frame, window_type_1d=dsp.utils.Window.HAMMING)
    radar_cube = radar_cube.reshape(
        adc_params.chirps, adc_params.tx * adc_params.rx, adc_params.samples
    )
    # 2) Doppler FFT
    det_matrix, aoa_input = dsp.doppler_processing(
        radar_cube,
        num_tx_antennas=adc_params.tx,
        clutter_removal_enabled=True,
        interleaved=False,
        window_type_2d=dsp.utils.Window.HAMMING,
    )
    det_matrix = np.fft.fftshift(det_matrix, axes=1)
    aoa_input = np.fft.fftshift(aoa_input, axes=2)

    # 3) CFAR and detections (same as baseline)
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
        dsp.ca_, axis=0, arr=fft2d_sum.T, l_bound=lb_d, guard_len=gl_d, noise_len=nl_d
    )
    thrR, noiseR = np.apply_along_axis(
        dsp.ca_, axis=0, arr=fft2d_sum, l_bound=lb_r, guard_len=gl_r, noise_len=nl_r
    )
    thrD = thrD.T
    full_mask = (det_matrix > thrD) & (det_matrix > thrR)
    det_peaks_indices = np.argwhere(full_mask)
    peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
    snr = peakVals - noiseR[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

    dtype_location = "(" + str(adc_params.tx) + ",)<f4"
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

    detObj2D = dsp.prune_to_peaks(
        detObj2DRaw, det_matrix, adc_params.chirps, reserve_neighbor=True
    )
    detObj2D = dsp.peak_grouping_along_doppler(detObj2D, det_matrix, adc_params.chirps)

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

    # 4) 2D MUSIC per detection
    num_det = (
        len(detObj2D["rangeIdx"])
        if isinstance(detObj2D, np.void) or isinstance(detObj2D, dict)
        else detObj2D.shape[0]
    )
    if num_det == 0:
        return {
            "range_doppler": det_matrix,
            "range_azimuth": None,
            "x_pos": np.array([]),
            "y_pos": np.array([]),
            "z_pos": np.array([]),
            "velocities": np.array([]),
            "snrs": np.array([]),
            "cluster_labels": np.array([]),
        }

    positions_wl = _virtual_array_positions_1843_in_wavelengths(
        adc_params.tx, adc_params.rx
    )
    az_grid = np.arange(az_range[0], az_range[1] + 1, az_step, dtype=np.float32)
    el_grid = np.arange(el_range[0], el_range[1] + 1, el_step, dtype=np.float32)

    xs = np.zeros(num_det, dtype=np.float32)
    ys = np.zeros(num_det, dtype=np.float32)
    zs = np.zeros(num_det, dtype=np.float32)

    # Doppler snapshots for covariance
    half = max(0, int(doppler_halfspan))
    for i in range(num_det):
        r = int(detObj2D["rangeIdx"][i])
        k = int(detObj2D["dopplerIdx"][i])
        k0 = max(0, k - half)
        k1 = min(aoa_input.shape[2] - 1, k + half)
        X = aoa_input[r, :, k0 : k1 + 1].astype(np.complex64)  # (M, snapshots)
        if X.ndim == 1:
            X = X[:, None]
        # Covariance + forward-backward averaging
        Rxx = (X @ X.conj().T) / max(1, X.shape[1])
        # FB averaging
        J = np.fliplr(np.eye(Rxx.shape[0], dtype=np.float32))
        Rfb = 0.5 * (Rxx + J @ Rxx.conj() @ J)

        az_peak, el_peak = _music_2d_peak(
            Rfb, positions_wl, az_grid, el_grid, num_sources=1
        )
        # Convert to unit vector consistent with steering mapping above
        azr = np.deg2rad(az_peak)
        elr = np.deg2rad(el_peak)
        ux = np.cos(elr) * np.sin(azr)
        uy = np.sin(elr)
        uz = np.cos(elr) * np.cos(azr)
        rng_m = adc_params.range_resolution * r
        # World axes: forward=y, right=x, up=z
        xs[i] = ux * rng_m  # right/left
        ys[i] = uz * rng_m  # forward (always >=0 near boresight)
        zs[i] = uy * rng_m  # vertical up/down

    # RA heatmap is not essential here; set to None to simplify
    range_azimuth = None

    velocities = detObj2D["dopplerIdx"] * adc_params.doppler_resolution
    snrs = detObj2D["SNR"]

    total_time = time.perf_counter() - function_start
    logger.info(f"MUSIC-2D Runtime: {total_time:.4f}s for {num_det} detections")

    return {
        # Return RD transposed to match analyser RD SHM shape (chirps, samples) before UI rotation
        "range_doppler": det_matrix.T,
        "range_azimuth": range_azimuth,
        "x_pos": xs,
        "y_pos": ys,
        "z_pos": zs,
        "velocities": velocities,
        "snrs": snrs,
        "cluster_labels": np.array([]),
    }
