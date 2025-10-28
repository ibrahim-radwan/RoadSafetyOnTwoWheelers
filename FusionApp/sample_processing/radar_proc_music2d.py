"""2D-MUSIC-based 3D radar processing pipeline.

Key points:
- World frame: radar at (0,0,0) facing +y; +x right, +z up. y is forward.
- RD orientation: we return `det_matrix.T` to match analyser expectations.
- AoA: 2D MUSIC over azimuth/elevation using TI 1843/2243 virtual array layout.
- Optimization: coarse-to-fine peak search with per-frame precomputed coarse/fine steering caches.
"""

import time
from typing import Optional, Tuple

import numpy as np
from mmwave import dsp

from sample_processing.radar_proc import logger  # reuse existing logger


_POSITIONS_CACHE = {}
_STEERING_CACHE = {}


def _get_positions_wl_cached(num_tx: int, num_rx: int) -> np.ndarray:
    """Return cached virtual array positions in wavelength units for given (tx, rx)."""
    key = (int(num_tx), int(num_rx))
    pos = _POSITIONS_CACHE.get(key)
    if pos is None:
        pos = _virtual_array_positions_1843_in_wavelengths(int(num_tx), int(num_rx))
        _POSITIONS_CACHE[key] = pos
    return pos


def _get_or_build_steering_cache(
    num_tx: int,
    num_rx: int,
    az_range: Tuple[int, int],
    el_range: Tuple[int, int],
    fine_az_step: int,
    fine_el_step: int,
    coarse_az_step: int,
    coarse_el_step: int,
    fine_half_win_az: int,
    fine_half_win_el: int,
):
    """Get or build per-parameter steering caches to reuse across frames.

    Returns: (positions_wl, az_grid, el_grid, coarse_az_grid, coarse_el_grid, A_coarse, fine_cache)
    """
    key = (
        int(num_tx),
        int(num_rx),
        int(az_range[0]),
        int(az_range[1]),
        int(el_range[0]),
        int(el_range[1]),
        int(max(1, int(fine_az_step))),
        int(max(1, int(fine_el_step))),
        int(max(1, int(coarse_az_step))),
        int(max(1, int(coarse_el_step))),
        int(max(1, int(fine_half_win_az))),
        int(max(1, int(fine_half_win_el))),
    )
    entry = _STEERING_CACHE.get(key)
    if entry is not None:
        return entry

    positions_wl = _get_positions_wl_cached(num_tx, num_rx)
    az_grid = np.arange(
        az_range[0], az_range[1] + 1, max(1, int(fine_az_step)), dtype=np.float32
    )
    el_grid = np.arange(
        el_range[0], el_range[1] + 1, max(1, int(fine_el_step)), dtype=np.float32
    )
    coarse_az_grid = np.arange(
        az_range[0], az_range[1] + 1, max(1, int(coarse_az_step)), dtype=np.float32
    )
    coarse_el_grid = np.arange(
        el_range[0], el_range[1] + 1, max(1, int(coarse_el_step)), dtype=np.float32
    )
    A_coarse, fine_cache = _build_coarse_fine_steering_cache(
        positions_wl,
        coarse_az_grid,
        coarse_el_grid,
        (int(az_grid.min()), int(az_grid.max())),
        (int(el_grid.min()), int(el_grid.max())),
        fine_az_step=max(1, int(fine_az_step)),
        fine_el_step=max(1, int(fine_el_step)),
        fine_half_win_az=int(fine_half_win_az),
        fine_half_win_el=int(fine_half_win_el),
    )
    entry = (
        positions_wl,
        az_grid,
        el_grid,
        coarse_az_grid,
        coarse_el_grid,
        A_coarse,
        fine_cache,
    )
    _STEERING_CACHE[key] = entry
    return entry


def _virtual_array_positions_1843_in_wavelengths(
    num_tx: int, num_rx: int
) -> np.ndarray:
    """Virtual antenna positions (wavelength units) for TI xWR1843/2243 3TXx4RX.

    Layout:
    - Azimuth: 8 elems at 0.5λ spacing along x, y=0.
    - Elevation: 4 elems at 0.5λ along x, y=0.5.
    - Planar array: z=0. Shape (M,3), M=num_tx*num_rx, columns (x,y,z).
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


def _prepare_tesseract_assets(
    adc_params,
    az_range: Tuple[int, int],
    el_range: Tuple[int, int],
    fine_az_step: int,
    fine_el_step: int,
):
    """Prepare assets for full 4D pseudospectrum computation.

    Returns (positions_wl, az_grid, el_grid, A_full) or (None, None, None, None) on failure.
    """
    try:
        positions_wl = _get_positions_wl_cached(adc_params.tx, adc_params.rx)
        az_grid = np.arange(
            az_range[0], az_range[1] + 1, max(1, int(fine_az_step)), dtype=np.float32
        )
        el_grid = np.arange(
            el_range[0], el_range[1] + 1, max(1, int(fine_el_step)), dtype=np.float32
        )
        A_full, _, _ = _gen_steering_matrix_2d(positions_wl, az_grid, el_grid)
        return positions_wl, az_grid, el_grid, A_full
    except Exception:
        return None, None, None, None


def _compute_tesseract(
    aoa_input: np.ndarray,
    positions_wl: np.ndarray,
    A_full: np.ndarray,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    doppler_halfspan: int,
    music_diag_load: float,
):
    """Compute full 4D MUSIC pseudospectrum tensor (D, R, E, A).

    Returns tensor as float32 or None on error.
    """
    try:
        Nr = aoa_input.shape[0]
        Nd = aoa_input.shape[2]
        Na = int(az_grid.size)
        Ne = int(el_grid.size)
        A_full = A_full.astype(np.complex64, copy=False)
        M = positions_wl.shape[0]
        half = max(0, int(doppler_halfspan))
        J_fb_full = np.fliplr(np.eye(M, dtype=np.float32))
        tesseract = np.empty((Nd, Nr, Ne, Na), dtype=np.float32)
        for r in range(Nr):
            for k in range(Nd):
                k0 = max(0, k - half)
                k1 = min(Nd - 1, k + half)
                X = aoa_input[r, :, k0 : k1 + 1].astype(np.complex64)
                if X.ndim == 1:
                    X = X[:, None]
                Rxx = (X @ X.conj().T) / max(1, X.shape[1])
                Rfb = 0.5 * (Rxx + J_fb_full @ Rxx.conj() @ J_fb_full)
                if music_diag_load and float(music_diag_load) > 0.0:
                    tr = float(np.trace(Rfb).real)
                    Rfb = Rfb + np.eye(M, dtype=Rfb.dtype) * (
                        float(music_diag_load) * tr / M
                    )
                # Noise subspace (d=1)
                _, v = np.linalg.eigh(Rfb)
                En = v[:, : (M - 1)]
                vprod = En.conj().T @ A_full
                denom = np.sum(np.abs(vprod) ** 2, axis=0).real + 1e-12
                P = (1.0 / denom).astype(np.float32)
                tesseract[k, r] = P.reshape((Ne, Na), order="C")
        return tesseract
    except Exception:
        return None


def _gen_steering_matrix_2d(
    positions_wl: np.ndarray, az_grid: np.ndarray, el_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D steering matrix for azimuth/elevation grids.

    Mapping: u=[ux,uy,uz]=[cos(el)sin(az), sin(el), cos(el)cos(az)].
    Flattening: columns enumerate (el,az) with Na fastest; A shape (M, Na*Ne).
    """
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
    """Compute 2D MUSIC peak (az, el) from Hermitian covariance Rxx."""
    # Eigen-decomposition (Hermitian)
    _, v = np.linalg.eigh(Rxx.astype(np.complex64))
    # Noise subspace = eigenvectors associated with smallest M - d eigenvalues
    M = Rxx.shape[0]
    d = max(1, min(num_sources, M - 1))
    En = v[:, : (M - d)]  # ascending order from eigh

    # Precompute steering matrix for grids
    A, _, _ = _gen_steering_matrix_2d(positions_wl, az_grid, el_grid)
    # P(θ, φ) = 1 / ||En^H a||^2
    vprod = En.conj().T @ A  # ((M-d), Na*Ne)
    denom = np.sum(np.abs(vprod) ** 2, axis=0).real + 1e-12
    P = 1.0 / denom
    # Argmax
    idx = int(np.argmax(P))
    Na = az_grid.size
    # Our flattening followed el as outer? We built ux,uy with el as first dim then az; so flatten order matches (el, az)
    el_idx = idx // Na
    az_idx = idx % Na
    return float(az_grid[az_idx]), float(el_grid[el_idx])


def _music_peak_with_A(
    En: np.ndarray, A: np.ndarray, az_grid: np.ndarray, el_grid: np.ndarray
) -> Tuple[float, float]:
    """MUSIC peak using noise subspace En and precomputed A."""
    vprod = En.conj().T @ A  # ((M-d), Na*Ne)
    denom = np.sum(np.abs(vprod) ** 2, axis=0).real + 1e-12
    P = 1.0 / denom
    idx = int(np.argmax(P))
    Na = az_grid.size
    el_idx = idx // Na
    az_idx = idx % Na
    return float(az_grid[az_idx]), float(el_grid[el_idx])


def _build_coarse_fine_steering_cache(
    positions_wl: np.ndarray,
    coarse_az_grid: np.ndarray,
    coarse_el_grid: np.ndarray,
    az_range: Tuple[int, int],
    el_range: Tuple[int, int],
    fine_az_step: int,
    fine_el_step: int,
    fine_half_win_az: int,
    fine_half_win_el: int,
):
    """Precompute coarse steering and per-coarse-bin fine steering.

    Returns: (A_coarse, fine_cache) keyed by (el_idx, az_idx) -> (A_fine, fine_az, fine_el).
    Built once per frame; reused for all detections.
    """
    A_coarse, _, _ = _gen_steering_matrix_2d(
        positions_wl, coarse_az_grid, coarse_el_grid
    )
    cache = {}
    for ie, el0 in enumerate(coarse_el_grid):
        el_min = max(el_range[0], int(np.round(el0)) - fine_half_win_el)
        el_max = min(el_range[1], int(np.round(el0)) + fine_half_win_el)
        fine_el_grid = np.arange(
            el_min, el_max + 1, max(1, int(fine_el_step)), dtype=np.float32
        )
        for ia, az0 in enumerate(coarse_az_grid):
            az_min = max(az_range[0], int(np.round(az0)) - fine_half_win_az)
            az_max = min(az_range[1], int(np.round(az0)) + fine_half_win_az)
            fine_az_grid = np.arange(
                az_min, az_max + 1, max(1, int(fine_az_step)), dtype=np.float32
            )
            A_fine, _, _ = _gen_steering_matrix_2d(
                positions_wl, fine_az_grid, fine_el_grid
            )
            cache[(ie, ia)] = (A_fine, fine_az_grid, fine_el_grid)
    return A_coarse, cache


def _music_2d_peak_coarse_to_fine_cached(
    Rxx: np.ndarray,
    A_coarse: np.ndarray,
    num_coarse_az: int,
    fine_cache: dict,
    num_sources: int = 1,
) -> Tuple[float, float]:
    """Coarse-to-fine peak with precomputed fine steering cache.

    `num_coarse_az` is the number of azimuth points in the coarse grid (Na).
    """
    # Noise subspace
    _, v = np.linalg.eigh(Rxx.astype(np.complex64))
    M = Rxx.shape[0]
    d = max(1, min(num_sources, M - 1))
    En = v[:, : (M - d)]
    # Coarse search in element space
    vprod = En.conj().T @ A_coarse
    denom = np.sum(np.abs(vprod) ** 2, axis=0).real + 1e-12
    P = 1.0 / denom
    idx = int(np.argmax(P))
    ie = idx // num_coarse_az
    ia = idx % num_coarse_az
    # Fine lookup
    A_fine, fine_az_grid, fine_el_grid = fine_cache[(ie, ia)]
    az, el = _music_peak_with_A(En, A_fine, fine_az_grid, fine_el_grid)
    return az, el


def _estimate_xyz_music2d(
    aoa_input: np.ndarray,
    r: int,
    k: int,
    doppler_halfspan: int,
    positions_wl: np.ndarray,
    range_resolution: float,
    diag_load: float = 0.0,
    *,
    # Fallback full-grid search inputs
    az_grid: Optional[np.ndarray] = None,
    el_grid: Optional[np.ndarray] = None,
    # Coarse-to-fine inputs
    A_coarse: Optional[np.ndarray] = None,
    num_coarse_az: Optional[int] = None,
    fine_cache: Optional[dict] = None,
    # Optional precomputed FB-averaging permutation matrix
    J_fb: Optional[np.ndarray] = None,
) -> tuple:
    """Estimate (x,y,z) for one detection via 2D MUSIC.

    Inputs: `aoa_input[r, :, k±half]` snapshots and array geometry.
    Modes:
      - Coarse-to-fine (preferred): provide A_coarse, num_coarse_az, fine_cache (2D cache keyed by (el_idx, az_idx)).
      - Fallback full-grid: provide az_grid and el_grid.
    Output: (x,y,z) in world frame (+y forward, +x right, +z up).
    """
    M = positions_wl.shape[0]
    half = max(0, int(doppler_halfspan))
    k0 = max(0, k - half)
    k1 = min(aoa_input.shape[2] - 1, k + half)
    X = aoa_input[r, :, k0 : k1 + 1].astype(np.complex64)
    if X.ndim == 1:
        X = X[:, None]
    # Covariance + forward-backward averaging
    Rxx = (X @ X.conj().T) / max(1, X.shape[1])
    if J_fb is None:
        J_fb = np.fliplr(np.eye(M, dtype=np.float32))
    Rfb = 0.5 * (Rxx + J_fb @ Rxx.conj() @ J_fb)
    if diag_load and diag_load > 0.0:
        tr = float(np.trace(Rfb).real)
        Rfb = Rfb + np.eye(M, dtype=Rfb.dtype) * (diag_load * tr / M)

    if (
        A_coarse is not None
        and fine_cache is not None
        and isinstance(num_coarse_az, int)
    ):
        az_peak, el_peak = _music_2d_peak_coarse_to_fine_cached(
            Rfb, A_coarse, int(num_coarse_az), fine_cache, num_sources=1
        )
    else:
        if az_grid is None or el_grid is None:
            raise ValueError(
                "Full-grid MUSIC requires az_grid and el_grid when coarse cache is absent"
            )
        az_peak, el_peak = _music_2d_peak(
            Rfb, positions_wl, az_grid, el_grid, num_sources=1
        )

    # Convert to unit vector consistent with steering mapping used in this module
    azr = np.deg2rad(az_peak)
    elr = np.deg2rad(el_peak)
    ux = np.cos(elr) * np.sin(azr)
    uy = np.sin(elr)
    uz = np.cos(elr) * np.cos(azr)
    rng_m = range_resolution * r
    # World axes: forward=y, right=x, up=z
    x = ux * rng_m
    y = uz * rng_m
    z = uy * rng_m
    return x, y, z


def process_3D_radar_frame_music_2d(
    frame,
    adc_params,
    tuning: Optional[dict] = None,
    az_range: Tuple[int, int] = (-53, 53),
    fine_az_step: int = 2,
    el_range: Tuple[int, int] = (-18, 18),
    fine_el_step: int = 4,
    doppler_halfspan: int = 2,
    # Coarse-to-fine settings (first optimization)
    coarse_az_step: int = 8,
    coarse_el_step: int = 12,
    fine_half_win_az: int = 8,
    fine_half_win_el: int = 8,
    music_diag_load: float = 0.01,
    *,
    compute_tesseract: bool = False,
):
    """3D processing using 2D MUSIC AoA per detection.

    Args:
        frame: Complex radar frame (chirps, tx, rx, samples) or equivalent.
        adc_params: Object with tx, rx, chirps, samples, range_resolution, doppler_resolution.
        tuning: Optional dict for CFAR/pruning knobs (same shape as baseline pipeline).
        az_range: Azimuth bounds [deg] for the search grid.
        fine_az_step: Fine azimuth grid step [deg] used for refinement and fallback.
        el_range: Elevation bounds [deg] for the search grid.
        fine_el_step: Fine elevation grid step [deg] used for refinement and fallback.
        doppler_halfspan: Number of Doppler bins on each side of the detection index
            to include as snapshots when forming Rxx. Total snapshots ≈ 2*halfspan+1.
            Larger: smoother covariance, better SNR, but more compute and potential
            motion smearing. Smaller: faster, but noisier estimates. Typical 1–3.
        coarse_az_step: Coarse azimuth step [deg] for first-stage search.
        coarse_el_step: Coarse elevation step [deg] for first-stage search.
        fine_half_win_az: Refinement half-window [deg] around the coarse azimuth peak.
            Larger: more robust to coarse-peak errors but more compute; Smaller: faster
            but may miss the true peak if coarse is off. Start at 6–10°; increase if
            coarse_az_step is large or targets are off-boresight.
        fine_half_win_el: Refinement half-window [deg] around the coarse elevation peak.
            Similar trade-off as az. Start at 6–12° depending on expected elevation span.
        music_diag_load: Diagonal loading fraction (trace(R)/M * fraction) added to
            covariance for robustness. 0.0 disables. Try 0.01 if estimates are unstable.

    Notes:
        - World frame: +y forward, +x right, +z up. Returned RD is transposed for the analyser.
        - Coarse-to-fine uses per-frame cached steering; if cache is absent, falls back to full fine grid.

    Returns:
        dict: range_doppler, range_azimuth (None), x_pos, y_pos, z_pos, velocities, snrs, cluster_labels.
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
    t_range = time.perf_counter() - step_start

    # 2) Doppler FFT
    step_start = time.perf_counter()
    det_matrix, aoa_input = dsp.doppler_processing(
        radar_cube,
        num_tx_antennas=adc_params.tx,
        clutter_removal_enabled=False,
        interleaved=False,
        window_type_2d=dsp.utils.Window.HAMMING,
    )
    det_matrix = np.fft.fftshift(det_matrix, axes=1)
    aoa_input = np.fft.fftshift(aoa_input, axes=2)
    t_doppler = time.perf_counter() - step_start

    # Optionally precompute full-grid steering for tesseract computation (no CFAR/max)
    # We do this before CFAR so the 4D tensor reflects raw spectrum across all bins.
    # NOTE: The tesseract (D,R,E,A) always uses a fixed 1° azimuth/elevation grid, independent
    # of the fine search steps used for per-detection MUSIC refinement. This guarantees
    # consistent angular resolution for offline analysis and zyx cube generation.
    positions_wl_pre, az_grid_pre, el_grid_pre, A_full_pre = (None, None, None, None)
    tesseract = None
    tesseract_time_s = None
    if compute_tesseract:
        # Force 1° resolution for tesseract irrespective of fine_az_step/fine_el_step
        positions_wl_pre, az_grid_pre, el_grid_pre, A_full_pre = (
            _prepare_tesseract_assets(
                adc_params,
                az_range,
                el_range,
                1,
                1,
            )
        )
        if (
            A_full_pre is not None
            and positions_wl_pre is not None
            and az_grid_pre is not None
            and el_grid_pre is not None
        ):
            _t0_tess = time.perf_counter()
            tesseract = _compute_tesseract(
                aoa_input,
                positions_wl_pre,
                A_full_pre,
                az_grid_pre,
                el_grid_pre,
                doppler_halfspan,
                music_diag_load,
            )
            tesseract_time_s = time.perf_counter() - _t0_tess

    # 3) CFAR and detections (same as baseline)
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
    t_cfar = time.perf_counter() - step_start

    # 4) Data structure creation for raw detections
    step_start = time.perf_counter()
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
    t_struct = time.perf_counter() - step_start

    step_start = time.perf_counter()
    detObj2D = dsp.prune_to_peaks(
        detObj2DRaw, det_matrix, adc_params.chirps, reserve_neighbor=True
    )
    detObj2D = dsp.peak_grouping_along_doppler(detObj2D, det_matrix, adc_params.chirps)
    t_group = time.perf_counter() - step_start

    # 5) Pruning based on SNR and peak value
    step_start = time.perf_counter()
    th3d = (tuning or {}).get("thresholds_3d", {}) if isinstance(tuning, dict) else {}
    SNRThresholds2 = np.array(
        th3d.get("snr_table", [[2, 10.5], [10, 7.5], [35, 5.0]]), dtype=np.float32
    )
    t_prune = time.perf_counter() - step_start
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

    # 6) 2D MUSIC per detection
    num_det = (
        len(detObj2D["rangeIdx"])
        if isinstance(detObj2D, np.void) or isinstance(detObj2D, dict)
        else detObj2D.shape[0]
    )
    if num_det == 0:
        return {
            "range_doppler": det_matrix.T,
            "range_azimuth": None,
            "x_pos": np.array([]),
            "y_pos": np.array([]),
            "z_pos": np.array([]),
            "velocities": np.array([]),
            "snrs": np.array([]),
            "cluster_labels": np.array([]),
            "tesseract": tesseract,
            "tesseract_az_grid_deg": az_grid_pre if compute_tesseract else None,
            "tesseract_el_grid_deg": el_grid_pre if compute_tesseract else None,
            "tesseract_time_s": tesseract_time_s,
        }

    step_start = time.perf_counter()
    (
        positions_wl,
        az_grid,
        el_grid,
        coarse_az_grid,
        coarse_el_grid,
        A_coarse,
        fine_cache,
    ) = _get_or_build_steering_cache(
        adc_params.tx,
        adc_params.rx,
        az_range,
        el_range,
        fine_az_step,
        fine_el_step,
        coarse_az_step,
        coarse_el_step,
        fine_half_win_az,
        fine_half_win_el,
    )
    t_music_setup = time.perf_counter() - step_start

    xs = np.zeros(num_det, dtype=np.float32)
    ys = np.zeros(num_det, dtype=np.float32)
    zs = np.zeros(num_det, dtype=np.float32)

    # Doppler snapshots for covariance
    music_t_total = 0.0
    # Precompute FB-averaging permutation once
    M = positions_wl.shape[0]
    J_fb = np.fliplr(np.eye(M, dtype=np.float32))

    # Batched coarse-stage search across detections
    t0_music = time.perf_counter()
    half = max(0, int(doppler_halfspan))
    r_idx = detObj2D["rangeIdx"].astype(int)
    k_idx = detObj2D["dopplerIdx"].astype(int)
    Rfbs = np.empty((num_det, M, M), dtype=np.complex64)
    for i in range(num_det):
        r = int(r_idx[i])
        k = int(k_idx[i])
        k0 = max(0, k - half)
        k1 = min(aoa_input.shape[2] - 1, k + half)
        X = aoa_input[r, :, k0 : k1 + 1].astype(np.complex64)
        if X.ndim == 1:
            X = X[:, None]
        Rxx = (X @ X.conj().T) / max(1, X.shape[1])
        Rfb = 0.5 * (Rxx + J_fb @ Rxx.conj() @ J_fb)
        if music_diag_load and float(music_diag_load) > 0.0:
            tr = float(np.trace(Rfb).real)
            Rfb = Rfb + np.eye(M, dtype=Rfb.dtype) * (float(music_diag_load) * tr / M)
        Rfbs[i] = Rfb.astype(np.complex64, copy=False)

    # Compute noise subspaces En for all detections, then batch multiply against A_coarse
    d = 1
    p = M - d
    Ens = np.empty((num_det, M, p), dtype=np.complex64)
    for i in range(num_det):
        _, v = np.linalg.eigh(Rfbs[i])
        Ens[i] = v[:, :p]
    EnH = np.transpose(Ens.conj(), (0, 2, 1))  # (N, p, M)
    A_coarse_mat = A_coarse.astype(np.complex64, copy=False)  # (M, G)
    vprod = EnH @ A_coarse_mat  # (N, p, G)
    denom = np.sum(np.abs(vprod) ** 2, axis=1).real + 1e-12  # (N, G)
    P = 1.0 / denom
    coarse_idx = np.argmax(P, axis=1).astype(int)
    Na_coarse = int(coarse_az_grid.size)
    el_idx_coarse = coarse_idx // Na_coarse
    az_idx_coarse = coarse_idx % Na_coarse

    # Fine refinement per detection using cached fine steering
    for i in range(num_det):
        ie = int(el_idx_coarse[i])
        ia = int(az_idx_coarse[i])
        A_fine, fine_az_grid, fine_el_grid = fine_cache[(ie, ia)]
        az_peak, el_peak = _music_peak_with_A(
            Ens[i], A_fine, fine_az_grid, fine_el_grid
        )
        # Convert to xyz
        azr = np.deg2rad(az_peak)
        elr = np.deg2rad(el_peak)
        ux = np.cos(elr) * np.sin(azr)
        uy = np.sin(elr)
        uz = np.cos(elr) * np.cos(azr)
        rng_m = adc_params.range_resolution * float(r_idx[i])
        xs[i] = ux * rng_m
        ys[i] = uz * rng_m
        zs[i] = uy * rng_m
    music_t_total = time.perf_counter() - t0_music

    # RA heatmap is not essential here; set to None to simplify
    range_azimuth = None

    velocities = detObj2D["dopplerIdx"] * adc_params.doppler_resolution
    snrs = detObj2D["SNR"]

    # tesseract already computed earlier if requested

    total_time = time.perf_counter() - function_start
    try:
        logger.info(
            "[MUSIC2D] total=%.3fs | range=%.3fs, doppler=%.3fs, cfar=%.3fs, struct=%.3fs, group=%.3fs, prune=%.3fs, setup=%.3fs, music=%.3fs (avg=%.2fms, N=%d)",
            total_time,
            t_range,
            t_doppler,
            t_cfar,
            t_struct,
            t_group,
            t_prune,
            t_music_setup,
            music_t_total,
            (music_t_total / max(1, num_det)) * 1e3,
            num_det,
        )
    except Exception:
        pass

    return {
        # Return RD transposed to match analyser RD SHM shape (chirps, samples) before UI rotation
        "range_doppler": det_matrix.T,
        "range_azimuth": range_azimuth,
        "x_pos": ys,
        "y_pos": xs,
        "z_pos": zs,
        "velocities": velocities,
        "snrs": snrs,
        "cluster_labels": np.array([]),
        # Optional full 4D MUSIC pseudospectrum tensor: (doppler, range, elevation, azimuth)
        "tesseract": tesseract if compute_tesseract else None,
        "tesseract_az_grid_deg": az_grid_pre if compute_tesseract else None,
        "tesseract_el_grid_deg": el_grid_pre if compute_tesseract else None,
        "tesseract_time_s": tesseract_time_s if compute_tesseract else None,
    }
