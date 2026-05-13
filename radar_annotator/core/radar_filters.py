"""Shared radar point-cloud filtering helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np


ReflectionMode = str
RoiBounds = Tuple[float, float, float, float, Optional[float], Optional[float]]
ThresholdValue = Union[float, Literal["auto"], None]

COL_X = 0
COL_Y = 1
COL_Z = 2
COL_RCS = 3
COL_VR = 4
COL_VR_COMP = 5
COL_TIME = 6
COL_SNR = 7


@dataclass(frozen=True)
class RadarFilterResult:
    points: np.ndarray
    mask: np.ndarray


def _finite_xyz_mask(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    return np.all(np.isfinite(points[:, :3]), axis=1)


def _adaptive_threshold(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("inf")
    return float(np.percentile(finite, percentile))


def _threshold_value(values: np.ndarray, value: ThresholdValue, percentile: float) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "auto":
        return _adaptive_threshold(values, percentile)
    return float(value)


def _local_density_mask(points_xyz: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    """Small point-cloud density filter without scipy/sklearn dependencies."""
    n = points_xyz.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    if n == 1:
        return np.array([min_neighbors <= 1], dtype=bool)

    radius2 = float(radius) * float(radius)
    keep = np.zeros(n, dtype=bool)
    # Chunking keeps memory bounded for dense frames.
    chunk = 512
    for start in range(0, n, chunk):
        block = points_xyz[start:start + chunk]
        diff = block[:, None, :] - points_xyz[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        counts = np.count_nonzero(dist2 <= radius2, axis=1)
        keep[start:start + chunk] = counts >= min_neighbors
    return keep


def apply_radar_filters(
    points: np.ndarray,
    *,
    z_range: Optional[Tuple[float, float]] = None,
    roi: Optional[RoiBounds] = None,
    reflection_mode: ReflectionMode = "all",
    custom_rcs_min: ThresholdValue = None,
    snr_min: ThresholdValue = None,
) -> RadarFilterResult:
    """Return points surviving the active display/export filters.

    Expected VoD radar columns are x, y, z, RCS, v_r, v_r_compensated, time.
    If a dataset appends SNR as an 8th column, ``snr_min`` filters by it.
    ``custom_rcs_min`` and ``snr_min`` may be ``"auto"`` to use a robust
    percentile threshold from the currently active finite/ROI/Z points.
    For non-VoD clouds the helper degrades gracefully: modes that need RCS,
    velocity, or SNR become simple finite/ROI/Z filtering instead of rejecting
    all data.
    """
    pts = np.asarray(points if points is not None else np.zeros((0, 3)), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return RadarFilterResult(pts.reshape(0, pts.shape[-1] if pts.ndim == 2 else 3), np.zeros(0, dtype=bool))

    mask = _finite_xyz_mask(pts)

    if z_range is not None and pts.shape[1] >= 3:
        z_min, z_max = z_range
        mask &= (pts[:, 2] >= float(z_min)) & (pts[:, 2] <= float(z_max))

    if roi is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = roi
        mask &= (
            (pts[:, 0] >= float(min(x_min, x_max)))
            & (pts[:, 0] <= float(max(x_min, x_max)))
            & (pts[:, 1] >= float(min(y_min, y_max)))
            & (pts[:, 1] <= float(max(y_min, y_max)))
        )
        if z_min is not None and z_max is not None:
            mask &= (
                (pts[:, 2] >= float(min(z_min, z_max)))
                & (pts[:, 2] <= float(max(z_min, z_max)))
            )

    if snr_min is not None and pts.shape[1] > COL_SNR:
        snr = pts[:, COL_SNR]
        snr_threshold = _threshold_value(snr[mask], snr_min, 55.0)
        if snr_threshold is not None:
            mask &= np.isfinite(snr) & (snr >= snr_threshold)

    mode = (reflection_mode or "all").lower()
    if mode != "all" and pts.shape[1] > COL_RCS:
        rcs = pts[:, COL_RCS]
        if mode == "strong":
            mask &= rcs >= _adaptive_threshold(rcs[mask], 70.0)
        elif mode == "object":
            active = np.where(mask)[0]
            if active.size > 0:
                active_rcs = rcs[active]
                rcs_keep = active_rcs >= _adaptive_threshold(active_rcs, 45.0)
                density_keep = _local_density_mask(pts[active, :3], radius=1.25, min_neighbors=2)
                mode_keep = rcs_keep & density_keep
                tmp = np.zeros_like(mask)
                tmp[active] = mode_keep
                mask &= tmp
        elif mode == "moving":
            if pts.shape[1] > COL_VR_COMP:
                vel = pts[:, COL_VR_COMP]
            elif pts.shape[1] > COL_VR:
                vel = pts[:, COL_VR]
            else:
                vel = np.zeros(pts.shape[0], dtype=np.float64)
            mask &= (np.abs(vel) >= 0.25) & (rcs >= _adaptive_threshold(rcs[mask], 35.0))
        elif mode == "custom" and custom_rcs_min is not None:
            rcs_threshold = _threshold_value(rcs[mask], custom_rcs_min, 55.0)
            if rcs_threshold is not None:
                mask &= np.isfinite(rcs) & (rcs >= rcs_threshold)

    return RadarFilterResult(pts[mask], mask)


def describe_filter_settings(
    *,
    z_range: Optional[Tuple[float, float]],
    roi: Optional[RoiBounds],
    reflection_mode: ReflectionMode,
    custom_rcs_min: ThresholdValue,
    snr_min: ThresholdValue = None,
) -> dict:
    return {
        "z_range": None if z_range is None else [float(z_range[0]), float(z_range[1])],
        "roi": None if roi is None else [
            float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]),
            None if roi[4] is None else float(roi[4]),
            None if roi[5] is None else float(roi[5]),
        ],
        "reflection_mode": reflection_mode,
        "custom_rcs_min": custom_rcs_min,
        "snr_min": snr_min,
    }
