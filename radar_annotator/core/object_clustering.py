"""Initial object proposals from radar point clusters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .geometry import Box3D


@dataclass(frozen=True)
class ClusterConfig:
    eps_m: float = 2.0
    min_samples: int = 3
    max_objects: int = 30
    duplicate_iou: float = 0.35
    ambiguous_iou: float = 0.10
    use_rcs_feature: bool = True
    # When True, clamp box L/W to the cluster's own spatial extent
    size_from_extent: bool = True


@dataclass(frozen=True)
class ClusterResult:
    created: List[Box3D]
    duplicates: int
    ambiguous: List[Box3D]
    noise_points: int


def suggest_eps(points: np.ndarray, factor: float = 3.5) -> float:
    """Return a data-driven DBSCAN epsilon (meters) based on median NN distance.

    Samples up to 300 XY points for speed, then scales the median nearest-
    neighbour distance by *factor*.  Clamped to [0.8, 4.0] m.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] < 2:
        return 2.0
    rng = np.random.default_rng(0)
    idx = rng.choice(pts.shape[0], min(300, pts.shape[0]), replace=False)
    sample = pts[idx, :2]
    diff = sample[:, None, :] - sample[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    np.fill_diagonal(d2, np.inf)
    nn_d = np.sqrt(np.min(d2, axis=1))
    return float(np.clip(np.median(nn_d) * factor, 0.8, 4.0))


def propose_initial_objects(
    points: np.ndarray,
    *,
    class_name: str,
    default_size: tuple[float, float, float],
    existing_boxes: Sequence[Box3D],
    config: ClusterConfig = ClusterConfig(),
) -> ClusterResult:
    pts = np.asarray(points if points is not None else np.zeros((0, 3)), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return ClusterResult([], 0, [], 0)

    finite = np.all(np.isfinite(pts[:, :3]), axis=1)
    pts = pts[finite]
    if pts.shape[0] == 0:
        return ClusterResult([], 0, [], 0)

    labels = _dbscan_labels(_cluster_features(pts, config), eps=1.0, min_samples=config.min_samples)
    noise_points = int(np.count_nonzero(labels < 0))
    boxes = []
    for label in sorted(v for v in set(labels.tolist()) if v >= 0):
        cluster = pts[labels == label]
        if cluster.shape[0] < config.min_samples:
            continue
        boxes.append(_box_from_cluster(cluster, class_name, default_size, config))

    boxes.sort(key=lambda b: int(np.count_nonzero(b.points_inside(pts[:, :3]))), reverse=True)
    created: List[Box3D] = []
    ambiguous: List[Box3D] = []
    duplicates = 0
    accepted_existing: List[Box3D] = list(existing_boxes)
    for box in boxes:
        max_iou = _max_bev_iou(box, accepted_existing)
        if max_iou >= config.duplicate_iou:
            duplicates += 1
            continue
        if max_iou >= config.ambiguous_iou:
            ambiguous.append(box)
            continue
        created.append(box)
        accepted_existing.append(box)
        if len(created) >= config.max_objects:
            break

    return ClusterResult(created, duplicates, ambiguous, noise_points)


def _cluster_features(points: np.ndarray, config: ClusterConfig) -> np.ndarray:
    xyz = points[:, :3].copy()
    features = [xyz[:, 0], xyz[:, 1], xyz[:, 2] * 0.5]
    if config.use_rcs_feature and points.shape[1] > 3:
        rcs = points[:, 3]
        finite = rcs[np.isfinite(rcs)]
        if finite.size > 0:
            med = float(np.median(finite))
            iqr = float(np.percentile(finite, 75) - np.percentile(finite, 25))
            scale = max(iqr, 1.0)
            rcs_norm = np.clip((np.nan_to_num(rcs, nan=med) - med) / scale, -3.0, 3.0)
            features.append(rcs_norm * 0.75)
    feat = np.column_stack(features)
    # Divide all spatial columns (x, y, z·0.5) by eps_m so that DBSCAN eps=1.0
    # means "within eps_m metres in xy, within 2·eps_m metres in z" regardless
    # of the eps_m value.  Only the first 3 columns are spatial; RCS is already
    # a dimensionless normalised score and must not be divided by eps_m.
    feat[:, :3] /= max(float(config.eps_m), 1e-6)
    return feat


# Maximum rows processed at once for the distance matrix to cap memory use.
_DBSCAN_CHUNK = 512


def _dbscan_labels(features: np.ndarray, *, eps: float, min_samples: int) -> np.ndarray:
    """Pure-NumPy DBSCAN.

    Fixes vs. the original:
    * Chunked distance matrix — avoids O(N²·D) memory for large clouds.
    * O(1) seed-set membership check via a companion set, not list.contains().
    """
    n = features.shape[0]
    labels = np.full(n, -99, dtype=int)
    if n == 0:
        return labels

    eps2 = eps * eps

    # Build neighbor lists in memory-safe chunks
    neighbors: list[np.ndarray] = [np.empty(0, dtype=int)] * n
    for start in range(0, n, _DBSCAN_CHUNK):
        end = min(start + _DBSCAN_CHUNK, n)
        diff = features[start:end, None, :] - features[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        for local_i in range(end - start):
            neighbors[start + local_i] = np.flatnonzero(dist2[local_i] <= eps2)

    cluster_id = 0
    for i in range(n):
        if labels[i] != -99:
            continue
        if neighbors[i].size < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        seeds: list[int] = list(neighbors[i])
        seeds_set: set[int] = set(seeds)  # O(1) membership checks
        k = 0
        while k < len(seeds):
            j = seeds[k]
            if labels[j] == -1:
                labels[j] = cluster_id
            if labels[j] != -99:
                k += 1
                continue
            labels[j] = cluster_id
            if neighbors[j].size >= min_samples:
                for nb in neighbors[j]:
                    nb_int = int(nb)
                    if labels[nb_int] in (-99, -1) and nb_int not in seeds_set:
                        seeds.append(nb_int)
                        seeds_set.add(nb_int)
            k += 1
        cluster_id += 1

    labels[labels == -99] = -1
    return labels


def _box_from_cluster(
    points: np.ndarray,
    class_name: str,
    default_size: tuple[float, float, float],
    config: ClusterConfig,
) -> Box3D:
    length, width, height = default_size
    center = np.median(points[:, :3], axis=0)
    yaw = _estimate_yaw(points[:, :2])

    if config.size_from_extent and points.shape[0] >= 3:
        # Rotate cluster into box-local frame and measure 5th–95th percentile extent
        c, s = float(np.cos(-yaw)), float(np.sin(-yaw))
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        local = (R @ (points[:, :2] - center[:2]).T).T
        ext_l = float(np.percentile(local[:, 0], 95) - np.percentile(local[:, 0], 5))
        ext_w = float(np.percentile(local[:, 1], 95) - np.percentile(local[:, 1], 5))
        # Clamp to [0.7×, 1.6×] of the class default so impossible sizes are rejected
        length = float(np.clip(ext_l * 1.15, length * 0.70, length * 1.60))
        width = float(np.clip(ext_w * 1.15, width * 0.70, width * 1.60))

    # Use the 20th-percentile Z of the cluster as an estimate of the object's
    # lower surface, then position the box centre height/2 above it.
    # This correctly handles elevated objects while defaulting to ground level
    # when radar returns are near z=0.
    z_base = max(float(np.percentile(points[:, 2], 20)), 0.0)
    z_center = z_base + height * 0.5

    return Box3D(
        x=float(center[0]),
        y=float(center[1]),
        z=z_center,
        length=float(length),
        width=float(width),
        height=float(height),
        yaw=float(yaw),
        class_name=class_name,
        confidence=0.6,
        manual_placement=True,
        notes="Initial radar cluster proposal",
    )


def _estimate_yaw(xy: np.ndarray) -> float:
    """Estimate heading from XY point spread via PCA.

    Returns 0.0 when the cluster is too round to reliably determine orientation
    (eigenvalue ratio < 2.5) or when there are fewer than 3 points.
    """
    if xy.shape[0] < 3:
        return 0.0
    centered = xy - np.mean(xy, axis=0)
    cov = centered.T @ centered / max(xy.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)
    minor_val = max(float(vals[order[0]]), 1e-9)
    major_val = float(vals[order[-1]])
    # Require principal direction to be at least 2.5× stronger than minor axis;
    # below this the cluster is too isotropic for reliable yaw estimation.
    if major_val < minor_val * 2.5:
        return 0.0
    principal = vecs[:, order[-1]]
    return float(np.arctan2(principal[1], principal[0]))


def _max_bev_iou(box: Box3D, others: Iterable[Box3D]) -> float:
    return max((_bev_iou(box, other) for other in others), default=0.0)


def _bev_iou(a: Box3D, b: Box3D) -> float:
    pa = _footprint_xy(a)
    pb = _footprint_xy(b)
    inter = _polygon_area(_clip_polygon(pa, pb))
    if inter <= 0.0:
        return 0.0
    area_a = _polygon_area(pa)
    area_b = _polygon_area(pb)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def _footprint_xy(box: Box3D) -> np.ndarray:
    l2, w2 = box.length * 0.5, box.width * 0.5
    local = np.array([[-l2, -w2], [l2, -w2], [l2, w2], [-l2, w2]], dtype=np.float64)
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (rot @ local.T).T + np.array([box.x, box.y], dtype=np.float64)


def _polygon_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = subject.copy()
    for i in range(clip.shape[0]):
        a = clip[i]
        b = clip[(i + 1) % clip.shape[0]]
        if output.shape[0] == 0:
            break
        input_poly = output
        output_points = []
        prev = input_poly[-1]
        prev_inside = _inside_half_plane(prev, a, b)
        for cur in input_poly:
            cur_inside = _inside_half_plane(cur, a, b)
            if cur_inside:
                if not prev_inside:
                    output_points.append(_line_intersection(prev, cur, a, b))
                output_points.append(cur)
            elif prev_inside:
                output_points.append(_line_intersection(prev, cur, a, b))
            prev = cur
            prev_inside = cur_inside
        output = np.asarray(output_points, dtype=np.float64)
    return output


def _inside_half_plane(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    return _cross2(b - a, p - a) >= -1e-9


def _line_intersection(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = p2 - p1
    s = b - a
    denom = _cross2(r, s)
    if abs(denom) < 1e-12:
        return p2
    t = _cross2(a - p1, s) / denom
    return p1 + t * r


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])
