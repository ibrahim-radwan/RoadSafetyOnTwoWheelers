"""
Core geometry module.

Implements the single-master-box rule from the design spec:
one canonical 3D box per object lives in the master (radar / ego) frame.
All other representations — BEV, camera wireframe, 2D bbox — are derived.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .calibration import Calibration
import numpy as np
import uuid


@dataclass
class Box3D:
    """Canonical 3D bounding box in the master (radar/ego) coordinate frame.

    Coordinate convention (master frame):
        x forward, y left, z up  (ROS / ego-vehicle style).
    `yaw` rotates around +z. `pitch`/`roll` default to zero and are rarely edited.

    This object is the *single source of truth*. Never store an independent
    camera-frame box for the same object — always derive it.
    """
    # Pose & size
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    length: float = 4.0   # along local x (forward)
    width: float = 1.8    # along local y
    height: float = 1.6   # along local z
    yaw: float = 0.0      # radians, around +z
    pitch: float = 0.0
    roll: float = 0.0

    # Identity & semantics
    object_id: int = 0
    track_id: Optional[int] = None
    class_name: str = "Car"
    confidence: float = 1.0
    occlusion: int = 0        # 0=visible, 1=partial, 2=heavy, 3=unknown
    truncation: float = 0.0   # 0..1
    notes: str = ""

    # True when pose was set purely from radar / manual workflows that must not
    # rely on inverse camera extrinsics (still overlay via forward projection).
    manual_placement: bool = False

    # Derived / cached (not persisted as primary state)
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    # --- Geometry helpers ---------------------------------------------------

    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix from yaw/pitch/roll (ZYX intrinsic)."""
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cr, sr = np.cos(self.roll), np.sin(self.roll)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Rz @ Ry @ Rx

    def corners(self) -> np.ndarray:
        """Return the 8 corners of the box in the master frame. Shape (8, 3).

        Corner order (for consistent wireframe drawing):
            0: -l/2 -w/2 -h/2   (rear-right-bottom)
            1: +l/2 -w/2 -h/2
            2: +l/2 +w/2 -h/2
            3: -l/2 +w/2 -h/2
            4..7: same but top (+h/2)
        """
        l, w, h = self.length / 2.0, self.width / 2.0, self.height / 2.0
        local = np.array([
            [-l, -w, -h], [ l, -w, -h], [ l,  w, -h], [-l,  w, -h],
            [-l, -w,  h], [ l, -w,  h], [ l,  w,  h], [-l,  w,  h],
        ])
        R = self.rotation_matrix()
        world = (R @ local.T).T + np.array([self.x, self.y, self.z])
        return world

    def center(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def size(self) -> np.ndarray:
        return np.array([self.length, self.width, self.height])

    # --- Point-in-box check (for "num radar points inside") -----------------

    def points_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        """Boolean mask of which points lie inside the oriented box."""
        if points_xyz.size == 0:
            return np.zeros(0, dtype=bool)
        # Translate to box origin then rotate into box-local frame
        R = self.rotation_matrix()
        local = (points_xyz - self.center()) @ R  # R^T applied on the right
        hl, hw, hh = self.length / 2, self.width / 2, self.height / 2
        return (
            (np.abs(local[:, 0]) <= hl)
            & (np.abs(local[:, 1]) <= hw)
            & (np.abs(local[:, 2]) <= hh)
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Box3D":
        # Ignore unknown keys for forward compatibility
        known = {f for f in Box3D.__dataclass_fields__}
        return Box3D(**{k: v for k, v in d.items() if k in known})


def box_bottom_center_master(box: Box3D) -> np.ndarray:
    """Mean of the four bottom-face corners in the master frame.

    KITTI labels use this reference point (after ``T_cam_from_master``), not the
    geometric centre ``(box.x, box.y, box.z)``.
    """
    c = box.corners()
    return c[[0, 1, 2, 3]].mean(axis=0)


def box_location_cam_kitti(box: Box3D, cal: Calibration) -> np.ndarray:
    """Bottom-centre of ``box`` in rectified camera coordinates.

    Matches the three ``location`` scalars written by :mod:`io.kitti_export`
    (before ``rotation_y``). Axes are camera: X right, Y down, Z forward — not
    the master ``x`` forward / ``y`` left convention.
    """
    p = box_bottom_center_master(box)
    h = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    return (cal.T @ h)[:3]


def radar_returns_median_xyz(
    box: Box3D,
    radar_points: np.ndarray,
) -> Optional[np.ndarray]:
    """Median ``(x, y, z)`` of radar returns inside ``box``, or ``None``."""
    if radar_points.size == 0:
        return None
    mask = box.points_inside(radar_points[:, :3])
    if not np.any(mask):
        return None
    xyz = np.asarray(radar_points[mask, :3], dtype=np.float64)
    return np.median(xyz, axis=0)


def radar_display_xyz_m(
    box: Box3D,
    radar_points: np.ndarray,
) -> Tuple[float, float, float]:
    """UI/display position (m): radar median inside box, else box centre."""
    med = radar_returns_median_xyz(box, radar_points)
    if med is None:
        return float(box.x), float(box.y), float(box.z)
    return float(med[0]), float(med[1]), float(med[2])


def radar_range_from_x_m(box: Box3D, radar_points: np.ndarray) -> float:
    """Along-track range (m): median radar X of returns inside the box.

    Uses Cartesian X as forward range (common automotive radar convention).
    Falls back to box centre X when no returns lie inside the box.
    """
    med = radar_returns_median_xyz(box, radar_points)
    if med is None:
        return float(box.x)
    return float(med[0])


def radar_azimuth_deg_from_xy(x: float, y: float) -> float:
    """Azimuth in degrees in the master horizontal plane: ``atan2(y, x)``."""
    return float(np.rad2deg(np.arctan2(y, x)))


# ---------------------------------------------------------------------------
# Projection utilities
# ---------------------------------------------------------------------------

# The 12 edges of a cuboid given the corner ordering above.
BOX_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),   # top
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
)


def project_points(points_master: np.ndarray,
                   T_cam_from_master: np.ndarray,
                   K: np.ndarray,
                   P_rect: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D master-frame points into the image.

    Returns (uv, depth) where uv is (N,2) pixel coordinates (may be outside
    image) and depth is (N,) homogeneous scale (equals camera ``z`` when P's third
    row is ``[0,0,1,0]``). Points with depth <= 0 are behind the camera.

    When ``P_rect`` is the full KITTI ``P2`` (3×4), uses ``P @ [x,y,z,1]`` — required
    for correct boxes versus ``K @ [x,y,z] / z`` (baseline column in ``P``).
    """
    N = points_master.shape[0]
    homog = np.hstack([points_master, np.ones((N, 1))])         # (N,4)
    cam = (T_cam_from_master @ homog.T).T[:, :3]                # (N,3)
    if P_rect is not None:
        P = np.asarray(P_rect, dtype=np.float64).reshape(3, 4)
        cam_h = np.hstack([cam, np.ones((N, 1))])
        uv_h = (P @ cam_h.T).T
        w = uv_h[:, 2]
        safe = np.where(np.abs(w) < 1e-6, 1e-6, w)
        uv = uv_h[:, :2] / safe[:, None]
        depth = w.copy()
        return uv, depth
    depth = cam[:, 2].copy()
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    uv_h = (K @ cam.T).T                                        # (N,3)
    uv = uv_h[:, :2] / safe[:, None]
    return uv, depth


def project_points_calibration(points_master: np.ndarray, cal: Calibration):
    """Convenience: project using optional ``cal.P_rect`` when loaded from KITTI txt."""
    return project_points(points_master, cal.T, cal.K, cal.P_rect)


def project_box(box: Box3D,
                T_cam_from_master: np.ndarray,
                K: np.ndarray,
                P_rect: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Project a Box3D into the image. Returns (corners_uv (8,2), depths (8,))."""
    corners = box.corners()
    return project_points(corners, T_cam_from_master, K, P_rect)


def project_box_calibration(box: Box3D, cal: Calibration):
    """Convenience: project using optional ``cal.P_rect`` when loaded from KITTI txt."""
    return project_box(box, cal.T, cal.K, cal.P_rect)


def box_2d_from_projection(corners_uv: np.ndarray,
                           depths: np.ndarray,
                           image_size_wh: Tuple[int, int]
                           ) -> Optional[Tuple[float, float, float, float]]:
    """Derive an axis-aligned 2D bbox (xmin, ymin, xmax, ymax) from projected
    corners. Returns None if the box is fully behind the camera or fully
    outside the image.
    """
    W, H = image_size_wh
    if np.all(depths <= 0):
        return None
    # Only use corners that are in front of the camera for extent estimation
    front = depths > 0
    uv = corners_uv[front]
    if uv.size == 0:
        return None
    xmin, ymin = uv.min(axis=0)
    xmax, ymax = uv.max(axis=0)
    # Clip to image
    xmin_c = max(0.0, float(xmin))
    ymin_c = max(0.0, float(ymin))
    xmax_c = min(float(W - 1), float(xmax))
    ymax_c = min(float(H - 1), float(ymax))
    if xmax_c <= xmin_c or ymax_c <= ymin_c:
        return None
    return (xmin_c, ymin_c, xmax_c, ymax_c)
