"""
Radar / BEV view.

A top-down (bird's-eye) canvas showing the radar point cloud and the 3D boxes
projected onto the ground plane. The user creates, drags, rotates, and resizes
boxes here; edits propagate to the canonical Box3D immediately so the image
view updates.

Coordinate mapping (master frame):
    x forward -> screen up
    y left    -> screen left  (so the visualisation matches what the driver
                               sees looking forward)
    z up      -> not shown (BEV)

Interactions:
    Left click empty   : deselect
    Left click on box  : select
    Left drag inside   : translate (X, Y in master frame)
    Left drag on edge  : resize front/back/left/right (no Shift)
    Left drag on corner: resize length+width from opposite corner (selected box)
    Left drag on heading arrow: rotate yaw (same as right-drag)
    Right drag on box  : rotate yaw
    Shift+drag edge    : still resizes (same as plain edge drag)
    Scroll wheel       : zoom
    Middle drag / Space+drag : pan
    Keys (selected): WASD move, Q/E yaw, Z/X length, C/V width (Shift = finer)
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (QColor, QPainter, QPen, QBrush, QFont, QWheelEvent,
                           QMouseEvent, QKeyEvent, QPaintEvent)
from PySide6.QtWidgets import QWidget, QSizePolicy

from ..core.geometry import Box3D
from ..core.annotation_model import AnnotationModel
from ..core.object_clustering import ClusterConfig, _cluster_features, _dbscan_labels
from .annotation_classes import CLASS_COLORS, class_default_box_size


# Editing modes while dragging
_MODE_NONE = 0
_MODE_TRANSLATE = 1
_MODE_ROTATE = 2
_MODE_RESIZE = 3
_MODE_PAN = 4
_MODE_CREATE = 5
_MODE_CORNER = 6


class RadarBEVView(QWidget):
    """Top-down editable view of radar points + 3D boxes."""

    status_message = Signal(str)

    DEFAULT_COLOR = QColor(100, 116, 139)
    SELECTED_COLOR = QColor(255, 255, 255)
    BG_COLOR = QColor(17, 24, 39)              # dark BEV canvas on light panel
    GRID_COLOR = QColor(55, 65, 81)
    AXIS_COLOR = QColor(156, 163, 175)
    POINT_CMAP_LOW = QColor(139, 92, 246)      # violet for low points
    POINT_CMAP_MID = QColor(34, 197, 94)       # green mid
    POINT_CMAP_HIGH = QColor(234, 179, 8)      # yellow high

    def __init__(self, model: AnnotationModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._points: np.ndarray = np.zeros((0, 3))

        # View transform: meters per pixel + center of view in master frame
        self._scale = 8.0            # pixels per meter
        self._center_master = np.array([15.0, 0.0])  # (x, y) centered ahead
        self._panning = False
        self._last_mouse = QPointF()

        # Drag state
        self._mode = _MODE_NONE
        self._drag_uid: Optional[str] = None
        self._drag_start_box: Optional[Box3D] = None
        self._drag_start_master: Optional[np.ndarray] = None
        self._resize_edge: Optional[str] = None  # 'front','back','left','right','rotate_heading'
        self._corner_idx: Optional[int] = None  # 0..3 for _MODE_CORNER
        self._corner_fix_master: Optional[np.ndarray] = None  # opposite corner (xy), fixed during corner drag

        # Creation state
        self._create_mode = False
        self._create_class = "Car"

        # Visibility toggles
        self._show_boxes = True
        self._show_labels = True
        self._show_possible_objects = False
        self._possible_object_regions = []

        # Optional height (z) filter applied to displayed points (min, max)
        self._height_filter: Optional[Tuple[float, float]] = None
        self._roi_bounds = None
        self._display_filter = "__all__"

        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.model.objects_changed.connect(self.update)
        self.model.selection_changed.connect(lambda _uid: self.update())
        self.model.frame_loaded.connect(self._on_frame_loaded)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_points(self, points: np.ndarray) -> None:
        """Pass the current radar point cloud (master frame, at least xyz)."""
        self._points = points if points is not None else np.zeros((0, 3))
        self._update_possible_object_regions()
        self.update()

    def begin_create_mode(self, class_name: str = "Car") -> None:
        """Next left-click will initialise a new box at the cursor."""
        self._create_mode = True
        self._create_class = class_name
        self.setCursor(Qt.CrossCursor)
        self.status_message.emit(
            f"Click on the radar view to place a new {class_name} box "
            f"(press Esc to cancel)."
        )

    def cancel_create_mode(self) -> None:
        self._create_mode = False
        self.unsetCursor()

    def set_height_filter(self, z_min: Optional[float], z_max: Optional[float]) -> None:
        if z_min is None or z_max is None:
            self._height_filter = None
        else:
            self._height_filter = (float(z_min), float(z_max))
        self.update()

    def set_display_filter(self, filter_key: Optional[str]) -> None:
        self._display_filter = filter_key or "__all__"
        self.update()

    def set_roi_bounds(self, roi_bounds) -> None:
        self._roi_bounds = roi_bounds
        self.update()

    def set_show_boxes(self, on: bool) -> None:
        self._show_boxes = bool(on)
        self.update()

    def set_show_labels(self, on: bool) -> None:
        self._show_labels = bool(on)
        self.update()

    def set_show_possible_objects(self, on: bool) -> None:
        self._show_possible_objects = bool(on)
        self._update_possible_object_regions()
        self.update()

    def _box_is_visible(self, box: Box3D) -> bool:
        if self._display_filter == "__all__":
            return True
        if self._display_filter == "__selected__":
            return box.uid == self.model.selected_uid
        return box.class_name == self._display_filter

    def _visible_boxes(self) -> list[Box3D]:
        return [box for box in self.model.objects if self._box_is_visible(box)]

    def _update_possible_object_regions(self) -> None:
        pts = np.asarray(self._points if self._points is not None else np.zeros((0, 3)))
        self._possible_object_regions = []
        cfg = ClusterConfig(eps_m=1.5, min_samples=4, use_rcs_feature=True)
        if not self._show_possible_objects or pts.ndim != 2 or pts.shape[1] < 3:
            return

        finite = np.all(np.isfinite(pts[:, :3]), axis=1)
        candidates = pts[finite]
        if candidates.shape[0] < cfg.min_samples:
            return

        # Keep only strong returns (≥ 60th-percentile RCS) — weak reflections
        # are ground clutter or multipath, not real objects.
        if candidates.shape[1] > 3:
            rcs = candidates[:, 3]
            finite_rcs = rcs[np.isfinite(rcs)]
            if finite_rcs.size > 0:
                thresh = float(np.percentile(finite_rcs, 60.0))
                keep = np.isfinite(rcs) & (rcs >= thresh)
                candidates = candidates[keep]
        if candidates.shape[0] < cfg.min_samples:
            return

        # DBSCAN with tight eps — each physical object gets its own circle.
        # min_samples=4 and 60th-percentile RCS threshold ensure only high-
        # confidence, well-supported clusters are surfaced as candidates.
        feats = _cluster_features(candidates, cfg)
        labels = _dbscan_labels(feats, eps=1.0, min_samples=cfg.min_samples)

        regions = []
        xy = candidates[:, :2]
        for lbl in sorted(set(labels.tolist())):
            if lbl < 0:
                continue
            cluster = xy[labels == lbl]
            count = cluster.shape[0]
            if count < cfg.min_samples:
                continue
            center = np.median(cluster, axis=0)
            spread = np.linalg.norm(cluster - center[None, :], axis=1)
            radius = max(float(np.percentile(spread, 85.0)) + 0.4, 0.9)
            regions.append((center, radius, count))

        # Keep only the top-N most-populated clusters; cap at 12 so the canvas
        # shows distinct, high-confidence hints rather than a sea of circles.
        self._possible_object_regions = sorted(
            regions, key=lambda item: item[2], reverse=True)[:12]

    # ------------------------------------------------------------------
    # Coordinate transforms (master <-> screen)
    # ------------------------------------------------------------------
    def _master_to_screen(self, xy: np.ndarray) -> QPointF:
        w = self.width()
        h = self.height()
        # master x (forward) goes UP on screen; master y (left) goes LEFT
        dx = xy[0] - self._center_master[0]
        dy = xy[1] - self._center_master[1]
        sx = w / 2.0 - dy * self._scale
        sy = h / 2.0 - dx * self._scale
        return QPointF(sx, sy)

    def _screen_to_master(self, pt: QPointF) -> np.ndarray:
        w = self.width()
        h = self.height()
        dy = -(pt.x() - w / 2.0) / self._scale
        dx = -(pt.y() - h / 2.0) / self._scale
        return np.array([self._center_master[0] + dx,
                         self._center_master[1] + dy])

    # ------------------------------------------------------------------
    # Hit-testing
    # ------------------------------------------------------------------
    def _box_footprint_master(self, box: Box3D) -> np.ndarray:
        """Return the 4 ground-plane corners (master xy) in order:
           [rear-right, front-right, front-left, rear-left]."""
        l2, w2 = box.length / 2.0, box.width / 2.0
        local = np.array([[-l2, -w2], [ l2, -w2], [ l2,  w2], [-l2,  w2]])
        c, s = np.cos(box.yaw), np.sin(box.yaw)
        R = np.array([[c, -s], [s, c]])
        return (R @ local.T).T + np.array([box.x, box.y])

    def _point_in_box_xy(self, pt_master: np.ndarray, box: Box3D) -> bool:
        c, s = np.cos(-box.yaw), np.sin(-box.yaw)
        R = np.array([[c, -s], [s, c]])
        local = R @ (pt_master - np.array([box.x, box.y]))
        return (abs(local[0]) <= box.length / 2.0 and
                abs(local[1]) <= box.width / 2.0)

    @staticmethod
    def _dist_point_to_segment_sq(px: float, py: float, ax: float, ay: float,
                                   bx: float, by: float) -> float:
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-12:
            return apx * apx + apy * apy
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
        qx, qy = ax + t * abx, ay + t * aby
        dx, dy = px - qx, py - qy
        return dx * dx + dy * dy

    def _hit_test_heading_rotate(
        self, screen_pt: QPointF, box: Box3D, master_pt: np.ndarray
    ) -> bool:
        """True if click is near the heading segment (not on box centre — avoids steal)."""
        c, s = np.cos(-box.yaw), np.sin(-box.yaw)
        R = np.array([[c, -s], [s, c]])
        local = R @ (master_pt - np.array([box.x, box.y]))
        if local[0] < box.length * 0.12:
            return False
        center = self._master_to_screen(np.array([box.x, box.y]))
        fwd_world = np.array([
            box.x + np.cos(box.yaw) * box.length * 0.55,
            box.y + np.sin(box.yaw) * box.length * 0.55,
        ])
        tip = self._master_to_screen(fwd_world)
        # Along full heading line to front face midpoint for a fatter target
        front_mid = self._master_to_screen(np.array([
            box.x + np.cos(box.yaw) * box.length * 0.5,
            box.y + np.sin(box.yaw) * box.length * 0.5,
        ]))
        thr = 10.0  # px
        d1 = self._dist_point_to_segment_sq(
            screen_pt.x(), screen_pt.y(),
            center.x(), center.y(), front_mid.x(), front_mid.y(),
        )
        d2 = self._dist_point_to_segment_sq(
            screen_pt.x(), screen_pt.y(),
            front_mid.x(), front_mid.y(), tip.x(), tip.y(),
        )
        return min(d1, d2) <= thr * thr

    def _hit_test_corner(self, screen_pt: QPointF, box: Box3D) -> Optional[int]:
        """Return corner index 0..3 if near a footprint corner (screen px)."""
        footprint = self._box_footprint_master(box)
        thr = 11.0
        thr2 = thr * thr
        for i, fp in enumerate(footprint):
            sc = self._master_to_screen(fp)
            dx = screen_pt.x() - sc.x()
            dy = screen_pt.y() - sc.y()
            if dx * dx + dy * dy <= thr2:
                return i
        return None

    def _hit_test(self, screen_pt: QPointF) -> Tuple[Optional[Box3D], Optional[str]]:
        """Return (box, edge) where edge is 'front'/'back'/'left'/'right' if
        the click is near a box edge (for resizing), else None.
        Special edges: 'rotate_heading' (near heading arrow), 'corner0'..'corner3'."""
        master_pt = self._screen_to_master(screen_pt)
        # Prefer selected box if still hit
        sel = self.model.selected()
        ordered = []
        if sel is not None and self._box_is_visible(sel):
            ordered.append(sel)
        ordered += [b for b in self._visible_boxes() if b is not sel]

        edge_threshold_m = 12.0 / self._scale  # ~12 px — easier edge grabs

        for box in ordered:
            # Corners first (small targets), then heading arrow, then edges / body.
            ci = self._hit_test_corner(screen_pt, box)
            if ci is not None:
                return box, f"corner{ci}"
            if not self._point_in_box_xy(master_pt, box):
                # Near a side from outside the footprint (still resizable in one click)
                c, s = np.cos(-box.yaw), np.sin(-box.yaw)
                R = np.array([[c, -s], [s, c]])
                local = R @ (master_pt - np.array([box.x, box.y]))
                dx = abs(local[0]) - box.length / 2.0
                dy = abs(local[1]) - box.width / 2.0
                if (dx < edge_threshold_m and abs(local[1]) < box.width / 2.0 + edge_threshold_m):
                    edge = 'front' if local[0] > 0 else 'back'
                    return box, edge
                if (dy < edge_threshold_m and abs(local[0]) < box.length / 2.0 + edge_threshold_m):
                    edge = 'left' if local[1] > 0 else 'right'
                    return box, edge
                if self._hit_test_heading_rotate(screen_pt, box, master_pt):
                    return box, "rotate_heading"
                continue
            # Inside: check edge proximity for resize
            c, s = np.cos(-box.yaw), np.sin(-box.yaw)
            R = np.array([[c, -s], [s, c]])
            local = R @ (master_pt - np.array([box.x, box.y]))
            d_front = box.length / 2.0 - local[0]
            d_back = local[0] + box.length / 2.0
            d_left = box.width / 2.0 - local[1]
            d_right = local[1] + box.width / 2.0
            min_d = min(d_front, d_back, d_left, d_right)
            if min_d < edge_threshold_m:
                if min_d == d_front: return box, 'front'
                if min_d == d_back:  return box, 'back'
                if min_d == d_left:  return box, 'left'
                if min_d == d_right: return box, 'right'
            if self._hit_test_heading_rotate(screen_pt, box, master_pt):
                return box, "rotate_heading"
            return box, None
        return None, None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_frame_loaded(self) -> None:
        # Auto-fit view to radar extent
        if self._points.size > 0:
            xs = self._points[:, 0]
            ys = self._points[:, 1]
            x_center = float((xs.min() + xs.max()) / 2.0)
            y_center = float((ys.min() + ys.max()) / 2.0)
            self._center_master = np.array([x_center, y_center])
            span = max(xs.max() - xs.min(), ys.max() - ys.min(), 20.0)
            self._scale = 0.8 * min(self.width(), self.height()) / span
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def paintEvent(self, ev: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        # Dark BEV canvas (sits inside the light panel for contrast)
        p.fillRect(self.rect(), self.BG_COLOR)

        self._draw_grid_with_ticks(p)
        self._draw_ego_marker(p)
        self._draw_points(p)
        self._draw_possible_object_regions(p)
        self._draw_roi(p)
        if self._show_boxes:
            self._draw_boxes(p)
        self._draw_bev_label(p)
        self._draw_axis_indicator(p)
        self._draw_scale_bar(p)

    def _draw_bev_label(self, p: QPainter) -> None:
        """Top-left title badge."""
        p.setPen(QColor(229, 231, 235))
        f = QFont(); f.setPointSize(10); f.setBold(True)
        p.setFont(f)
        p.drawText(14, 22, "BEV (Top-Down)")

    def _draw_axis_indicator(self, p: QPainter) -> None:
        """Coord-axis triad in bottom-left: red X forward, green Y left."""
        cx, cy = 30, self.height() - 30
        # X axis (red, points right on screen = forward in BEV view)
        p.setPen(QPen(QColor(239, 68, 68), 2))
        p.drawLine(QPointF(cx, cy), QPointF(cx + 26, cy))
        p.drawLine(QPointF(cx + 26, cy), QPointF(cx + 20, cy - 4))
        p.drawLine(QPointF(cx + 26, cy), QPointF(cx + 20, cy + 4))
        p.setPen(QColor(239, 68, 68))
        f = QFont(); f.setPointSize(8); f.setBold(True); p.setFont(f)
        p.drawText(QPointF(cx + 30, cy + 4), "X")
        # Y axis (green, points up on screen = left in BEV view)
        p.setPen(QPen(QColor(34, 197, 94), 2))
        p.drawLine(QPointF(cx, cy), QPointF(cx, cy - 26))
        p.drawLine(QPointF(cx, cy - 26), QPointF(cx - 4, cy - 20))
        p.drawLine(QPointF(cx, cy - 26), QPointF(cx + 4, cy - 20))
        p.setPen(QColor(34, 197, 94))
        p.drawText(QPointF(cx - 10, cy - 30), "Y")

    def _draw_grid_with_ticks(self, p: QPainter) -> None:
        """Grid lines with numeric axis labels.

        Grid step and label frequency are chosen dynamically so labels stay
        at least ~50 px apart and never land in reserved overlay zones
        (title top-left, axis-indicator bottom-left, scale-bar bottom-right).
        """
        w, h = self.width(), self.height()
        tl = self._screen_to_master(QPointF(0, 0))
        br = self._screen_to_master(QPointF(w, h))
        x_min, x_max = min(tl[0], br[0]), max(tl[0], br[0])
        y_min, y_max = min(tl[1], br[1]), max(tl[1], br[1])

        # Pick the smallest "nice" step that keeps neighbouring labels ≥ 50 px apart.
        _NICE = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
        _MIN_PX = 50.0
        step = 10.0
        for s in _NICE:
            if s * self._scale >= _MIN_PX:
                step = s
                break
        minor_step = step / 2.0

        # Minor grid (subtle)
        minor_pen = QPen(QColor(35, 42, 55), 1)
        p.setPen(minor_pen)
        x0 = np.floor(x_min / minor_step) * minor_step
        while x0 <= x_max:
            p.drawLine(self._master_to_screen(np.array([x0, y_min])),
                       self._master_to_screen(np.array([x0, y_max])))
            x0 += minor_step
        y0 = np.floor(y_min / minor_step) * minor_step
        while y0 <= y_max:
            p.drawLine(self._master_to_screen(np.array([x_min, y0])),
                       self._master_to_screen(np.array([x_max, y0])))
            y0 += minor_step

        # Major grid lines
        major_pen = QPen(self.GRID_COLOR, 1)
        p.setPen(major_pen)
        x0 = np.floor(x_min / step) * step
        while x0 <= x_max:
            p.drawLine(self._master_to_screen(np.array([x0, y_min])),
                       self._master_to_screen(np.array([x0, y_max])))
            x0 += step
        y0 = np.floor(y_min / step) * step
        while y0 <= y_max:
            p.drawLine(self._master_to_screen(np.array([x_min, y0])),
                       self._master_to_screen(np.array([x_max, y0])))
            y0 += step

        # ── Axis labels ───────────────────────────────────────────────────
        # Reserved screen zones (pixels) where we must NOT draw tick labels:
        #   top-left title    : x < 170, y < 32
        #   bottom-left axis  : x < 72,  y > h - 68
        #   bottom-right scale: x > w - 145
        #   bottom strip      : y > h - 6  (too close to edge)
        label_font = QFont(); label_font.setPointSize(8)
        p.setFont(label_font)
        fm = p.fontMetrics()

        label_color = QColor(156, 163, 175)

        # X-axis labels along the bottom edge (y = h - 10)
        last_sx = -9999.0
        x0 = np.floor(x_min / step) * step
        while x0 <= x_max:
            sx = float(self._master_to_screen(np.array([x0, 0])).x())
            text = f"{x0:.0f}"
            tw = fm.horizontalAdvance(text)
            lx = sx - tw / 2.0
            # Skip if: too close to last label, in scale-bar zone, or near left edge
            if (sx - last_sx >= _MIN_PX and
                    lx + tw < w - 140 and
                    lx > 8):
                p.setPen(label_color)
                p.drawText(QPointF(lx, h - 10), text)
                last_sx = sx
            x0 += step

        # X-axis caption centred at bottom (clear of scale bar and axis indicator)
        p.setPen(QColor(100, 116, 139))
        f_cap = QFont(); f_cap.setPointSize(8); f_cap.setItalic(True); p.setFont(f_cap)
        cap_x = "X (fwd, m)"
        cap_xw = fm.horizontalAdvance(cap_x)
        p.drawText(QPointF(w / 2.0 - cap_xw / 2.0, h - 24), cap_x)

        # Y-axis labels along the left edge (x = 6)
        p.setFont(label_font)
        last_sy = 9999.0
        y0 = np.floor(y_min / step) * step
        while y0 <= y_max:
            sy = float(self._master_to_screen(np.array([0, y0])).y())
            text = f"{y0:.0f}"
            # Skip if: too close to last, in title zone (top), in axis-indicator zone (bottom)
            if (abs(sy - last_sy) >= _MIN_PX and
                    sy > 32 and
                    sy < h - 68):
                p.setPen(label_color)
                p.drawText(QPointF(6, sy + 4), text)
                last_sy = sy
            y0 += step

        # Y-axis caption rotated at the left side
        p.setPen(QColor(100, 116, 139))
        f_cap2 = QFont(); f_cap2.setPointSize(8); f_cap2.setItalic(True); p.setFont(f_cap2)
        p.save()
        p.translate(16, h / 2.0 + 28)
        p.rotate(-90)
        p.drawText(QPointF(0, 0), "Y (left, m)")
        p.restore()

    def _draw_ego_marker(self, p: QPainter) -> None:
        origin = self._master_to_screen(np.array([0.0, 0.0]))
        # A small vehicle icon at ego origin
        pen = QPen(QColor(229, 231, 235), 1.5)
        p.setPen(pen)
        p.setBrush(QBrush(QColor(229, 231, 235, 40)))
        r = 6.0
        p.drawEllipse(origin, r, r)
        p.setPen(QPen(QColor(229, 231, 235), 1.5))
        fwd = self._master_to_screen(np.array([3.0, 0.0]))
        p.drawLine(origin, fwd)
        # Arrowhead
        p.drawLine(fwd, self._master_to_screen(np.array([2.3, 0.6])))
        p.drawLine(fwd, self._master_to_screen(np.array([2.3, -0.6])))

    def _draw_points(self, p: QPainter) -> None:
        if self._points.size == 0:
            p.setPen(QColor(156, 163, 175))
            p.drawText(14, 44, "No radar points loaded.")
            return

        pts = self._points
        # Apply height filter if set
        if self._height_filter is not None and pts.shape[1] >= 3:
            zmin, zmax = self._height_filter
            mask = (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
            pts = pts[mask]

        if pts.size == 0:
            return

        zs = pts[:, 2] if pts.shape[1] >= 3 else np.zeros(len(pts))
        z_min, z_max = float(zs.min()), float(zs.max())
        z_range = max(z_max - z_min, 1e-3)

        for i in range(pts.shape[0]):
            x, y = pts[i, 0], pts[i, 1]
            z = zs[i] if zs.size else 0.0
            t = (z - z_min) / z_range
            # Violet -> green -> yellow ramp
            if t < 0.5:
                k = t * 2
                r = int((1 - k) * self.POINT_CMAP_LOW.red()   + k * self.POINT_CMAP_MID.red())
                g = int((1 - k) * self.POINT_CMAP_LOW.green() + k * self.POINT_CMAP_MID.green())
                b = int((1 - k) * self.POINT_CMAP_LOW.blue()  + k * self.POINT_CMAP_MID.blue())
            else:
                k = (t - 0.5) * 2
                r = int((1 - k) * self.POINT_CMAP_MID.red()   + k * self.POINT_CMAP_HIGH.red())
                g = int((1 - k) * self.POINT_CMAP_MID.green() + k * self.POINT_CMAP_HIGH.green())
                b = int((1 - k) * self.POINT_CMAP_MID.blue()  + k * self.POINT_CMAP_HIGH.blue())
            color = QColor(r, g, b)
            p.setPen(QPen(color, 1.5))
            pt = self._master_to_screen(np.array([x, y]))
            p.drawPoint(pt)

    def _draw_possible_object_regions(self, p: QPainter) -> None:
        if not self._show_possible_objects or not self._possible_object_regions:
            return
        p.setBrush(QBrush(QColor(245, 158, 11, 32)))
        p.setPen(QPen(QColor(251, 191, 36, 220), 2, Qt.DashLine))
        for center, radius_m, count in self._possible_object_regions:
            screen_center = self._master_to_screen(center)
            radius_px = max(5.0, float(radius_m) * self._scale)
            p.drawEllipse(screen_center, radius_px, radius_px)
            p.setPen(QPen(QColor(253, 230, 138), 1))
            f = QFont()
            f.setPointSize(8)
            f.setBold(True)
            p.setFont(f)
            p.drawText(screen_center + QPointF(radius_px + 4, -4), f"{count}")
            p.setPen(QPen(QColor(251, 191, 36, 220), 2, Qt.DashLine))

    def _draw_roi(self, p: QPainter) -> None:
        if self._roi_bounds is None:
            return
        x_min, x_max, y_min, y_max, _z_min, _z_max = self._roi_bounds
        x0, x1 = min(x_min, x_max), max(x_min, x_max)
        y0, y1 = min(y_min, y_max), max(y_min, y_max)
        corners = [
            np.array([x0, y0]),
            np.array([x1, y0]),
            np.array([x1, y1]),
            np.array([x0, y1]),
        ]
        from PySide6.QtGui import QPolygonF
        pts = [self._master_to_screen(c) for c in corners]
        p.setPen(QPen(QColor(56, 189, 248), 2, Qt.DashLine))
        p.setBrush(QBrush(QColor(56, 189, 248, 22)))
        p.drawPolygon(QPolygonF(pts))
        p.setPen(QColor(186, 230, 253))
        f = QFont()
        f.setPointSize(8)
        f.setBold(True)
        p.setFont(f)
        anchor = self._master_to_screen(np.array([x1, y1]))
        p.drawText(anchor + QPointF(6, -6), "ROI")

    def _draw_boxes(self, p: QPainter) -> None:
        for box in self._visible_boxes():
            footprint = self._box_footprint_master(box)
            is_selected = (box.uid == self.model.selected_uid)
            color = CLASS_COLORS.get(box.class_name, self.DEFAULT_COLOR)
            pen = QPen(color, 3 if is_selected else 2)
            p.setPen(pen)
            brush = QBrush(QColor(color.red(), color.green(), color.blue(), 45))
            p.setBrush(brush)

            pts = [self._master_to_screen(fp) for fp in footprint]
            from PySide6.QtGui import QPolygonF
            poly = QPolygonF(pts)
            p.drawPolygon(poly)

            # Heading indicator
            center = self._master_to_screen(np.array([box.x, box.y]))
            fwd_world = np.array([
                box.x + np.cos(box.yaw) * box.length / 2.0,
                box.y + np.sin(box.yaw) * box.length / 2.0
            ])
            fwd_screen = self._master_to_screen(fwd_world)
            p.setPen(QPen(color, 2))
            p.drawLine(center, fwd_screen)

            # Corner handles for the selected box
            if is_selected:
                p.setBrush(QBrush(QColor(255, 255, 255)))
                p.setPen(QPen(color, 1))
                hs = 5.0
                for corner in pts:
                    p.drawRect(
                        QRectF(corner.x() - hs, corner.y() - hs, 2 * hs, 2 * hs)
                    )
                # Edge midpoints (resize targets)
                for i in range(4):
                    a, b = pts[i], pts[(i + 1) % 4]
                    mx = (a.x() + b.x()) * 0.5
                    my = (a.y() + b.y()) * 0.5
                    p.setBrush(QBrush(QColor(255, 255, 255, 220)))
                    p.drawEllipse(QPointF(mx, my), 4.0, 4.0)

            # Label badge above the box
            if self._show_labels:
                label_text = f"{box.class_name} (ID: {box.object_id})"
                # Find top-right-ish anchor
                anchor_master = np.array([box.x + box.length * 0.2,
                                          box.y + box.width * 0.8])
                anchor = self._master_to_screen(anchor_master)
                self._draw_label_badge(p, anchor, label_text, color)

    def _draw_label_badge(self, p: QPainter, pt: QPointF, text: str,
                          color: QColor) -> None:
        font = QFont(); font.setPointSize(8); font.setBold(True)
        p.setFont(font)
        fm = p.fontMetrics()
        pad_x, pad_y = 6, 3
        text_w = fm.horizontalAdvance(text)
        text_h = fm.height()
        rect = QRectF(pt.x() - text_w / 2 - pad_x, pt.y() - text_h - pad_y * 2,
                      text_w + pad_x * 2, text_h + pad_y * 2)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(color))
        p.drawRoundedRect(rect, 3, 3)
        p.setPen(QColor(255, 255, 255))
        p.drawText(QPointF(rect.x() + pad_x, rect.y() + text_h),
                   text)

    def _draw_scale_bar(self, p: QPainter) -> None:
        """Bottom-right '10 m' scale indicator."""
        length_m = 10.0
        length_px = length_m * self._scale
        y = self.height() - 30
        x1 = self.width() - 30
        x0 = x1 - length_px
        # Rounded background
        pad = 10
        bg_rect = QRectF(x0 - pad, y - 20, length_px + pad * 2, 32)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(31, 41, 55, 200)))
        p.drawRoundedRect(bg_rect, 6, 6)
        p.setPen(QPen(QColor(229, 231, 235), 2))
        p.drawLine(QPointF(x0, y), QPointF(x1, y))
        p.drawLine(QPointF(x0, y - 4), QPointF(x0, y + 4))
        p.drawLine(QPointF(x1, y - 4), QPointF(x1, y + 4))
        p.setPen(QColor(229, 231, 235))
        f = QFont(); f.setPointSize(9); f.setBold(True); p.setFont(f)
        fm = p.fontMetrics()
        label = f"{length_m:.0f} m"
        p.drawText(QPointF(x0 + (length_px - fm.horizontalAdvance(label)) / 2,
                           y - 6), label)

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def wheelEvent(self, ev: QWheelEvent) -> None:
        # Zoom around mouse position
        old_master = self._screen_to_master(ev.position())
        factor = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self._scale = max(0.5, min(200.0, self._scale * factor))
        new_master = self._screen_to_master(ev.position())
        # Adjust center so the point under cursor stays under cursor
        self._center_master += (old_master - new_master)
        self.update()

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        self.setFocus()
        self._last_mouse = ev.position()

        # Creation mode takes priority
        if self._create_mode and ev.button() == Qt.LeftButton:
            master_pt = self._screen_to_master(ev.position())
            from ..core.geometry import Box3D
            new_box = Box3D(
                x=float(master_pt[0]), y=float(master_pt[1]), z=0.8,
                class_name=self._create_class,
            )
            l, w, h = class_default_box_size(self._create_class)
            new_box.length, new_box.width, new_box.height = l, w, h
            new_box.z = h / 2.0
            self.model.add(new_box)
            self.cancel_create_mode()
            self.status_message.emit(f"Created {self._create_class} #{new_box.object_id}")
            return

        if ev.button() == Qt.MiddleButton:
            self._mode = _MODE_PAN
            return

        box, edge = self._hit_test(ev.position())

        if ev.button() == Qt.LeftButton:
            if box is None:
                self.model.select(None)
                return
            self.model.select(box.uid)
            self._drag_uid = box.uid
            import copy
            self._drag_start_box = copy.deepcopy(box)
            self._drag_start_master = self._screen_to_master(ev.position())
            if edge == "rotate_heading":
                self._mode = _MODE_ROTATE
                self._resize_edge = None
                self._corner_idx = None
                self._corner_fix_master = None
            elif edge is not None and edge.startswith("corner"):
                self._mode = _MODE_CORNER
                self._corner_idx = int(edge.replace("corner", ""))
                foot = self._box_footprint_master(self._drag_start_box)
                self._corner_fix_master = foot[(self._corner_idx + 2) % 4].copy()
                self._resize_edge = None
            elif edge in ("front", "back", "left", "right"):
                self._mode = _MODE_RESIZE
                self._resize_edge = edge
                self._corner_idx = None
                self._corner_fix_master = None
            else:
                self._mode = _MODE_TRANSLATE
                self._resize_edge = None
                self._corner_idx = None
                self._corner_fix_master = None
            return

        if ev.button() == Qt.RightButton:
            if box is None:
                return
            self.model.select(box.uid)
            self._drag_uid = box.uid
            import copy
            self._drag_start_box = copy.deepcopy(box)
            self._drag_start_master = self._screen_to_master(ev.position())
            self._mode = _MODE_ROTATE
            return

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._mode == _MODE_PAN:
            delta = ev.position() - self._last_mouse
            self._center_master[0] += delta.y() / self._scale
            self._center_master[1] += delta.x() / self._scale
            self._last_mouse = ev.position()
            self.update()
            return

        if self._mode in (_MODE_TRANSLATE, _MODE_ROTATE, _MODE_RESIZE, _MODE_CORNER):
            if self._drag_uid is None or self._drag_start_box is None:
                return
            box = self.model.find(self._drag_uid)
            if box is None:
                return
            cur_master = self._screen_to_master(ev.position())

            if self._mode == _MODE_TRANSLATE:
                dx = cur_master[0] - self._drag_start_master[0]
                dy = cur_master[1] - self._drag_start_master[1]
                box.x = self._drag_start_box.x + dx
                box.y = self._drag_start_box.y + dy
            elif self._mode == _MODE_ROTATE:
                # Angle from box center to current cursor, relative to start
                v0 = self._drag_start_master - np.array(
                    [self._drag_start_box.x, self._drag_start_box.y])
                v1 = cur_master - np.array([box.x, box.y])
                a0 = np.arctan2(v0[1], v0[0])
                a1 = np.arctan2(v1[1], v1[0])
                box.yaw = self._drag_start_box.yaw + (a1 - a0)
            elif self._mode == _MODE_RESIZE:
                # Transform cursor into box-local frame of the START pose
                sb = self._drag_start_box
                c, s = np.cos(-sb.yaw), np.sin(-sb.yaw)
                R = np.array([[c, -s], [s, c]])
                local_cur = R @ (cur_master - np.array([sb.x, sb.y]))
                # Adjust the appropriate half-extent
                if self._resize_edge == 'front':
                    new_l = max(0.2, local_cur[0] + sb.length / 2.0)
                    # Keep opposite edge fixed
                    shift = (new_l - sb.length) / 2.0
                    box.length = new_l
                    # Move center forward in box-local x by shift
                    box.x = sb.x + np.cos(sb.yaw) * shift
                    box.y = sb.y + np.sin(sb.yaw) * shift
                elif self._resize_edge == 'back':
                    new_l = max(0.2, sb.length / 2.0 - local_cur[0])
                    shift = -(new_l - sb.length) / 2.0
                    box.length = new_l
                    box.x = sb.x + np.cos(sb.yaw) * shift
                    box.y = sb.y + np.sin(sb.yaw) * shift
                elif self._resize_edge == 'left':
                    new_w = max(0.2, local_cur[1] + sb.width / 2.0)
                    shift = (new_w - sb.width) / 2.0
                    box.width = new_w
                    # Perpendicular to heading (to the left)
                    box.x = sb.x + (-np.sin(sb.yaw)) * shift
                    box.y = sb.y + ( np.cos(sb.yaw)) * shift
                elif self._resize_edge == 'right':
                    new_w = max(0.2, sb.width / 2.0 - local_cur[1])
                    shift = -(new_w - sb.width) / 2.0
                    box.width = new_w
                    box.x = sb.x + (-np.sin(sb.yaw)) * shift
                    box.y = sb.y + ( np.cos(sb.yaw)) * shift
            elif self._mode == _MODE_CORNER and self._corner_fix_master is not None:
                sb = self._drag_start_box
                P_fix = self._corner_fix_master
                M = cur_master
                c = (P_fix + M) * 0.5
                cy, sy = np.cos(-sb.yaw), np.sin(-sb.yaw)
                R = np.array([[cy, -sy], [sy, cy]])
                local = R @ (M - c)
                box.length = max(0.2, 2.0 * abs(float(local[0])))
                box.width = max(0.2, 2.0 * abs(float(local[1])))
                box.x = float(c[0])
                box.y = float(c[1])
                box.yaw = sb.yaw

            # Push to model without polluting undo stack mid-drag
            self.model.update(box, snapshot=False)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if self._mode in (_MODE_TRANSLATE, _MODE_ROTATE, _MODE_RESIZE, _MODE_CORNER):
            # Commit one undo entry for this full drag
            self.model.commit()
        self._mode = _MODE_NONE
        self._drag_uid = None
        self._drag_start_box = None
        self._drag_start_master = None
        self._resize_edge = None
        self._corner_idx = None
        self._corner_fix_master = None

    def keyPressEvent(self, ev: QKeyEvent) -> None:
        if ev.key() == Qt.Key_Escape and self._create_mode:
            self.cancel_create_mode()
            self.status_message.emit("Create mode cancelled.")
            return

        sel = self.model.selected()
        if sel is None:
            super().keyPressEvent(ev)
            return

        import copy
        before = copy.deepcopy(sel)
        step = 0.2 if not (ev.modifiers() & Qt.ShiftModifier) else 0.05
        rot_step = np.deg2rad(5.0 if not (ev.modifiers() & Qt.ShiftModifier) else 1.5)
        changed = True

        k = ev.key()
        if k == Qt.Key_W:
            sel.x += step
        elif k == Qt.Key_S:
            sel.x -= step
        elif k == Qt.Key_A:
            sel.y += step
        elif k == Qt.Key_D:
            sel.y -= step
        elif k == Qt.Key_R:
            sel.z += step
        elif k == Qt.Key_F:
            sel.z -= step
        elif k == Qt.Key_Q:
            sel.yaw += rot_step
        elif k == Qt.Key_E:
            sel.yaw -= rot_step
        elif k == Qt.Key_Z:
            sel.length = max(0.2, sel.length - step)
        elif k == Qt.Key_X:
            sel.length += step
        elif k == Qt.Key_C:
            sel.width = max(0.2, sel.width - step)
        elif k == Qt.Key_V:
            sel.width += step
        elif k == Qt.Key_G:
            sel.z = sel.height / 2.0  # snap base to z=0
        else:
            changed = False

        if changed:
            self.model.update(sel, snapshot=True)
        else:
            super().keyPressEvent(ev)
