"""
3D perspective radar view.

Provides an orbitable view of the radar point cloud and annotated boxes so the
user can inspect the scene geometry from outside the strict top-down BEV.
"""
from __future__ import annotations

from typing import Optional, Tuple
import math
import numpy as np

from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QMouseEvent, QPaintEvent,
                           QPainter, QPen, QPolygonF, QWheelEvent)
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..core.annotation_model import AnnotationModel
from ..core.geometry import BOX_EDGES
from .annotation_classes import CLASS_COLORS

_BOX_FACE_FRONT = (1, 2, 6, 5)
_BOX_FACE_TOP = (4, 5, 6, 7)


class RadarPerspectiveView(QWidget):
    """Simple software-rendered 3D view of radar points and boxes."""

    status_message = Signal(str)

    BG_COLOR = QColor(17, 24, 39)
    GRID_COLOR = QColor(55, 65, 81)
    TEXT_COLOR = QColor(226, 232, 240)
    DEFAULT_COLOR = QColor(100, 116, 139)
    SELECTED_COLOR = QColor(255, 255, 255)

    def __init__(self, model: AnnotationModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._points = np.zeros((0, 3), dtype=np.float64)
        self._height_filter: Optional[Tuple[float, float]] = None
        self._roi_bounds = None
        self._show_grid = True
        self._show_axes = True
        self._show_boxes = True
        self._display_filter = "__all__"

        self._target = np.array([12.0, 0.0, 0.8], dtype=np.float64)
        self._distance = 42.0
        self._azimuth = math.radians(58.0)
        self._elevation = math.radians(24.0)

        self._drag_mode: Optional[str] = None
        self._last_mouse = QPointF()

        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)

        self.model.objects_changed.connect(self.update)
        self.model.selection_changed.connect(lambda _uid: self.update())
        self.model.frame_loaded.connect(self.auto_fit)

    def set_points(self, points: np.ndarray) -> None:
        self._points = (points if points is not None else np.zeros((0, 3))).astype(
            np.float64, copy=False
        )
        self.auto_fit()

    def set_height_filter(self, z_min: Optional[float], z_max: Optional[float]) -> None:
        if z_min is None or z_max is None:
            self._height_filter = None
        else:
            self._height_filter = (float(z_min), float(z_max))
        self.update()

    def set_show_grid(self, on: bool) -> None:
        self._show_grid = bool(on)
        self.update()

    def set_show_axes(self, on: bool) -> None:
        self._show_axes = bool(on)
        self.update()

    def set_show_boxes(self, on: bool) -> None:
        self._show_boxes = bool(on)
        self.update()

    def set_display_filter(self, filter_key: Optional[str]) -> None:
        self._display_filter = filter_key or "__all__"
        self.update()

    def set_roi_bounds(self, roi_bounds) -> None:
        self._roi_bounds = roi_bounds
        self.update()

    def _box_is_visible(self, box) -> bool:
        if self._display_filter == "__all__":
            return True
        if self._display_filter == "__selected__":
            return box.uid == self.model.selected_uid
        return box.class_name == self._display_filter

    def _visible_boxes(self) -> list:
        return [box for box in self.model.objects if self._box_is_visible(box)]

    def zoom_by(self, delta: int) -> None:
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._distance = max(8.0, min(180.0, self._distance / factor))
        self.update()

    def auto_fit(self) -> None:
        pts = self._filtered_points()
        xs = [0.0]
        ys = [0.0]
        zs = [0.0]
        if pts.size > 0:
            xs += [float(pts[:, 0].min()), float(pts[:, 0].max())]
            ys += [float(pts[:, 1].min()), float(pts[:, 1].max())]
            zs += [float(pts[:, 2].min()), float(pts[:, 2].max())]
        for box in self._visible_boxes():
            corners = box.corners()
            xs += [float(corners[:, 0].min()), float(corners[:, 0].max())]
            ys += [float(corners[:, 1].min()), float(corners[:, 1].max())]
            zs += [float(corners[:, 2].min()), float(corners[:, 2].max())]

        self._target = np.array(
            [
                0.5 * (min(xs) + max(xs)),
                0.5 * (min(ys) + max(ys)),
                max(0.5, 0.5 * (min(zs) + max(zs))),
            ],
            dtype=np.float64,
        )
        span = max(max(xs) - min(xs), max(ys) - min(ys), 18.0)
        self._distance = max(14.0, min(180.0, span * 1.35))
        self.update()

    def _filtered_points(self) -> np.ndarray:
        pts = self._points
        if pts.size == 0:
            return pts
        if self._height_filter is None or pts.shape[1] < 3:
            return pts
        z_min, z_max = self._height_filter
        mask = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
        return pts[mask]

    def _camera_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cos_e = math.cos(self._elevation)
        cam_pos = self._target + np.array(
            [
                self._distance * cos_e * math.cos(self._azimuth),
                self._distance * cos_e * math.sin(self._azimuth),
                self._distance * math.sin(self._elevation),
            ],
            dtype=np.float64,
        )
        forward = self._target - cam_pos
        forward /= max(np.linalg.norm(forward), 1e-6)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        right /= max(np.linalg.norm(right), 1e-6)
        up = np.cross(right, forward)
        up /= max(np.linalg.norm(up), 1e-6)
        return cam_pos, forward, right, up

    def _project_points(self, world_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if world_pts.size == 0:
            return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
        cam_pos, forward, right, up = self._camera_basis()
        rel = world_pts - cam_pos[None, :]
        cam_x = rel @ right
        cam_y = rel @ up
        cam_z = rel @ forward
        focal = 0.82 * min(max(self.width(), 1), max(self.height(), 1))
        safe_z = np.where(np.abs(cam_z) < 1e-6, 1e-6, cam_z)
        uv = np.column_stack(
            [
                self.width() * 0.5 + focal * cam_x / safe_z,
                self.height() * 0.5 - focal * cam_y / safe_z,
            ]
        )
        return uv, cam_z

    def _draw_badges(self, p: QPainter) -> None:
        p.setPen(self.TEXT_COLOR)
        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        p.setFont(title_font)
        p.drawText(14, 22, "3D Radar View")

        help_font = QFont()
        help_font.setPointSize(8)
        p.setFont(help_font)
        p.setPen(QColor(148, 163, 184))
        p.drawText(
            QRectF(12, self.height() - 28, self.width() - 24, 16),
            Qt.AlignLeft | Qt.AlignVCenter,
            "Left drag: orbit   Right drag: pan   Wheel: zoom",
        )

    def _draw_ground_grid(self, p: QPainter) -> None:
        p.setPen(QPen(self.GRID_COLOR, 1))
        x_center = float(self._target[0])
        y_center = float(self._target[1])
        x_min = math.floor((x_center - 32.0) / 5.0) * 5.0
        x_max = math.ceil((x_center + 32.0) / 5.0) * 5.0
        y_min = math.floor((y_center - 32.0) / 5.0) * 5.0
        y_max = math.ceil((y_center + 32.0) / 5.0) * 5.0

        for x in np.arange(x_min, x_max + 0.1, 5.0):
            pts = np.array([[x, y_min, 0.0], [x, y_max, 0.0]], dtype=np.float64)
            uv, depth = self._project_points(pts)
            if np.all(depth > 0.05):
                p.drawLine(QPointF(*uv[0]), QPointF(*uv[1]))
        for y in np.arange(y_min, y_max + 0.1, 5.0):
            pts = np.array([[x_min, y, 0.0], [x_max, y, 0.0]], dtype=np.float64)
            uv, depth = self._project_points(pts)
            if np.all(depth > 0.05):
                p.drawLine(QPointF(*uv[0]), QPointF(*uv[1]))

    def _draw_axes(self, p: QPainter) -> None:
        axis_len = 8.0
        axes = [
            ("X", QColor(239, 68, 68), np.array([[0.0, 0.0, 0.0], [axis_len, 0.0, 0.0]])),
            ("Y", QColor(34, 197, 94), np.array([[0.0, 0.0, 0.0], [0.0, axis_len, 0.0]])),
            ("Z", QColor(59, 130, 246), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, axis_len * 0.45]])),
        ]
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        p.setFont(font)
        for label, color, pts in axes:
            uv, depth = self._project_points(pts)
            if not np.all(depth > 0.05):
                continue
            start = QPointF(*uv[0])
            end = QPointF(*uv[1])
            p.setPen(QPen(color, 3, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(start, end)

            vx = end.x() - start.x()
            vy = end.y() - start.y()
            length = max(1e-6, math.hypot(vx, vy))
            vx /= length
            vy /= length
            hx, hy = -vy, vx
            head = 9.0
            p.drawLine(
                end,
                QPointF(end.x() - vx * head + hx * head * 0.45,
                        end.y() - vy * head + hy * head * 0.45),
            )
            p.drawLine(
                end,
                QPointF(end.x() - vx * head - hx * head * 0.45,
                        end.y() - vy * head - hy * head * 0.45),
            )
            p.setPen(color)
            p.drawText(QPointF(end.x() + 5, end.y() - 5), label)

    def _draw_points(self, p: QPainter) -> None:
        pts = self._filtered_points()
        if pts.size == 0:
            p.setPen(QColor(156, 163, 175))
            p.drawText(14, 42, "No radar points loaded.")
            return

        uv, depth = self._project_points(pts[:, :3])
        z_vals = pts[:, 2] if pts.shape[1] >= 3 else np.zeros(len(pts))
        z_min = float(np.min(z_vals)) if len(z_vals) else 0.0
        z_max = float(np.max(z_vals)) if len(z_vals) else 1.0
        z_span = max(z_max - z_min, 1e-6)

        order = np.argsort(depth)[::-1]
        for idx in order:
            if depth[idx] <= 0.05:
                continue
            x, y = uv[idx]
            if x < -4 or x > self.width() + 4 or y < -4 or y > self.height() + 4:
                continue
            t = (float(z_vals[idx]) - z_min) / z_span
            r = int(120 + 110 * t)
            g = int(90 + 140 * (1.0 - abs(t - 0.45) * 1.4))
            b = int(230 - 130 * t)
            radius = max(1.0, min(3.2, 4.4 - depth[idx] * 0.05))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(r, g, b, 170))
            p.drawEllipse(QPointF(x, y), radius, radius)

    def _draw_box_face(
        self,
        p: QPainter,
        color: QColor,
        corners_uv: np.ndarray,
        depths: np.ndarray,
        face_indices: tuple[int, ...],
        alpha: int,
    ) -> None:
        if not np.all(depths[list(face_indices)] > 0.05):
            return
        poly = QPolygonF([QPointF(*corners_uv[i]) for i in face_indices])
        fill = QColor(color)
        fill.setAlpha(alpha)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(fill))
        p.drawPolygon(poly)

    def _draw_boxes(self, p: QPainter) -> None:
        for box in self._visible_boxes():
            corners = box.corners()
            corners_uv, depths = self._project_points(corners)
            if np.all(depths <= 0.05):
                continue

            color = CLASS_COLORS.get(box.class_name, self.DEFAULT_COLOR)
            is_selected = box.uid == self.model.selected_uid

            self._draw_box_face(
                p, color, corners_uv, depths, _BOX_FACE_TOP, 48 if not is_selected else 72
            )
            front_fill = QColor(
                min(255, color.red() + 45),
                min(255, color.green() + 45),
                min(255, color.blue() + 25),
            )
            self._draw_box_face(
                p,
                front_fill,
                corners_uv,
                depths,
                _BOX_FACE_FRONT,
                105 if not is_selected else 135,
            )

            for i, j in BOX_EDGES:
                if depths[i] <= 0.05 or depths[j] <= 0.05:
                    continue
                width = 3 if {i, j} in ({1, 2}, {2, 6}, {6, 5}, {5, 1}) else (
                    2 if is_selected else 1
                )
                p.setPen(QPen(color, width))
                p.setBrush(Qt.NoBrush)
                p.drawLine(QPointF(*corners_uv[i]), QPointF(*corners_uv[j]))

            visible = corners_uv[depths > 0.05]
            if visible.size == 0:
                continue
            label_x = float(np.min(visible[:, 0]))
            label_y = float(np.min(visible[:, 1])) - 8.0
            text = f"{box.class_name} (ID: {box.object_id})"
            font = QFont()
            font.setPointSize(8)
            font.setBold(True)
            p.setFont(font)
            fm = p.fontMetrics()
            pad_x = 6
            pad_y = 3
            rect = QRectF(
                label_x,
                label_y - fm.height() - pad_y * 2,
                fm.horizontalAdvance(text) + pad_x * 2,
                fm.height() + pad_y * 2,
            )
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(color))
            p.drawRoundedRect(rect, 3, 3)
            p.setPen(QColor(255, 255, 255))
            p.drawText(QPointF(rect.x() + pad_x, rect.y() + fm.height()), text)

    def paintEvent(self, ev: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), self.BG_COLOR)
        if self._show_grid:
            self._draw_ground_grid(p)
        if self._show_axes:
            self._draw_axes(p)
        self._draw_points(p)
        self._draw_roi(p)
        if self._show_boxes:
            self._draw_boxes(p)
            self._draw_badges(p)

    def _draw_roi(self, p: QPainter) -> None:
        if self._roi_bounds is None:
            return
        x_min, x_max, y_min, y_max, z_min, z_max = self._roi_bounds
        x0, x1 = min(x_min, x_max), max(x_min, x_max)
        y0, y1 = min(y_min, y_max), max(y_min, y_max)
        z0 = 0.0 if z_min is None else float(min(z_min, z_max))
        z1 = 0.0 if z_max is None else float(max(z_min, z_max))
        bottom = np.array(
            [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],
            dtype=np.float64,
        )
        top = np.array(
            [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
            dtype=np.float64,
        )
        p.setPen(QPen(QColor(56, 189, 248), 2, Qt.DashLine))
        for ring in (bottom, top):
            uv, depth = self._project_points(ring)
            for i in range(4):
                j = (i + 1) % 4
                if depth[i] > 0.05 and depth[j] > 0.05:
                    p.drawLine(QPointF(*uv[i]), QPointF(*uv[j]))
        if z1 != z0:
            uv_b, depth_b = self._project_points(bottom)
            uv_t, depth_t = self._project_points(top)
            for i in range(4):
                if depth_b[i] > 0.05 and depth_t[i] > 0.05:
                    p.drawLine(QPointF(*uv_b[i]), QPointF(*uv_t[i]))

    def wheelEvent(self, ev: QWheelEvent) -> None:
        self.zoom_by(1 if ev.angleDelta().y() > 0 else -1)

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        self.setFocus()
        self._last_mouse = ev.position()
        if ev.button() == Qt.LeftButton:
            self._drag_mode = "orbit"
        elif ev.button() == Qt.RightButton:
            self._drag_mode = "pan"

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._drag_mode is None:
            return
        delta = ev.position() - self._last_mouse
        self._last_mouse = ev.position()

        if self._drag_mode == "orbit":
            self._azimuth -= delta.x() * 0.01
            self._elevation = max(
                math.radians(8.0),
                min(math.radians(80.0), self._elevation + delta.y() * 0.008),
            )
        elif self._drag_mode == "pan":
            _, forward, right, _up = self._camera_basis()
            flat_forward = np.array([forward[0], forward[1], 0.0], dtype=np.float64)
            if np.linalg.norm(flat_forward) < 1e-6:
                flat_forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            flat_forward /= max(np.linalg.norm(flat_forward), 1e-6)
            flat_right = np.array([right[0], right[1], 0.0], dtype=np.float64)
            if np.linalg.norm(flat_right) < 1e-6:
                flat_right = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            flat_right /= max(np.linalg.norm(flat_right), 1e-6)
            meters_per_px = max(self._distance / max(min(self.width(), self.height()), 1), 0.02)
            self._target += (-flat_right * delta.x() + flat_forward * delta.y()) * meters_per_px
        self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        self._drag_mode = None
