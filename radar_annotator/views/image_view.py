"""
Camera image view.

Displays the current frame image with projected 3D boxes and camera-side
editing handles for translate, yaw, length, width, and height.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import numpy as np

from PySide6.QtCore import QPointF, Qt, QRectF, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetrics, QImage,
                           QMouseEvent, QPaintEvent, QPainter, QPen, QPixmap,
                           QPolygonF)
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..core.annotation_model import AnnotationModel
from ..core.calibration import Calibration
from ..core.geometry import (BOX_EDGES, Box3D, box_2d_from_projection,
                             project_box_calibration, project_points_calibration,
                             radar_display_xyz_m, radar_range_from_x_m)
from .annotation_classes import CLASS_COLORS

_BOX_FACE_FRONT = (1, 2, 6, 5)
_BOX_FACES_FILL = (
    ("back", (0, 3, 7, 4)),
    ("bottom", (0, 1, 2, 3)),
    ("top", (4, 5, 6, 7)),
    ("right", (0, 1, 5, 4)),
    ("left", (3, 2, 6, 7)),
    ("front", _BOX_FACE_FRONT),
)
_FRONT_EDGE_SET = frozenset(
    {
        frozenset((1, 2)),
        frozenset((2, 6)),
        frozenset((6, 5)),
        frozenset((5, 1)),
    }
)

_HANDLE_HIT_RADIUS = 12.0
# Drag: world metres spanned when moving one full image diagonal (uniform XY).
_DRAG_WORLD_METRES_ACROSS_DIAGONAL = 55.0
# Extra image pixels around projected box for hit-testing (fat finger / clipped edges).
_HIT_IMAGE_MARGIN = 14.0
# Additional margin when Manual placement is on (easier grab for camera-only edits).
_HIT_IMAGE_MARGIN_MANUAL_EXTRA = 22.0
# Arrow-key nudge in image pixels (scaled by _meters_per_image_pixel_drag).
_ARROW_NUDGE_IMAGE_PX = 8.0


@dataclass
class _HandleSpec:
    role: str
    point: QPointF
    world_dir: Optional[np.ndarray] = None
    axis_unit_px: Optional[np.ndarray] = None
    meters_per_px: Optional[float] = None


class CameraImageView(QWidget):
    """Fit-to-widget image display with projected 3D box overlays."""

    status_message = Signal(str)
    box_created_from_image = Signal(object)  # emits Box3D

    DEFAULT_COLOR = QColor(100, 116, 139)
    SELECTED_COLOR = QColor(255, 255, 255)

    def __init__(self, model: AnnotationModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._calibration: Calibration = Calibration.identity()
        self._image: Optional[QImage] = None
        self._image_size: Tuple[int, int] = (0, 0)

        self._img_scale = 1.0
        self._img_offset = QPointF(0, 0)

        self._show_2d_bbox = True
        self._show_wireframe = True
        self._show_labels = True
        self._show_box_metrics = True
        self._display_filter = "__all__"

        self._radar_points: Optional[np.ndarray] = None
        self._show_radar_projected = False

        # When True: image view ignores loaded radar→camera extrinsics for projection
        # (synthetic master↔display-camera swap, t=0); drag uses K + depth hint only.
        self._manual_placement_mode = False

        self._dragging = False
        self._drag_role: Optional[str] = None
        self._drag_uid: Optional[str] = None
        self._drag_start_box: Optional[Box3D] = None
        self._drag_start_mouse: Optional[QPointF] = None
        self._drag_handle: Optional[_HandleSpec] = None

        # Image-side box creation
        self._create_mode = False
        self._create_class = "Car"
        self._create_rect_start: Optional[QPointF] = None
        self._create_rect_end: Optional[QPointF] = None

        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)

        self.model.objects_changed.connect(self.update)
        self.model.selection_changed.connect(lambda _uid: self.update())

    def set_image(self, image_bgr_or_rgb: np.ndarray, is_rgb: bool = False) -> None:
        if image_bgr_or_rgb is None:
            self._image = None
            self._image_size = (0, 0)
            self.update()
            return
        arr = image_bgr_or_rgb
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        h, w, _ = arr.shape
        if not is_rgb:
            arr = arr[:, :, ::-1]
        arr = np.ascontiguousarray(arr)
        self._image = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._image_size = (w, h)
        self._recompute_fit()
        self.update()

    def set_calibration(self, calib: Calibration) -> None:
        self._calibration = calib
        self.update()

    def set_manual_placement_mode(self, on: bool) -> None:
        """Ignore loaded Tr_* for on-image geometry; use weak display extrinsics + real K."""
        self._manual_placement_mode = bool(on)
        self.update()

    def set_manual_image_adjust(self, on: bool) -> None:
        """Alias for toolbar compatibility."""
        self.set_manual_placement_mode(on)

    def _display_calibration(self) -> Calibration:
        """Calibration used for projection / hit-test / drag on this widget.

        Manual placement: same intrinsics ``K`` (scaled to current image size) but
        **no** loaded ``Tr_velo_to_cam`` — only a fixed axis swap (master x→display z)
        with **zero** translation, and no ``P_rect`` baseline column. Lets you move the
        box freely on the photo without wrong fusion extrinsics. Export / other panels
        still use the real calibration from the dataset.
        """
        if not self._manual_placement_mode:
            return self._calibration
        W, H = self._image_size
        if W <= 0 or H <= 0:
            W, H = int(self._calibration.image_size[0]), int(self._calibration.image_size[1])
        base = self._calibration
        if (W, H) != (int(base.image_size[0]), int(base.image_size[1])):
            kcal = base.rescaled_to_image_size((W, H))
        else:
            kcal = base
        Rm = np.array(
            [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        Tm = np.eye(4, dtype=np.float64)
        Tm[:3, :3] = Rm
        Tm[:3, 3] = 0.0
        return Calibration(
            kcal.K.copy(),
            Tm,
            (W, H),
            f"{kcal.id}:manual_view",
            P_rect=None,
        )

    def set_show_2d_bbox(self, on: bool) -> None:
        self._show_2d_bbox = on
        self.update()

    def set_show_wireframe(self, on: bool) -> None:
        self._show_wireframe = on
        self.update()

    def set_show_labels(self, on: bool) -> None:
        self._show_labels = on
        self.update()

    def set_show_box_metrics(self, on: bool) -> None:
        self._show_box_metrics = bool(on)
        self.update()

    def begin_create_mode(self, class_name: str = "Car") -> None:
        """Enter image-draw mode: user draws a 2D rect → creates a 3D box."""
        self._create_mode = True
        self._create_class = class_name
        self._create_rect_start = None
        self._create_rect_end = None
        self.setCursor(Qt.CrossCursor)
        self.status_message.emit(
            f"Draw a rectangle on the image to place a new {class_name} box "
            f"(press Esc to cancel)."
        )

    def cancel_create_mode(self) -> None:
        self._create_mode = False
        self._create_rect_start = None
        self._create_rect_end = None
        self.unsetCursor()
        self.update()

    def set_display_filter(self, filter_key: Optional[str]) -> None:
        self._display_filter = filter_key or "__all__"
        self.update()

    def set_radar_points(self, points_xyz: Optional[np.ndarray]) -> None:
        if points_xyz is None or points_xyz.size == 0:
            self._radar_points = None
            self.update()
            return
        pts = np.asarray(points_xyz, dtype=np.float64)
        if pts.shape[0] > 45000:
            pts = pts[:: max(1, pts.shape[0] // 45000)]
        self._radar_points = pts
        self.update()

    def set_show_radar_projected(self, on: bool) -> None:
        self._show_radar_projected = bool(on)
        self.update()

    def _recompute_fit(self) -> None:
        if self._image is None or self.width() <= 0 or self.height() <= 0:
            return
        iw, ih = self._image_size
        sx = self.width() / iw
        sy = self.height() / ih
        self._img_scale = min(sx, sy)
        draw_w = iw * self._img_scale
        draw_h = ih * self._img_scale
        self._img_offset = QPointF(
            (self.width() - draw_w) / 2.0,
            (self.height() - draw_h) / 2.0,
        )

    def resizeEvent(self, ev) -> None:
        self._recompute_fit()
        super().resizeEvent(ev)

    def leaveEvent(self, ev) -> None:
        if not self._dragging:
            self.unsetCursor()
        super().leaveEvent(ev)

    def _box_is_visible(self, box: Box3D) -> bool:
        if self._display_filter == "__all__":
            return True
        if self._display_filter == "__selected__":
            return box.uid == self.model.selected_uid
        return box.class_name == self._display_filter

    def _visible_boxes(self) -> list[Box3D]:
        return [box for box in self.model.objects if self._box_is_visible(box)]

    def _image_to_widget(self, uv: np.ndarray) -> QPointF:
        return QPointF(
            self._img_offset.x() + float(uv[0]) * self._img_scale,
            self._img_offset.y() + float(uv[1]) * self._img_scale,
        )

    def _widget_to_image(self, pt: QPointF) -> np.ndarray:
        return np.array(
            [
                (pt.x() - self._img_offset.x()) / max(self._img_scale, 1e-6),
                (pt.y() - self._img_offset.y()) / max(self._img_scale, 1e-6),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _raw_projected_bbox(
        corners_uv: np.ndarray, depths: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        """Axis-aligned bbox in image pixels from corners in front of camera (no clip)."""
        if np.all(depths <= 0.05):
            return None
        front = depths > 0.05
        if not np.any(front):
            return None
        uv = corners_uv[front]
        return (
            float(uv[:, 0].min()),
            float(uv[:, 1].min()),
            float(uv[:, 0].max()),
            float(uv[:, 1].max()),
        )

    def _meters_per_image_pixel_drag(self) -> float:
        """Uniform scale: drag tracks mouse 1:1 across the whole photo (weak coupling)."""
        W, H = self._image_size
        if W <= 0 or H <= 0:
            W, H = 640, 480
        diag = float(math.hypot(W, H))
        return _DRAG_WORLD_METRES_ACROSS_DIAGONAL / max(diag, 1.0)

    def _project_world_points(
        self, points_world: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return project_points_calibration(points_world, self._display_calibration())

    def _project_widget_point(self, world_point: np.ndarray) -> Optional[QPointF]:
        uv, depth = self._project_world_points(world_point.reshape(1, 3))
        if depth[0] <= 0.05:
            return None
        return self._image_to_widget(uv[0])

    def _box_basis(self, box: Box3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Rm = box.rotation_matrix()
        forward = Rm @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        left = Rm @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        up = Rm @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return forward, left, up

    def _axis_drag_mapping(
        self, box: Box3D, world_dir: np.ndarray, sample_m: float
    ) -> tuple[Optional[np.ndarray], Optional[float]]:
        """Map a world-space direction to (widget unit vector, metres per widget pixel).

        Uses several sample lengths so near-degenerate projections (common in manual
        display mode) still yield a usable axis instead of disabling resize entirely.
        """
        center = np.array([box.x, box.y, box.z], dtype=np.float64)
        wdir = np.asarray(world_dir, dtype=np.float64).reshape(3)
        wn = float(np.linalg.norm(wdir))
        if wn < 1e-9:
            return None, None
        wdir = wdir / wn
        base_sm = max(0.15, float(sample_m))
        for sm in (
            base_sm,
            max(0.4, base_sm * 2.0),
            max(0.8, base_sm * 3.5),
            1.2,
            2.0,
            3.5,
        ):
            pts = np.vstack([center, center + wdir * sm])
            uv, depth = self._project_world_points(pts)
            if np.any(depth <= 0.05):
                continue
            a = self._image_to_widget(uv[0])
            b = self._image_to_widget(uv[1])
            delta = np.array([b.x() - a.x(), b.y() - a.y()], dtype=np.float64)
            px = float(np.linalg.norm(delta))
            if px < 0.18:
                continue
            return delta / px, float(sm / px)
        return None, None

    def _axis_drag_manual_fallback(
        self, role: str, world_dir: np.ndarray, box: Box3D
    ) -> tuple[np.ndarray, float]:
        """When projection-based axis fails (manual placement): stable widget axes + mpp."""
        mpp = self._meters_per_image_pixel_drag() * 1.2
        cy, sy = math.cos(float(box.yaw)), math.sin(float(box.yaw))
        # Crude image-plane proxies for master-frame length / width when pinhole axis fails.
        if role == "resize_height_top":
            return np.array([0.0, -1.0], dtype=np.float64), mpp
        if role == "resize_height_bottom":
            return np.array([0.0, 1.0], dtype=np.float64), mpp
        def _n2(v: np.ndarray) -> np.ndarray:
            n = float(np.linalg.norm(v))
            return v / n if n >= 1e-6 else np.array([1.0, 0.0], dtype=np.float64)

        if role == "resize_length_front":
            return _n2(np.array([sy, -cy], dtype=np.float64)), mpp
        if role == "resize_length_back":
            return _n2(np.array([-sy, cy], dtype=np.float64)), mpp
        if role == "resize_width_left":
            return _n2(np.array([-cy, -sy], dtype=np.float64)), mpp
        if role == "resize_width_right":
            return _n2(np.array([cy, sy], dtype=np.float64)), mpp
        u = np.asarray(world_dir[:2], dtype=np.float64).reshape(2)
        nu = float(np.linalg.norm(u))
        if nu >= 1e-6:
            u = u / nu
            return np.array([u[0], -u[1]], dtype=np.float64), mpp
        return np.array([1.0, 0.0], dtype=np.float64), mpp

    def _control_handles(self, box: Box3D) -> Dict[str, _HandleSpec]:
        center = np.array([box.x, box.y, box.z], dtype=np.float64)
        forward, left, up = self._box_basis(box)

        samples = {
            "front": max(0.4, box.length * 0.25),
            "back": max(0.4, box.length * 0.25),
            "left": max(0.3, box.width * 0.5),
            "right": max(0.3, box.width * 0.5),
            "top": max(0.25, box.height * 0.35),
        }
        world_points = {
            "front": center + forward * (box.length * 0.5),
            "back": center - forward * (box.length * 0.5),
            "left": center + left * (box.width * 0.5),
            "right": center - left * (box.width * 0.5),
            "top": center + up * (box.height * 0.5),
            "bottom": center - up * (box.height * 0.5),
            "rotate": center + forward * (box.length * 0.5 + max(0.5, box.length * 0.18)),
        }

        handles: Dict[str, _HandleSpec] = {}
        role_defs = {
            "resize_length_front": ("front", forward, samples["front"]),
            "resize_length_back": ("back", -forward, samples["back"]),
            "resize_width_left": ("left", left, samples["left"]),
            "resize_width_right": ("right", -left, samples["right"]),
            "resize_height_top": ("top", up, samples["top"]),
            "resize_height_bottom": ("bottom", -up, samples["top"]),
        }

        for role, (key, world_dir, sample_m) in role_defs.items():
            point = self._project_widget_point(world_points[key])
            if point is None:
                continue
            axis_unit_px, meters_per_px = self._axis_drag_mapping(box, world_dir, sample_m)
            if axis_unit_px is None or meters_per_px is None:
                # Height handles: always allow vertical fallback (projection often degenerates).
                if role in ("resize_height_top", "resize_height_bottom") or self._manual_placement_mode:
                    axis_unit_px, meters_per_px = self._axis_drag_manual_fallback(
                        role, world_dir, box
                    )
                else:
                    continue
            handles[role] = _HandleSpec(
                role=role,
                point=point,
                world_dir=world_dir,
                axis_unit_px=axis_unit_px,
                meters_per_px=meters_per_px,
            )

        rotate_point = self._project_widget_point(world_points["rotate"])
        if rotate_point is not None:
            handles["rotate"] = _HandleSpec(role="rotate", point=rotate_point)
        return handles

    def _mean_face_depth_cam(self, box: Box3D, indices: Tuple[int, ...]) -> float:
        corners = box.corners()[list(indices)]
        homog = np.hstack([corners, np.ones((len(corners), 1))])
        cam = (self._display_calibration().T @ homog.T).T[:, :3]
        return float(np.mean(cam[:, 2]))

    def _face_poly_widget(
        self, indices: Tuple[int, ...], corners_uv: np.ndarray, depths: np.ndarray
    ) -> Optional[QPolygonF]:
        for i in indices:
            if depths[i] <= 0.05:
                return None
        return QPolygonF([self._image_to_widget(corners_uv[i]) for i in indices])

    def paintEvent(self, ev: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(17, 24, 39))

        if self._image is None:
            p.setPen(QColor(156, 163, 175))
            p.drawText(self.rect(), Qt.AlignCenter, "No image loaded.")
            return

        pix = QPixmap.fromImage(self._image)
        target_w = self._image_size[0] * self._img_scale
        target_h = self._image_size[1] * self._img_scale
        p.drawPixmap(
            QRectF(self._img_offset.x(), self._img_offset.y(), target_w, target_h),
            pix,
            QRectF(0, 0, self._image_size[0], self._image_size[1]),
        )

        if self._show_radar_projected and self._radar_points is not None:
            self._draw_radar_points_overlay(p)

        for box in self._visible_boxes():
            self._draw_box_overlay(p, box)

        # Draw in-progress creation rectangle
        if self._create_mode and self._create_rect_start is not None and self._create_rect_end is not None:
            pen = QPen(QColor(255, 255, 80), 2, Qt.DashLine)
            p.setPen(pen)
            p.setBrush(QBrush(QColor(255, 255, 80, 30)))
            rect = QRectF(self._create_rect_start, self._create_rect_end).normalized()
            p.drawRect(rect)

        footer_font = QFont()
        footer_font.setPointSize(8)
        fm_f = QFontMetrics(footer_font)
        line_calib = (
            f"calib: {self._calibration.id}"
            + ("  · view: extrinsics off" if self._manual_placement_mode else "")
        )
        line_img = (
            f"img: {self._image_size[0]}×{self._image_size[1]}"
            + ("  · manual-placement" if self._manual_placement_mode else "")
        )
        pad_x, pad_y = 10.0, 5.0
        # Bottom-right kept clear for axis gizmo (lines + X/Y labels extend ~55px below origin).
        right_clear = 104.0
        strip_h = float(max(34, fm_f.lineSpacing() * 2 + int(pad_y * 2)))
        w_px = float(self.width())
        h_px = float(self.height())
        strip_w = max(0.0, w_px - right_clear)
        strip_rect = QRectF(0.0, h_px - strip_h, strip_w, strip_h)
        p.fillRect(strip_rect, QColor(17, 24, 39, 236))

        text_max_w = max(32, int(strip_w - pad_x * 2))
        p.setFont(footer_font)
        p.setPen(QColor(203, 213, 225))
        y1 = h_px - strip_h + pad_y + fm_f.ascent()
        y2 = y1 + fm_f.lineSpacing()
        p.drawText(
            QPointF(pad_x, y1),
            fm_f.elidedText(line_calib, Qt.ElideMiddle, text_max_w),
        )
        p.drawText(
            QPointF(pad_x, y2),
            fm_f.elidedText(line_img, Qt.ElideMiddle, text_max_w),
        )

        # Entire gizmo sits above the footer strip so Y/X glyphs never stack on footer lines.
        self._draw_axis_indicator(p, footer_strip_h=int(strip_h))

    def _draw_axis_indicator(self, p: QPainter, footer_strip_h: int = 0) -> None:
        margin = 12
        ox = float(self.width()) - margin - 52
        oy = float(self.height()) - float(footer_strip_h) - margin - 46
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        p.setFont(font)

        p.setPen(QPen(QColor(239, 68, 68), 2))
        p.drawLine(QPointF(ox, oy), QPointF(ox + 24, oy))
        p.drawLine(QPointF(ox + 24, oy), QPointF(ox + 18, oy - 4))
        p.drawLine(QPointF(ox + 24, oy), QPointF(ox + 18, oy + 4))
        p.setPen(QColor(239, 68, 68))
        p.drawText(QPointF(ox + 28, oy + 4), "X")

        p.setPen(QPen(QColor(34, 197, 94), 2))
        p.drawLine(QPointF(ox, oy), QPointF(ox, oy + 24))
        p.drawLine(QPointF(ox, oy + 24), QPointF(ox - 4, oy + 18))
        p.drawLine(QPointF(ox, oy + 24), QPointF(ox + 4, oy + 18))
        p.setPen(QColor(34, 197, 94))
        p.drawText(QPointF(ox - 12, oy + 34), "Y")

    def _draw_radar_points_overlay(self, p: QPainter) -> None:
        pts = self._radar_points
        if pts is None or pts.shape[0] == 0:
            return
        uv, depth = project_points_calibration(pts[:, :3], self._display_calibration())
        p.save()
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        radius = max(1.2, 1.8 * self._img_scale * 0.35)
        intensities = None
        if pts.shape[1] >= 4:
            intensities = pts[:, 3]
            i_min = float(np.min(intensities))
            i_max = float(np.max(intensities))
            i_span = max(i_max - i_min, 1e-6)
        for i in range(uv.shape[0]):
            if depth[i] <= 0.05:
                continue
            if (
                uv[i, 0] < 0.0
                or uv[i, 0] >= self._image_size[0]
                or uv[i, 1] < 0.0
                or uv[i, 1] >= self._image_size[1]
            ):
                continue
            wpt = self._image_to_widget(uv[i])
            if not self.rect().contains(wpt.toPoint()):
                continue
            if intensities is not None:
                t = (float(intensities[i]) - i_min) / i_span
                color = QColor(255, int(80 + 175 * (1.0 - t)), int(200 + 55 * t), 140)
            else:
                color = QColor(250, 204, 21, 130)
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawEllipse(wpt, radius, radius)
        p.restore()

    def _draw_heading_arrow(self, p: QPainter, box: Box3D) -> None:
        forward, _left, _up = self._box_basis(box)
        base = np.array([box.x, box.y, box.z], dtype=np.float64)
        tip = base + forward * 0.75
        uv0, d0 = self._project_world_points(base.reshape(1, 3))
        uv1, d1 = self._project_world_points(tip.reshape(1, 3))
        if d0[0] <= 0.05 or d1[0] <= 0.05:
            return
        pa = self._image_to_widget(uv0[0])
        pb = self._image_to_widget(uv1[0])
        p.setPen(QPen(QColor(255, 255, 255, 230), 3, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(pa, pb)
        vx = pb.x() - pa.x()
        vy = pb.y() - pa.y()
        length = max(1e-3, math.hypot(vx, vy))
        vx /= length
        vy /= length
        hx, hy = -vy, vx
        size = 10.0
        p.drawLine(
            pb,
            QPointF(pb.x() - vx * size + hx * size * 0.45, pb.y() - vy * size + hy * size * 0.45),
        )
        p.drawLine(
            pb,
            QPointF(pb.x() - vx * size - hx * size * 0.45, pb.y() - vy * size - hy * size * 0.45),
        )

    def _draw_selected_handles(self, p: QPainter, box: Box3D, color: QColor) -> None:
        handles = self._control_handles(box)
        if not handles:
            return

        if "rotate" in handles and "resize_length_front" in handles:
            p.setPen(QPen(QColor(255, 255, 255, 150), 1.5, Qt.DashLine))
            p.drawLine(handles["resize_length_front"].point, handles["rotate"].point)

        for role, spec in handles.items():
            if role == "rotate":
                p.setPen(QPen(QColor(255, 255, 255), 2))
                p.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 180)))
                p.drawEllipse(spec.point, 7.0, 7.0)
                p.setBrush(QBrush(QColor(17, 24, 39)))
                p.drawEllipse(spec.point, 3.0, 3.0)
                continue

            if role in ("resize_height_top", "resize_height_bottom"):
                rect = QRectF(spec.point.x() - 5, spec.point.y() - 5, 10, 10)
                p.setPen(QPen(QColor(255, 255, 255), 1.5))
                p.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 220)))
                p.drawRect(rect)
            else:
                p.setPen(QPen(QColor(255, 255, 255), 1.5))
                p.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 220)))
                p.drawEllipse(spec.point, 5.2, 5.2)

    def _draw_box_overlay(self, p: QPainter, box: Box3D) -> None:
        corners_uv, depths = project_box_calibration(box, self._display_calibration())
        if np.all(depths <= 0.05):
            return

        is_selected = box.uid == self.model.selected_uid
        color = CLASS_COLORS.get(box.class_name, self.DEFAULT_COLOR)

        if self._show_wireframe:
            p.save()
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            faces_ranked = []
            for name, inds in _BOX_FACES_FILL:
                if name == "front":
                    continue
                poly = self._face_poly_widget(inds, corners_uv, depths)
                if poly is not None:
                    faces_ranked.append((self._mean_face_depth_cam(box, inds), poly))
            faces_ranked.sort(key=lambda item: -item[0])

            for _depth, poly in faces_ranked:
                fill = QColor(color)
                fill.setAlpha(42 if not is_selected else 55)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(fill))
                p.drawPolygon(poly)

            front_poly = self._face_poly_widget(_BOX_FACE_FRONT, corners_uv, depths)
            if front_poly is not None:
                front_fill = QColor(
                    min(255, color.red() + 55),
                    min(255, color.green() + 55),
                    min(255, color.blue() + 40),
                )
                front_fill.setAlpha(115 if not is_selected else 145)
                p.setPen(QPen(QColor(255, 255, 255, 100), 2))
                p.setBrush(QBrush(front_fill))
                p.drawPolygon(front_poly)
            p.restore()

            for i, j in BOX_EDGES:
                if depths[i] <= 0.05 and depths[j] <= 0.05:
                    continue
                pa = self._image_to_widget(corners_uv[i])
                pb = self._image_to_widget(corners_uv[j])
                is_front = frozenset((i, j)) in _FRONT_EDGE_SET
                width = 3 if is_front else (2 if is_selected else 1)
                edge_style = Qt.DashLine if box.manual_placement else Qt.SolidLine
                p.setPen(QPen(color, width, edge_style))
                p.setBrush(Qt.NoBrush)
                p.drawLine(pa, pb)
            self._draw_heading_arrow(p, box)

        bbox = box_2d_from_projection(corners_uv, depths, self._image_size)
        if self._show_2d_bbox and bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            rect = QRectF(
                self._image_to_widget(np.array([xmin, ymin])),
                self._image_to_widget(np.array([xmax, ymax])),
            )
            p.setPen(QPen(color, 2 if is_selected else 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect)

        if self._show_labels and bbox is not None:
            xmin, ymin, _xmax, _ymax = bbox
            anchor = self._image_to_widget(np.array([xmin, max(0.0, ymin - 2.0)]))
            lbl = f"{box.class_name} (ID: {box.object_id})"
            if box.manual_placement:
                lbl += " [radar-ground]"
            lines = [lbl]
            if self._show_box_metrics:
                pts = self._radar_points
                if pts is None:
                    pts = np.zeros((0, 3), dtype=np.float64)
                dx, dy, dz = radar_display_xyz_m(box, pts)
                rng_x = radar_range_from_x_m(box, pts)
                lines.append(
                    f"x:{dx:.1f} y:{dy:.1f} z:{dz:.1f} d:{rng_x:.1f}m"
                )
                lines.append(
                    f"L:{box.length:.1f} W:{box.width:.1f} H:{box.height:.1f}m"
                )
            font = QFont()
            font.setPointSize(8)
            font.setBold(True)
            p.setFont(font)
            fm = p.fontMetrics()
            line_h = fm.height()
            text_w = max(fm.horizontalAdvance(line) for line in lines)
            rect = QRectF(
                anchor.x(),
                anchor.y() - line_h * len(lines) - 6,
                text_w + 12,
                line_h * len(lines) + 6,
            )
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(color))
            p.drawRoundedRect(rect, 3, 3)
            p.setPen(QColor(255, 255, 255))
            for row, line in enumerate(lines):
                y = rect.y() + line_h * (row + 1)
                p.drawText(QPointF(rect.x() + 6, y), line)

        if is_selected:
            self._draw_selected_handles(p, box, color)


    def _create_box_from_image_rect(self, p1: QPointF, p2: QPointF) -> None:
        """Ray-cast a drawn 2D rect back into 3D to create a Box3D."""
        from ..core.geometry import Box3D
        from .annotation_classes import class_default_box_size

        rect = QRectF(p1, p2).normalized()
        if rect.width() < 5 or rect.height() < 5:
            return

        K = self._calibration.K
        T = self._calibration.T
        fx = float(K[0, 0]); fy = float(K[1, 1])
        cx_k = float(K[0, 2]); cy_k = float(K[1, 2])

        l_def, w_def, h_def = class_default_box_size(self._create_class)

        img_p1 = self._widget_to_image(p1)
        img_p2 = self._widget_to_image(p2)
        r = np.array([
            min(img_p1[0], img_p2[0]), min(img_p1[1], img_p2[1]),
            max(img_p1[0], img_p2[0]), max(img_p1[1], img_p2[1]),
        ])

        if self._manual_placement_mode:
            nx: Optional[float] = None
            ny: Optional[float] = None
            from_radar = False
            if self._radar_points is not None and self._radar_points.shape[0] > 0:
                uv_all, dep_all = project_points_calibration(
                    self._radar_points[:, :3], self._display_calibration())
                inside = (
                    (uv_all[:, 0] >= r[0]) & (uv_all[:, 0] <= r[2])
                    & (uv_all[:, 1] >= r[1]) & (uv_all[:, 1] <= r[3])
                    & (dep_all > 0)
                )
                if np.any(inside):
                    pts_sel = self._radar_points[inside, :3]
                    nx = float(np.median(pts_sel[:, 0]))
                    ny = float(np.median(pts_sel[:, 1]))
                    from_radar = True
            if nx is None:
                # Camera-only: map rect centre to master XY (same scale as drag), no radar.
                img_center = self._widget_to_image(
                    QPointF((rect.left() + rect.right()) * 0.5,
                            (rect.top() + rect.bottom()) * 0.5))
                iw, ih = self._image_size if self._image_size[0] > 0 else (640, 480)
                mpp = self._meters_per_image_pixel_drag()
                nx = float((img_center[0] - 0.5 * iw) * mpp)
                ny = float(-(img_center[1] - 0.5 * ih) * mpp)
            new_box = Box3D(
                x=nx,
                y=ny,
                z=float(h_def / 2.0),
                length=float(l_def),
                width=float(w_def),
                height=float(h_def),
                class_name=self._create_class,
                manual_placement=True,
            )
            self.box_created_from_image.emit(new_box)
            self.cancel_create_mode()
            src = "radar in rect" if from_radar else "image only — refine on BEV/radar"
            self.status_message.emit(
                f"Created {self._create_class} ({src}) at "
                f"({new_box.x:.1f}, {new_box.y:.1f}, {new_box.z:.1f}) m")
            return

        # Centre of 2D rect in image coords (calibration-backed placement)
        img_center = self._widget_to_image(
            QPointF((rect.left() + rect.right()) / 2.0,
                    (rect.top() + rect.bottom()) / 2.0))

        T_inv = np.linalg.inv(T)
        depth = 10.0  # metres fallback

        if self._radar_points is not None and self._radar_points.shape[0] > 0:
            uv_all, dep_all = project_points_calibration(
                self._radar_points[:, :3], self._calibration)
            inside = ((uv_all[:, 0] >= r[0]) & (uv_all[:, 0] <= r[2]) &
                      (uv_all[:, 1] >= r[1]) & (uv_all[:, 1] <= r[3]) &
                      (dep_all > 0))
            if np.any(inside):
                depth = float(np.median(dep_all[inside]))

        cam_pt = np.array([(img_center[0] - cx_k) * depth / fx,
                            (img_center[1] - cy_k) * depth / fy,
                            depth, 1.0], dtype=np.float64)
        master_pt = T_inv @ cam_pt

        new_box = Box3D(
            x=float(master_pt[0]),
            y=float(master_pt[1]),
            z=float(h_def / 2.0),
            length=float(l_def),
            width=float(w_def),
            height=float(h_def),
            class_name=self._create_class,
        )

        if self._radar_points is not None and self._radar_points.shape[0] > 0:
            pts = self._radar_points[:, :3]
            center_xy = np.array([new_box.x, new_box.y])
            dist = np.linalg.norm(pts[:, :2] - center_xy, axis=1)
            order = np.argsort(dist)
            radius = max(l_def, w_def, 3.0)
            close_idx = order[:min(20, len(order))]
            close = pts[close_idx][dist[close_idx] <= radius]
            if close.shape[0] > 0:
                new_box.x = float(np.median(close[:, 0]))
                new_box.y = float(np.median(close[:, 1]))

        self.box_created_from_image.emit(new_box)
        self.cancel_create_mode()
        self.status_message.emit(
            f"Created {self._create_class} from image rect at "
            f"({new_box.x:.1f}, {new_box.y:.1f}, {new_box.z:.1f}) m")

    def _box_hit_test(self, widget_pt: QPointF) -> Optional[Box3D]:
        img_pt = self._widget_to_image(widget_pt)
        best: Optional[Box3D] = None
        best_area = float("inf")
        m = _HIT_IMAGE_MARGIN + (
            _HIT_IMAGE_MARGIN_MANUAL_EXTRA if self._manual_placement_mode else 0.0
        )
        for box in reversed(self._visible_boxes()):
            corners_uv, depths = project_box_calibration(box, self._display_calibration())
            raw = self._raw_projected_bbox(corners_uv, depths)
            if raw is None:
                continue
            xmin, ymin, xmax, ymax = raw
            if (
                xmin - m <= img_pt[0] <= xmax + m
                and ymin - m <= img_pt[1] <= ymax + m
            ):
                area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
                if area < best_area:
                    best_area = area
                    best = box
        return best

    def _hover_hit(self, widget_pt: QPointF) -> tuple[Optional[Box3D], Optional[str], Optional[_HandleSpec]]:
        sel = self.model.selected()
        if sel is not None and self._box_is_visible(sel):
            handles = self._control_handles(sel)
            for role, spec in handles.items():
                if math.hypot(widget_pt.x() - spec.point.x(), widget_pt.y() - spec.point.y()) <= _HANDLE_HIT_RADIUS:
                    return sel, role, spec
        hit_box = self._box_hit_test(widget_pt)
        if hit_box is not None:
            return hit_box, "move", None
        return None, None, None

    def _set_cursor_for_role(self, role: Optional[str]) -> None:
        if role == "move":
            self.setCursor(Qt.SizeAllCursor)
        elif role == "rotate":
            self.setCursor(Qt.OpenHandCursor)
        elif role in ("resize_height_top", "resize_height_bottom"):
            self.setCursor(Qt.SizeVerCursor)
        elif role is not None:
            self.setCursor(Qt.CrossCursor)
        else:
            self.unsetCursor()

    def _start_drag(self, box: Box3D, role: str, mouse_pos: QPointF, handle: Optional[_HandleSpec]) -> None:
        self._dragging = True
        self._drag_role = role
        self._drag_uid = box.uid
        self._drag_start_mouse = mouse_pos
        self._drag_handle = handle
        import copy

        self._drag_start_box = copy.deepcopy(box)
        self._set_cursor_for_role(role)

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        self.setFocus()

        # Creation mode: start drawing rect
        if self._create_mode and ev.button() == Qt.LeftButton:
            self._create_rect_start = ev.position()
            self._create_rect_end = ev.position()
            self.update()
            return

        hit_box, role, handle = self._hover_hit(ev.position())

        if ev.button() == Qt.RightButton:
            if hit_box is not None:
                self.model.select(hit_box.uid)
                self._start_drag(hit_box, "rotate", ev.position(), handle)
            return

        if ev.button() != Qt.LeftButton:
            return

        if hit_box is None:
            self.model.select(None)
            self.unsetCursor()
            return

        self.model.select(hit_box.uid)
        self._start_drag(hit_box, role or "move", ev.position(), handle)

    def _apply_translate_drag_manual(self, box: Box3D, ev: QMouseEvent) -> None:
        """Uniform metres per image pixel — same sensitivity everywhere on the photo."""
        if self._drag_start_box is None or self._drag_start_mouse is None:
            return
        cur_img = self._widget_to_image(ev.position())
        start_img = self._widget_to_image(self._drag_start_mouse)
        du = float(cur_img[0] - start_img[0])
        dv = float(cur_img[1] - start_img[1])
        mpp = self._meters_per_image_pixel_drag()
        # Image +v down; master +y left — flip dv to match BEV sense.
        box.x = float(self._drag_start_box.x + du * mpp)
        box.y = float(self._drag_start_box.y - dv * mpp)
        box.z = self._drag_start_box.z

    def _apply_translate_drag(self, box: Box3D, ev: QMouseEvent) -> None:
        if self._drag_start_box is None:
            return
        T_inv = np.linalg.inv(self._calibration.T)
        c_master_h = np.array(
            [self._drag_start_box.x, self._drag_start_box.y, self._drag_start_box.z, 1.0],
            dtype=np.float64,
        )
        c_cam = self._calibration.T @ c_master_h
        depth = float(c_cam[2])
        if depth <= 0.1:
            self._apply_translate_drag_manual(box, ev)
            return

        cur_img = self._widget_to_image(ev.position())
        fx = float(self._calibration.K[0, 0])
        fy = float(self._calibration.K[1, 1])
        cx = float(self._calibration.K[0, 2])
        cy = float(self._calibration.K[1, 2])
        cam_pt = np.array(
            [
                (cur_img[0] - cx) * depth / fx,
                (cur_img[1] - cy) * depth / fy,
                depth,
                1.0,
            ],
            dtype=np.float64,
        )
        master_pt = T_inv @ cam_pt
        box.x = float(master_pt[0])
        box.y = float(master_pt[1])
        box.z = self._drag_start_box.z

    def _apply_rotate_drag(self, box: Box3D, ev: QMouseEvent) -> None:
        if self._drag_start_box is None or self._drag_start_mouse is None:
            return
        center = np.array([self._drag_start_box.x, self._drag_start_box.y, self._drag_start_box.z], dtype=np.float64)
        center_pt = self._project_widget_point(center)
        if center_pt is None:
            return
        a0 = math.atan2(
            self._drag_start_mouse.y() - center_pt.y(),
            self._drag_start_mouse.x() - center_pt.x(),
        )
        a1 = math.atan2(ev.position().y() - center_pt.y(), ev.position().x() - center_pt.x())
        box.yaw = self._drag_start_box.yaw + (a1 - a0)

    def _apply_resize_drag(self, box: Box3D, ev: QMouseEvent) -> None:
        if self._drag_start_box is None or self._drag_start_mouse is None or self._drag_handle is None:
            return
        if self._drag_handle.axis_unit_px is None or self._drag_handle.meters_per_px is None:
            return

        delta_px = np.array(
            [
                ev.position().x() - self._drag_start_mouse.x(),
                ev.position().y() - self._drag_start_mouse.y(),
            ],
            dtype=np.float64,
        )
        delta_m = float(np.dot(delta_px, self._drag_handle.axis_unit_px)) * float(
            self._drag_handle.meters_per_px
        )
        world_dir = self._drag_handle.world_dir
        if world_dir is None:
            return

        start = self._drag_start_box
        if self._drag_role in {"resize_length_front", "resize_length_back"}:
            new_length = max(0.2, start.length + delta_m)
            shift = world_dir * ((new_length - start.length) * 0.5)
            box.length = new_length
            box.x = float(start.x + shift[0])
            box.y = float(start.y + shift[1])
            box.z = start.z
        elif self._drag_role in {"resize_width_left", "resize_width_right"}:
            new_width = max(0.2, start.width + delta_m)
            shift = world_dir * ((new_width - start.width) * 0.5)
            box.width = new_width
            box.x = float(start.x + shift[0])
            box.y = float(start.y + shift[1])
            box.z = start.z
        elif self._drag_role == "resize_height_top":
            base_z = start.z - start.height * 0.5
            new_height = max(0.2, start.height + delta_m)
            box.height = new_height
            box.z = float(base_z + new_height * 0.5)
        elif self._drag_role == "resize_height_bottom":
            # Keep top face fixed; grow/shrink downward in +master z.
            top_z = start.z + start.height * 0.5
            new_height = max(0.2, start.height + delta_m)
            box.height = new_height
            box.z = float(top_z - new_height * 0.5)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        # Update creation rect while drawing
        if self._create_mode and self._create_rect_start is not None:
            self._create_rect_end = ev.position()
            self.update()
            return

        if not self._dragging:
            _box, role, _handle = self._hover_hit(ev.position())
            self._set_cursor_for_role(role)
            return

        if self._drag_uid is None or self._drag_start_box is None:
            return
        box = self.model.find(self._drag_uid)
        if box is None:
            return

        if self._drag_role == "move":
            if self._manual_placement_mode:
                self._apply_translate_drag_manual(box, ev)
            else:
                self._apply_translate_drag(box, ev)
        elif self._drag_role == "rotate":
            self._apply_rotate_drag(box, ev)
        else:
            self._apply_resize_drag(box, ev)
        self.model.update(box, snapshot=False)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        # Finish creation rect
        if self._create_mode and ev.button() == Qt.LeftButton and self._create_rect_start is not None:
            self._create_box_from_image_rect(self._create_rect_start, ev.position())
            return

        if self._dragging:
            self.model.commit()
        self._dragging = False
        self._drag_role = None
        self._drag_uid = None
        self._drag_start_box = None
        self._drag_start_mouse = None
        self._drag_handle = None
        self.unsetCursor()

    def keyPressEvent(self, ev) -> None:
        from PySide6.QtCore import Qt as _Qt
        if ev.key() == _Qt.Key_Escape and self._create_mode:
            self.cancel_create_mode()
            self.status_message.emit("Image create mode cancelled.")
            ev.accept()
            return
        if (
            self._manual_placement_mode
            and not self._create_mode
            and not self._dragging
            and self.model.selected_uid is not None
        ):
            sel = self.model.find(self.model.selected_uid)
            if sel is not None:
                step_px = _ARROW_NUDGE_IMAGE_PX * (
                    0.25 if (ev.modifiers() & _Qt.ShiftModifier) else 1.0
                )
                mpp = self._meters_per_image_pixel_drag()
                du = dv = 0.0
                if ev.key() == _Qt.Key_Left:
                    du = -step_px
                elif ev.key() == _Qt.Key_Right:
                    du = step_px
                elif ev.key() == _Qt.Key_Up:
                    dv = -step_px
                elif ev.key() == _Qt.Key_Down:
                    dv = step_px
                if du != 0.0 or dv != 0.0:
                    sel.x = float(sel.x + du * mpp)
                    sel.y = float(sel.y - dv * mpp)
                    self.model.update(sel, snapshot=True)
                    self.update()
                    ev.accept()
                    return
        super().keyPressEvent(ev)
