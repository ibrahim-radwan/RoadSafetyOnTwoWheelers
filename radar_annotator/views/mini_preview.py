"""
Mini Preview widget shown at the bottom of the left panel.

Shows small thumbnails of the BEV and camera views with the class legend
beneath — matches the reference mock-up.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (QPainter, QPen, QBrush, QColor, QImage, QPixmap,
                           QFont, QPaintEvent)
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QFrame, QSizePolicy)

from ..core.annotation_model import AnnotationModel
from .annotation_classes import CLASS_COLORS


class _BEVThumb(QWidget):
    """Tiny top-down thumbnail of the radar + box footprints."""

    def __init__(self, model: AnnotationModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._points = np.zeros((0, 3))
        self.setFixedHeight(90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.model.objects_changed.connect(self.update)
        self.model.selection_changed.connect(lambda _u: self.update())

    def set_points(self, pts: np.ndarray) -> None:
        self._points = pts if pts is not None else np.zeros((0, 3))
        self.update()

    def _master_to_screen(self, xy: np.ndarray,
                          x_range, y_range) -> QPointF:
        pad = 6
        w = self.width() - 2 * pad
        h = self.height() - 2 * pad
        # x forward -> up, y left -> left
        x0, x1 = x_range; y0, y1 = y_range
        sx = pad + w * (1.0 - (xy[1] - y0) / max(y1 - y0, 1e-3))
        sy = pad + h * (1.0 - (xy[0] - x0) / max(x1 - x0, 1e-3))
        return QPointF(sx, sy)

    def paintEvent(self, ev: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        # Dark mini canvas
        rect = QRectF(0, 0, self.width(), self.height())
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(17, 24, 39)))
        p.drawRoundedRect(rect, 4, 4)

        # Label
        p.setPen(QColor(203, 213, 225))
        f = QFont(); f.setPointSize(7); f.setBold(True); p.setFont(f)
        p.drawText(6, 12, "BEV (Radar)")

        # Determine view bounds: enclose points + objects
        xs, ys = [0.0], [0.0]
        if self._points.size > 0:
            xs += [float(self._points[:, 0].min()), float(self._points[:, 0].max())]
            ys += [float(self._points[:, 1].min()), float(self._points[:, 1].max())]
        for b in self.model.objects:
            xs += [b.x - 3, b.x + 3]
            ys += [b.y - 3, b.y + 3]
        xr = (min(xs), max(xs))
        yr = (min(ys), max(ys))
        if xr[1] - xr[0] < 20: xr = (xr[0] - 10, xr[1] + 10)
        if yr[1] - yr[0] < 20: yr = (yr[0] - 10, yr[1] + 10)

        # Points (faint)
        if self._points.size > 0:
            p.setPen(QPen(QColor(148, 163, 184), 1))
            step = max(1, self._points.shape[0] // 300)
            for i in range(0, self._points.shape[0], step):
                pt = self._master_to_screen(self._points[i, :2], xr, yr)
                p.drawPoint(pt)

        # Box footprints
        from ..core.geometry import Box3D
        for box in self.model.objects:
            color = CLASS_COLORS.get(box.class_name, CLASS_COLORS["Other"])
            l2, w2 = box.length / 2.0, box.width / 2.0
            local = np.array([[-l2, -w2], [l2, -w2], [l2, w2], [-l2, w2]])
            c, s = np.cos(box.yaw), np.sin(box.yaw)
            R = np.array([[c, -s], [s, c]])
            world = (R @ local.T).T + np.array([box.x, box.y])
            pts = [self._master_to_screen(w, xr, yr) for w in world]
            from PySide6.QtGui import QPolygonF
            p.setPen(QPen(color, 1.5))
            p.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 70)))
            p.drawPolygon(QPolygonF(pts))


class _CameraThumb(QWidget):
    """Tiny thumbnail of the current camera frame with 2D bbox overlays."""

    def __init__(self, model: AnnotationModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._image: QImage = None
        self._image_size = (0, 0)
        self._calibration = None
        self.setFixedHeight(90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.model.objects_changed.connect(self.update)
        self.model.selection_changed.connect(lambda _u: self.update())

    def set_image(self, qimage: QImage, image_size) -> None:
        self._image = qimage
        self._image_size = tuple(image_size) if image_size else (0, 0)
        self.update()

    def set_calibration(self, calib) -> None:
        self._calibration = calib
        self.update()

    def paintEvent(self, ev: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(0, 0, self.width(), self.height())
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(17, 24, 39)))
        p.drawRoundedRect(rect, 4, 4)

        if self._image is not None:
            from PySide6.QtGui import QPixmap
            pix = QPixmap.fromImage(self._image)
            iw, ih = self._image_size
            if iw > 0 and ih > 0:
                scale = min(self.width() / iw, self.height() / ih)
                tw, th = iw * scale, ih * scale
                offx = (self.width() - tw) / 2
                offy = (self.height() - th) / 2
                p.drawPixmap(QRectF(offx, offy, tw, th),
                             pix, QRectF(0, 0, iw, ih))

                # Overlay projected 2D bboxes
                if self._calibration is not None:
                    from ..core.geometry import (
                        project_box_calibration, box_2d_from_projection)
                    for box in self.model.objects:
                        corners_uv, depths = project_box_calibration(
                            box, self._calibration)
                        bbox = box_2d_from_projection(
                            corners_uv, depths, self._image_size)
                        if bbox is None:
                            continue
                        xmin, ymin, xmax, ymax = bbox
                        color = CLASS_COLORS.get(
                            box.class_name, CLASS_COLORS["Other"])
                        p.setPen(QPen(color, 1.2))
                        p.setBrush(Qt.NoBrush)
                        p.drawRect(QRectF(
                            offx + xmin * scale,
                            offy + ymin * scale,
                            (xmax - xmin) * scale,
                            (ymax - ymin) * scale))
        p.setPen(QColor(203, 213, 225))
        f = QFont(); f.setPointSize(7); f.setBold(True); p.setFont(f)
        p.drawText(6, 12, "Camera")


class MiniPreview(QWidget):
    """Combined BEV + camera thumbnails + class legend."""

    def __init__(self, model: AnnotationModel, parent=None):
        super().__init__(parent)
        self.model = model

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        title = QLabel("Mini Preview")
        tf = title.font(); tf.setBold(True); tf.setPointSize(9)
        title.setFont(tf)
        root.addWidget(title)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.bev_thumb = _BEVThumb(model)
        self.cam_thumb = _CameraThumb(model)
        row.addWidget(self.bev_thumb, 1)
        row.addWidget(self.cam_thumb, 1)
        root.addLayout(row)

        # Legend
        legend = QHBoxLayout()
        legend.setSpacing(10)
        for cls in ("Car", "Pedestrian", "Cyclist", "Other", "Background"):
            dot = QLabel()
            if cls == "Background":
                dot.setStyleSheet("background: transparent; border: 1px solid #cbd5e1; "
                                  "border-radius: 4px; min-width: 8px; max-width: 8px; "
                                  "min-height: 8px; max-height: 8px;")
            else:
                c = CLASS_COLORS.get(cls, CLASS_COLORS["Other"])
                dot.setStyleSheet(
                    f"background: rgb({c.red()}, {c.green()}, {c.blue()}); "
                    f"border-radius: 4px; min-width: 8px; max-width: 8px; "
                    f"min-height: 8px; max-height: 8px;")
            lbl = QLabel(cls)
            lbl.setStyleSheet("color: #64748b; font-size: 10px;")
            item = QHBoxLayout()
            item.setSpacing(3)
            item.addWidget(dot)
            item.addWidget(lbl)
            wrap = QWidget(); wrap.setLayout(item)
            legend.addWidget(wrap)
        legend.addStretch(1)
        root.addLayout(legend)

    # Pass-through setters
    def set_points(self, pts) -> None:
        self.bev_thumb.set_points(pts)

    def set_image(self, qimage, size) -> None:
        self.cam_thumb.set_image(qimage, size)

    def set_calibration(self, calib) -> None:
        self.cam_thumb.set_calibration(calib)
