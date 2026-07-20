"""
Left panel — "Loaded Example Details" — matching the reference mock-up.

Shows dataset-level context, the full per-frame field list (timestamp, pair
index, calibration ID, image resolution, weather/scene tag, sync status, etc.),
and a Mini Preview widget with synchronised BEV + camera thumbnails.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                               QLabel, QScrollArea, QFrame, QSizePolicy,
                               QPushButton)

from ..core.annotation_model import AnnotationModel
from .mini_preview import MiniPreview


class FrameInfoPanelV2(QScrollArea):
    fix_requested = Signal(str, str)

    def __init__(self, model: AnnotationModel, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QFrame()
        container.setObjectName("leftPanel")
        container.setStyleSheet(
            "#leftPanel { background: #ffffff; border: 1px solid #e2e8f0; "
            "border-radius: 8px; }")
        root = QVBoxLayout(container)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)
        self.setWidget(container)

        # Title with chevron-right affordance (matches the image)
        header = QHBoxLayout()
        title = QLabel("Loaded Example Details")
        tf = title.font(); tf.setBold(True); tf.setPointSize(11)
        title.setFont(tf)
        title.setStyleSheet("color: #1e293b;")
        header.addWidget(title)
        header.addStretch(1)
        chev = QLabel("›")
        chev.setStyleSheet("color: #94a3b8; font-size: 18px;")
        header.addWidget(chev)
        root.addLayout(header)

        # Divider
        line = QFrame(); line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #e2e8f0; background: #e2e8f0; max-height: 1px;")
        root.addWidget(line)

        # Form of per-frame fields
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)

        self._fields = {}
        for key, label in [
            ("dataset_path", "Dataset Path"),
            ("frame_id", "Frame ID"),
            ("radar_file", "Radar File"),
            ("image_file", "Image File"),
            ("timestamp", "Timestamp"),
            ("pair_index", "Pair Index"),
            ("total_frames", "Total Frames"),
            ("calibration_id", "Calibration ID"),
            ("point_count", "Point Count"),
            ("image_resolution", "Image Resolution"),
            ("weather_scene", "Weather / Scene"),
            ("sync_status", "Sync Status"),
        ]:
            k = QLabel(label)
            k.setStyleSheet("color: #64748b; font-size: 11px;")
            v = QLabel("—")
            v.setTextInteractionFlags(Qt.TextSelectableByMouse)
            v.setStyleSheet("color: #1e293b; font-size: 11px;")
            v.setWordWrap(True)
            form.addRow(k, v)
            self._fields[key] = v

        # Sync status gets a colored dot. The sync status label was added as
        # a plain row above; we now replace that row with a dot-plus-label
        # composite. Create a fresh label (the original is destroyed when
        # the form row is removed below).
        self._sync_dot = QLabel()
        self._sync_dot.setStyleSheet(
            "background: #22c55e; border-radius: 4px; "
            "min-width: 8px; max-width: 8px; min-height: 8px; max-height: 8px;")
        sync_text = QLabel("Synchronized")
        sync_text.setStyleSheet("color: #16a34a; font-size: 11px;")
        sync_container = QWidget()
        sync_layout = QHBoxLayout(sync_container)
        sync_layout.setContentsMargins(0, 0, 0, 0)
        sync_layout.setSpacing(5)
        sync_layout.addWidget(self._sync_dot)
        sync_layout.addWidget(sync_text)
        sync_layout.addStretch(1)
        # Remove the placeholder row created above and re-add the composed one
        form.removeRow(form.rowCount() - 1)
        k = QLabel("Sync Status")
        k.setStyleSheet("color: #64748b; font-size: 11px;")
        form.addRow(k, sync_container)
        # Swap the stored reference so later setters update the new label
        self._fields["sync_status"] = sync_text

        root.addLayout(form)

        # Separator before Mini Preview
        line2 = QFrame(); line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("color: #e2e8f0; background: #e2e8f0; max-height: 1px;")
        root.addWidget(line2)

        # Mini Preview
        self.mini_preview = MiniPreview(model)
        root.addWidget(self.mini_preview)

        # Validation section (below mini preview)
        line3 = QFrame(); line3.setFrameShape(QFrame.HLine)
        line3.setStyleSheet("color: #e2e8f0; background: #e2e8f0; max-height: 1px;")
        root.addWidget(line3)

        val_title = QLabel("Validation")
        vf = val_title.font(); vf.setBold(True); vf.setPointSize(9)
        val_title.setFont(vf)
        val_title.setStyleSheet("color: #475569;")
        root.addWidget(val_title)

        status_row = QHBoxLayout()
        status_row.setSpacing(5)
        self._dirty_dot = QLabel()
        self._dirty_dot.setStyleSheet(
            "background: #22c55e; border-radius: 4px; "
            "min-width: 8px; max-width: 8px; min-height: 8px; max-height: 8px;")
        self._dirty_lbl = QLabel("saved")
        self._dirty_lbl.setStyleSheet("color: #16a34a; font-size: 11px;")
        status_row.addWidget(self._dirty_dot)
        status_row.addWidget(self._dirty_lbl)
        status_row.addStretch(1)
        root.addLayout(status_row)

        self._warnings_lbl = QLabel("none")
        self._warnings_lbl.setWordWrap(True)
        self._warnings_lbl.setStyleSheet("color: #64748b; font-size: 10px;")
        root.addWidget(self._warnings_lbl)
        self._warnings_container = QWidget()
        self._warnings_layout = QVBoxLayout(self._warnings_container)
        self._warnings_layout.setContentsMargins(0, 0, 0, 0)
        self._warnings_layout.setSpacing(4)
        root.addWidget(self._warnings_container)
        self._warnings_container.setVisible(False)

        root.addStretch(1)

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------
    def set_dataset(self, root: Optional[Path], calibration_id: str,
                    total_frames: int) -> None:
        self._fields["dataset_path"].setText(str(root) if root else "—")
        self._fields["calibration_id"].setText(calibration_id or "—")
        self._fields["total_frames"].setText(str(total_frames))

    def set_frame(self, *, frame_id: str, radar_name: str, image_name: str,
                  timestamp: str, pair_index: int,
                  point_count: int, image_resolution,
                  weather_scene: str = "—", synchronized: bool = True) -> None:
        self._fields["frame_id"].setText(frame_id)
        self._fields["radar_file"].setText(radar_name)
        self._fields["image_file"].setText(image_name)
        self._fields["timestamp"].setText(timestamp)
        self._fields["pair_index"].setText(str(pair_index))
        self._fields["point_count"].setText(f"{point_count:,}")
        if image_resolution and len(image_resolution) == 2:
            self._fields["image_resolution"].setText(
                f"{image_resolution[0]} × {image_resolution[1]}")
        else:
            self._fields["image_resolution"].setText("—")
        self._fields["weather_scene"].setText(weather_scene)

        if synchronized:
            self._sync_dot.setStyleSheet(
                "background: #22c55e; border-radius: 4px; "
                "min-width: 8px; max-width: 8px; min-height: 8px; max-height: 8px;")
            self._fields["sync_status"].setText("Synchronized")
            self._fields["sync_status"].setStyleSheet(
                "color: #16a34a; font-size: 11px;")
        else:
            self._sync_dot.setStyleSheet(
                "background: #f97316; border-radius: 4px; "
                "min-width: 8px; max-width: 8px; min-height: 8px; max-height: 8px;")
            self._fields["sync_status"].setText("Timestamp mismatch")
            self._fields["sync_status"].setStyleSheet(
                "color: #c2410c; font-size: 11px;")

    def set_dirty(self, dirty: bool) -> None:
        if dirty:
            self._dirty_dot.setStyleSheet(
                "background: #f59e0b; border-radius: 4px; "
                "min-width: 8px; max-width: 8px; min-height: 8px; max-height: 8px;")
            self._dirty_lbl.setText("unsaved changes")
            self._dirty_lbl.setStyleSheet("color: #d97706; font-size: 11px;")
        else:
            self._dirty_dot.setStyleSheet(
                "background: #22c55e; border-radius: 4px; "
                "min-width: 8px; max-width: 8px; min-height: 8px; max-height: 8px;")
            self._dirty_lbl.setText("saved")
            self._dirty_lbl.setStyleSheet("color: #16a34a; font-size: 11px;")

    def set_warnings(self, warnings) -> None:
        while self._warnings_layout.count():
            item = self._warnings_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not warnings:
            self._warnings_lbl.setText("no warnings")
            self._warnings_lbl.setStyleSheet("color: #16a34a; font-size: 10px;")
            self._warnings_lbl.setVisible(True)
            self._warnings_container.setVisible(False)
        else:
            self._warnings_lbl.setVisible(False)
            self._warnings_container.setVisible(True)
            for warning in warnings:
                if isinstance(warning, dict):
                    text = str(warning.get("text", ""))
                    uid = str(warning.get("uid", ""))
                    fix = str(warning.get("fix", ""))
                else:
                    text = str(warning)
                    uid = ""
                    fix = ""

                row = QWidget()
                h = QHBoxLayout(row)
                h.setContentsMargins(0, 0, 0, 0)
                h.setSpacing(4)
                if uid and fix:
                    btn = QPushButton("Fix")
                    btn.setToolTip("Try recommended values for this warning.")
                    btn.setFixedSize(30, 18)
                    btn.setStyleSheet(
                        "QPushButton { background: #fee2e2; color: #b91c1c; "
                        "border: 1px solid #fecaca; border-radius: 3px; "
                        "font-size: 9px; font-weight: 600; }"
                        "QPushButton:hover { background: #fecaca; }")
                    btn.clicked.connect(
                        lambda _checked=False, u=uid, f=fix:
                        self.fix_requested.emit(u, f))
                    h.addWidget(btn)

                lbl = QLabel(text)
                lbl.setWordWrap(True)
                lbl.setStyleSheet("color: #dc2626; font-size: 10px;")
                h.addWidget(lbl, 1)
                self._warnings_layout.addWidget(row)

    def set_points(self, pts) -> None:
        self.mini_preview.set_points(pts)

    def set_image(self, qimage, size) -> None:
        self.mini_preview.set_image(qimage, size)

    def set_calibration(self, calib) -> None:
        self.mini_preview.set_calibration(calib)
