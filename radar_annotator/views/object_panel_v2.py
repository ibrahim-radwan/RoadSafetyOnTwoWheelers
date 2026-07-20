"""
Right panel — "Selected Object Details" — matching the reference mock-up.

All per-object fields from the reference are present, including Doppler
velocity, radar median X/Y/Z, Range X (median contacts), Azimuth, Visibility %. Numeric edits are buffered
locally; "Apply" commits them to the shared model in one undoable step.
Duplicate and Delete buttons live at the bottom of the panel.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                               QLabel, QScrollArea, QFrame, QDoubleSpinBox,
                               QSpinBox, QComboBox, QLineEdit, QPlainTextEdit,
                               QPushButton, QMessageBox, QSizePolicy, QCheckBox)

from ..core.annotation_model import AnnotationModel
from ..core.geometry import (
    Box3D,
    box_2d_from_projection,
    box_location_cam_kitti,
    project_box_calibration,
    radar_azimuth_deg_from_xy,
    radar_range_from_x_m,
    radar_returns_median_xyz,
)
from ..core.radar_filters import COL_VR, COL_VR_COMP
from .annotation_classes import OBJECT_CLASS_CHOICES, class_default_box_size


class ObjectDetailsPanelV2(QScrollArea):
    edit_focus_requested = Signal()  # Table asks us to come to attention

    def __init__(self, model: AnnotationModel, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.model = model

        from ..core.calibration import Calibration
        self._calibration = Calibration.identity()
        self._radar_points = np.zeros((0, 3))
        self._image_size = (1920, 1080)
        self._updating_from_model = False
        self._has_pending_changes = False

        container = QFrame()
        container.setObjectName("rightPanel")
        container.setStyleSheet(
            "#rightPanel { background: #ffffff; border: 1px solid #e2e8f0; "
            "border-radius: 8px; }")
        root = QVBoxLayout(container)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)
        self.setWidget(container)

        # Header
        header = QHBoxLayout()
        self._title = QLabel("Selected Object Details")
        tf = self._title.font(); tf.setBold(True); tf.setPointSize(11)
        self._title.setFont(tf)
        self._title.setStyleSheet("color: #1e293b;")
        header.addWidget(self._title)
        header.addStretch(1)
        chev = QLabel("›")
        chev.setStyleSheet("color: #94a3b8; font-size: 18px;")
        header.addWidget(chev)
        root.addLayout(header)

        line = QFrame(); line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #e2e8f0; background: #e2e8f0; max-height: 1px;")
        root.addWidget(line)

        # Empty-state notice
        self._empty_lbl = QLabel(
            "No object selected.\n\n"
            "Click a box in either view or a row in the object list below, "
            "or press N to create a new one.")
        self._empty_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self._empty_lbl.setWordWrap(True)
        root.addWidget(self._empty_lbl)

        # Form
        self._form_container = QWidget()
        form = QFormLayout(self._form_container)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(5)
        form.setContentsMargins(0, 0, 0, 0)
        self._form = form
        self._inputs = {}

        def add_row(key, label, widget):
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #64748b; font-size: 11px;")
            widget.setStyleSheet(self._input_stylesheet())
            form.addRow(lbl, widget)
            self._inputs[key] = widget

        # Identity
        self._obj_id = QSpinBox(); self._obj_id.setRange(1, 1_000_000)
        add_row("object_id", "Object ID", self._obj_id)
        self._track_id = QLineEdit(); self._track_id.setPlaceholderText("(optional)")
        add_row("track_id", "Track ID", self._track_id)

        self._class = QComboBox(); self._class.setEditable(True)
        self._class.addItems(list(OBJECT_CLASS_CHOICES))
        add_row("class", "Class", self._class)

        self._manual_place = QCheckBox("Radar-ground geometry")
        self._manual_place.setToolTip(
            "Pose defined manually or from radar — creation/move skips inverse "
            "camera extrinsics where applicable. Overlay still uses forward projection.")
        lbl_mp = QLabel("Placement")
        lbl_mp.setStyleSheet("color: #64748b; font-size: 11px;")
        form.addRow(lbl_mp, self._manual_place)

        # Pose
        self._cx = self._make_dbl(-500, 500, 0.01, 2)
        self._cy = self._make_dbl(-500, 500, 0.01, 2)
        self._cz = self._make_dbl(-10, 20, 0.01, 2)
        self._cx.setToolTip(
            "Editable 3D box centre. Table columns X/Y/Z show radar medians "
            "inside the box when returns exist.")
        add_row("cx", "Centre X (m)", self._cx)
        self._cy.setToolTip(self._cx.toolTip())
        add_row("cy", "Centre Y (m)", self._cy)
        self._cz.setToolTip(self._cx.toolTip())
        add_row("cz", "Centre Z (m)", self._cz)

        # Size
        self._l = self._make_dbl(0.1, 30, 0.01, 2)
        self._w = self._make_dbl(0.1, 10, 0.01, 2)
        self._h = self._make_dbl(0.1, 10, 0.01, 2)
        add_row("length", "Length (m)", self._l)
        add_row("width", "Width (m)", self._w)
        add_row("height", "Height (m)", self._h)

        # Orientation
        self._yaw = self._make_dbl(-360, 360, 0.1, 2)
        self._pitch = self._make_dbl(-90, 90, 0.1, 2)
        self._roll = self._make_dbl(-90, 90, 0.1, 2)
        add_row("yaw", "Yaw (deg)", self._yaw)
        add_row("pitch", "Pitch (deg)", self._pitch)
        add_row("roll", "Roll (deg)", self._roll)

        # QA
        self._trunc = self._make_dbl(0, 1, 0.01, 2)
        add_row("truncation", "Truncation (0–1)", self._trunc)
        self._occ = QSpinBox(); self._occ.setRange(0, 3)
        add_row("occlusion", "Occlusion (0–3)", self._occ)
        self._conf = self._make_dbl(0, 1, 0.01, 2)
        add_row("confidence", "Confidence (0–1)", self._conf)

        # Derived (read-only)
        self._num_pts = QLabel("—")
        self._num_pts.setStyleSheet("color: #1e293b; font-size: 11px;")
        lb = QLabel("Number of Radar Points")
        lb.setStyleSheet("color: #64748b; font-size: 11px;")
        form.addRow(lb, self._num_pts)

        self._rad_rx = QLabel("—")
        self._rad_rx.setStyleSheet("color: #1e293b; font-size: 11px;")
        lrx = QLabel("Radar X median (m)")
        lrx.setStyleSheet("color: #64748b; font-size: 11px;")
        lrx.setToolTip("Median X of returns inside box (radar range axis); — if none.")
        form.addRow(lrx, self._rad_rx)

        self._rad_ry = QLabel("—")
        self._rad_ry.setStyleSheet("color: #1e293b; font-size: 11px;")
        lry = QLabel("Radar Y median (m)")
        lry.setStyleSheet("color: #64748b; font-size: 11px;")
        form.addRow(lry, self._rad_ry)

        self._rad_rz = QLabel("—")
        self._rad_rz.setStyleSheet("color: #1e293b; font-size: 11px;")
        lrz = QLabel("Radar Z median (m)")
        lrz.setStyleSheet("color: #64748b; font-size: 11px;")
        form.addRow(lrz, self._rad_rz)

        self._doppler = QLabel("—")
        self._doppler.setStyleSheet("color: #1e293b; font-size: 11px;")
        lb2 = QLabel("Doppler / Velocity (m/s)")
        lb2.setStyleSheet("color: #64748b; font-size: 11px;")
        form.addRow(lb2, self._doppler)

        self._distance = QLabel("—")
        self._distance.setStyleSheet("color: #1e293b; font-size: 11px;")
        lb3 = QLabel("Range X (m)")
        lb3.setToolTip(
            "Median radar X inside box (forward range axis); else box centre X.")
        lb3.setStyleSheet("color: #64748b; font-size: 11px;")
        form.addRow(lb3, self._distance)

        self._azimuth = QLabel("—")
        self._azimuth.setStyleSheet("color: #1e293b; font-size: 11px;")
        lb4 = QLabel("Azimuth (deg)")
        lb4.setToolTip("atan2(Y, X) using radar median XY when returns exist.")
        lb4.setStyleSheet("color: #64748b; font-size: 11px;")
        form.addRow(lb4, self._azimuth)

        self._visibility = QLabel("—")
        self._visibility.setStyleSheet("color: #1e293b; font-size: 11px;")
        lb5 = QLabel("Visibility in Image (%)")
        lb5.setStyleSheet("color: #64748b; font-size: 11px;")
        form.addRow(lb5, self._visibility)

        self._kitti_cam_xyz = QLabel("—")
        self._kitti_cam_xyz.setStyleSheet(
            "color: #1e293b; font-size: 11px; font-family: monospace;")
        lb_k = QLabel("VoD/KITTI export xyz (cam)")
        lb_k.setStyleSheet("color: #64748b; font-size: 11px;")
        lb_k.setToolTip(
            "Bottom-face centre in rectified camera coords — the three numbers "
            "before the final rotation in VoD/KITTI export. Camera: X right, Y down, Z forward. "
            "Compare to Centre X/Y/Z (master geometric centre) or Radar median only "
            "after transforming frames; they will not match numerically.")
        form.addRow(lb_k, self._kitti_cam_xyz)

        # Notes
        self._notes = QPlainTextEdit()
        self._notes.setFixedHeight(50)
        self._notes.setStyleSheet(
            "QPlainTextEdit { border: 1px solid #e2e8f0; border-radius: 4px; "
            "background: #ffffff; color: #1e293b; font-size: 11px; padding: 4px; }")
        lb_notes = QLabel("Notes")
        lb_notes.setStyleSheet("color: #64748b; font-size: 11px;")
        form.addRow(lb_notes, self._notes)

        root.addWidget(self._form_container)
        self._form_container.setVisible(False)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setStyleSheet(
            "QPushButton { background: #2563eb; color: white; "
            "border-radius: 4px; padding: 7px 16px; font-weight: 600; font-size: 11px; } "
            "QPushButton:hover { background: #1d4ed8; } "
            "QPushButton:disabled { background: #93c5fd; color: #e0e7ff; }")
        self._apply_btn.clicked.connect(self._on_apply)

        self._reset_btn = QPushButton("Reset Dims")
        self._reset_btn.setToolTip(
            "Reset Length / Width / Height to class defaults")
        self._reset_btn.setStyleSheet(
            "QPushButton { background: #ffffff; color: #0369a1; "
            "border: 1px solid #7dd3fc; border-radius: 4px; padding: 7px 10px; "
            "font-size: 11px; } QPushButton:hover { background: #f0f9ff; }")
        self._reset_btn.clicked.connect(self._on_reset_dimensions)

        self._duplicate_btn = QPushButton("Duplicate")
        self._duplicate_btn.setStyleSheet(
            "QPushButton { background: #ffffff; color: #475569; "
            "border: 1px solid #cbd5e1; border-radius: 4px; padding: 7px 16px; "
            "font-size: 11px; } QPushButton:hover { background: #f1f5f9; }")
        self._duplicate_btn.clicked.connect(self._on_duplicate)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setStyleSheet(
            "QPushButton { background: #ffffff; color: #dc2626; "
            "border: 1px solid #fca5a5; border-radius: 4px; padding: 7px 16px; "
            "font-size: 11px; } QPushButton:hover { background: #fef2f2; }")
        self._delete_btn.clicked.connect(self._on_delete)

        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(self._reset_btn)
        btn_row.addWidget(self._duplicate_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._delete_btn)
        self._btn_container = QWidget(); self._btn_container.setLayout(btn_row)
        root.addWidget(self._btn_container)
        self._btn_container.setVisible(False)

        root.addStretch(1)

        # Wire signals
        self.model.selection_changed.connect(lambda _u: self._on_selection_changed())
        self.model.objects_changed.connect(self._on_model_changed)

        # Any edit marks "pending" so Apply becomes active
        self._manual_place.stateChanged.connect(self._mark_dirty)

        for w in (self._obj_id, self._track_id, self._class, self._cx,
                  self._cy, self._cz, self._l, self._w, self._h, self._yaw,
                  self._pitch, self._roll, self._trunc, self._occ, self._conf):
            if isinstance(w, QLineEdit):
                w.textEdited.connect(self._mark_dirty)
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(self._mark_dirty)
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.valueChanged.connect(self._mark_dirty)
        self._notes.textChanged.connect(self._mark_dirty)
        self._class.currentTextChanged.connect(self._on_class_changed)

        self._set_apply_enabled(False)

    # ------------------------------------------------------------------
    def _input_stylesheet(self) -> str:
        return (
            "QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox { "
            "background: #ffffff; color: #1e293b; "
            "border: 1px solid #e2e8f0; border-radius: 4px; "
            "padding: 4px 6px; font-size: 11px; min-width: 90px; "
            "}"
            "QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, "
            "QComboBox:focus { border: 1px solid #2563eb; }"
            "QComboBox::drop-down { border: none; }"
        )

    def _make_dbl(self, lo, hi, step, dp) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi); s.setSingleStep(step); s.setDecimals(dp)
        return s

    def set_context(self, calibration, radar_points, image_size) -> None:
        self._calibration = calibration
        self._radar_points = radar_points if radar_points is not None else np.zeros((0, 3))
        self._image_size = tuple(image_size)
        self._refresh_derived()

    # ------------------------------------------------------------------
    def _on_selection_changed(self) -> None:
        self._load_from_model()

    def _on_model_changed(self) -> None:
        # Radar view / drag -> update primitive values without marking as
        # pending Apply. Derived read-outs also refresh.
        if self._updating_from_model:
            return
        # If an Apply is pending (user is mid-typing), don't yank their
        # typed values back from the model.
        if self._has_pending_changes:
            self._refresh_derived()
            return
        self._load_from_model()

    def _load_from_model(self) -> None:
        sel = self.model.selected()
        if sel is None:
            self._empty_lbl.setVisible(True)
            self._form_container.setVisible(False)
            self._btn_container.setVisible(False)
            self._title.setText("Selected Object Details")
            self._has_pending_changes = False
            self._set_apply_enabled(False)
            return

        self._updating_from_model = True
        try:
            self._empty_lbl.setVisible(False)
            self._form_container.setVisible(True)
            self._btn_container.setVisible(True)
            self._title.setText("Selected Object Details")

            self._obj_id.setValue(sel.object_id)
            self._track_id.setText("" if sel.track_id is None else str(sel.track_id))

            idx = self._class.findText(sel.class_name)
            if idx >= 0:
                self._class.setCurrentIndex(idx)
            else:
                self._class.setEditText(sel.class_name)

            self._manual_place.setChecked(sel.manual_placement)

            self._cx.setValue(sel.x)
            self._cy.setValue(sel.y)
            self._cz.setValue(sel.z)
            self._l.setValue(sel.length)
            self._w.setValue(sel.width)
            self._h.setValue(sel.height)
            self._yaw.setValue(np.rad2deg(sel.yaw))
            self._pitch.setValue(np.rad2deg(sel.pitch))
            self._roll.setValue(np.rad2deg(sel.roll))

            self._trunc.setValue(sel.truncation)
            self._occ.setValue(sel.occlusion)
            self._conf.setValue(sel.confidence)
            if self._notes.toPlainText() != sel.notes:
                self._notes.setPlainText(sel.notes)

            self._refresh_derived()

            self._has_pending_changes = False
            self._set_apply_enabled(False)
        finally:
            self._updating_from_model = False

    def _refresh_derived(self) -> None:
        sel = self.model.selected()
        if sel is None:
            return
        # Radar points inside
        if self._radar_points.size > 0:
            n = int(np.count_nonzero(sel.points_inside(self._radar_points[:, :3])))
        else:
            n = 0
        self._num_pts.setText(str(n))

        med = radar_returns_median_xyz(sel, self._radar_points)
        if med is None:
            self._rad_rx.setText("—")
            self._rad_ry.setText("—")
            self._rad_rz.setText("—")
            rx, ry = float(sel.x), float(sel.y)
        else:
            self._rad_rx.setText(f"{float(med[0]):.2f}")
            self._rad_ry.setText(f"{float(med[1]):.2f}")
            self._rad_rz.setText(f"{float(med[2]):.2f}")
            rx, ry = float(med[0]), float(med[1])

        dist = radar_range_from_x_m(sel, self._radar_points)
        az = radar_azimuth_deg_from_xy(rx, ry)
        self._distance.setText(f"{dist:.2f}")
        self._azimuth.setText(f"{az:.2f}")

        # Mean Doppler of points inside. VoD columns:
        # x, y, z, RCS, v_r, v_r_compensated, time.
        if self._radar_points.size > 0 and self._radar_points.shape[1] >= 5:
            mask = sel.points_inside(self._radar_points[:, :3])
            if np.any(mask):
                vel_col = (
                    COL_VR_COMP
                    if self._radar_points.shape[1] > COL_VR_COMP
                    else COL_VR
                )
                self._doppler.setText(
                    f"{float(self._radar_points[mask, vel_col].mean()):+.2f}")
            else:
                self._doppler.setText("—")
        else:
            self._doppler.setText("—")

        # Visibility % (fraction of projected corners in image)
        if self._calibration is not None and self._image_size[0] > 0:
            corners_uv, depths = project_box_calibration(sel, self._calibration)
            if np.any(depths > 0):
                front = depths > 0
                uv = corners_uv[front]
                W, H = self._image_size
                in_img = ((uv[:, 0] >= 0) & (uv[:, 0] < W) &
                          (uv[:, 1] >= 0) & (uv[:, 1] < H))
                vis = 100.0 * float(np.count_nonzero(in_img)) / max(front.sum(), 1)
                self._visibility.setText(f"{vis:.1f}")
            else:
                self._visibility.setText("0.0")
        else:
            self._visibility.setText("—")

        lc = box_location_cam_kitti(sel, self._calibration)
        self._kitti_cam_xyz.setText(
            f"{lc[0]:.2f}   {lc[1]:.2f}   {lc[2]:.2f}")

    # ------------------------------------------------------------------
    def _mark_dirty(self, *_) -> None:
        if self._updating_from_model:
            return
        self._has_pending_changes = True
        self._set_apply_enabled(True)

    def _set_apply_enabled(self, enabled: bool) -> None:
        self._apply_btn.setEnabled(enabled)

    def _on_class_changed(self, class_name: str) -> None:
        if self._updating_from_model:
            return
        length, width, height = class_default_box_size(class_name.strip())
        base_z = self._cz.value() - self._h.value() * 0.5
        self._l.setValue(length)
        self._w.setValue(width)
        self._h.setValue(height)
        self._cz.setValue(base_z + height * 0.5)
        self._mark_dirty()

    def _on_apply(self) -> None:
        sel = self.model.selected()
        if sel is None:
            return

        sel.object_id = self._obj_id.value()
        t = self._track_id.text().strip()
        sel.track_id = None if t == "" else (int(t) if t.isdigit() or
                                              (t.startswith("-") and t[1:].isdigit())
                                              else None)
        sel.class_name = self._class.currentText().strip() or "Object"

        sel.x = self._cx.value(); sel.y = self._cy.value(); sel.z = self._cz.value()
        sel.length = self._l.value(); sel.width = self._w.value(); sel.height = self._h.value()
        sel.yaw = np.deg2rad(self._yaw.value())
        sel.pitch = np.deg2rad(self._pitch.value())
        sel.roll = np.deg2rad(self._roll.value())

        sel.truncation = self._trunc.value()
        sel.occlusion = self._occ.value()
        sel.confidence = self._conf.value()
        sel.notes = self._notes.toPlainText()
        sel.manual_placement = self._manual_place.isChecked()

        self.model.update(sel, snapshot=True)
        self._has_pending_changes = False
        self._set_apply_enabled(False)

    def _on_reset_dimensions(self) -> None:
        """Reset L/W/H fields to class defaults (does not auto-apply)."""
        class_name = self._class.currentText().strip()
        length, width, height = class_default_box_size(class_name)
        base_z = self._cz.value() - self._h.value() * 0.5
        self._updating_from_model = True
        try:
            self._l.setValue(length)
            self._w.setValue(width)
            self._h.setValue(height)
            self._cz.setValue(base_z + height * 0.5)
        finally:
            self._updating_from_model = False
        self._mark_dirty()

    def _on_duplicate(self) -> None:
        uid = self.model.selected_uid
        if uid:
            self.model.duplicate(uid)

    def _on_delete(self) -> None:
        sel = self.model.selected()
        if sel is None:
            return
        r = QMessageBox.question(
            self, "Delete object",
            f"Delete {sel.class_name} #{sel.object_id}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r == QMessageBox.Yes:
            self.model.delete(sel.uid)
