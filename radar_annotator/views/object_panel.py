"""
Right panel: Selected Object Details (Section 3.4 of spec).

Shows read-only fields and editable numeric spinners / class dropdown for the
currently selected box. All edits write back to the shared AnnotationModel.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QLabel,
                               QDoubleSpinBox, QSpinBox, QComboBox, QGroupBox,
                               QLineEdit, QPlainTextEdit, QScrollArea, QFrame)

from ..core.annotation_model import AnnotationModel
from ..core.geometry import Box3D, project_box_calibration, box_2d_from_projection
from ..core.calibration import Calibration
from .annotation_classes import DEFAULT_CLASSES


class ObjectDetailsPanel(QScrollArea):
    def __init__(self, model: AnnotationModel, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.model = model
        self._calibration = Calibration.identity()
        self._radar_points = np.zeros((0, 3))
        self._image_size = (1920, 1080)
        self._updating_from_model = False  # guard against feedback loop

        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(8)
        self.setWidget(self._container)

        self._title = QLabel("Selected Object")
        title_font = self._title.font()
        title_font.setBold(True)
        title_font.setPointSize(11)
        self._title.setFont(title_font)
        self._layout.addWidget(self._title)

        self._no_selection = QLabel("No object selected.\n\n"
                                    "Click a box in either view, or press N "
                                    "to create a new one.")
        self._no_selection.setStyleSheet("color: #999;")
        self._layout.addWidget(self._no_selection)

        self._details = self._build_details_widget()
        self._layout.addWidget(self._details)
        self._details.setVisible(False)

        self._layout.addStretch(1)

        self.model.selection_changed.connect(self._on_selection_changed)
        self.model.objects_changed.connect(self._refresh_from_model)

    # ------------------------------------------------------------------
    def set_context(self, calibration: Calibration,
                    radar_points: np.ndarray,
                    image_size) -> None:
        self._calibration = calibration
        self._radar_points = radar_points if radar_points is not None else np.zeros((0, 3))
        self._image_size = tuple(image_size)
        self._refresh_from_model()

    # ------------------------------------------------------------------
    def _build_details_widget(self) -> QWidget:
        w = QFrame()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Identity ---
        id_group = QGroupBox("Identity")
        id_form = QFormLayout(id_group)
        self._obj_id_spin = QSpinBox()
        self._obj_id_spin.setRange(1, 1_000_000)
        self._obj_id_spin.valueChanged.connect(self._on_field_changed)
        id_form.addRow("Object ID", self._obj_id_spin)

        self._track_id_edit = QLineEdit()
        self._track_id_edit.setPlaceholderText("(optional)")
        self._track_id_edit.editingFinished.connect(self._on_field_changed)
        id_form.addRow("Track ID", self._track_id_edit)

        self._class_combo = QComboBox()
        self._class_combo.setEditable(True)
        self._class_combo.addItems(DEFAULT_CLASSES)
        self._class_combo.currentTextChanged.connect(self._on_field_changed)
        id_form.addRow("Class", self._class_combo)

        layout.addWidget(id_group)

        # --- Pose ---
        pose_group = QGroupBox("Pose (master frame)")
        pose_form = QFormLayout(pose_group)
        self._x_spin = self._make_double(-500, 500, 0.1, 3)
        self._y_spin = self._make_double(-500, 500, 0.1, 3)
        self._z_spin = self._make_double(-10, 20, 0.05, 3)
        pose_form.addRow("X (m)", self._x_spin)
        pose_form.addRow("Y (m)", self._y_spin)
        pose_form.addRow("Z (m)", self._z_spin)

        self._yaw_spin = self._make_double(-360, 360, 1.0, 2, suffix=" °")
        self._pitch_spin = self._make_double(-90, 90, 0.5, 2, suffix=" °")
        self._roll_spin = self._make_double(-90, 90, 0.5, 2, suffix=" °")
        pose_form.addRow("Yaw", self._yaw_spin)
        pose_form.addRow("Pitch", self._pitch_spin)
        pose_form.addRow("Roll", self._roll_spin)

        layout.addWidget(pose_group)

        # --- Size ---
        size_group = QGroupBox("Dimensions (m)")
        size_form = QFormLayout(size_group)
        self._l_spin = self._make_double(0.1, 30, 0.05, 3)
        self._w_spin = self._make_double(0.1, 10, 0.05, 3)
        self._h_spin = self._make_double(0.1, 10, 0.05, 3)
        size_form.addRow("Length", self._l_spin)
        size_form.addRow("Width", self._w_spin)
        size_form.addRow("Height", self._h_spin)
        layout.addWidget(size_group)

        # --- QA ---
        qa_group = QGroupBox("QA & attributes")
        qa_form = QFormLayout(qa_group)
        self._occ_combo = QComboBox()
        self._occ_combo.addItems(["0 - visible", "1 - partial",
                                  "2 - heavy", "3 - unknown"])
        self._occ_combo.currentIndexChanged.connect(self._on_field_changed)
        qa_form.addRow("Occlusion", self._occ_combo)

        self._trunc_spin = self._make_double(0, 1, 0.05, 2)
        qa_form.addRow("Truncation", self._trunc_spin)

        self._conf_spin = self._make_double(0, 1, 0.05, 2)
        qa_form.addRow("Confidence", self._conf_spin)

        self._notes_edit = QPlainTextEdit()
        self._notes_edit.setFixedHeight(60)
        self._notes_edit.textChanged.connect(self._on_field_changed)
        qa_form.addRow("Notes", self._notes_edit)

        layout.addWidget(qa_group)

        # --- Derived (read-only) ---
        derived_group = QGroupBox("Derived")
        derived_form = QFormLayout(derived_group)
        self._num_pts_label = QLabel("—")
        self._bbox2d_label = QLabel("—")
        self._bbox2d_label.setWordWrap(True)
        derived_form.addRow("Radar points inside", self._num_pts_label)
        derived_form.addRow("Projected 2D bbox", self._bbox2d_label)
        layout.addWidget(derived_group)

        return w

    def _make_double(self, lo, hi, step, dp, suffix: str = "") -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(dp)
        if suffix:
            sp.setSuffix(suffix)
        sp.valueChanged.connect(self._on_field_changed)
        return sp

    # ------------------------------------------------------------------
    def _on_selection_changed(self, _uid) -> None:
        self._refresh_from_model()

    def _refresh_from_model(self) -> None:
        sel = self.model.selected()
        if sel is None:
            self._no_selection.setVisible(True)
            self._details.setVisible(False)
            self._title.setText("Selected Object")
            return

        self._updating_from_model = True
        try:
            self._no_selection.setVisible(False)
            self._details.setVisible(True)
            self._title.setText(f"Selected: {sel.class_name} #{sel.object_id}")

            self._obj_id_spin.setValue(sel.object_id)
            self._track_id_edit.setText(
                "" if sel.track_id is None else str(sel.track_id))

            # Class combo
            idx = self._class_combo.findText(sel.class_name)
            if idx >= 0:
                self._class_combo.setCurrentIndex(idx)
            else:
                self._class_combo.setEditText(sel.class_name)

            self._x_spin.setValue(sel.x)
            self._y_spin.setValue(sel.y)
            self._z_spin.setValue(sel.z)
            self._yaw_spin.setValue(np.rad2deg(sel.yaw))
            self._pitch_spin.setValue(np.rad2deg(sel.pitch))
            self._roll_spin.setValue(np.rad2deg(sel.roll))

            self._l_spin.setValue(sel.length)
            self._w_spin.setValue(sel.width)
            self._h_spin.setValue(sel.height)

            self._occ_combo.setCurrentIndex(max(0, min(3, sel.occlusion)))
            self._trunc_spin.setValue(sel.truncation)
            self._conf_spin.setValue(sel.confidence)

            if self._notes_edit.toPlainText() != sel.notes:
                self._notes_edit.setPlainText(sel.notes)

            # Derived
            if self._radar_points.size > 0:
                n = int(np.count_nonzero(
                    sel.points_inside(self._radar_points[:, :3])))
            else:
                n = 0
            self._num_pts_label.setText(str(n))

            corners_uv, depths = project_box_calibration(sel, self._calibration)
            bbox2d = box_2d_from_projection(corners_uv, depths, self._image_size)
            if bbox2d is None:
                self._bbox2d_label.setText("out of view")
            else:
                x0, y0, x1, y1 = bbox2d
                self._bbox2d_label.setText(
                    f"[{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}]")
        finally:
            self._updating_from_model = False

    # ------------------------------------------------------------------
    def _on_field_changed(self, *args) -> None:
        if self._updating_from_model:
            return
        sel = self.model.selected()
        if sel is None:
            return

        sel.object_id = self._obj_id_spin.value()

        track_text = self._track_id_edit.text().strip()
        if track_text == "":
            sel.track_id = None
        else:
            try:
                sel.track_id = int(track_text)
            except ValueError:
                sel.track_id = None

        sel.class_name = self._class_combo.currentText().strip() or "Object"

        sel.x = self._x_spin.value()
        sel.y = self._y_spin.value()
        sel.z = self._z_spin.value()
        sel.yaw = np.deg2rad(self._yaw_spin.value())
        sel.pitch = np.deg2rad(self._pitch_spin.value())
        sel.roll = np.deg2rad(self._roll_spin.value())

        sel.length = self._l_spin.value()
        sel.width = self._w_spin.value()
        sel.height = self._h_spin.value()

        sel.occlusion = self._occ_combo.currentIndex()
        sel.truncation = self._trunc_spin.value()
        sel.confidence = self._conf_spin.value()
        sel.notes = self._notes_edit.toPlainText()

        self.model.update(sel, snapshot=True)
