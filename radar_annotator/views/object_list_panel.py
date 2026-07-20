"""
Object List panel (bottom of the window) — a synchronised table view of
every Box3D in the current frame with per-row Edit / Delete actions.

Click a row -> selects the box in the shared model (every other view updates
automatically). Edit pencil -> focuses the right-hand Selected Object panel.
Trash -> deletes after confirmation.
"""
from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPixmap, QPainter, QBrush, QPen
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QAbstractItemView, QPushButton, QMessageBox,
                               QSizePolicy)

from ..core.annotation_model import AnnotationModel
from ..core.geometry import (
    box_2d_from_projection,
    project_box_calibration,
    radar_azimuth_deg_from_xy,
    radar_display_xyz_m,
    radar_range_from_x_m,
)
from .annotation_classes import class_rgb


COLUMNS = [
    ("ID", 44), ("Track ID", 72), ("Class", 110),
    ("X (m)", 70), ("Y (m)", 70), ("Z (m)", 70),
    ("L (m)", 70), ("W (m)", 70), ("H (m)", 70),
    ("Yaw (deg)", 86), ("Radar Pts", 82), ("Visible (%)", 88),
    ("Occl.", 56), ("Conf.", 60),
    ("Range X (m)", 100), ("Azimuth (deg)", 110),
    ("Actions", 80),
]


class ObjectListPanel(QWidget):
    """Synchronised table of frame-level objects."""

    edit_requested = Signal()   # user clicked the pencil — main window focuses right panel

    def __init__(self, model: AnnotationModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._syncing = False   # guard: table -> selection vs selection -> table
        self._calibration = None
        self._image_size = (0, 0)
        self._radar_points = np.zeros((0, 3))

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(6)

        title = QLabel("Object List (All Objects in Frame)")
        f = title.font(); f.setBold(True); f.setPointSize(10)
        title.setFont(f)
        root.addWidget(title)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([c[0] for c in COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setFocusPolicy(Qt.ClickFocus)
        self.table.setStyleSheet("""
            QTableWidget {
                background: #ffffff;
                alternate-background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                color: #1e293b;
                gridline-color: #e2e8f0;
            }
            QTableWidget::item:selected {
                background: #dbeafe;
                color: #1e293b;
            }
            QHeaderView::section {
                background: #f1f5f9;
                color: #475569;
                border: none;
                border-right: 1px solid #e2e8f0;
                padding: 6px 8px;
                font-weight: 600;
            }
            QHeaderView { background: #f1f5f9; }
        """)

        header = self.table.horizontalHeader()
        for i, (_, w) in enumerate(COLUMNS):
            header.setSectionResizeMode(i, QHeaderView.Interactive)
            self.table.setColumnWidth(i, w)
        # Let the Actions column stay at fixed width
        self.table.setMinimumHeight(160)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)

        root.addWidget(self.table)

        # Wire model signals
        self.model.objects_changed.connect(self._refresh)
        self.model.selection_changed.connect(self._sync_selection_to_table)
        self.model.frame_loaded.connect(self._refresh)

    # ------------------------------------------------------------------
    def set_context(self, calibration, radar_points, image_size) -> None:
        self._calibration = calibration
        self._radar_points = radar_points if radar_points is not None else np.zeros((0, 3))
        self._image_size = tuple(image_size)
        self._refresh()

    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        self._syncing = True
        try:
            self.table.setRowCount(len(self.model.objects))
            for row, box in enumerate(self.model.objects):
                self._populate_row(row, box)
            self._sync_selection_to_table(self.model.selected_uid)
        finally:
            self._syncing = False

    def _populate_row(self, row: int, box) -> None:
        # Compute derived fields
        if self._radar_points.size > 0:
            n_pts = int(np.count_nonzero(
                box.points_inside(self._radar_points[:, :3])))
        else:
            n_pts = 0

        rx, ry, rz = radar_display_xyz_m(box, self._radar_points)
        radar_rng = radar_range_from_x_m(box, self._radar_points)
        azimuth = radar_azimuth_deg_from_xy(rx, ry)

        visible = 0.0
        if self._calibration is not None and self._image_size[0] > 0:
            corners_uv, depths = project_box_calibration(box, self._calibration)
            W, H = self._image_size
            if np.any(depths > 0):
                # Fraction of corners that land inside the image
                front = depths > 0
                uv = corners_uv[front]
                in_img = (
                    (uv[:, 0] >= 0) & (uv[:, 0] < W) &
                    (uv[:, 1] >= 0) & (uv[:, 1] < H)
                )
                visible = 100.0 * float(np.count_nonzero(in_img)) / max(front.sum(), 1)

        # Cell values
        values = [
            str(box.object_id),
            str(box.track_id) if box.track_id is not None else "—",
            None,  # Class with dot — handled below
            f"{rx:.2f}", f"{ry:.2f}", f"{rz:.2f}",
            f"{box.length:.2f}", f"{box.width:.2f}", f"{box.height:.2f}",
            f"{np.rad2deg(box.yaw):.2f}",
            str(n_pts),
            f"{visible:.1f}",
            str(box.occlusion),
            f"{box.confidence:.2f}",
            f"{radar_rng:.2f}",
            f"{azimuth:.2f}",
        ]
        for col, val in enumerate(values):
            if val is None:
                continue
            item = QTableWidgetItem(val)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setForeground(QColor("#1e293b"))
            item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            item.setData(Qt.UserRole, box.uid)
            self.table.setItem(row, col, item)

        # Class cell with colored dot
        class_widget = QWidget()
        hl = QHBoxLayout(class_widget)
        hl.setContentsMargins(8, 0, 8, 0)
        hl.setSpacing(6)
        dot = QLabel()
        r, g, b = class_rgb(box.class_name)
        dot.setStyleSheet(
            f"background: rgb({r}, {g}, {b}); border-radius: 5px; "
            f"min-width: 10px; max-width: 10px; "
            f"min-height: 10px; max-height: 10px;")
        hl.addWidget(dot)
        lbl = QLabel(box.class_name)
        lbl.setStyleSheet("color: #1e293b;")
        hl.addWidget(lbl)
        hl.addStretch(1)
        self.table.setCellWidget(row, 2, class_widget)

        # Actions column
        action_widget = QWidget()
        ah = QHBoxLayout(action_widget)
        ah.setContentsMargins(6, 0, 6, 0)
        ah.setSpacing(4)
        edit_btn = QPushButton("✎")
        edit_btn.setToolTip("Edit (focus property panel)")
        edit_btn.setFixedSize(26, 22)
        edit_btn.setStyleSheet(
            "QPushButton { background: #e2e8f0; color: #334155; border-radius: 3px; "
            "font-size: 12px; } QPushButton:hover { background: #cbd5e1; }")
        edit_btn.clicked.connect(lambda _, uid=box.uid: self._on_edit_clicked(uid))

        del_btn = QPushButton("🗑")
        del_btn.setToolTip("Delete")
        del_btn.setFixedSize(26, 22)
        del_btn.setStyleSheet(
            "QPushButton { background: #fee2e2; color: #b91c1c; border-radius: 3px; "
            "font-size: 12px; } QPushButton:hover { background: #fecaca; }")
        del_btn.clicked.connect(lambda _, uid=box.uid: self._on_delete_clicked(uid))

        ah.addWidget(edit_btn)
        ah.addWidget(del_btn)
        ah.addStretch(1)
        self.table.setCellWidget(row, len(COLUMNS) - 1, action_widget)

    # ------------------------------------------------------------------
    def _sync_selection_to_table(self, uid) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.table.clearSelection()
            if uid is None:
                return
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                if item and item.data(Qt.UserRole) == uid:
                    self.table.selectRow(row)
                    break
        finally:
            self._syncing = False

    def _on_table_selection_changed(self) -> None:
        if self._syncing:
            return
        selected = self.table.selectedItems()
        if not selected:
            return
        uid = selected[0].data(Qt.UserRole)
        if uid:
            self._syncing = True
            try:
                self.model.select(uid)
            finally:
                self._syncing = False

    def _on_edit_clicked(self, uid: str) -> None:
        self.model.select(uid)
        self.edit_requested.emit()

    def _on_delete_clicked(self, uid: str) -> None:
        box = self.model.find(uid)
        if box is None:
            return
        r = QMessageBox.question(
            self, "Delete object",
            f"Delete {box.class_name} #{box.object_id}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r == QMessageBox.Yes:
            self.model.delete(uid)
