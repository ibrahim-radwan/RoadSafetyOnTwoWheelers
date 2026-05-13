"""
Main window v2 — matches the reference mock-up.

Layout:
    [Top bar: title + window controls]
    [Toolbar: Select Folder | Load Calibration | Prev | Next | Go To |
              Save | Save All | Export KITTI | New Box |
              Delete Box | Snap to Ground | Auto-Fit | view-mode switcher]
    ┌─────────────────┬────────────────────────┬───────────────────┬─────────┐
    │ Loaded Example  │ Radar / BEV / Point    │ Camera Image View │ Selected│
    │ Details         │ Cloud View             │                   │ Object  │
    │                 │                        │                   │ Details │
    │ (fields)        │ (BEV canvas)           │ (image canvas)    │ (form + │
    │                 │                        │                   │ Apply / │
    │ Mini Preview    │                        │                   │ Dup /   │
    │ (BEV + Cam)     │                        │                   │ Delete) │
    ├─────────────────┴────────────────────────┴───────────────────┴─────────┤
    │ Object List (All Objects in Frame)                                     │
    │ (table: ID | Track | Class | X/Y/Z | L/W/H | Yaw | Radar Pts | ... )   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Status bar: Ready | Frame X / Y | Synchronized | Radar: N | Auto-Save  │
    └─────────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import sys
import traceback
import json
from pathlib import Path

# Running `python .../main_window_v2.py` sets __name__ == "__main__" with no
# package context; relative imports need the parent of this folder on sys.path
# and __package__ set to this directory's name (same layout as `python -m`).
if not __package__:
    _pkg_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(_pkg_dir.parent))
    __package__ = _pkg_dir.name
from typing import List, Optional
import datetime as _dt
import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (QAction, QKeySequence, QIcon, QPalette, QColor,
                           QFont, QPixmap, QPainter, QBrush, QPen, QImage)
from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                               QSplitter, QFileDialog, QMessageBox, QToolBar,
                               QComboBox, QLabel, QStatusBar, QApplication,
                               QInputDialog, QPushButton, QFrame, QSpinBox,
                               QDoubleSpinBox, QSizePolicy, QToolButton,
                               QStyle, QWidgetAction, QCheckBox, QStackedWidget)

from .core.annotation_model import AnnotationModel
from .core.calibration import Calibration, iter_calibration_candidate_paths
from .core.dataset import Dataset, load_radar_points, Frame
from .core.radar_filters import apply_radar_filters, describe_filter_settings
from .core.object_clustering import ClusterConfig, propose_initial_objects, suggest_eps
from .core.geometry import Box3D, project_box_calibration, box_2d_from_projection
from .views.radar_view import RadarBEVView
from .views.radar_perspective_view import RadarPerspectiveView
from .views.image_view import CameraImageView
from .views.object_panel_v2 import ObjectDetailsPanelV2
from .views.annotation_classes import (
    OBJECT_CLASS_CHOICES,
    class_default_box_size,
)
from .views.frame_info_panel_v2 import FrameInfoPanelV2
from .views.object_list_panel import ObjectListPanel
from .io.internal_json import save_frame_labels, load_frame_labels
from .io.kitti_export import export_kitti_label_dir
from .io.image_bbox_export import export_annotations


# ---------------------------------------------------------------------------
# Simple icon factory — draws glyphs onto QPixmaps so we don't depend on a
# bundled icon theme. Each function returns a QIcon.
# ---------------------------------------------------------------------------

def _icon_pixmap(draw_fn, size: int = 18) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    draw_fn(p, size)
    p.end()
    return QIcon(pix)


def _stroke(p: QPainter, color="#475569", width=1.6) -> None:
    pen = QPen(QColor(color)); pen.setWidthF(width)
    pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen); p.setBrush(Qt.NoBrush)


def icon_folder():
    def draw(p, s):
        _stroke(p)
        p.drawRoundedRect(2, 5, s - 4, s - 7, 2, 2)
        p.drawLine(2, 7, s // 2, 7)
        p.drawLine(s // 2, 4, s // 2, 7)
    return _icon_pixmap(draw)


def icon_calib():
    def draw(p, s):
        _stroke(p)
        p.drawRect(3, 3, s - 6, s - 6)
        p.drawLine(s // 2, 4, s // 2, s - 4)
        p.drawLine(4, s // 2, s - 4, s // 2)
    return _icon_pixmap(draw)


def icon_prev():
    def draw(p, s):
        _stroke(p)
        p.drawLine(11, 4, 5, s // 2)
        p.drawLine(5, s // 2, 11, s - 4)
    return _icon_pixmap(draw)


def icon_next():
    def draw(p, s):
        _stroke(p)
        p.drawLine(7, 4, 13, s // 2)
        p.drawLine(13, s // 2, 7, s - 4)
    return _icon_pixmap(draw)


def icon_save():
    def draw(p, s):
        _stroke(p)
        p.drawRoundedRect(3, 3, s - 6, s - 6, 1.5, 1.5)
        p.drawLine(6, 3, 6, 7)
        p.drawLine(s - 6, 3, s - 6, 7)
        p.drawLine(5, s - 5, s - 5, s - 5)
    return _icon_pixmap(draw)


def icon_export():
    def draw(p, s):
        _stroke(p)
        p.drawLine(s // 2, 3, s // 2, s - 6)
        p.drawLine(s // 2 - 4, s - 10, s // 2, s - 6)
        p.drawLine(s // 2 + 4, s - 10, s // 2, s - 6)
        p.drawLine(3, s - 3, s - 3, s - 3)
    return _icon_pixmap(draw)


def icon_plus():
    def draw(p, s):
        _stroke(p, "#16a34a", 2.0)
        p.drawEllipse(3, 3, s - 6, s - 6)
        p.drawLine(s // 2, 6, s // 2, s - 6)
        p.drawLine(6, s // 2, s - 6, s // 2)
    return _icon_pixmap(draw)


def icon_delete():
    def draw(p, s):
        _stroke(p, "#dc2626", 2.0)
        p.drawEllipse(3, 3, s - 6, s - 6)
        p.drawLine(7, 7, s - 7, s - 7)
        p.drawLine(s - 7, 7, 7, s - 7)
    return _icon_pixmap(draw)


def icon_snap():
    def draw(p, s):
        _stroke(p)
        p.drawRect(5, 4, s - 10, s - 12)
        p.drawLine(3, s - 4, s - 3, s - 4)
    return _icon_pixmap(draw)


def icon_autofit():
    def draw(p, s):
        _stroke(p)
        # Corner brackets
        p.drawLine(3, 3, 3, 7); p.drawLine(3, 3, 7, 3)
        p.drawLine(s - 3, 3, s - 3, 7); p.drawLine(s - 3, 3, s - 7, 3)
        p.drawLine(3, s - 3, 3, s - 7); p.drawLine(3, s - 3, 7, s - 3)
        p.drawLine(s - 3, s - 3, s - 3, s - 7); p.drawLine(s - 3, s - 3, s - 7, s - 3)
    return _icon_pixmap(draw)


# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar–Camera 3D Annotation Tool")
        self.setMinimumSize(1280, 760)
        self.resize(1680, 960)

        self.model = AnnotationModel()
        self.dataset: Optional[Dataset] = None
        self.current_index: int = -1
        self.calibration: Calibration = Calibration.identity()
        self.current_radar_points: np.ndarray = np.zeros((0, 3))
        self.current_filtered_radar_points: np.ndarray = np.zeros((0, 3))
        self.current_image_size = (1920, 1080)
        self.current_image_qimage: Optional[QImage] = None
        self._auto_save = True
        self._active_z_range = (-3.0, 2.0)
        self._active_reflection_mode = "all"
        self._custom_rcs_min = "auto"

        self._build_ui()
        self._build_toolbar()
        self._wire_signals()
        self._apply_styles()

        self.statusBar().showMessage(
            "Ready — Select a dataset folder (Ctrl+O) to begin."
        )

    # ======================================================================
    # UI construction
    # ======================================================================
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ----- Upper split: left info | center views | right details -------
        upper_split = QSplitter(Qt.Horizontal)
        upper_split.setHandleWidth(6)
        upper_split.setChildrenCollapsible(False)

        # LEFT
        self.frame_info = FrameInfoPanelV2(self.model)
        self.frame_info.setMinimumWidth(260)
        upper_split.addWidget(self.frame_info)

        # CENTER — side-by-side radar + image
        center_split = QSplitter(Qt.Horizontal)
        center_split.setChildrenCollapsible(False)
        center_split.setHandleWidth(6)

        # Radar pane (BEV)
        radar_container = self._build_pane_container(
            "Radar / BEV / Point Cloud View",
            self._build_radar_pane())
        center_split.addWidget(radar_container)

        # Camera pane
        camera_container = self._build_pane_container(
            "Camera Image View",
            self._build_camera_pane())
        center_split.addWidget(camera_container)

        center_split.setSizes([600, 600])
        upper_split.addWidget(center_split)

        # RIGHT
        self.object_panel = ObjectDetailsPanelV2(self.model)
        self.object_panel.setMinimumWidth(330)
        upper_split.addWidget(self.object_panel)

        upper_split.setSizes([280, 1060, 360])
        upper_split.setStretchFactor(0, 0)
        upper_split.setStretchFactor(1, 1)
        upper_split.setStretchFactor(2, 0)

        root.addWidget(upper_split, 1)

        # ----- Lower: object table ----------------------------------------
        lower_frame = QFrame()
        lower_frame.setObjectName("lowerPanel")
        lower_frame.setStyleSheet(
            "#lowerPanel { background: #ffffff; border: 1px solid #e2e8f0; "
            "border-radius: 8px; }")
        lower_layout = QVBoxLayout(lower_frame)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        self.object_list = ObjectListPanel(self.model)
        lower_layout.addWidget(self.object_list)
        root.addWidget(lower_frame)

        # ----- Status bar --------------------------------------------------
        self._build_status_bar()

    def _build_pane_container(self, title: str, inner: QWidget) -> QWidget:
        """Wraps a canvas pane in a titled rounded card."""
        outer = QFrame()
        outer.setObjectName("paneCard")
        outer.setStyleSheet(
            "#paneCard { background: #ffffff; border: 1px solid #e2e8f0; "
            "border-radius: 8px; }")
        v = QVBoxLayout(outer)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(8)

        header = QHBoxLayout()
        lbl = QLabel(title)
        f = lbl.font(); f.setBold(True); f.setPointSize(11); lbl.setFont(f)
        lbl.setStyleSheet("color: #1e293b;")
        header.addWidget(lbl)
        header.addStretch(1)
        v.addLayout(header)

        # Pane toolbars: two rows each so controls do not overlap when the pane is narrow.
        if "Radar" in title:
            self._radar_tb_row1 = QHBoxLayout()
            self._radar_tb_row1.setContentsMargins(0, 0, 0, 0)
            self._radar_tb_row1.setSpacing(5)
            self._radar_tb_row2 = QHBoxLayout()
            self._radar_tb_row2.setContentsMargins(0, 0, 0, 0)
            self._radar_tb_row2.setSpacing(5)
            self._build_radar_pane_toolbar()
            tb_wrap = QVBoxLayout()
            tb_wrap.setSpacing(4)
            tb_wrap.setContentsMargins(0, 0, 0, 0)
            tb_wrap.addLayout(self._radar_tb_row1)
            tb_wrap.addLayout(self._radar_tb_row2)
            v.addLayout(tb_wrap)
        else:
            self._camera_tb_row1 = QHBoxLayout()
            self._camera_tb_row1.setSpacing(6)
            self._camera_tb_row2 = QHBoxLayout()
            self._camera_tb_row2.setSpacing(6)
            self._build_camera_pane_toolbar()
            tb_cam = QVBoxLayout()
            tb_cam.setSpacing(4)
            tb_cam.setContentsMargins(0, 0, 0, 0)
            tb_cam.addLayout(self._camera_tb_row1)
            tb_cam.addLayout(self._camera_tb_row2)
            v.addLayout(tb_cam)

        v.addWidget(inner, 1)
        return outer

    def _build_radar_pane(self) -> QWidget:
        self.radar_view = RadarBEVView(self.model)
        self.radar_3d_view = RadarPerspectiveView(self.model)
        self.radar_stack = QStackedWidget()
        self.radar_stack.addWidget(self.radar_view)
        self.radar_stack.addWidget(self.radar_3d_view)
        self.radar_view.setMinimumHeight(420)
        self.radar_3d_view.setMinimumHeight(420)
        self.radar_view.setStyleSheet("border-radius: 6px;")
        self.radar_3d_view.setStyleSheet("border-radius: 6px;")
        return self.radar_stack

    def _build_camera_pane(self) -> QWidget:
        self.image_view = CameraImageView(self.model)
        self.image_view.setMinimumHeight(420)
        return self.image_view

    def _build_radar_pane_toolbar(self) -> None:
        """Two-row BEV toolbar.

        Row 1 — view controls: Pan, Select, Zoom, Z-height, View mode,
                                Boxes, Labels, Grid (3D), Axes (3D).
        Row 2 — annotation/filter: Candidates, Cluster Objects,
                                    Echo filter, RCS threshold,
                                    ROI toggle + Fit, Save Filtered.
        ROI bound spinboxes (roi_x_min/max, roi_y_min/max) are kept as
        invisible attributes so _current_roi() keeps working; the Fit
        button is the primary way to set the bounds.
        """
        cb_style = (
            "QCheckBox { color: #475569; font-size: 11px; spacing: 4px; "
            "padding-left: 2px; }"
            "QCheckBox::indicator { width: 13px; height: 13px; }"
        )
        btn_style = (
            "QPushButton { background: #f1f5f9; color: #334155; "
            "border: 1px solid #e2e8f0; border-radius: 4px; "
            "padding: 4px 8px; font-size: 11px; min-width: 24px; }"
            "QPushButton:hover { background: #e2e8f0; }"
            "QPushButton:checked { background: #dbeafe; border-color: #93c5fd; "
            "color: #1d4ed8; }"
        )
        combo_style = (
            "QComboBox { background: #ffffff; color: #1e293b; "
            "border: 1px solid #e2e8f0; border-radius: 4px; "
            "padding: 3px 6px; font-size: 11px; }"
            "QComboBox QAbstractItemView { background: #ffffff; color: #1e293b; "
            "selection-background-color: #dbeafe; }"
        )

        # ── Row 1: Zoom | Z | View mode | Candidates Boxes Labels Grid Axes | Echo combo | →stretch
        for text, delta in [("+", +1), ("−", -1)]:
            b = QPushButton(text); b.setStyleSheet(btn_style); b.setFixedSize(26, 24)
            b.clicked.connect(lambda _, d=delta: self._zoom_radar(d))
            self._radar_tb_row1.addWidget(b)

        self._radar_tb_row1.addSpacing(5)
        hf_label = QLabel("Z")
        hf_label.setToolTip("Height filter"); hf_label.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600;")
        self._radar_tb_row1.addWidget(hf_label)

        self.height_filter_combo = QComboBox()
        self.height_filter_combo.setToolTip("Filter radar points by height.")
        self.height_filter_combo.setFixedWidth(80)
        self.height_filter_combo.setStyleSheet(combo_style)
        self.height_filter_combo.addItems(["All Z", "-3..2 m", "-1..1 m", "0..2.5 m", "Ground+"])
        self.height_filter_combo.setCurrentIndex(1)
        self.height_filter_combo.currentIndexChanged.connect(self._on_height_filter_changed)
        self._radar_tb_row1.addWidget(self.height_filter_combo)

        self._radar_tb_row1.addSpacing(5)
        self.radar_mode_combo = QComboBox()
        self.radar_mode_combo.setToolTip("Switch radar view mode.")
        self.radar_mode_combo.setFixedWidth(130)
        self.radar_mode_combo.setStyleSheet(combo_style)
        self.radar_mode_combo.addItems(["BEV (Top-Down)", "3D Perspective"])
        self.radar_mode_combo.currentIndexChanged.connect(self._on_radar_mode_changed)
        self._radar_tb_row1.addWidget(self.radar_mode_combo)

        self._radar_tb_row1.addSpacing(8)

        # Display toggles (checkboxes) — all on row 1
        def _add_cb1(attr, label, tip, checked=True, *slots):
            cb = QCheckBox(label); cb.setChecked(checked)
            cb.setToolTip(tip); cb.setStyleSheet(cb_style)
            for slot in slots:
                cb.toggled.connect(slot)
            self._radar_tb_row1.addWidget(cb)
            setattr(self, attr, cb)

        _add_cb1("show_possible_objects_cb", "Candidates",
                 "Show high-confidence object candidate circles in BEV (uses active echo + ROI).",
                 False,
                 self.radar_view.set_show_possible_objects)
        _add_cb1("show_radar_boxes_cb", "Boxes",
                 "Show/hide annotation boxes on radar view.",
                 True,
                 self.radar_view.set_show_boxes, self.radar_3d_view.set_show_boxes)
        _add_cb1("show_radar_labels_cb", "Labels",
                 "Show/hide class/ID label badges on radar view.",
                 True,
                 self.radar_view.set_show_labels)
        _add_cb1("show_3d_grid_cb", "Grid",
                 "Show the ground grid in the 3D radar view.",
                 True,
                 self.radar_3d_view.set_show_grid)
        _add_cb1("show_3d_axes_cb", "Axes",
                 "Show X/Y/Z axes in the 3D radar view.",
                 True,
                 self.radar_3d_view.set_show_axes)

        self._radar_tb_row1.addSpacing(8)
        echo_label = QLabel("Echo")
        echo_label.setToolTip("Filter radar returns by reflection type.")
        echo_label.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600;")
        self._radar_tb_row1.addWidget(echo_label)

        self.reflection_filter_combo = QComboBox()
        self.reflection_filter_combo.setToolTip(
            "Object-like: RCS + local density.  Strong: high-RCS only.  Moving: compensated velocity.")
        self.reflection_filter_combo.setFixedWidth(100)
        self.reflection_filter_combo.setStyleSheet(combo_style)
        self.reflection_filter_combo.addItem("All", "all")
        self.reflection_filter_combo.addItem("Object-like", "object")
        self.reflection_filter_combo.addItem("Strong", "strong")
        self.reflection_filter_combo.addItem("Moving", "moving")
        self.reflection_filter_combo.addItem("Custom RCS", "custom")
        self.reflection_filter_combo.currentIndexChanged.connect(self._on_radar_filter_changed)
        self._radar_tb_row1.addWidget(self.reflection_filter_combo)

        self._radar_tb_row1.addStretch(1)

        # ── Row 2: Cluster | RCS≥ spin Auto | ROI X bounds Y bounds | Fit | Save | →stretch
        cluster_btn = QPushButton("Cluster")
        cluster_btn.setToolTip("Auto-detect objects from radar clusters (uses active filters + ROI).")
        cluster_btn.setStyleSheet(btn_style); cluster_btn.setFixedHeight(24)
        cluster_btn.clicked.connect(self._on_cluster_objects)
        self._radar_tb_row2.addWidget(cluster_btn)

        self._radar_tb_row2.addSpacing(8)
        self.rcs_threshold_label = QLabel("RCS≥")
        self.rcs_threshold_label.setToolTip("Minimum RCS (Custom RCS mode only).")
        self.rcs_threshold_label.setStyleSheet("color: #64748b; font-size: 10px;")
        self._radar_tb_row2.addWidget(self.rcs_threshold_label)

        self.custom_rcs_spin = QDoubleSpinBox()
        self.custom_rcs_spin.setRange(-100.0, 100.0); self.custom_rcs_spin.setDecimals(1)
        self.custom_rcs_spin.setSingleStep(1.0); self.custom_rcs_spin.setValue(0.0)
        self.custom_rcs_spin.setFixedWidth(54)
        self.custom_rcs_spin.setStyleSheet(combo_style)
        self.custom_rcs_spin.valueChanged.connect(self._on_radar_filter_changed)
        self._radar_tb_row2.addWidget(self.custom_rcs_spin)

        self.rcs_auto_cb = QCheckBox("Auto")
        self.rcs_auto_cb.setChecked(True)
        self.rcs_auto_cb.setToolTip("Auto-compute RCS threshold (Custom RCS mode).")
        self.rcs_auto_cb.setStyleSheet(cb_style)
        self.rcs_auto_cb.toggled.connect(self._on_radar_filter_changed)
        self._radar_tb_row2.addWidget(self.rcs_auto_cb)

        self._radar_tb_row2.addSpacing(6)
        self.roi_enabled_cb = QCheckBox("ROI")
        self.roi_enabled_cb.setChecked(False)
        self.roi_enabled_cb.setToolTip(
            "Restrict radar display and clustering to the ROI box.\n"
            "Edit X/Y bounds below or use Fit to auto-set from point cloud.")
        self.roi_enabled_cb.setStyleSheet(cb_style)
        self.roi_enabled_cb.toggled.connect(self._on_radar_filter_changed)
        self._radar_tb_row2.addWidget(self.roi_enabled_cb)

        # ROI bound spinboxes — compact, 48 px each
        self.roi_x_min = self._make_roi_spin(-200.0)
        self.roi_x_max = self._make_roi_spin(200.0)
        self.roi_y_min = self._make_roi_spin(-80.0)
        self.roi_y_max = self._make_roi_spin(80.0)
        _lbl_style = "color: #64748b; font-size: 10px; padding: 0 1px;"
        for axis_lbl, w_min, w_max in (
            ("X", self.roi_x_min, self.roi_x_max),
            ("Y", self.roi_y_min, self.roi_y_max),
        ):
            al = QLabel(axis_lbl); al.setStyleSheet(_lbl_style)
            self._radar_tb_row2.addWidget(al)
            for widget in (w_min, w_max):
                widget.setFixedWidth(48)
                widget.valueChanged.connect(self._on_radar_filter_changed)
                self._radar_tb_row2.addWidget(widget)

        fit_roi_btn = QPushButton("Fit")
        fit_roi_btn.setToolTip("Auto-fit ROI bounds to the raw point cloud extent, then enable ROI.")
        fit_roi_btn.setStyleSheet(btn_style); fit_roi_btn.setFixedSize(32, 24)
        fit_roi_btn.clicked.connect(self._fit_roi_to_points)
        self._radar_tb_row2.addWidget(fit_roi_btn)

        save_btn = QPushButton("Save")
        save_btn.setToolTip("Export the current filtered/ROI point cloud to disk.")
        save_btn.setStyleSheet(btn_style); save_btn.setFixedHeight(24)
        save_btn.clicked.connect(self._on_save_filtered_cloud)
        self._radar_tb_row2.addWidget(save_btn)

        self._radar_tb_row2.addStretch(1)
        self._sync_threshold_control_state()

    def _make_roi_spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1000.0, 1000.0)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setValue(value)
        spin.setFixedWidth(58)
        spin.setStyleSheet(self.height_filter_combo.styleSheet())
        return spin

    def _build_camera_pane_toolbar(self) -> None:
        style = (
            "QPushButton { background: #f1f5f9; color: #334155; "
            "border: 1px solid #e2e8f0; border-radius: 4px; "
            "padding: 4px 8px; font-size: 11px; min-width: 24px; }"
            "QPushButton:hover { background: #e2e8f0; }"
            "QPushButton:checked { background: #dbeafe; border-color: #93c5fd; "
            "color: #1d4ed8; }"
        )
        for text, tip in [("☀", "Brightness"),
                          ("⌖", "Center / auto-fit")]:
            b = QPushButton(text); b.setToolTip(tip); b.setStyleSheet(style)
            b.setFixedSize(28, 24)
            self._camera_tb_row1.addWidget(b)

        self._camera_tb_row1.addSpacing(8)
        for text in ("+", "−"):
            b = QPushButton(text); b.setStyleSheet(style); b.setFixedSize(28, 24)
            self._camera_tb_row1.addWidget(b)

        self._camera_tb_row1.addSpacing(8)
        disp_label = QLabel("Show")
        disp_label.setStyleSheet("color: #64748b; font-size: 11px;")
        self._camera_tb_row1.addWidget(disp_label)

        self.display_combo = QComboBox()
        self.display_combo.setMinimumWidth(140)
        self.display_combo.setMaximumWidth(260)
        self.display_combo.setStyleSheet(
            "QComboBox { background: #ffffff; color: #1e293b; "
            "border: 1px solid #e2e8f0; border-radius: 4px; "
            "padding: 3px 6px; font-size: 11px; }"
            "QComboBox QAbstractItemView { background: #ffffff; color: #1e293b; "
            "selection-background-color: #dbeafe; }")
        self.display_combo.addItem("All", "__all__")
        self.display_combo.addItem("Selected", "__selected__")
        self.display_combo.insertSeparator(self.display_combo.count())
        for class_name in OBJECT_CLASS_CHOICES:
            self.display_combo.addItem(class_name, class_name)
        self.display_combo.currentIndexChanged.connect(
            self._on_camera_display_changed)
        self._camera_tb_row1.addWidget(self.display_combo)
        self._camera_tb_row1.addStretch(1)

        self.show_projections_cb = QCheckBox("Boxes")
        self.show_projections_cb.setChecked(True)
        self.show_projections_cb.setMinimumWidth(54)
        self.show_projections_cb.setToolTip("Show/hide 3D wireframe boxes on camera image.")
        self.show_projections_cb.setStyleSheet(
            "QCheckBox { color: #475569; font-size: 11px; spacing: 3px; }"
            "QCheckBox::indicator { width: 13px; height: 13px; }")
        self.show_projections_cb.toggled.connect(self._on_show_projections_toggled)
        self._camera_tb_row2.addWidget(self.show_projections_cb)

        self.show_cam_labels_cb = QCheckBox("Labels")
        self.show_cam_labels_cb.setChecked(True)
        self.show_cam_labels_cb.setMinimumWidth(56)
        self.show_cam_labels_cb.setToolTip("Show/hide class/ID label badges on camera image.")
        self.show_cam_labels_cb.setStyleSheet(
            "QCheckBox { color: #475569; font-size: 11px; spacing: 3px; }"
            "QCheckBox::indicator { width: 13px; height: 13px; }")
        self.show_cam_labels_cb.toggled.connect(self.image_view.set_show_labels)
        self._camera_tb_row2.addWidget(self.show_cam_labels_cb)

        self.show_radar_on_cam_cb = QCheckBox("Radar")
        self.show_radar_on_cam_cb.setChecked(False)
        self.show_radar_on_cam_cb.setMinimumWidth(56)
        self.show_radar_on_cam_cb.setToolTip(
            "Project radar points into the camera image (semi-transparent dots).")
        self.show_radar_on_cam_cb.setStyleSheet(
            "QCheckBox { color: #475569; font-size: 11px; spacing: 3px; }"
            "QCheckBox::indicator { width: 13px; height: 13px; }")
        self.show_radar_on_cam_cb.toggled.connect(
            self.image_view.set_show_radar_projected)
        self._camera_tb_row2.addWidget(self.show_radar_on_cam_cb)

        self.show_box_metrics_cb = QCheckBox("Metrics")
        self.show_box_metrics_cb.setChecked(True)
        self.show_box_metrics_cb.setMinimumWidth(66)
        self.show_box_metrics_cb.setToolTip(
            "Display x/y/z, camera depth, and L/W/H on projected boxes.")
        self.show_box_metrics_cb.setStyleSheet(
            "QCheckBox { color: #475569; font-size: 11px; spacing: 3px; }"
            "QCheckBox::indicator { width: 13px; height: 13px; }")
        self.show_box_metrics_cb.toggled.connect(
            self.image_view.set_show_box_metrics)
        self._camera_tb_row2.addWidget(self.show_box_metrics_cb)

        self.manual_img_move_cb = QCheckBox("Manual placement")
        self.manual_img_move_cb.setChecked(False)
        self.manual_img_move_cb.setMinimumWidth(116)
        self.manual_img_move_cb.setToolTip(
            "Camera pane: ignores loaded extrinsics for overlay/drag (uniform move "
            "across the photo). Draw a new box with or without radar (no returns → "
            "image-only XY). BEV refines radar pose separately. Arrow keys nudge the "
            "selected box when this pane has focus; Shift = finer step. KITTI export "
            "still uses the real calibration.")
        self.manual_img_move_cb.setStyleSheet(
            "QCheckBox { color: #475569; font-size: 11px; spacing: 3px; }"
            "QCheckBox::indicator { width: 13px; height: 13px; }")
        self.manual_img_move_cb.toggled.connect(self.image_view.set_manual_image_adjust)
        self._camera_tb_row2.addWidget(self.manual_img_move_cb)

        self._camera_tb_row2.addStretch(1)

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        bar.setStyleSheet(
            "QStatusBar { background: #ffffff; border-top: 1px solid #e2e8f0; "
            "color: #475569; font-size: 11px; }"
            "QStatusBar::item { border: none; }")
        self.setStatusBar(bar)

        self._status_ready = self._make_status_dot("Ready", "#22c55e")
        bar.addWidget(self._status_ready)
        bar.addWidget(self._make_separator())

        self._status_frame = QLabel("Frame — / —")
        self._status_frame.setStyleSheet("color: #475569;")
        bar.addWidget(self._status_frame)
        bar.addWidget(self._make_separator())

        self._status_sync = self._make_status_dot("Waiting for dataset", "#94a3b8")
        bar.addWidget(self._status_sync)
        bar.addWidget(self._make_separator())

        self._status_points = QLabel("Radar Points: 0")
        self._status_points.setStyleSheet("color: #475569;")
        bar.addWidget(self._status_points)
        bar.addWidget(self._make_separator())

        self._status_objects = QLabel("Objects: 0")
        self._status_objects.setStyleSheet("color: #475569;")
        bar.addWidget(self._status_objects)

        # right side
        bar.addPermanentWidget(self._make_autosave_widget())

    def _make_separator(self) -> QWidget:
        s = QFrame(); s.setFrameShape(QFrame.VLine)
        s.setStyleSheet("color: #cbd5e1; max-width: 1px;")
        return s

    def _make_status_dot(self, text: str, color: str) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w)
        h.setContentsMargins(6, 2, 6, 2); h.setSpacing(5)
        dot = QLabel()
        dot.setStyleSheet(
            f"background: {color}; border-radius: 4px; "
            f"min-width: 8px; max-width: 8px; min-height: 8px; max-height: 8px;")
        lbl = QLabel(text); lbl.setStyleSheet("color: #475569;")
        w.setProperty("dot", dot); w.setProperty("lbl", lbl)
        h.addWidget(dot); h.addWidget(lbl)
        return w

    def _set_status_dot(self, widget: QWidget, text: str, color: str) -> None:
        dot = widget.property("dot")
        lbl = widget.property("lbl")
        if dot is not None:
            dot.setStyleSheet(
                f"background: {color}; border-radius: 4px; "
                f"min-width: 8px; max-width: 8px; min-height: 8px; max-height: 8px;")
        if lbl is not None:
            lbl.setText(text)

    def _make_autosave_widget(self) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w)
        h.setContentsMargins(6, 2, 6, 2); h.setSpacing(5)
        dot = QLabel()
        dot.setStyleSheet(
            "background: #22c55e; border-radius: 4px; "
            "min-width: 8px; max-width: 8px; min-height: 8px; max-height: 8px;")
        lbl = QLabel("Auto-Save: ON")
        lbl.setStyleSheet("color: #16a34a; font-weight: 600;")
        h.addWidget(dot); h.addWidget(lbl)
        self._autosave_dot = dot
        self._autosave_lbl = lbl
        return w

    # ======================================================================
    # Toolbar
    # ======================================================================
    def _build_toolbar(self) -> None:
        toolbar_qss = (
            "QToolBar { background: #ffffff; border-bottom: 1px solid #e2e8f0; "
            "spacing: 3px; padding: 5px 7px; }"
            "QToolButton { background: #ffffff; color: #334155; "
            "border: 1px solid #e2e8f0; border-radius: 4px; "
            "padding: 4px 7px; font-size: 11px; }"
            "QToolButton:hover { background: #f8fafc; border-color: #cbd5e1; }"
            "QToolButton:checked { background: #dbeafe; border-color: #93c5fd; "
            "color: #1d4ed8; }"
            "QToolBar::separator { background: #e2e8f0; width: 1px; "
            "margin: 4px 6px; }"
        )
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(self._scaled_icon_size())
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        tb.setStyleSheet(toolbar_qss)
        self.addToolBar(tb)

        def add_action(icon, text, slot=None, shortcut=None, tip=None):
            act = QAction(icon, text, self)
            if slot is not None:
                act.triggered.connect(slot)
            if shortcut is not None:
                act.setShortcut(QKeySequence(shortcut))
            if tip is not None:
                act.setToolTip(tip)
            tb.addAction(act)
            return act

        # --- Dataset ---
        add_action(icon_folder(), "Select Dataset Folder",
                   self._on_open_folder, "Ctrl+O")
        add_action(icon_calib(), "Load Calibration",
                   self._on_load_calibration)
        tb.addSeparator()

        # --- Navigation ---
        add_action(icon_prev(), "Previous",
                   lambda: self._go_to(self.current_index - 1), Qt.Key_Left)
        add_action(icon_next(), "Next",
                   lambda: self._go_to(self.current_index + 1), Qt.Key_Right)

        tb.addWidget(QLabel("  Go To Frame  "))
        self.goto_spin = QSpinBox()
        self.goto_spin.setRange(1, 1)
        self.goto_spin.setStyleSheet(
            "QSpinBox { background: #ffffff; color: #1e293b; "
            "border: 1px solid #e2e8f0; border-radius: 4px; padding: 3px 6px; "
            "min-width: 70px; font-size: 11px; }"
            "QSpinBox:focus { border: 1px solid #2563eb; }")
        self.goto_spin.editingFinished.connect(
            lambda: self._go_to(self.goto_spin.value() - 1))
        tb.addWidget(self.goto_spin)
        self.goto_total_lbl = QLabel(" / 0")
        self.goto_total_lbl.setStyleSheet("color: #64748b; padding: 0 6px;")
        tb.addWidget(self.goto_total_lbl)
        tb.addSeparator()

        # --- Save / export ---
        add_action(icon_save(), "Save Annotation",
                   self._save_current_frame, "Ctrl+S")
        add_action(icon_save(), "Save All", self._on_save_all)

        act_kitti = add_action(icon_export(), "Export VoD/KITTI (txt)",
                               self._on_export_kitti)
        act_kitti.setToolTip(
            "Writes label_2/<frame>.txt. Location fields are bottom-centre in "
            "camera coords — not the same numbers as JSON centre_master / radar X.")

        act_img_bbox = add_action(icon_export(), "Export Annotations",
                                  self._on_export_image_bbox)
        act_img_bbox.setToolTip(
            "Exports confirmed 3D boxes with full camera + radar details: "
            "3D pose, projected 8-corner wireframe, 2D pixel bbox, "
            "per-box radar return stats and raw points inside. "
            "Output: JSON (full fidelity) + CSV (flat aggregate).")
        tb.addSeparator()

        # --- Annotation ---
        add_action(icon_plus(), "New Box (Radar)", self._on_new_box, "N")
        add_action(icon_plus(), "New Box (Image)", self._on_new_box_image, "I")
        add_action(icon_delete(), "Delete Box", self._on_delete_box,
                   Qt.Key_Delete)
        add_action(icon_snap(), "Snap to Ground", self._on_snap_ground, "G")
        add_action(icon_autofit(), "Auto-Fit", self._on_auto_fit)
        tb.addSeparator()

        # --- Undo / redo (less visible but shortcutable) ---
        undo = QAction("Undo", self); undo.setShortcut(QKeySequence.Undo)
        undo.triggered.connect(self.model.undo); self.addAction(undo)
        redo = QAction("Redo", self); redo.setShortcut(QKeySequence.Redo)
        redo.triggered.connect(self.model.redo); self.addAction(redo)

        # Second row: class picker stays wide — keeping it on the main row caused
        # overlap with navigation/save actions on smaller windows.
        self.addToolBarBreak(Qt.TopToolBarArea)
        tb_class = QToolBar("Class row")
        tb_class.setMovable(False)
        tb_class.setFloatable(False)
        tb_class.setIconSize(self._scaled_icon_size())
        tb_class.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        tb_class.setStyleSheet(toolbar_qss)
        self.addToolBar(tb_class)

        class_lbl = QLabel("Class: ")
        class_lbl.setStyleSheet("color: #64748b;")
        tb_class.addWidget(class_lbl)
        self.class_combo = QComboBox()
        self.class_combo.setEditable(True)
        self.class_combo.addItems(list(OBJECT_CLASS_CHOICES))
        self.class_combo.setMinimumWidth(160)
        self.class_combo.setMaximumWidth(400)
        self.class_combo.setStyleSheet(
            "QComboBox { background: #ffffff; color: #1e293b; "
            "border: 1px solid #e2e8f0; border-radius: 4px; "
            "padding: 3px 8px; font-size: 11px; }"
            "QComboBox QAbstractItemView { background: #ffffff; color: #1e293b; "
            "selection-background-color: #dbeafe; }")
        tb_class.addWidget(self.class_combo)
        spacer_class = QWidget()
        spacer_class.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb_class.addWidget(spacer_class)

    def _scaled_icon_size(self):
        from PySide6.QtCore import QSize
        return QSize(16, 16)

    # ======================================================================
    # Signal wiring
    # ======================================================================
    def _wire_signals(self) -> None:
        self.model.objects_changed.connect(self._on_objects_changed)
        self.radar_view.status_message.connect(self.statusBar().showMessage)
        self.radar_3d_view.status_message.connect(self.statusBar().showMessage)
        self.image_view.status_message.connect(self.statusBar().showMessage)
        self.image_view.box_created_from_image.connect(self._on_box_created_from_image)
        self.object_list.edit_requested.connect(
            lambda: self.object_panel.setFocus())
        self.frame_info.fix_requested.connect(self._on_validation_fix_requested)

    def _apply_styles(self) -> None:
        # Global light palette
        app = QApplication.instance()
        if app is None:
            return
        pal = QPalette()
        pal.setColor(QPalette.Window,         QColor("#f1f5f9"))
        pal.setColor(QPalette.WindowText,     QColor("#1e293b"))
        pal.setColor(QPalette.Base,           QColor("#ffffff"))
        pal.setColor(QPalette.AlternateBase,  QColor("#f8fafc"))
        pal.setColor(QPalette.Text,           QColor("#1e293b"))
        pal.setColor(QPalette.Button,         QColor("#ffffff"))
        pal.setColor(QPalette.ButtonText,     QColor("#334155"))
        pal.setColor(QPalette.Highlight,      QColor("#3b82f6"))
        pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        pal.setColor(QPalette.ToolTipBase,    QColor("#1e293b"))
        pal.setColor(QPalette.ToolTipText,    QColor("#ffffff"))
        app.setPalette(pal)
        self.setStyleSheet("QMainWindow { background: #f1f5f9; }")
        self._on_height_filter_changed()

    # ======================================================================
    # Radar / camera pane toolbar handlers
    # ======================================================================
    def _zoom_radar(self, delta: int) -> None:
        if self.radar_stack.currentWidget() is self.radar_3d_view:
            self.radar_3d_view.zoom_by(delta)
            return
        factor = 1.2 if delta > 0 else 1 / 1.2
        self.radar_view._scale = max(0.5, min(200.0, self.radar_view._scale * factor))
        self.radar_view.update()

    def _on_height_filter_changed(self, *_) -> None:
        text = self.height_filter_combo.currentText()
        # Must match strings in height_filter_combo.addItems(...) exactly.
        if text == "All Z":
            self._active_z_range = None
        elif text == "-3..2 m":
            self._active_z_range = (-3.0, 2.0)
        elif text == "-1..1 m":
            self._active_z_range = (-1.0, 1.0)
        elif text == "0..2.5 m":
            self._active_z_range = (0.0, 2.5)
        elif text == "Ground+":
            self._active_z_range = (0.0, 10.0)
        self._apply_radar_filters_to_views()

    def _current_roi(self):
        if not hasattr(self, "roi_enabled_cb") or not self.roi_enabled_cb.isChecked():
            return None
        return (
            self.roi_x_min.value(),
            self.roi_x_max.value(),
            self.roi_y_min.value(),
            self.roi_y_max.value(),
            None,
            None,
        )

    def _current_filter_settings(self) -> dict:
        return describe_filter_settings(
            z_range=self._active_z_range,
            roi=self._current_roi(),
            reflection_mode=self._active_reflection_mode,
            custom_rcs_min=self._custom_rcs_min,
        )

    def _on_radar_filter_changed(self, *_) -> None:
        if hasattr(self, "reflection_filter_combo"):
            self._active_reflection_mode = (
                self.reflection_filter_combo.currentData() or "all")
        if hasattr(self, "custom_rcs_spin"):
            self._custom_rcs_min = (
                "auto"
                if hasattr(self, "rcs_auto_cb") and self.rcs_auto_cb.isChecked()
                else float(self.custom_rcs_spin.value())
            )
        self._sync_threshold_control_state()
        self._apply_radar_filters_to_views()

    def _sync_threshold_control_state(self) -> None:
        if hasattr(self, "custom_rcs_spin"):
            auto_rcs = hasattr(self, "rcs_auto_cb") and self.rcs_auto_cb.isChecked()
            custom_mode = self._active_reflection_mode == "custom"
            self.custom_rcs_spin.setEnabled(custom_mode and not auto_rcs)
            if hasattr(self, "rcs_threshold_label"):
                # Keep the label short and stable; disable spin when not needed
                self.rcs_threshold_label.setText("RCS≥")

    def _apply_radar_filters_to_views(self) -> None:
        pts = self.current_radar_points
        result = apply_radar_filters(
            pts,
            z_range=self._active_z_range,
            roi=self._current_roi(),
            reflection_mode=self._active_reflection_mode,
            custom_rcs_min=self._custom_rcs_min,
        )
        self.current_filtered_radar_points = result.points

        # Views receive already-filtered points; keep their legacy local Z filters off.
        self.radar_view.set_height_filter(None, None)
        self.radar_3d_view.set_height_filter(None, None)
        self.radar_view.set_points(self.current_filtered_radar_points)
        self.radar_3d_view.set_points(self.current_filtered_radar_points)
        self.image_view.set_radar_points(self.current_filtered_radar_points)

        roi = self._current_roi()
        if hasattr(self.radar_view, "set_roi_bounds"):
            self.radar_view.set_roi_bounds(roi)
        if hasattr(self.radar_3d_view, "set_roi_bounds"):
            self.radar_3d_view.set_roi_bounds(roi)

        if hasattr(self, "_status_points"):
            raw_n = int(pts.shape[0]) if pts is not None and pts.ndim == 2 else 0
            shown_n = int(self.current_filtered_radar_points.shape[0])
            self._status_points.setText(
                f"Radar Points: {shown_n:,} shown / {raw_n:,} raw")
            if hasattr(self.frame_info, "_fields"):
                self.frame_info._fields["point_count"].setText(
                    f"{shown_n:,} shown / {raw_n:,} raw")
        self._refresh_context()
        if hasattr(self, "model"):
            self._update_validation()

    def _fit_roi_to_points(self) -> None:
        pts = self.current_radar_points
        if pts is None or pts.size == 0:
            return
        xyz = pts[:, :3]
        finite = np.all(np.isfinite(xyz), axis=1)
        if not np.any(finite):
            return
        xy = xyz[finite, :2]
        pad = 1.0
        self.roi_x_min.blockSignals(True); self.roi_x_max.blockSignals(True)
        self.roi_y_min.blockSignals(True); self.roi_y_max.blockSignals(True)
        try:
            self.roi_x_min.setValue(float(xy[:, 0].min() - pad))
            self.roi_x_max.setValue(float(xy[:, 0].max() + pad))
            self.roi_y_min.setValue(float(xy[:, 1].min() - pad))
            self.roi_y_max.setValue(float(xy[:, 1].max() + pad))
            self.roi_enabled_cb.setChecked(True)
        finally:
            self.roi_x_min.blockSignals(False); self.roi_x_max.blockSignals(False)
            self.roi_y_min.blockSignals(False); self.roi_y_max.blockSignals(False)
        self._apply_radar_filters_to_views()

    def _on_save_filtered_cloud(self) -> None:
        if self.dataset is None or self.current_index < 0:
            QMessageBox.information(self, "No dataset", "Open a dataset folder first.")
            return
        frame = self.dataset.frame(self.current_index)
        default = self.dataset.root / f"{frame.frame_id}_filtered.npz"
        out, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save filtered radar point cloud",
            str(default),
            "NumPy archive (*.npz);;NumPy array (*.npy);;CSV (*.csv)",
        )
        if not out:
            return
        path = Path(out)
        points = np.asarray(self.current_filtered_radar_points, dtype=np.float64)
        try:
            if path.suffix.lower() == ".npy":
                np.save(path, points)
            elif path.suffix.lower() == ".csv":
                np.savetxt(path, points, delimiter=",")
            else:
                if path.suffix.lower() != ".npz":
                    path = path.with_suffix(".npz")
                np.savez_compressed(
                    path,
                    points=points,
                    source_file=str(frame.radar_path),
                    frame_id=frame.frame_id,
                    filter_settings=json.dumps(self._current_filter_settings()),
                )
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Save filtered cloud failed", str(e))
            return
        self.statusBar().showMessage(
            f"Saved filtered radar cloud: {path.name} ({points.shape[0]:,} points)",
            6000,
        )

    def _on_radar_mode_changed(self, index: int) -> None:
        self.radar_stack.setCurrentIndex(index)
        if index == 1:
            self.radar_3d_view.auto_fit()

    def _on_camera_display_changed(self, *_args) -> None:
        filter_key = self.display_combo.currentData()
        self.image_view.set_display_filter(filter_key)
        self.radar_view.set_display_filter(filter_key)
        self.radar_3d_view.set_display_filter(filter_key)

    def _on_show_projections_toggled(self, on: bool) -> None:
        self.image_view.set_show_wireframe(on)
        self.image_view.set_show_2d_bbox(on)

    # ======================================================================
    # Dataset / frame loading
    # ======================================================================
    def _on_open_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select dataset folder")
        if not d:
            return
        try:
            self.dataset = Dataset(Path(d))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open dataset:\n{e}")
            return

        self.calibration = Calibration.identity()

        total = len(self.dataset)
        self.frame_info.set_dataset(Path(d), self.calibration.id, total)
        self.goto_spin.setRange(1, max(1, total))
        self.goto_total_lbl.setText(f" / {total}")

        if total == 0:
            QMessageBox.warning(
                self, "Empty dataset",
                "No matching radar/image file pairs were found.\n\n"
                "Expected folders: 'radar/' and 'image/'.\n"
                "Files are paired by identical filename stem "
                "(e.g. 000001.bin ↔ 000001.png).")
            return

        # Global calib files (skip if every frame has calib/<stem>.* — KITTI-style)
        if self.dataset.calib_dir is not None and \
                not self.dataset.all_frames_have_stem_calibration():
            last_err: Exception | None = None
            for p in iter_calibration_candidate_paths(self.dataset.calib_dir):
                try:
                    self.calibration = Calibration.load(p)
                    self.statusBar().showMessage(
                        f"Loaded calibration: {p.name}", 4000)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
            if last_err is not None:
                QMessageBox.warning(
                    self, "Calibration",
                    f"No usable global calibration in calib/ "
                    f"(tried JSON/YAML/KITTI). Last error:\n{last_err}")

        self.frame_info.set_dataset(Path(d), self.calibration.id, total)
        self._go_to(0)

    def _on_load_calibration(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Load calibration", "",
            "Calibration (*.json *.txt *.yaml *.yml);;"
            "JSON (*.json);;Text (*.txt);;YAML (*.yaml *.yml);;All files (*)")
        if not p:
            return
        try:
            self.calibration = Calibration.load(Path(p))
        except Exception as e:
            QMessageBox.critical(self, "Calibration", f"Failed to load:\n{e}")
            return
        self.image_view.set_calibration(self.calibration)
        self.frame_info.set_calibration(self.calibration)
        if self.dataset is not None:
            self.frame_info.set_dataset(
                self.dataset.root, self.calibration.id, len(self.dataset))
        self._refresh_context()
        self.statusBar().showMessage(f"Calibration loaded: {self.calibration.id}", 4000)

    def _go_to(self, index: int) -> None:
        if self.dataset is None or len(self.dataset) == 0:
            return
        if self._auto_save and self.model.is_dirty and self.current_index >= 0:
            self._save_current_frame(quiet=True)

        index = max(0, min(len(self.dataset) - 1, index))
        frame = self.dataset.frame(index)
        self.current_index = index

        # Radar
        try:
            pts = load_radar_points(frame.radar_path)
        except Exception as e:
            traceback.print_exc()
            pts = np.zeros((0, 3))
            QMessageBox.warning(self, "Radar",
                                f"Failed to load {frame.radar_path.name}:\n{e}")
        self.current_radar_points = pts
        self._apply_radar_filters_to_views()

        # Image
        img = self._load_image(frame.image_path)
        self.current_image_qimage = None
        if img is not None:
            self.current_image_size = (img.shape[1], img.shape[0])
            if frame.calib_path is not None:
                try:
                    cal = Calibration.load(frame.calib_path)
                    if self.current_image_size != cal.image_size:
                        cal = cal.rescaled_to_image_size(self.current_image_size)
                    self.calibration = cal
                except Exception as e:
                    traceback.print_exc()
                    QMessageBox.warning(
                        self, "Calibration",
                        f"Failed to load {frame.calib_path.name}:\n{e}")
            elif self.calibration.id == "identity_default" and \
                    self.current_image_size != self.calibration.image_size:
                self.calibration = Calibration.identity(self.current_image_size)
            elif self.current_image_size != self.calibration.image_size:
                self.calibration = self.calibration.rescaled_to_image_size(
                    self.current_image_size)
            self.image_view.set_calibration(self.calibration)
            self.image_view.set_image(img, is_rgb=False)
            # Build a QImage thumbnail for the mini preview
            arr = img[:, :, ::-1].copy()  # BGR -> RGB
            h, w, _ = arr.shape
            self.current_image_qimage = QImage(
                arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        else:
            self.current_image_size = (0, 0)
            self.image_view.set_image(None)

        # Labels
        labels_path = None
        if self.dataset.labels_dir is not None:
            labels_path = self.dataset.labels_dir / f"{frame.frame_id}.json"
        boxes = (load_frame_labels(labels_path)
                 if labels_path is not None and labels_path.exists() else [])
        self.model.replace_all(boxes)

        self._refresh_context()
        self._refresh_frame_info(frame)

        # Update goto spin & status bar
        self.goto_spin.blockSignals(True)
        self.goto_spin.setValue(index + 1)
        self.goto_spin.blockSignals(False)
        self._status_frame.setText(f"Frame {index + 1:04d} / {len(self.dataset)}")
        self._status_points.setText(
            f"Radar Points: {self.current_filtered_radar_points.shape[0]:,} shown / "
            f"{pts.shape[0]:,} raw")
        self._set_status_dot(self._status_sync, "Synchronized via Calibration", "#22c55e")
        self._set_status_dot(self._status_ready, "Ready", "#22c55e")

        self.statusBar().showMessage(
            f"Loaded frame {index + 1}/{len(self.dataset)} — {frame.frame_id}", 4000)

    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        try:
            import cv2
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            return img  # BGR or None
        except ImportError:
            qimg = QImage(str(path))
            if qimg.isNull():
                return None
            qimg = qimg.convertToFormat(QImage.Format_RGB888)
            w, h = qimg.width(), qimg.height()
            ptr = qimg.constBits()
            arr = np.frombuffer(ptr, np.uint8).reshape(h, w, 3).copy()
            return arr[:, :, ::-1].copy()

    def _refresh_context(self) -> None:
        display_points = self.current_filtered_radar_points
        self.object_panel.set_context(
            self.calibration, display_points, self.current_image_size)
        self.object_list.set_context(
            self.calibration, display_points, self.current_image_size)
        self.frame_info.set_calibration(self.calibration)
        self.frame_info.set_points(display_points)
        self.frame_info.set_image(self.current_image_qimage, self.current_image_size)

    def _refresh_frame_info(self, frame: Frame) -> None:
        # Use file mtime as a stand-in "timestamp" when the dataset doesn't
        # provide explicit timestamps. Real datasets can replace this via
        # per-frame metadata later.
        ts = ""
        try:
            mt = frame.radar_path.stat().st_mtime
            ts = _dt.datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

        self.frame_info.set_frame(
            frame_id=frame.frame_id,
            radar_name=frame.radar_path.name,
            image_name=frame.image_path.name,
            timestamp=ts or "—",
            pair_index=self.current_index,
            point_count=self.current_filtered_radar_points.shape[0],
            image_resolution=self.current_image_size,
            weather_scene="—",
            synchronized=True,
        )
        self.frame_info._fields["point_count"].setText(
            f"{self.current_filtered_radar_points.shape[0]:,} shown / "
            f"{self.current_radar_points.shape[0]:,} raw")
        self._update_validation()

    def _update_validation(self) -> None:
        warnings: List[dict] = []
        display_points = self.current_filtered_radar_points
        for box in self.model.objects:
            if display_points.size > 0:
                n = int(np.count_nonzero(
                    box.points_inside(display_points[:, :3])))
                if n == 0:
                    warnings.append({
                        "uid": box.uid,
                        "fix": "points",
                        "text": f"#{box.object_id} {box.class_name}: 0 radar points inside",
                    })
            default_l, default_w, default_h = class_default_box_size(box.class_name)
            if (
                box.length < default_l * 0.55
                or box.length > default_l * 1.75
                or box.width < default_w * 0.55
                or box.width > default_w * 1.75
                or box.height < default_h * 0.55
                or box.height > default_h * 1.75
            ):
                warnings.append({
                    "uid": box.uid,
                    "fix": "dimensions",
                    "text": (
                        f"#{box.object_id} {box.class_name}: dimensions "
                        f"{box.length:.1f} x {box.width:.1f} x {box.height:.1f}m "
                        "look unusual"
                    ),
                })
            if self.current_image_size[0] > 0:
                corners_uv, depths = project_box_calibration(box, self.calibration)
                bbox = box_2d_from_projection(
                    corners_uv, depths, self.current_image_size)
                if bbox is None:
                    warnings.append({
                        "uid": box.uid,
                        "fix": "projection",
                        "text": f"#{box.object_id}: doesn't project into the image",
                    })
        self.frame_info.set_warnings(warnings)

    def _apply_default_dimensions(self, box: Box3D) -> None:
        length, width, height = class_default_box_size(box.class_name)
        base_z = box.z - box.height * 0.5
        box.length = float(length)
        box.width = float(width)
        box.height = float(height)
        box.z = float(base_z + height * 0.5)

    def _move_box_to_nearby_radar_points(self, box: Box3D) -> int:
        if self.current_filtered_radar_points.size == 0:
            return 0
        pts = np.asarray(self.current_filtered_radar_points[:, :3], dtype=np.float64)
        center_xy = np.array([box.x, box.y], dtype=np.float64)
        dist = np.linalg.norm(pts[:, :2] - center_xy, axis=1)
        order = np.argsort(dist)
        if order.size == 0:
            return 0
        nearest = pts[order[: min(12, order.size)]]
        max_dist = max(box.length, box.width, 2.5)
        cluster = nearest[dist[order[: min(12, order.size)]] <= max_dist]
        if cluster.shape[0] == 0:
            cluster = nearest[:1]
        box.x = float(np.median(cluster[:, 0]))
        box.y = float(np.median(cluster[:, 1]))
        box.z = float(max(np.median(cluster[:, 2]), 0.0) + box.height * 0.5)
        return int(cluster.shape[0])

    def _on_validation_fix_requested(self, uid: str, fix: str) -> None:
        box = self.model.find(uid)
        if box is None:
            return
        self.model.select(uid)
        if fix in {"dimensions", "projection"}:
            self._apply_default_dimensions(box)
        if fix in {"points", "projection"}:
            count = self._move_box_to_nearby_radar_points(box)
            if count == 0 and fix == "points":
                self.statusBar().showMessage(
                    "Auto-fix skipped: no radar points are loaded.", 5000)
                return
        self.model.update(box, snapshot=True)
        self.statusBar().showMessage(
            f"Auto-fixed {box.class_name} #{box.object_id} using recommended values",
            5000,
        )

    # ======================================================================
    # Model -> UI updates
    # ======================================================================
    def _on_objects_changed(self) -> None:
        self.frame_info.set_dirty(self.model.is_dirty)
        self._status_objects.setText(f"Objects: {len(self.model.objects)}")
        self._update_validation()

    # ======================================================================
    # Actions
    # ======================================================================
    def _on_new_box(self) -> None:
        if self.dataset is None:
            QMessageBox.information(self, "No dataset",
                                    "Open a dataset folder first.")
            return
        cls = self.class_combo.currentText().strip() or "Car"
        self.radar_view.begin_create_mode(cls)

    def _on_new_box_image(self) -> None:
        if self.dataset is None:
            QMessageBox.information(self, "No dataset",
                                    "Open a dataset folder first.")
            return
        cls = self.class_combo.currentText().strip() or "Car"
        self.image_view.begin_create_mode(cls)

    def _on_box_created_from_image(self, new_box) -> None:
        """Receives a Box3D created from a drawn rectangle on the camera image."""
        self.model.add(new_box)
        self.model.select(new_box.uid)

    def _on_cluster_objects(self) -> None:
        if self.dataset is None:
            QMessageBox.information(self, "No dataset", "Open a dataset folder first.")
            return
        points = self.current_filtered_radar_points
        if points is None or points.size == 0:
            QMessageBox.information(
                self,
                "Cluster objects",
                "No visible radar points are available. Adjust filters or load a frame first.",
            )
            return

        class_name = self.class_combo.currentText().strip() or "Car"
        # Derive eps from the point cloud density so the algorithm adapts
        # automatically to sparse highway scenes and dense parking lots.
        # min_samples=4 matches the Candidates threshold so that "Cluster Objects"
        # and the candidate circles agree on what counts as a real object.
        eps = suggest_eps(points)
        config = ClusterConfig(eps_m=eps, min_samples=4, max_objects=30)
        result = propose_initial_objects(
            points,
            class_name=class_name,
            default_size=class_default_box_size(class_name),
            existing_boxes=self.model.objects,
            config=config,
        )
        if not result.created:
            if result.ambiguous and hasattr(self, "show_possible_objects_cb"):
                self.show_possible_objects_cb.setChecked(True)
            QMessageBox.information(
                self,
                "Cluster objects",
                "No new objects were created. "
                f"Duplicates: {result.duplicates}; ambiguous clusters: {len(result.ambiguous)}; "
                f"noise points: {result.noise_points}.",
            )
            return

        self.model.add_many(result.created)
        if result.ambiguous and hasattr(self, "show_possible_objects_cb"):
            self.show_possible_objects_cb.setChecked(True)
        self.statusBar().showMessage(
            f"Created {len(result.created)} clustered {class_name} object(s)  "
            f"[eps={eps:.2f} m]  —  "
            f"{result.duplicates} duplicate and {len(result.ambiguous)} ambiguous cluster(s) skipped.",
            7000,
        )

    def _on_delete_box(self) -> None:
        uid = self.model.selected_uid
        if uid is not None:
            self.model.delete(uid)

    def _on_snap_ground(self) -> None:
        sel = self.model.selected()
        if sel is None:
            return
        sel.z = sel.height / 2.0
        self.model.update(sel, snapshot=True)

    def _on_auto_fit(self) -> None:
        """Re-center and rescale the BEV view to enclose points + objects."""
        display_points = self.current_filtered_radar_points
        if display_points.size == 0 and not self.model.objects:
            return
        xs, ys = [], []
        if display_points.size > 0:
            xs += [float(display_points[:, 0].min()),
                   float(display_points[:, 0].max())]
            ys += [float(display_points[:, 1].min()),
                   float(display_points[:, 1].max())]
        for b in self.model.objects:
            xs += [b.x - 3, b.x + 3]; ys += [b.y - 3, b.y + 3]
        if not xs:
            xs = [-10, 30]; ys = [-15, 15]
        x_c = (min(xs) + max(xs)) / 2
        y_c = (min(ys) + max(ys)) / 2
        span = max(max(xs) - min(xs), max(ys) - min(ys), 20.0) * 1.1
        self.radar_view._center_master = np.array([x_c, y_c])
        self.radar_view._scale = 0.8 * min(
            self.radar_view.width(), self.radar_view.height()) / span
        self.radar_view.update()
        self.radar_3d_view.auto_fit()

    def _save_current_frame(self, quiet: bool = False) -> None:
        if self.dataset is None or self.current_index < 0:
            return
        frame = self.dataset.frame(self.current_index)
        labels_dir = self.dataset.root / "labels_internal"
        labels_dir.mkdir(exist_ok=True)
        out = labels_dir / f"{frame.frame_id}.json"
        save_frame_labels(
            out, frame.frame_id, frame.radar_path.name, frame.image_path.name,
            self.calibration.id, self.model.objects)
        self.model.mark_clean()
        self.frame_info.set_dirty(False)
        if not quiet:
            self.statusBar().showMessage(f"Saved: {out.name}", 4000)

    def _on_save_all(self) -> None:
        # Save current frame; other frames are only modified if loaded, so
        # this is effectively the same as Save. Hook is here for parity
        # with the mock.
        self._save_current_frame()

    def _parse_frame_selection(self, text: str) -> List[int]:
        if self.dataset is None:
            return []
        total = len(self.dataset)
        selected: set[int] = set()
        for raw_part in text.replace(" ", "").split(","):
            if not raw_part:
                continue
            if "-" in raw_part:
                left, right = raw_part.split("-", 1)
                start = int(left)
                end = int(right)
                if start > end:
                    start, end = end, start
                selected.update(range(start - 1, end))
            else:
                selected.add(int(raw_part) - 1)
        return sorted(i for i in selected if 0 <= i < total)

    def _choose_kitti_export_indices(self) -> Optional[List[int]]:
        if self.dataset is None or len(self.dataset) == 0:
            return None
        options = [
            "Current frame only",
            "All frames",
            "Frame range / list",
            "Annotated frames only",
        ]
        choice, ok = QInputDialog.getItem(
            self,
            "VoD/KITTI export scope",
            "Frames to export:",
            options,
            0,
            False,
        )
        if not ok:
            return None
        if choice == "Current frame only":
            return [self.current_index if self.current_index >= 0 else 0]
        if choice == "All frames":
            return list(range(len(self.dataset)))
        if choice == "Annotated frames only":
            indices = []
            for i in range(len(self.dataset)):
                fr = self.dataset.frame(i)
                labels_path = (
                    self.dataset.labels_dir / f"{fr.frame_id}.json"
                    if self.dataset.labels_dir is not None
                    else None
                )
                if i == self.current_index and self.model.objects:
                    indices.append(i)
                elif labels_path is not None and labels_path.exists():
                    boxes = load_frame_labels(labels_path)
                    if boxes:
                        indices.append(i)
            return indices

        default = f"{self.current_index + 1}" if self.current_index >= 0 else "1"
        text, range_ok = QInputDialog.getText(
            self,
            "VoD/KITTI export frame range",
            "Use 1-based frame numbers. Examples: 1-10 or 1,4,8-12",
            text=default,
        )
        if not range_ok:
            return None
        try:
            indices = self._parse_frame_selection(text)
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid frame range",
                "Enter frame numbers like 1-10 or 1,4,8-12.",
            )
            return None
        if not indices:
            QMessageBox.warning(
                self,
                "No frames selected",
                "The requested frame selection did not match any loaded frame.",
            )
            return None
        return indices

    def _kitti_frame_payload(self, i: int) -> dict:
        if self.dataset is None:
            raise RuntimeError("No dataset loaded")
        fr = self.dataset.frame(i)
        labels_path = (
            self.dataset.labels_dir / f"{fr.frame_id}.json"
            if self.dataset.labels_dir is not None
            else None
        )
        if i == self.current_index:
            boxes = list(self.model.objects)
        elif labels_path is not None and labels_path.exists():
            boxes = load_frame_labels(labels_path)
        else:
            boxes = []

        img = self._load_image(fr.image_path)
        image_size = (
            (img.shape[1], img.shape[0])
            if img is not None
            else self.calibration.image_size
        )

        cal: Calibration = self.calibration
        if fr.calib_path is not None:
            try:
                cal = Calibration.load(fr.calib_path)
            except Exception:
                cal = self.calibration
        if (int(image_size[0]), int(image_size[1])) != (
                int(cal.image_size[0]), int(cal.image_size[1])):
            cal = cal.rescaled_to_image_size(
                (int(image_size[0]), int(image_size[1])))

        image_file = fr.image_path.name if fr.image_path is not None else f"{fr.frame_id}.png"
        return {
            "frame_id": fr.frame_id,
            "image_file": image_file,
            "image_size": image_size,
            "boxes": boxes,
            "calibration": cal,
        }

    def _full_frame_payload(self, i: int) -> dict:
        """Like _kitti_frame_payload but adds filtered radar points for the frame."""
        payload = self._kitti_frame_payload(i)
        if i == self.current_index:
            payload["radar_points"] = self.current_filtered_radar_points
        else:
            fr = self.dataset.frame(i)
            try:
                raw = load_radar_points(fr.radar_path)
                result = apply_radar_filters(
                    raw,
                    z_range=self._active_z_range,
                    roi=self._current_roi(),
                    reflection_mode=self._active_reflection_mode,
                    custom_rcs_min=self._custom_rcs_min,
                )
                payload["radar_points"] = result.points
            except Exception:
                payload["radar_points"] = np.zeros((0, 3))
        return payload

    def _on_export_kitti(self) -> None:
        if self.dataset is None or len(self.dataset) == 0:
            return
        if self.model.is_dirty:
            self._save_current_frame(quiet=True)
        export_indices = self._choose_kitti_export_indices()
        if export_indices is None:
            return
        if not export_indices:
            QMessageBox.information(
                self,
                "VoD/KITTI export",
                "No annotated frames were found for this export scope.",
            )
            return
        default_dir = str(self.dataset.root)
        out_root = QFileDialog.getExistingDirectory(
            self,
            "Select folder for VoD/KITTI export (creates label_2/ inside it)",
            default_dir,
        )
        if not out_root:
            return
        label_dir = Path(out_root) / "label_2"
        try:
            frames_payload = [
                self._kitti_frame_payload(i) for i in export_indices
            ]
            export_kitti_label_dir(label_dir, frames_payload, self.calibration)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Export failed", str(e))
            return
        QMessageBox.information(
            self, "Export complete",
            f"VoD/KITTI-style labels written to:\n{label_dir}\n\n"
            f"Exported {len(frames_payload)} frame(s).\n\n"
            "The three location numbers per line are the box bottom-centre in "
            "**camera** coordinates (X right, Y down, Z forward), not the "
            "master-frame geometric centre stored in internal JSON "
            "(X forward, Y left, Z up). Radar table \"Range X\" is median radar "
            "forward distance — also a different quantity.")
        self.statusBar().showMessage(f"Exported KITTI labels: {label_dir}", 6000)

    def _on_export_image_bbox(self) -> None:
        if self.dataset is None or len(self.dataset) == 0:
            return
        if self.calibration is None:
            QMessageBox.warning(
                self, "No calibration",
                "Load a calibration file before exporting 2D bounding boxes.")
            return
        if self.model.is_dirty:
            self._save_current_frame(quiet=True)
        export_indices = self._choose_kitti_export_indices()
        if export_indices is None:
            return
        if not export_indices:
            QMessageBox.information(
                self,
                "Export Annotations",
                "No annotated frames were found for this export scope.",
            )
            return
        default_dir = str(self.dataset.root)
        out_dir = QFileDialog.getExistingDirectory(
            self,
            "Select output folder for annotation export",
            default_dir,
        )
        if not out_dir:
            return
        try:
            frames_payload = [
                self._full_frame_payload(i) for i in export_indices
            ]
            result = export_annotations(Path(out_dir), frames_payload, self.calibration)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Export failed", str(e))
            return
        QMessageBox.information(
            self, "Export complete",
            f"Full annotations written to:\n{out_dir}\n\n"
            f"  Annotations exported: {result['exported']}\n"
            f"  Boxes not visible in image: {result['skipped_projection']}\n\n"
            f"Files:\n  {result['json'].name}\n  {result['csv'].name}\n\n"
            "JSON contains: 3D box pose, 8 projected image corners, all radar "
            "return stats and raw points inside each box.\n"
            "CSV contains: same data flattened (no raw points).")
        self.statusBar().showMessage(
            f"Exported {result['exported']} annotations to {out_dir}", 6000)

    # ------------------------------------------------------------------
    def closeEvent(self, ev) -> None:
        if self.model.is_dirty and self.dataset is not None:
            r = QMessageBox.question(
                self, "Unsaved changes",
                "You have unsaved changes in the current frame. "
                "Save before exiting?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save)
            if r == QMessageBox.Save:
                self._save_current_frame(quiet=True)
            elif r == QMessageBox.Cancel:
                ev.ignore()
                return
        super().closeEvent(ev)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
