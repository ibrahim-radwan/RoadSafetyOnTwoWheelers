"""
Left panel: Loaded Example / Frame Details (Section 3.2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QLabel,
                               QGroupBox, QScrollArea, QFrame)


class FrameInfoPanel(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)

        container = QWidget()
        root = QVBoxLayout(container)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        self.setWidget(container)

        title = QLabel("Frame Details")
        f = title.font()
        f.setBold(True); f.setPointSize(11)
        title.setFont(f)
        root.addWidget(title)

        # Dataset group
        ds = QGroupBox("Dataset")
        ds_form = QFormLayout(ds)
        self._root_lbl = self._val("—", word_wrap=True)
        self._split_lbl = self._val("—")
        self._calib_lbl = self._val("—")
        ds_form.addRow("Root:", self._root_lbl)
        ds_form.addRow("Split:", self._split_lbl)
        ds_form.addRow("Calibration:", self._calib_lbl)
        root.addWidget(ds)

        # Frame group
        fr = QGroupBox("Current frame")
        fr_form = QFormLayout(fr)
        self._index_lbl = self._val("—")
        self._frame_id_lbl = self._val("—")
        self._radar_name_lbl = self._val("—", word_wrap=True)
        self._image_name_lbl = self._val("—", word_wrap=True)
        self._image_size_lbl = self._val("—")
        self._radar_count_lbl = self._val("—")
        self._objects_lbl = self._val("—")
        fr_form.addRow("Index:", self._index_lbl)
        fr_form.addRow("Frame ID:", self._frame_id_lbl)
        fr_form.addRow("Radar file:", self._radar_name_lbl)
        fr_form.addRow("Image file:", self._image_name_lbl)
        fr_form.addRow("Image size:", self._image_size_lbl)
        fr_form.addRow("Radar points:", self._radar_count_lbl)
        fr_form.addRow("Objects:", self._objects_lbl)
        root.addWidget(fr)

        # Validation group
        vg = QGroupBox("Validation")
        vg_form = QFormLayout(vg)
        self._dirty_lbl = self._val("saved", color="#6cc070")
        self._warnings_lbl = self._val("—", word_wrap=True)
        vg_form.addRow("Status:", self._dirty_lbl)
        vg_form.addRow("Warnings:", self._warnings_lbl)
        root.addWidget(vg)

        root.addStretch(1)

    def _val(self, text: str, *, word_wrap: bool = False,
             color: Optional[str] = None) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(word_wrap)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        if color:
            lbl.setStyleSheet(f"color: {color};")
        return lbl

    # ------------------------------------------------------------------
    def set_dataset(self, root: Optional[Path], calib_id: str) -> None:
        self._root_lbl.setText(str(root) if root else "—")
        self._calib_lbl.setText(calib_id or "—")

    def set_frame(self, *, index: int, total: int, frame_id: str,
                  radar_name: str, image_name: str,
                  image_size, radar_points: int,
                  object_count: int) -> None:
        self._index_lbl.setText(f"{index + 1} / {total}")
        self._frame_id_lbl.setText(frame_id)
        self._radar_name_lbl.setText(radar_name)
        self._image_name_lbl.setText(image_name)
        self._image_size_lbl.setText(f"{image_size[0]} × {image_size[1]}")
        self._radar_count_lbl.setText(str(radar_points))
        self._objects_lbl.setText(str(object_count))

    def set_dirty(self, dirty: bool) -> None:
        if dirty:
            self._dirty_lbl.setText("unsaved changes")
            self._dirty_lbl.setStyleSheet("color: #e0a040;")
        else:
            self._dirty_lbl.setText("saved")
            self._dirty_lbl.setStyleSheet("color: #6cc070;")

    def set_warnings(self, warnings) -> None:
        if not warnings:
            self._warnings_lbl.setText("none")
            self._warnings_lbl.setStyleSheet("color: #6cc070;")
        else:
            self._warnings_lbl.setText(" • " + "\n • ".join(warnings))
            self._warnings_lbl.setStyleSheet("color: #e07070;")

    def set_object_count(self, n: int) -> None:
        self._objects_lbl.setText(str(n))
