"""
Annotation model.

Stores the list of Box3D objects for the currently loaded frame, provides
undo/redo via snapshots, and emits Qt signals when the state changes so
every view can re-render.
"""
from __future__ import annotations

import copy
from typing import List, Optional
from PySide6.QtCore import QObject, Signal

from .geometry import Box3D


class AnnotationModel(QObject):
    """Per-frame annotation container.

    Signals:
        objects_changed       : emitted after any add/delete/modify
        selection_changed(uid): emitted when the selected object changes
        frame_loaded          : emitted after replace_all() on frame load
    """
    objects_changed = Signal()
    selection_changed = Signal(object)  # str uid or None
    frame_loaded = Signal()

    MAX_HISTORY = 50

    def __init__(self):
        super().__init__()
        self._objects: List[Box3D] = []
        self._selected_uid: Optional[str] = None
        self._next_object_id: int = 1
        self._undo_stack: List[List[Box3D]] = []
        self._redo_stack: List[List[Box3D]] = []
        self._dirty: bool = False
        self._drag_checkpoint_open: bool = False

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def objects(self) -> List[Box3D]:
        return self._objects

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def mark_clean(self) -> None:
        self._dirty = False

    @property
    def selected_uid(self) -> Optional[str]:
        return self._selected_uid

    def selected(self) -> Optional[Box3D]:
        if self._selected_uid is None:
            return None
        for b in self._objects:
            if b.uid == self._selected_uid:
                return b
        return None

    def find(self, uid: str) -> Optional[Box3D]:
        for b in self._objects:
            if b.uid == uid:
                return b
        return None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def _snapshot(self) -> None:
        self._undo_stack.append(copy.deepcopy(self._objects))
        if len(self._undo_stack) > self.MAX_HISTORY:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._drag_checkpoint_open = False

    def replace_all(self, boxes: List[Box3D]) -> None:
        """Called on frame load. Does NOT push to undo; resets history."""
        self._objects = list(boxes)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._selected_uid = None
        self._next_object_id = max((b.object_id for b in boxes), default=0) + 1
        self._dirty = False
        self._drag_checkpoint_open = False
        self.frame_loaded.emit()
        self.objects_changed.emit()
        self.selection_changed.emit(None)

    def add(self, box: Box3D) -> None:
        self._snapshot()
        if box.object_id == 0:
            box.object_id = self._next_object_id
            self._next_object_id += 1
        self._objects.append(box)
        self._selected_uid = box.uid
        self._dirty = True
        self.objects_changed.emit()
        self.selection_changed.emit(box.uid)

    def add_many(self, boxes: List[Box3D]) -> None:
        if not boxes:
            return
        self._snapshot()
        for box in boxes:
            if box.object_id == 0:
                box.object_id = self._next_object_id
                self._next_object_id += 1
            self._objects.append(box)
        self._selected_uid = boxes[-1].uid
        self._dirty = True
        self.objects_changed.emit()
        self.selection_changed.emit(self._selected_uid)

    def delete(self, uid: str) -> None:
        if not any(b.uid == uid for b in self._objects):
            return
        self._snapshot()
        self._objects = [b for b in self._objects if b.uid != uid]
        if self._selected_uid == uid:
            self._selected_uid = None
            self.selection_changed.emit(None)
        self._dirty = True
        self.objects_changed.emit()

    def duplicate(self, uid: str) -> Optional[Box3D]:
        src = self.find(uid)
        if src is None:
            return None
        new = copy.deepcopy(src)
        import uuid as _u
        new.uid = _u.uuid4().hex[:8]
        new.object_id = self._next_object_id
        self._next_object_id += 1
        # Offset slightly so the copy is visible
        new.x += 1.0
        self._snapshot()
        self._objects.append(new)
        self._selected_uid = new.uid
        self._dirty = True
        self.objects_changed.emit()
        self.selection_changed.emit(new.uid)
        return new

    def update(self, box: Box3D, snapshot: bool = True) -> None:
        """Replace the object with the same uid. Call with snapshot=False for
        continuous drags (avoids polluting undo stack); caller should invoke
        commit() when the drag ends."""
        for i, b in enumerate(self._objects):
            if b.uid == box.uid:
                if snapshot:
                    self._snapshot()
                elif not self._drag_checkpoint_open:
                    self._snapshot()
                    self._drag_checkpoint_open = True
                self._objects[i] = box
                self._dirty = True
                self.objects_changed.emit()
                return

    def commit(self) -> None:
        """Close an in-progress drag/edit checkpoint started by snapshot=False."""
        self._drag_checkpoint_open = False

    def select(self, uid: Optional[str]) -> None:
        if uid == self._selected_uid:
            return
        self._selected_uid = uid
        self.selection_changed.emit(uid)

    def select_by_object_id(self, object_id: int) -> None:
        for b in self._objects:
            if b.object_id == object_id:
                self.select(b.uid)
                return

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------
    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(copy.deepcopy(self._objects))
        self._objects = self._undo_stack.pop()
        self._drag_checkpoint_open = False
        # Selection may no longer exist
        if self._selected_uid and not self.find(self._selected_uid):
            self._selected_uid = None
            self.selection_changed.emit(None)
        self._dirty = True
        self.objects_changed.emit()
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(copy.deepcopy(self._objects))
        self._objects = self._redo_stack.pop()
        self._drag_checkpoint_open = False
        if self._selected_uid and not self.find(self._selected_uid):
            self._selected_uid = None
            self.selection_changed.emit(None)
        self._dirty = True
        self.objects_changed.emit()
        return True
