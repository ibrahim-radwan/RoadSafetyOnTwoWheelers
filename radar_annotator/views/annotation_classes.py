"""
Shared class names and RGB palette for BEV + camera views and exporters.
"""
from __future__ import annotations

from PySide6.QtGui import QColor

VOD_CLASSES = [
    "Car",
    "Pedestrian",
    "Cyclist",
    "Rider",
    "Unused bicycle",
    "Bicycle rack",
    "Human depiction (e.g. statues)",
    "Moped or scooter",
    "Motor",
    "Truck",
    "Other ride",
    "Other vehicle",
    "Uncertain ride",
]

DEFAULT_CLASSES = [
    "Car",
    "Pedestrian",
    "Cyclist",
    "Rider",
    "Bike",
    "Bike rack",
    "Human depiction / statue",
    "E-Scooter / Moped",
    "Motorbike",
    "Truck",
    "Other ride",
    "Other vehicle",
    "Uncertain ride",
]

# RGB 0–255; keep distinct hues for adjacent rows in lists.
_CLASS_RGB: dict[str, tuple[int, int, int]] = {
    "Car": (59, 130, 246),
    "Pedestrian": (34, 197, 94),
    "Cyclist": (236, 72, 153),
    "Rider": (14, 165, 233),
    "Bike": (20, 184, 166),
    "Bike rack": (45, 212, 191),
    "Human depiction / statue": (52, 211, 153),
    "E-Scooter / Moped": (244, 63, 94),
    "Motorbike": (239, 68, 68),
    "Unused bicycle": (20, 184, 166),
    "Bicycle rack": (45, 212, 191),
    "Human depiction (e.g. statues)": (52, 211, 153),
    "Moped or scooter": (244, 63, 94),
    "Motor": (239, 68, 68),
    "Truck": (249, 115, 22),
    "Other ride": (217, 70, 239),
    "Other vehicle": (168, 85, 247),
    "Uncertain ride": (100, 116, 139),
    # Compatibility aliases for labels created before the VoD class list.
    "Human depiction": (52, 211, 153),
    "Bus": (234, 179, 8),
    "Van": (139, 92, 246),
    "Motorcycle": (239, 68, 68),
    "Bicycle": (20, 184, 166),
    "Scooter": (244, 63, 94),
    "E-Scooter": (217, 70, 239),
    "Person": (52, 211, 153),
    "Animal": (168, 85, 247),
    "Traffic_Cone": (251, 146, 60),
    "Barrier": (148, 163, 184),
    "Debris": (120, 113, 108),
    "Unknown": (100, 116, 139),
    "Other": (100, 116, 139),
}

DEFAULT_CLASS_RGB = (100, 116, 139)

DEFAULT_BOX_SPECS: dict[str, tuple[float, float, float]] = {
    "Car": (4.3, 1.8, 1.6),
    "Pedestrian": (0.6, 0.6, 1.75),
    "Cyclist": (1.8, 0.6, 1.75),
    "Rider": (0.7, 0.7, 1.75),
    "Bike": (1.8, 0.6, 1.2),
    "Bike rack": (2.0, 0.7, 1.2),
    "Human depiction / statue": (0.7, 0.7, 1.75),
    "E-Scooter / Moped": (1.9, 0.75, 1.45),
    "Motorbike": (2.1, 0.8, 1.45),
    "Unused bicycle": (1.8, 0.6, 1.2),
    "Bicycle rack": (2.0, 0.7, 1.2),
    "Human depiction (e.g. statues)": (0.7, 0.7, 1.75),
    "Moped or scooter": (1.9, 0.75, 1.45),
    "Motor": (2.1, 0.8, 1.45),
    "Truck": (8.0, 2.5, 3.2),
    "Other ride": (2.0, 0.8, 1.5),
    "Other vehicle": (4.5, 1.9, 1.8),
    "Uncertain ride": (2.0, 0.8, 1.5),
    # Compatibility aliases for labels created before the VoD class list.
    "Human depiction": (0.7, 0.7, 1.75),
    "Bus": (10.0, 2.6, 3.2),
    "Van": (5.0, 2.0, 2.0),
    "Motorcycle": (2.0, 0.8, 1.5),
    "Bicycle": (1.8, 0.6, 1.6),
    "Scooter": (1.7, 0.7, 1.5),
    "E-Scooter": (1.4, 0.6, 1.5),
    "Person": (0.7, 0.7, 1.75),
    "Animal": (1.2, 0.6, 1.0),
    "Traffic_Cone": (0.5, 0.5, 0.9),
    "Barrier": (1.5, 0.6, 1.0),
    "Debris": (0.8, 0.8, 0.6),
    "Unknown": (2.5, 1.2, 1.5),
    "Other": (2.5, 1.2, 1.5),
}


def class_rgb(name: str) -> tuple[int, int, int]:
    return _CLASS_RGB.get(name, DEFAULT_CLASS_RGB)


def class_qcolor(name: str) -> QColor:
    r, g, b = class_rgb(name)
    return QColor(r, g, b)


def class_default_box_size(name: str) -> tuple[float, float, float]:
    return DEFAULT_BOX_SPECS.get(name, DEFAULT_BOX_SPECS["Other"])


CLASS_COLORS = {k: QColor(*v) for k, v in _CLASS_RGB.items()}


def _compute_object_class_choices() -> tuple[str, ...]:
    classes = list(DEFAULT_CLASSES)
    classes.extend(sorted(name for name in _CLASS_RGB if name not in classes))
    return tuple(classes)


# Shared by Show filter, toolbar Class, object panel, exporters.
OBJECT_CLASS_CHOICES = _compute_object_class_choices()


def object_class_choices() -> list[str]:
    return list(OBJECT_CLASS_CHOICES)
