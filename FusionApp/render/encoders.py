import numpy as np
import cv2
from typing import Optional, Dict, Any


def encode_jpeg(bgr: np.ndarray, quality: int = 80) -> Optional[bytes]:
    try:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, buf = cv2.imencode(".jpg", bgr, encode_params)
        if not ok:
            return None
        return buf.tobytes()
    except Exception:
        return None


def heatmap_to_png(array2d: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> Optional[bytes]:
    try:
        a = np.asarray(array2d)
        if a.ndim != 2 or a.size == 0:
            return None
        a = a.astype(np.float32, copy=False)
        finite = np.isfinite(a)
        if not np.any(finite):
            return None
        vmin = float(np.percentile(a[finite], 1.0))
        vmax = float(np.percentile(a[finite], 99.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return None
        if vmax <= vmin:
            vmax = vmin + 1.0
        af = np.nan_to_num(a, nan=vmin, posinf=vmax, neginf=vmin)
        af = np.clip(af, vmin, vmax, out=None)
        norm = (af - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0.0, 1.0, out=None)
        img8 = (norm * 255.0).astype(np.uint8, copy=False)
        if colormap is not None:
            img_color = cv2.applyColorMap(img8, colormap)
        else:
            img_color = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        try:
            target_w, target_h = 640, 480
            h, w = img_color.shape[:2]
            scale = min(target_w / max(1, w), target_h / max(1, h))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            if new_w != w or new_h != h:
                img_resized = cv2.resize(img_color, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img_color
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            off_x = (target_w - new_w) // 2
            off_y = (target_h - new_h) // 2
            canvas[off_y : off_y + new_h, off_x : off_x + new_w] = img_resized
            img_color = canvas
        except Exception:
            pass
        ok, buf = cv2.imencode(".png", img_color)
        if not ok:
            return None
        return buf.tobytes()
    except Exception:
        return None


def render_point_cloud_png(point_cloud: Dict[str, Any], width: int = 640, height: int = 480) -> Optional[bytes]:
    try:
        if not isinstance(point_cloud, dict):
            return None
        x = point_cloud.get("x")
        y = point_cloud.get("y")
        if x is None or y is None:
            return None
        x = np.asarray(x).astype(np.float32)
        y = np.asarray(y).astype(np.float32)
        if x.size == 0 or y.size == 0:
            img = np.zeros((height, width, 3), dtype=np.uint8)
            ok, buf = cv2.imencode(".png", img)
            return buf.tobytes() if ok else None
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            return None
        x = x[finite]
        y = y[finite]
        x_min = float(np.percentile(x, 1.0))
        x_max = float(np.percentile(x, 99.0))
        y_min = float(np.percentile(y, 1.0))
        y_max = float(np.percentile(y, 99.0))
        if (
            not np.isfinite(x_min)
            or not np.isfinite(x_max)
            or not np.isfinite(y_min)
            or not np.isfinite(y_max)
        ):
            return None
        if x_max <= x_min:
            x_max = x_min + 1e-3
        if y_max <= y_min:
            y_max = y_min + 1e-3
        nx = (x - x_min) / (x_max - x_min)
        ny = (y - y_min) / (y_max - y_min)
        px = np.clip((nx * (width - 1)).astype(np.int32), 0, width - 1)
        py = np.clip(((1.0 - ny) * (height - 1)).astype(np.int32), 0, height - 1)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for xi, yi in zip(px, py):
            cv2.circle(img, (int(xi), int(yi)), 2, (0, 255, 0), -1)
        try:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except Exception:
            pass
        ok, buf = cv2.imencode(".png", img)
        return buf.tobytes() if ok else None
    except Exception:
        return None


