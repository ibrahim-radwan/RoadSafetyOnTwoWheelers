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


def heatmap_to_png(
    array2d: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    extents: Optional[tuple] = None,
    *,
    force_square: bool = False,
    target_size: tuple = (640, 480),
) -> Optional[bytes]:
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
            target_w, target_h = int(target_size[0]), int(target_size[1])
            if force_square:
                side = min(target_w, target_h)
                target_w, target_h = side, side
            h, w = img_color.shape[:2]
            # If extents provided, prefer their aspect ratio (unless force_square)
            if (not force_square) and extents is not None and len(extents) == 4:
                xmin, xmax, ymin, ymax = extents
                ex = float(xmax) - float(xmin)
                ey = float(ymax) - float(ymin)
                ex = max(ex, 1e-6)
                ey = max(ey, 1e-6)
                aspect = ex / ey
                # Choose new size that fits target while matching aspect
                if target_w / target_h >= aspect:
                    new_h = target_h
                    new_w = max(1, int(round(aspect * new_h)))
                else:
                    new_w = target_w
                    new_h = max(1, int(round(new_w / aspect)))
            else:
                if force_square:
                    # Fill the square by stretching
                    new_w = target_w
                    new_h = target_h
                else:
                    # Keep original array aspect
                    aspect = w / max(1, h)
                    if target_w / target_h >= aspect:
                        new_h = target_h
                        new_w = max(1, int(round(aspect * new_h)))
                    else:
                        new_w = target_w
                        new_h = max(1, int(round(new_w / aspect)))

            img_resized = cv2.resize(
                img_color, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
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


def render_point_cloud_png(
    point_cloud: Dict[str, Any],
    width: int = 640,
    height: int = 480,
    *,
    cam_pos: tuple = (0.0, -1.0, 2.0),
    cam_yaw_deg: float = 0.0,
) -> Optional[bytes]:
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
            img = np.full((height, width, 3), 255, dtype=np.uint8)
            ok, buf = cv2.imencode(".png", img)
            return buf.tobytes() if ok else None

        # Optional z and intensity
        z_in = point_cloud.get("z")
        if z_in is None:
            z = np.zeros_like(x, dtype=np.float32)
        else:
            z = np.asarray(z_in).astype(np.float32)
            if z.shape != x.shape:
                try:
                    z = np.resize(z, x.shape)
                except Exception:
                    z = np.zeros_like(x, dtype=np.float32)
        intensity = point_cloud.get("intensity", point_cloud.get("snr", None))
        if intensity is None or len(intensity) != len(x):
            inten = np.ones_like(x, dtype=np.float32)
        else:
            inten = np.asarray(intensity, dtype=np.float32)

        # Optional extents from analyser
        max_range = point_cloud.get("max_range", None)
        max_speed = point_cloud.get("max_speed", None)

        # Finite mask across used dims
        finite2d = np.isfinite(x) & np.isfinite(y)
        finite3d = finite2d & np.isfinite(z)
        # Decide 2D/3D based on whether algorithm provided z
        use_3d = z_in is not None and np.any(np.isfinite(z))
        finite = finite3d if use_3d else finite2d
        if not np.any(finite):
            return None
        x = x[finite]
        y = y[finite]
        z = z[finite]
        inten = inten[finite]

        img = np.full((height, width, 3), 255, dtype=np.uint8)

        if use_3d:
            # Perspective projection from camera parameters (yaw around Z, position cam_pos)
            try:
                cx, cy, cz = float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])
            except Exception:
                cx, cy, cz = 0.0, -1.0, 2.0
            yaw = float(cam_yaw_deg) * (np.pi / 180.0)
            c, s = np.cos(-yaw), np.sin(-yaw)  # rotate world into camera yaw frame
            Rz = np.array(
                [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )

            P = np.stack([x - cx, y - cy, z - cz], axis=0)  # shape (3, N)
            Q = Rz @ P
            x_cam = Q[0, :]
            y_cam = Q[1, :]
            z_cam = Q[2, :]

            # Keep points in front of camera (positive forward along +Y in camera frame)
            fmask = y_cam > 1e-6
            if not np.any(fmask):
                ok, buf = cv2.imencode(".png", img)
                return buf.tobytes() if ok else None
            x_cam = x_cam[fmask]
            y_cam = y_cam[fmask]
            z_cam = z_cam[fmask]
            inten_cam = inten[fmask]

            # Perspective projection (pinhole) onto image plane at y=1
            u = x_cam / y_cam
            v = z_cam / y_cam

            # Robust scaling to fit into canvas
            max_abs = float(np.max(np.abs(np.concatenate([u, v])))) if u.size else 1.0
            if not np.isfinite(max_abs) or max_abs <= 0:
                max_abs = 1.0
            scale = 0.45 * float(min(width, height)) / max_abs
            cx_pix = (width - 1) * 0.5
            cy_pix = (height - 1) * 0.5
            px = np.clip((cx_pix + u * scale).astype(np.int32), 0, width - 1)
            py = np.clip((cy_pix - v * scale).astype(np.int32), 0, height - 1)

            # Intensity to grayscale (high intensity -> dark gray; low -> light gray)
            inten_f = inten_cam.astype(np.float32)
            finite_i = np.isfinite(inten_f)
            if np.any(finite_i):
                vals = inten_f[finite_i]
                vmin = float(np.percentile(vals, 1.0))
                vmax = float(np.percentile(vals, 99.0))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                    vmin = float(np.min(vals))
                    vmax = float(np.max(vals))
                    if vmin >= vmax:
                        vmax = vmin + 1.0
                norm = (inten_f - vmin) / (vmax - vmin)
                norm = np.clip(norm, 0.0, 1.0)
            else:
                norm = np.zeros_like(inten_f, dtype=np.float32)
            # Map: 1.0 -> ~30 (dark), 0.0 -> ~230 (light)
            gray = (255.0 - (norm * 220.0 + 20.0)).astype(np.int32)
            gray = np.clip(gray, 0, 255)
            for xi, yi, gi in zip(px, py, gray):
                g = int(gi)
                cv2.circle(img, (int(xi), int(yi)), 2, (g, g, g), -1)
        else:
            # Original 2D rendering path (ignore z)
            if (
                isinstance(max_range, (int, float))
                and np.isfinite(max_range)
                and max_range > 0
            ):
                x_min, x_max = -float(max_range), float(max_range)
                y_min, y_max = 0.0, float(max_range)
            else:
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
            for xi, yi in zip(px, py):
                cv2.circle(img, (int(xi), int(yi)), 2, (0, 180, 0), -1)

        try:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except Exception:
            pass

        # Draw axes and ranges after rotation for correct orientation
        try:
            h, w = img.shape[:2]
            base_thickness = max(1, int(round(min(h, w) / 240)))
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Make labels half the previous size
            font_scale = max(0.2, min(0.4, base_thickness * 0.35))
            margin = max(8, base_thickness * 6)

            if use_3d:
                # Projected axes through camera perspective
                axis_len_world = (
                    max(1.0, float(max_range) * 0.2)
                    if isinstance(max_range, (int, float)) and max_range > 0
                    else 2.0
                )
                # Build world points for axis endpoints from world origin (0,0,0)
                axes_world = np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [axis_len_world, 0.0, 0.0],  # +X
                        [0.0, axis_len_world, 0.0],  # +Y
                        [0.0, 0.0, axis_len_world],  # +Z
                    ],
                    dtype=np.float32,
                ).T  # shape (3,4)
                # Transform to camera frame
                cx, cy, cz = float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])
                yaw = float(cam_yaw_deg) * (np.pi / 180.0)
                c_, s_ = np.cos(-yaw), np.sin(-yaw)
                Rz = np.array(
                    [[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
                )
                Pw = axes_world - np.array([[cx], [cy], [cz]], dtype=np.float32)
                Pc = Rz @ Pw
                # Perspective projection
                y_cam_axis = Pc[1, :]
                mask = y_cam_axis > 1e-6
                if np.sum(mask) >= 2:
                    u = Pc[0, mask] / y_cam_axis[mask]
                    v = Pc[2, mask] / y_cam_axis[mask]
                    max_abs = (
                        float(np.max(np.abs(np.concatenate([u, v])))) if u.size else 1.0
                    )
                    if not np.isfinite(max_abs) or max_abs <= 0:
                        max_abs = 1.0
                    scale = 0.45 * float(min(width, height)) / max_abs
                    cx_pix = (width - 1) * 0.5
                    cy_pix = (height - 1) * 0.5
                    px_axis = (cx_pix + u * scale).astype(np.int32)
                    py_axis = (cy_pix - v * scale).astype(np.int32)
                    # Map back into full list order with fallbacks
                    pts = [(int(px_axis[0]), int(py_axis[0]))] + [None, None]
                    if len(px_axis) > 1:
                        pts[1] = (int(px_axis[1]), int(py_axis[1]))
                    if len(px_axis) > 2:
                        pts[2] = (int(px_axis[2]), int(py_axis[2]))
                    if len(px_axis) > 3:
                        pts.append((int(px_axis[3]), int(py_axis[3])))
                    # Draw lines from origin to endpoints if they were projected
                    origin_pt = pts[0]
                    if origin_pt and len(pts) >= 4:
                        if pts[1]:
                            cv2.arrowedLine(
                                img,
                                origin_pt,
                                pts[1],
                                (0, 0, 255),
                                base_thickness,
                                tipLength=0.2,
                            )
                            cv2.putText(
                                img,
                                "X",
                                (pts[1][0] + 4, pts[1][1] + 4),
                                font,
                                font_scale,
                                (0, 0, 255),
                                base_thickness,
                            )
                        if pts[2]:
                            cv2.arrowedLine(
                                img,
                                origin_pt,
                                pts[2],
                                (0, 128, 0),
                                base_thickness,
                                tipLength=0.2,
                            )
                            cv2.putText(
                                img,
                                "Y",
                                (pts[2][0] + 4, pts[2][1] + 4),
                                font,
                                font_scale,
                                (0, 128, 0),
                                base_thickness,
                            )
                        if pts[3]:
                            cv2.arrowedLine(
                                img,
                                origin_pt,
                                pts[3],
                                (255, 0, 0),
                                base_thickness,
                                tipLength=0.2,
                            )
                            cv2.putText(
                                img,
                                "Z",
                                (pts[3][0] + 4, pts[3][1] + 4),
                                font,
                                font_scale,
                                (255, 0, 0),
                                base_thickness,
                            )

                # Ranges text
                try:
                    x_min = float(np.percentile(x, 1.0))
                    x_max = float(np.percentile(x, 99.0))
                    y_min = float(np.percentile(y, 1.0))
                    y_max = float(np.percentile(y, 99.0))
                    z_min = float(np.percentile(z, 1.0))
                    z_max = float(np.percentile(z, 99.0))
                except Exception:
                    x_min = x_max = y_min = y_max = z_min = z_max = 0.0
                text = f"X:[{x_min:.1f},{x_max:.1f}]  Y:[{y_min:.1f},{y_max:.1f}]  Z:[{z_min:.1f},{z_max:.1f}] m"
                cv2.putText(
                    img,
                    text,
                    (margin, margin + int(16 * font_scale)),
                    font,
                    font_scale,
                    (0, 0, 0),
                    base_thickness,
                )
            else:
                # Draw X and Y axes and labels (image coordinates: origin top-left)
                # Center Y-axis at mid X per requirement
                x0 = w // 2
                y0 = h - margin
                x1 = w - margin
                y1 = margin
                # X axis
                cv2.line(img, (margin, y0), (x1, y0), (0, 0, 0), base_thickness)
                # Y axis through center X (thinner)
                y_th = max(1, base_thickness // 2)
                cv2.line(img, (x0, y0), (x0, y1), (0, 0, 0), y_th)

                # Ticks
                # No ticks for now per requirement

                # Range labels
                try:
                    if (
                        isinstance(max_range, (int, float))
                        and np.isfinite(max_range)
                        and max_range > 0
                    ):
                        x_min, x_max = -float(max_range), float(max_range)
                        y_min, y_max = 0.0, float(max_range)
                    else:
                        x_min = float(np.percentile(x, 1.0))
                        x_max = float(np.percentile(x, 99.0))
                        y_min = float(np.percentile(y, 1.0))
                        y_max = float(np.percentile(y, 99.0))
                except Exception:
                    x_min = x_max = y_min = y_max = 0.0
                label_x = f"X: [{x_min:.1f}, {x_max:.1f}] m"
                label_y = f"Y: [{y_min:.1f}, {y_max:.1f}] m"
                cv2.putText(
                    img,
                    label_x,
                    (x0 + 6, y0 - 6),
                    font,
                    font_scale,
                    (0, 0, 0),
                    base_thickness,
                )
                cv2.putText(
                    img,
                    label_y,
                    (x0 + 6, y1 + int(16 * font_scale)),
                    font,
                    font_scale,
                    (0, 0, 0),
                    base_thickness,
                )
        except Exception:
            pass
        ok, buf = cv2.imencode(".png", img)
        return buf.tobytes() if ok else None
    except Exception:
        return None
