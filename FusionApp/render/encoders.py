import numpy as np
import cv2
import io
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


def _render_point_cloud_mpl_png(
    x: np.ndarray,
    y: np.ndarray,
    intensity: np.ndarray,
    width: int,
    height: int,
    max_range: Optional[float],
) -> Optional[bytes]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm

        dpi = 100
        fig_w = max(1.0, width / dpi)
        fig_h = max(1.0, height / dpi)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.set_facecolor("white")
        ax.grid(False)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        # Set limits (and reapply every frame to ensure visibility)
        if (
            isinstance(max_range, (int, float))
            and np.isfinite(max_range)
            and max_range > 0
        ):
            ax.set_xlim([-float(max_range), float(max_range)])
            ax.set_ylim([0.0, float(max_range)])
        else:
            if x.size and y.size:
                xmin, xmax = float(np.min(x)), float(np.max(x))
                ymin, ymax = float(np.min(y)), float(np.max(y))
                if xmax - xmin < 1e-6:
                    pad = 0.5
                    xmin, xmax = xmin - pad, xmax + pad
                if ymax - ymin < 1e-6:
                    pad = 0.5
                    ymin, ymax = ymin - pad, ymax + pad
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])

        inten = np.asarray(intensity, dtype=float)
        finite = np.isfinite(inten)
        if np.any(finite):
            vals = inten[finite]
            vmin = float(np.percentile(vals, 1.0))
            vmax = float(np.percentile(vals, 99.0))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                vmin = float(np.min(vals))
                vmax = float(np.max(vals)) if np.max(vals) > vmin else (vmin + 1.0)
        else:
            vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.Greys_r
        colors = cmap(norm(inten)) if inten.size else np.zeros((0, 4))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-9:
            colors = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]]), (inten.size, 1))
        ax.scatter(x, y, s=6, c=colors, edgecolors="none", linewidths=0)

        fig.tight_layout(pad=0.1)
        buf = io.BytesIO()
        fig.savefig(
            buf, format="png", facecolor="white", bbox_inches="tight", pad_inches=0.1
        )
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


def _render_point_cloud_o3d_png(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    intensity: np.ndarray,
    width: int,
    height: int,
    cam_pos: tuple,
    cam_yaw_deg: float,
    max_range: Optional[float],
    two_d: bool = False,
) -> Optional[bytes]:
    try:
        import open3d as o3d

        # Build point cloud (allow empty)
        if two_d:
            pts = np.stack([x, y, np.zeros_like(x)], axis=1).astype(np.float32)
        else:
            pts = np.stack([x, y, z], axis=1).astype(np.float32)
        num_pts = pts.shape[0]
        pcd = None
        if num_pts > 0:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            # Simple, high-contrast coloring: black points on white background
            colors = np.zeros((num_pts, 3), dtype=np.float64)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Offscreen renderer
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        try:
            mat.point_size = 6.0
        except Exception:
            pass
        scene = renderer.scene
        scene.set_background([1.0, 1.0, 1.0, 1.0])
        if pcd is not None:
            scene.add_geometry("pc", pcd, mat)

        # Add 3D axes as line geometry (perspective-projected in 3D mode)
        # X: [-R, +R], Y: [0, +R], Z: [-R, +R]
        R_default = 10.0
        if (
            isinstance(max_range, (int, float))
            and np.isfinite(max_range)
            and max_range > 0
        ):
            R = float(max_range)
        else:
            # Fallback based on data span
            span_x = float(np.ptp(x)) if x.size else 0.0
            span_y = float(np.ptp(y)) if y.size else 0.0
            span_z = float(np.ptp(z)) if (not two_d and z.size) else 0.0
            guess = max(span_x, span_y, span_z, R_default)
            R = guess if guess > 1e-3 else R_default
        axis_pts = [
            [-R, 0.0, 0.0],
            [R, 0.0, 0.0],  # X
            [0.0, 0.0, 0.0],
            [0.0, R, 0.0],  # Y (positive only)
        ]
        axis_lines = [[0, 1], [2, 3]]
        axis_colors = [
            [0.0, 0.0, 0.0],  # X black
            [
                0.0,
                0.0,
                0.0,
            ],  # Y black (thin desired; renderer may not support per-line width)
        ]
        if not two_d:
            # Add Z axis for 3D visualization
            axis_pts += [[0.0, 0.0, -R], [0.0, 0.0, R]]
            axis_lines += [[4, 5]]
            axis_colors += [[0.0, 0.0, 0.0]]
        axes = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array(axis_pts, dtype=np.float32)),
            lines=o3d.utility.Vector2iVector(np.array(axis_lines, dtype=np.int32)),
        )
        axes.colors = o3d.utility.Vector3dVector(
            np.array(axis_colors, dtype=np.float32)
        )
        mat_axes = o3d.visualization.rendering.MaterialRecord()
        mat_axes.shader = "unlitLine"
        try:
            mat_axes.line_width = 1.0  # Best-effort thinner line
        except Exception:
            pass
        scene.add_geometry("axes", axes, mat_axes)

        # Camera
        if two_d:
            # Orthographic top-down on XY: eye at (0, -d, 0), look towards +Y
            d = (
                float(max_range) * 1.2
                if isinstance(max_range, (int, float)) and max_range > 0
                else 20.0
            )
            # Aim at data centroid for better framing
            cx0 = float(np.median(x)) if x.size else 0.0
            cy0 = float(np.median(y)) if y.size else 0.0
            cz0 = 0.0
            eye = np.array([cx0, cy0 - d, cz0], dtype=np.float32)
            center = np.array([cx0, cy0, cz0], dtype=np.float32)
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            scene.camera.look_at(center.tolist(), eye.tolist(), up.tolist())
            # Lock orthographic width to physical extent X in [-R, +R]
            ortho_width = R * 2.0
            scene.camera.set_projection(
                ortho_width,
                width / max(1, height),
                0.01,
                1000.0,
                o3d.visualization.rendering.Camera.FovType.Orthographic,
            )
        else:
            cxp, cyp, czp = float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])
            yaw = float(cam_yaw_deg) * (np.pi / 180.0)
            fwd = np.array([np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float32)
            eye = np.array([cxp, cyp, czp], dtype=np.float32)
            # Look at data centroid rather than an infinite ray
            cx0 = float(np.median(x)) if x.size else 0.0
            cy0 = float(np.median(y)) if y.size else 0.0
            cz0 = float(np.median(z)) if z.size else 0.0
            center = np.array([cx0, cy0, cz0], dtype=np.float32)
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            far = (
                float(max_range) * 2.0
                if isinstance(max_range, (int, float)) and max_range > 0
                else 50.0
            )
            near = 0.01
            scene.camera.look_at(center.tolist(), eye.tolist(), up.tolist())
            scene.camera.set_projection(
                60.0,
                width / max(1, height),
                near,
                far,
                o3d.visualization.rendering.Camera.FovType.Vertical,
            )

        img = renderer.render_to_image()
        if img is None:
            # Produce empty canvas if rendering yielded nothing
            blank = np.full((height, width, 3), 255, dtype=np.uint8)
            try:
                txt = f"pts={pts.shape[0]}"
                cv2.putText(
                    blank,
                    txt,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.rectangle(blank, (20, 20), (width - 20, height - 20), (0, 0, 0), 2)
            except Exception:
                pass
            ok, out = cv2.imencode(".png", blank)
            return out.tobytes() if ok else None
        # Convert to numpy and encode
        np_img = np.asarray(img)
        if np_img.dtype != np.uint8:
            np_img = np.clip(np_img * 255.0, 0, 255).astype(np.uint8)
        if np_img.ndim == 2:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
        elif np_img.shape[2] == 4:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
        elif np_img.shape[2] == 3:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        # Overlay simple debug text and frame; also show axis labels/ranges
        try:
            txt = f"pts={pts.shape[0]}"
            cv2.putText(
                np_img,
                txt,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            h, w = np_img.shape[:2]
            margin = 20
            cv2.rectangle(
                np_img, (margin, margin), (w - margin, h - margin), (0, 0, 0), 4
            )
            # Draw pixel-space axes overlay to guarantee visibility
            y0 = h - margin  # Y=0 at bottom
            x0 = w // 2  # X=0 at center
            cv2.line(np_img, (margin, y0), (w - margin, y0), (0, 0, 0), 3)
            cv2.line(np_img, (x0, h - margin), (x0, margin), (0, 0, 0), 3)
            # Strong diagonals for visibility confirmation
            cv2.line(
                np_img, (margin, h - margin), (w - margin, margin), (200, 200, 200), 1
            )
            cv2.line(
                np_img, (margin, margin), (w - margin, h - margin), (200, 200, 200), 1
            )
            # Axis labels (small)
            label_scale = 0.6
            label_th = 2
            r_txt = f"R={R:.1f}m"
            cv2.putText(
                np_img,
                "X [-R, +R] m",
                (margin + 8, h - margin - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_scale,
                (0, 0, 0),
                label_th,
                cv2.LINE_AA,
            )
            cv2.putText(
                np_img,
                "Y [0, +R] m",
                (margin + 8, margin + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_scale,
                (0, 0, 0),
                label_th,
                cv2.LINE_AA,
            )
            # Overlay 2D projection of points to guarantee visibility
            if x.size and y.size:
                # Map X in [-R, +R] to [margin, w - margin]
                sx = (w - 2 * margin) / max(1e-6, (2.0 * R))
                # Map Y in [0, +R] to [h - margin, margin]
                sy = (h - 2 * margin) / max(1e-6, R)
                px = (x + R) * sx + margin
                py = (R - np.clip(y, 0.0, R)) * sy + margin
                pts2d = np.stack([px, py], axis=1).astype(np.int32)
                for p in pts2d:
                    cx_i = int(np.clip(p[0], margin, w - margin))
                    cy_i = int(np.clip(p[1], margin, h - margin))
                    cv2.circle(
                        np_img, (cx_i, cy_i), 5, (0, 0, 0), -1, lineType=cv2.LINE_AA
                    )
            # Big watermark for debug visibility
            cv2.putText(
                np_img,
                "PC",
                (w - 80, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
        except Exception:
            pass
        ok, out = cv2.imencode(".png", np_img)
        return out.tobytes() if ok else None
    except Exception:
        # Always return an empty image on error to avoid 204s with empty plots
        try:
            blank = np.full((height, width, 3), 255, dtype=np.uint8)
            ok, out = cv2.imencode(".png", blank)
            return out.tobytes() if ok else None
        except Exception:
            return None


def _render_point_cloud_cv_png(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    intensity: np.ndarray,
    width: int,
    height: int,
    cam_pos: tuple,
    cam_yaw_deg: float,
    max_range: Optional[float],
    two_d: bool = False,
) -> Optional[bytes]:
    try:
        canvas = np.full((int(height), int(width), 3), 255, dtype=np.uint8)
        h, w = canvas.shape[:2]
        margin = 20
        # Border and axes
        cv2.rectangle(canvas, (margin, margin), (w - margin, h - margin), (0, 0, 0), 2)
        # Axis labels
        R = (
            float(max_range)
            if isinstance(max_range, (int, float))
            and np.isfinite(max_range)
            and max_range > 0
            else 10.0
        )
        cv2.putText(
            canvas,
            "X [-R, +R] m",
            (margin + 8, h - margin - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Y [0, +R] m",
            (margin + 8, margin + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        # Always draw axes in image space
        y0 = h - margin
        x0 = w // 2
        cv2.line(canvas, (margin, y0), (w - margin, y0), (0, 0, 0), 2)
        cv2.line(canvas, (x0, h - margin), (x0, margin), (0, 0, 0), 2)

        # Colors from intensity (dark for high, light for low)
        inten = (
            np.asarray(intensity, dtype=np.float32)
            if intensity is not None
            else np.ones_like(x, dtype=np.float32)
        )
        if inten.size:
            vals = inten[np.isfinite(inten)]
            if vals.size:
                vmin = float(np.percentile(vals, 1.0))
                vmax = float(np.percentile(vals, 99.0))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                    vmin = float(vals.min())
                    vmax = float(vals.max()) if float(vals.max()) > vmin else vmin + 1.0
            else:
                vmin, vmax = 0.0, 1.0
            norm = np.clip((inten - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
            gray = 1.0 - (0.86 * norm + 0.08)
            colors = (gray * 255.0).clip(0, 255).astype(np.uint8)
        else:
            colors = np.full_like(x, 0, dtype=np.uint8)

        if two_d:
            # Map X in [-R, +R] to [margin, w - margin]; Y in [0, +R] to [h - margin, margin]
            sx = (w - 2 * margin) / max(1e-6, (2.0 * R))
            sy = (h - 2 * margin) / max(1e-6, R)
            px = (np.clip(x, -R, R) + R) * sx + margin
            py = (R - np.clip(y, 0.0, R)) * sy + margin
            pts2d = (
                np.stack([px, py], axis=1).astype(np.int32)
                if px.size
                else np.zeros((0, 2), dtype=np.int32)
            )
        else:
            # Simple 3D perspective projection with yaw around Z and camera at cam_pos
            cx, cy, cz = float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])
            yaw = float(cam_yaw_deg) * (np.pi / 180.0)
            # Translate
            X = x - cx
            Y = y - cy
            Z = z - cz
            # Rotate by -yaw around Z
            cos_y = np.cos(-yaw)
            sin_y = np.sin(-yaw)
            Xr = cos_y * X - sin_y * Y
            Yr = sin_y * X + cos_y * Y
            Zr = Z
            # Define camera depth as forward along +Y' -> map to Zc, and Xc=X', Yc=Z (height)
            Zc = Yr
            Xc = Xr
            Yc = Zr
            # Perspective parameters
            fov_deg = 60.0
            fx = 0.5 * w / np.tan(np.deg2rad(fov_deg * 0.5))
            fy = fx
            cx_i = w * 0.5
            cy_i = h * 0.8  # lift horizon higher
            near = 0.1
            far = max(10.0, 4.0 * R)
            valid = np.isfinite(Zc) & (Zc > near) & (Zc < far)
            Xc = Xc[valid]
            Yc = Yc[valid]
            Zc = Zc[valid]
            colv = (
                colors[valid]
                if colors.size == valid.size
                else np.full_like(Xc, 0, dtype=np.uint8)
            )
            if Zc.size:
                u = fx * (Xc / Zc) + cx_i
                v = fy * (-Yc / Zc) + cy_i
                pts2d = np.stack([u, v], axis=1).astype(np.int32)
                pts2d[:, 0] = np.clip(pts2d[:, 0], margin, w - margin)
                pts2d[:, 1] = np.clip(pts2d[:, 1], margin, h - margin)
                colors = colv
            else:
                pts2d = np.zeros((0, 2), dtype=np.int32)

        # Draw points
        for idx in range(pts2d.shape[0]):
            p = pts2d[idx]
            c = int(colors[idx]) if idx < len(colors) else 0
            cv2.circle(
                canvas, (int(p[0]), int(p[1])), 4, (c, c, c), -1, lineType=cv2.LINE_AA
            )

        # Debug text
        cv2.putText(
            canvas,
            f"pts={int(x.size)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        ok, out = cv2.imencode(".png", canvas)
        return out.tobytes() if ok else None
    except Exception:
        try:
            blank = np.full((height, width, 3), 255, dtype=np.uint8)
            ok, out = cv2.imencode(".png", blank)
            return out.tobytes() if ok else None
        except Exception:
            return None


def _transform_ra_to_polar(ra_data: np.ndarray, extents: tuple) -> np.ndarray:
    """
    Transform rectangular range-azimuth data to polar coordinates for display.
    
    Args:
        ra_data: Range-Azimuth data, shape (n_range, n_azimuth)
        extents: (az_min, az_max, range_min, range_max) in degrees and meters
        
    Returns:
        Polar image for display, shape (height, width)
    """
    try:
        from scipy.ndimage import map_coordinates
    except ImportError:
        # Fallback to rectangular if scipy not available
        return ra_data
    
    # Data comes in as (azimuth, range) after .T in radar_proc_kradar.py
    # Transpose to get (range, azimuth) for proper polar mapping
    ra_data = ra_data.T
    n_range, n_azimuth = ra_data.shape
    az_min, az_max, range_min, range_max = extents
    
    # Create output Cartesian grid
    # Height (y-axis): 0 to range_max (bottom to top)
    # Width (x-axis): -range_max to +range_max (left to right)
    height = 400  # pixels
    width = 800   # pixels
    
    # Create Cartesian meshgrid
    # X: lateral distance from -range_max (left) to +range_max (right)
    # Y: forward distance from 0 (bottom/radar) to range_max (top)
    x = np.linspace(-range_max, range_max, width)
    y = np.linspace(0, range_max, height)  # 0 at bottom (radar position), max at top
    X, Y = np.meshgrid(x, y)
    
    # Convert Cartesian (x, y) to polar (range, azimuth)
    R = np.sqrt(X**2 + Y**2)
    Az_rad = np.arctan2(X, Y)  # atan2(x, y) gives azimuth from y-axis
    Az_deg = np.rad2deg(Az_rad)
    
    # Create output image (initialize with MAXIMUM value for areas outside coverage = white/bright)
    polar_image = np.full((height, width), np.max(ra_data), dtype=np.float32)
    
    # Mask for valid polar region (within azimuth coverage and range)
    mask = (Az_deg >= az_min) & (Az_deg <= az_max) & (R >= range_min) & (R <= range_max)
    
    if np.any(mask):
        # Normalize to indices
        range_idx = (R[mask] - range_min) / (range_max - range_min) * (n_range - 1)
        az_idx = (Az_deg[mask] - az_min) / (az_max - az_min) * (n_azimuth - 1)
        
        # Clip to valid indices
        range_idx = np.clip(range_idx, 0, n_range - 1)
        az_idx = np.clip(az_idx, 0, n_azimuth - 1)
        
        # Bilinear interpolation
        coords = np.array([range_idx, az_idx])
        interpolated = map_coordinates(ra_data, coords, order=1, mode='nearest')
        
        polar_image[mask] = interpolated
    
    # Flip vertically: image row 0 should be at top (y=range_max), row height-1 at bottom (y=0)
    polar_image = np.flipud(polar_image)
    
    return polar_image


def heatmap_to_png(
    array2d: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    extents: Optional[tuple] = None,
    *,
    force_square: bool = False,
    target_size: tuple = (640, 480),
    polar: bool = False,
) -> Optional[bytes]:
    try:
        a = np.asarray(array2d)
        if a.ndim != 2 or a.size == 0:
            return None
        a = a.astype(np.float32, copy=False)
        
        # Apply polar transformation for range-azimuth data if requested
        if polar and extents is not None and len(extents) == 4:
            a = _transform_ra_to_polar(a, extents)
            # Update extents for polar display (Cartesian space)
            az_min, az_max, range_min, range_max = extents
            extents = (-range_max, range_max, 0, range_max)
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
    backend: str = "auto",
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
        # Even with zero points we still want to draw axes

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
        if np.any(finite):
            x = x[finite]
            y = y[finite]
            z = z[finite]
            inten = inten[finite]
        else:
            # Keep arrays empty; renderer will draw axes on white canvas
            x = x[:0]
            y = y[:0]
            z = z[:0]
            inten = inten[:0]

        # Choose high-quality renderer: Matplotlib for 2D, Open3D for 3D
        be = (backend or "auto").lower()
        if not use_3d:
            # 2D path uses Matplotlib Agg for quality
            return _render_point_cloud_mpl_png(
                x,
                y,
                inten,
                width,
                height,
                max_range,
            )
        # 3D path prefers Open3D
        return _render_point_cloud_o3d_png(
            x,
            y,
            z,
            inten,
            width,
            height,
            cam_pos,
            cam_yaw_deg,
            max_range,
            two_d=False,
        )
    except Exception:
        return None
