import numpy as np
import cv2
from typing import Optional


def encode_jpeg(bgr: np.ndarray, quality: int = 80) -> Optional[bytes]:
    try:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, buf = cv2.imencode(".jpg", bgr, encode_params)
        if not ok:
            return None
        return buf.tobytes()
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
    width = 800  # pixels

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
        interpolated = map_coordinates(ra_data, coords, order=1, mode="nearest")

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
