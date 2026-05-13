"""
Generate a small synthetic radar + camera dataset for testing the tool.

Usage:
    python -m radar_annotator.tools.generate_sample_data ./sample_dataset

Creates:
    <out>/radar/000001.npy ... 000005.npy    (radar point clouds)
    <out>/image/000001.png ... 000005.png    (camera images, synthetic)
    <out>/calib/calib.json                   (calibration matching the scene)
"""
from __future__ import annotations

import sys
from pathlib import Path
import json
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def make_scene(seed: int):
    """Build one scene: a list of (x, y, z, length, width, height, yaw, class)
    tuples for ground-truth objects placed ahead of the ego."""
    rng = np.random.default_rng(seed)
    objects = []
    n = rng.integers(2, 5)
    for _ in range(n):
        x = float(rng.uniform(8, 35))     # forward distance
        y = float(rng.uniform(-6, 6))     # lateral
        cls = rng.choice(["Car", "Truck", "Pedestrian"], p=[0.6, 0.2, 0.2])
        if cls == "Car":
            l, w, h = rng.uniform(4.0, 4.7), rng.uniform(1.7, 1.9), rng.uniform(1.4, 1.7)
        elif cls == "Truck":
            l, w, h = rng.uniform(6.5, 9.0), rng.uniform(2.3, 2.6), rng.uniform(2.8, 3.5)
        else:
            l, w, h = 0.6, 0.6, 1.75
        yaw = float(rng.uniform(-0.3, 0.3))
        z = h / 2.0
        objects.append((x, y, z, l, w, h, yaw, cls))
    return objects


def rotate_z(points: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return points @ R.T


def sample_radar_points(objects, rng) -> np.ndarray:
    """Generate a realistic-looking sparse radar point cloud for a scene."""
    pts = []
    # Ground clutter
    n_clutter = rng.integers(200, 400)
    for _ in range(n_clutter):
        r = rng.uniform(3, 50)
        a = rng.uniform(-np.pi / 3, np.pi / 3)
        x, y = r * np.cos(a), r * np.sin(a)
        z = rng.normal(0.0, 0.15)
        pts.append([x, y, z])

    # Object returns
    for (x, y, z, l, w, h, yaw, cls) in objects:
        n = rng.integers(15, 45)
        local = np.column_stack([
            rng.uniform(-l / 2, l / 2, n),
            rng.uniform(-w / 2, w / 2, n),
            rng.uniform(-h / 2, h / 2, n),
        ])
        world = rotate_z(local, yaw) + np.array([x, y, z])
        # Add a bit of range noise
        world += rng.normal(0, 0.08, world.shape)
        pts.extend(world.tolist())

    return np.array(pts, dtype=np.float32)


def identity_calibration(image_size):
    """The same default calibration used by Calibration.identity()."""
    W, H = image_size
    fx = fy = 0.7 * W
    K = [[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1]]
    R = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    t = [0.0, -1.2, 0.0]
    T = np.eye(4)
    T[:3, :3] = np.array(R)
    T[:3, 3] = np.array(t)
    return K, T.tolist()


def render_scene(objects, image_size, K, T) -> np.ndarray:
    W, H = image_size
    img = np.full((H, W, 3), (40, 45, 55), dtype=np.uint8)

    # Horizon
    img[: H // 2] = (60, 65, 75)
    # Road
    for y in range(H // 2, H):
        t = (y - H / 2) / (H / 2)
        shade = int(40 + 30 * t)
        img[y, :] = (shade, shade, shade)
    # Lane markings
    for k in range(-2, 3):
        for row in range(H // 2 + 10, H, 40):
            x_master = 5.0 + (row - H / 2) * 0.15
            y_master = k * 3.5
            cam = T @ np.array([x_master, y_master, 0.0, 1.0])
            if cam[2] <= 0:
                continue
            u = (K[0][0] * cam[0] + K[0][2] * cam[2]) / cam[2]
            v = (K[1][1] * cam[1] + K[1][2] * cam[2]) / cam[2]
            if 0 <= int(u) < W and 0 <= int(v) < H:
                cv2.rectangle(img, (int(u) - 2, int(v) - 1),
                              (int(u) + 2, int(v) + 1), (200, 200, 200), -1)

    # Draw a simple colored rectangle per object (a very rough "vehicle")
    for (x, y, z, l, w, h, yaw, cls) in objects:
        corners_local = np.array([
            [-l/2, -w/2, -h/2], [ l/2, -w/2, -h/2], [ l/2,  w/2, -h/2], [-l/2,  w/2, -h/2],
            [-l/2, -w/2,  h/2], [ l/2, -w/2,  h/2], [ l/2,  w/2,  h/2], [-l/2,  w/2,  h/2],
        ])
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        world = corners_local @ R.T + np.array([x, y, z])
        homog = np.hstack([world, np.ones((8, 1))])
        cam = (T @ homog.T).T
        if np.any(cam[:, 2] <= 0.1):
            continue
        u = (K[0][0] * cam[:, 0] + K[0][2] * cam[:, 2]) / cam[:, 2]
        v = (K[1][1] * cam[:, 1] + K[1][2] * cam[:, 2]) / cam[:, 2]
        xmin, xmax = int(max(0, u.min())), int(min(W - 1, u.max()))
        ymin, ymax = int(max(0, v.min())), int(min(H - 1, v.max()))
        if xmax <= xmin or ymax <= ymin:
            continue
        color = {"Car": (80, 160, 255), "Truck": (80, 140, 255),
                 "Pedestrian": (120, 230, 120)}.get(cls, (180, 180, 180))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, -1)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
        cv2.putText(img, cls, (xmin + 4, ymin + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return img


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m radar_annotator.tools.generate_sample_data <output_dir>")
        sys.exit(1)
    if cv2 is None:
        print("opencv-python is required to generate sample images.")
        sys.exit(1)

    out = Path(sys.argv[1])
    (out / "radar").mkdir(parents=True, exist_ok=True)
    (out / "image").mkdir(parents=True, exist_ok=True)
    (out / "calib").mkdir(parents=True, exist_ok=True)

    image_size = (1280, 720)
    K, T = identity_calibration(image_size)

    with open(out / "calib" / "calib.json", "w") as f:
        json.dump({
            "id": "sample_front_cam",
            "image_size": list(image_size),
            "K": K,
            "T_cam_from_master": T,
        }, f, indent=2)

    rng = np.random.default_rng(0)
    for i in range(1, 6):
        frame_id = f"{i:06d}"
        objects = make_scene(i)
        pts = sample_radar_points(objects, rng)
        np.save(out / "radar" / f"{frame_id}.npy", pts)

        img = render_scene(objects, image_size, K, np.array(T))
        cv2.imwrite(str(out / "image" / f"{frame_id}.png"), img)
        print(f"  wrote {frame_id}: {len(pts)} radar points, {len(objects)} objects")

    print(f"\nSample dataset written to: {out}")
    print(f"Open it in the tool with: python -m radar_annotator  (then Ctrl+O)")


if __name__ == "__main__":
    main()
