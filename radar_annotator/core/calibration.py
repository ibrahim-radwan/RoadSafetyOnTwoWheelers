"""
Calibration manager.

Loads calibration from JSON (default), YAML when PyYAML is installed,
KITTI-style plaintext (``P2``, ``R0_rect``, ``Tr_velo_to_cam`` / ``Tr_radar_to_cam``),
or JSON embedded in a ``.txt`` file.

Exposes:
    - K: 3x3 camera intrinsics
    - T_cam_from_master: 4x4 extrinsic (master frame -> camera frame)

Expected schema (flexible; missing fields default to identity):
{
    "id": "front_radar_cam_v1",
    "image_size": [1920, 1080],
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "T_cam_from_master": [[...4x4...]]   # or provide "R" (3x3) + "t" (3,)
}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple
import numpy as np

_KITTI_LINE = re.compile(
    r"^(?P<key>P\d|R0_rect|Tr_velo_to_cam|Tr_radar_to_cam|Tr_lidar_to_cam)\s*:\s*(?P<body>.*)$"
)

# Extensions scanned under calib/ after preferred filenames (order matters).
_CALIB_SUFFIXES = frozenset({".json", ".txt", ".yaml", ".yml"})


def iter_calibration_candidate_paths(calib_dir: Path) -> Iterator[Path]:
    """Yield calibration file paths to try: fixed names first, then other
    matches by suffix (sorted) for stable behaviour.
    """
    root = Path(calib_dir)
    if not root.is_dir():
        return
    preferred = (
        "calib.json",
        "calibration.json",
        "calib.txt",
        "calibration.txt",
        "calib.yaml",
        "calibration.yaml",
        "calib.yml",
        "calibration.yml",
    )
    seen: set[Path] = set()
    for name in preferred:
        p = root / name
        if p.is_file():
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                yield p
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        rp = p.resolve()
        if rp in seen:
            continue
        if p.suffix.lower() in _CALIB_SUFFIXES:
            seen.add(rp)
            yield p


class Calibration:
    def __init__(self,
                 K: np.ndarray,
                 T_cam_from_master: np.ndarray,
                 image_size: Tuple[int, int] = (1920, 1080),
                 calib_id: str = "default",
                 P_rect: Optional[np.ndarray] = None):
        """If ``P_rect`` is set (KITTI ``P2`` as 3×4), projection uses ``P@[x,y,z,1]``
        instead of ``K@[x,y,z]/z``, matching the official devkit (baseline column matters).
        """
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.T = np.asarray(T_cam_from_master, dtype=np.float64).reshape(4, 4)
        self.image_size = tuple(image_size)
        self.id = calib_id
        self.P_rect: Optional[np.ndarray] = None
        if P_rect is not None:
            self.P_rect = np.asarray(P_rect, dtype=np.float64).reshape(3, 4)

    @staticmethod
    def identity(image_size: Tuple[int, int] = (1920, 1080)) -> "Calibration":
        """Fallback calibration used when no file is loaded. The camera is
        placed slightly above the master origin looking forward along +x
        of the master frame.
        """
        W, H = image_size
        fx = fy = 0.7 * W
        K = np.array([[fx, 0, W / 2.0],
                      [0, fy, H / 2.0],
                      [0,  0, 1.0]])

        # Master -> camera: rotate axes so master +x (forward) -> camera +z,
        # master -y (right) -> camera +x, master -z (down) -> camera +y.
        # This is the standard automotive radar-to-camera frame swap.
        R = np.array([[ 0, -1,  0],
                      [ 0,  0, -1],
                      [ 1,  0,  0]], dtype=np.float64)
        t = np.array([0.0, -1.2, 0.0])  # camera ~1.2m above radar
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return Calibration(K, T, image_size, "identity_default", P_rect=None)

    def rescaled_to_image_size(self, image_size: Tuple[int, int]) -> "Calibration":
        """Scale intrinsics when the actual image resolution differs from the
        one embedded in the calibration file (common with KITTI-style txt).
        """
        W1, H1 = int(image_size[0]), int(image_size[1])
        W0, H0 = int(self.image_size[0]), int(self.image_size[1])
        if (W0, H0) == (W1, H1):
            return self
        sx = W1 / float(W0)
        sy = H1 / float(H0)
        K = self.K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        P_new = None
        if self.P_rect is not None:
            S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            P_new = S @ self.P_rect
        return Calibration(K, self.T, (W1, H1), self.id, P_rect=P_new)

    @staticmethod
    def _looks_like_kitti_calib(text: str) -> bool:
        for line in text.splitlines():
            m = _KITTI_LINE.match(line.strip())
            if m is not None:
                body = m.group("body").strip()
                if body:
                    return True
        return False

    @staticmethod
    def _strip_kitti_optional_meta(
        raw: str,
    ) -> Tuple[str, Optional[Tuple[int, int]]]:
        """Remove optional ``# image_size W H`` header (our generator / tooling).

        VoD/KITTI viewers ignore unknown lines; we read only the first such header.
        """
        lines = raw.splitlines()
        wh: Optional[Tuple[int, int]] = None
        kept: list[str] = []
        for line in lines:
            s = line.strip()
            parts_sp = s.split()
            if wh is None and len(parts_sp) >= 4 and parts_sp[0] == "#" \
                    and parts_sp[1].lower() == "image_size":
                wh = (int(parts_sp[2]), int(parts_sp[3]))
                continue
            kept.append(line)
        return "\n".join(kept), wh

    @staticmethod
    def _parse_kitti_plaintext(raw: str, stem_id: str) -> Dict[str, Any]:
        raw, explicit_wh = Calibration._strip_kitti_optional_meta(raw)
        matrices: Dict[str, list[float]] = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            m = _KITTI_LINE.match(line)
            if m is None:
                continue
            key = m.group("key")
            body = m.group("body").strip()
            if not body:
                continue
            parts = body.replace(",", " ").split()
            matrices[key] = [float(x) for x in parts]

        p_vals = None
        for pk in ("P2", "P1", "P0", "P3"):
            v = matrices.get(pk)
            if v is not None and len(v) >= 12:
                p_vals = v
                break
        if p_vals is None:
            raise ValueError("KITTI-style calib: need a P0–P2 line with 12 floats")

        P_rect = np.array(
            [
                [p_vals[0], p_vals[1], p_vals[2], p_vals[3]],
                [p_vals[4], p_vals[5], p_vals[6], p_vals[7]],
                [p_vals[8], p_vals[9], p_vals[10], p_vals[11]],
            ],
            dtype=np.float64,
        )
        # Legacy callers still expect K ≈ left block (fine-tuning / fallbacks).
        K = P_rect[:, :3].copy()

        r0 = matrices.get("R0_rect")
        if r0 is None or len(r0) < 9:
            R0 = np.eye(3, dtype=np.float64)
        else:
            R0 = np.array(r0[:9], dtype=np.float64).reshape(3, 3)

        tr_vals = None
        for tk in ("Tr_velo_to_cam", "Tr_radar_to_cam", "Tr_lidar_to_cam"):
            v = matrices.get(tk)
            if v is not None and len(v) >= 12:
                tr_vals = v
                break
        if tr_vals is None:
            raise ValueError(
                "KITTI-style calib: need Tr_velo_to_cam, Tr_radar_to_cam, "
                "or Tr_lidar_to_cam with 12 floats"
            )

        # KITTI stores Tr as 12 floats in 3×4 row-major order [R | t] (same as P).
        R_tr = np.array(
            [
                [tr_vals[0], tr_vals[1], tr_vals[2]],
                [tr_vals[4], tr_vals[5], tr_vals[6]],
                [tr_vals[8], tr_vals[9], tr_vals[10]],
            ],
            dtype=np.float64,
        )
        t_tr = np.array([tr_vals[3], tr_vals[7], tr_vals[11]], dtype=np.float64)
        R_full = R0 @ R_tr
        t_full = R0 @ t_tr
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_full
        T[:3, 3] = t_full

        # KITTI/VOD *.txt files omit image resolution. A hard-coded (1920,1080)
        # breaks rescaling when intrinsics were calibrated at another size (e.g.
        # 640×480 Fusion exports): rescaled_to_image_size would crush fx,cy and
        # projections drift far from objects. Infer nominal size from principal
        # point (reasonable for automotive front cameras; VoD ~1936≈2*cx).
        if explicit_wh is not None:
            W_infer, H_infer = explicit_wh
        else:
            cx_pp = float(K[0, 2])
            cy_pp = float(K[1, 2])
            W_infer = max(int(round(2.0 * cx_pp)), 1)
            H_infer = max(int(round(2.0 * cy_pp)), 1)

        return {
            "id": stem_id,
            "image_size": [W_infer, H_infer],
            "K": K.tolist(),
            "T_cam_from_master": T.tolist(),
            "P_rect": P_rect.tolist(),
        }

    @staticmethod
    def _parse_mapping(raw: str, path: Path) -> Dict[str, Any]:
        text = raw.lstrip("\ufeff").strip()
        last_json_err: json.JSONDecodeError | None = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            last_json_err = e
            parsed = None
        if isinstance(parsed, dict):
            return parsed

        if Calibration._looks_like_kitti_calib(raw):
            try:
                return Calibration._parse_kitti_plaintext(raw, path.stem)
            except ValueError as e:
                raise ValueError(
                    f"KITTI-style calibration in {path.name} could not be read: {e}"
                ) from e

        try:
            import yaml  # type: ignore
        except ImportError:
            yaml = None  # type: ignore
        if yaml is not None:
            try:
                yobj = yaml.safe_load(raw)
            except Exception:
                yobj = None
            if isinstance(yobj, dict):
                return yobj

        hint = ""
        if yaml is None and last_json_err is not None:
            hint = (
                " Not valid JSON; install PyYAML (`pip install pyyaml`) "
                "if this file is YAML."
            )
        detail = ""
        if last_json_err is not None:
            detail = f" ({last_json_err})"
        raise ValueError(
            f"Could not parse calibration {path.name} as JSON, YAML, or "
            f"KITTI plaintext.{detail}{hint}"
        )

    @staticmethod
    def load(path: Path) -> "Calibration":
        path = Path(path)
        raw = path.read_text(encoding="utf-8")
        data = Calibration._parse_mapping(raw, path)
        K = np.array(data["K"], dtype=np.float64)
        if "T_cam_from_master" in data:
            T = np.array(data["T_cam_from_master"], dtype=np.float64)
        else:
            R = np.array(data.get("R", np.eye(3)), dtype=np.float64)
            t = np.array(data.get("t", [0, 0, 0]), dtype=np.float64).reshape(3)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
        image_size = tuple(data.get("image_size", (1920, 1080)))
        pr_raw = data.get("P_rect")
        P_rect = None
        if pr_raw is not None:
            P_rect = np.asarray(pr_raw, dtype=np.float64).reshape(3, 4)
        return Calibration(
            K,
            T,
            image_size,
            data.get("id", Path(path).stem),
            P_rect=P_rect,
        )

    def save(self, path: Path) -> None:
        payload = {
            "id": self.id,
            "image_size": list(self.image_size),
            "K": self.K.tolist(),
            "T_cam_from_master": self.T.tolist(),
        }
        if self.P_rect is not None:
            payload["P_rect"] = self.P_rect.tolist()
        Path(path).write_text(json.dumps(payload, indent=2))

    def __repr__(self) -> str:
        return f"Calibration(id={self.id!r}, image_size={self.image_size})"
