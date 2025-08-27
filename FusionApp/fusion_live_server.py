"""
Fusion live web server: exposes REST/SSE endpoints and a simple web UI.

Features (initial version):
- Start FusionEngine in a background process and read latest results
- REST control: start/stop recording
- Status endpoint
- Video endpoints: single frame JPEG and MJPEG stream

Next steps (planned):
- Attach to radar results SHM for RD/RA endpoints as PNG
- Detections JSON endpoint (radar point cloud)
- Status Server-Sent Events (SSE) stream at 1s cadence
"""

import os
import sys
import time
import atexit
import signal
import threading
import queue
import json
from typing import Optional, Dict, Any, Generator, List
from collections import deque

import numpy as np
import cv2
from flask import Flask, Response, jsonify, request, send_file, make_response
from flask import render_template_string

# Local imports
from engine.fusion_factory import FusionFactory
from engine.fusion_engine import FusionEngine
from sample_processing.radar_params import ADCParams
from config_params import CFGS
from utils import setup_logger, disable_shm_resource_tracker

# Low-level radar control
import fpga_udp
from mmwave import dsp


logger = setup_logger("fusion_live_server")


def _ensure_jsonable(value):
    try:
        import numpy as _np  # local alias

        if isinstance(value, _np.ndarray):
            return value.tolist()
        if isinstance(value, _np.generic):
            try:
                return value.item()
            except Exception:
                pass
    except Exception:
        pass
    if isinstance(value, dict):
        return {k: _ensure_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ensure_jsonable(v) for v in value]
    return value


class FusionRunner:
    """
    Starts FusionEngine in background and exposes latest data to the web server.
    """

    def __init__(self):
        self.logger = setup_logger("FusionRunner")
        self._stop_event = threading.Event()

        # Queues for inter-process data
        from multiprocessing import Queue, Event, Process

        self._radar_results_queue = Queue(maxsize=3)
        self._camera_results_queue = Queue(maxsize=2)
        self._control_queue = Queue()
        self._engine_stop_event = Event()

        # Latest data buffers
        self._latest_frame_bgr: Optional[np.ndarray] = None
        self._latest_frame_ts: float = 0.0
        self._latest_camera_objects: Optional[Any] = None  # Not used yet

        # Radar outputs (point cloud JSON-friendly)
        self._latest_point_cloud: Optional[Dict[str, Any]] = None
        self._latest_radar_ts: float = 0.0

        # RD/RA data; support both in-band arrays and SHM-based
        self._latest_rd: Optional[np.ndarray] = None
        self._latest_ra: Optional[np.ndarray] = None
        self._rd_blocks = []  # type: ignore[var-annotated]
        self._ra_blocks = []  # type: ignore[var-annotated]
        self._rd_shape: Optional[tuple] = None
        self._ra_shape: Optional[tuple] = None
        self._rd_dtype: str = "float32"
        self._ra_dtype: str = "float32"
        self._last_res_slot: Optional[int] = None
        self._last_res_seq: int = 0

        # Background threads/process
        self._engine_process: Optional[Process] = None
        self._camera_reader_thread: Optional[threading.Thread] = None
        self._radar_reader_thread: Optional[threading.Thread] = None
        self._engine_monitor_thread: Optional[threading.Thread] = None

        # Stats
        self._stats_lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "start_time": time.time(),
            "last_camera_update": None,
            "last_radar_update": None,
            "camera_drops": 0,
            "radar_drops": 0,
        }

        # Rolling windows (last 60s) for rates/drops
        self._radar_drop_hist: deque = deque()
        self._radar_update_times: deque = deque()
        self._camera_update_times: deque = deque()

        # Lifecycle state
        self._starting: bool = False
        self._running: bool = False
        self._failed: bool = False
        self._failure_reason: Optional[str] = None
        self._radar_connected: bool = False
        self._camera_connected: bool = False
        self._radar_only: bool = False

    def start(self, radar_only: bool = False) -> bool:
        if self._running or self._starting:
            return False
        self._failed = False
        self._failure_reason = None
        self._starting = True
        self._radar_connected = False
        self._camera_connected = False

        # Reset stop events and queues for a fresh run (previous stop() sets these)
        try:
            # Thread stop flag for reader/monitor loops
            self._stop_event = threading.Event()
            # Engine stop flag passed to child process
            from multiprocessing import Event as _MPEvent

            self._engine_stop_event = _MPEvent()
            # Fresh result queues to avoid stale/closed state from prior run
            from multiprocessing import Queue as _MPQueue

            self._radar_results_queue = _MPQueue(maxsize=3)
            self._camera_results_queue = _MPQueue(maxsize=2)
        except Exception:
            pass

        # Start fusion engine in a child process
        fusion_engine: FusionEngine
        if radar_only:
            fusion_engine = FusionFactory.create_live_radar_only()
        else:
            fusion_engine = FusionFactory.create_live_fusion()

        from multiprocessing import Process

        self._engine_process = Process(
            target=fusion_engine.run,
            args=(
                self._radar_results_queue,
                None if radar_only else self._camera_results_queue,
                self._engine_stop_event,
                self._control_queue,
                None,  # status_queue not used in live mode
            ),
        )
        try:
            # Initialize radar HW only on demand
            ok_hw = _radar_hw_init()
            if not ok_hw:
                self.logger.error(
                    "Radar HW init failed; proceeding to start engine anyway"
                )
            self._engine_process.start()
            self.logger.info(
                f"FusionEngine process started: pid={self._engine_process.pid}"
            )
            self._radar_only = bool(radar_only)
        except Exception as e:
            self._failed = True
            self._failure_reason = f"engine start error: {e}"
            self._starting = False
            self._running = False
            return False

        # Reset timing window start
        with self._stats_lock:
            self._stats["start_time"] = time.time()

        # Start queue reader threads
        if not radar_only:
            self._camera_reader_thread = threading.Thread(
                target=self._camera_reader_loop, name="CameraReader", daemon=True
            )
            self._camera_reader_thread.start()

        self._radar_reader_thread = threading.Thread(
            target=self._radar_reader_loop, name="RadarReader", daemon=True
        )
        self._radar_reader_thread.start()

        # Monitor engine process liveness
        self._engine_monitor_thread = threading.Thread(
            target=self._engine_monitor_loop, name="EngineMonitor", daemon=True
        )
        self._engine_monitor_thread.start()

        self._starting = False
        self._running = True
        return True

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._engine_stop_event.set()
        except Exception:
            pass
        # Best-effort join
        if self._engine_process is not None:
            self._engine_process.join(timeout=3)
            if self._engine_process.is_alive():
                try:
                    self._engine_process.kill()
                except Exception:
                    pass
                self._engine_process.join(timeout=1)
            self._engine_process = None
        # Join reader threads
        try:
            if self._camera_reader_thread is not None:
                self._camera_reader_thread.join(timeout=1)
        except Exception:
            pass
        try:
            if self._radar_reader_thread is not None:
                self._radar_reader_thread.join(timeout=1)
        except Exception:
            pass
        self._camera_reader_thread = None
        self._radar_reader_thread = None
        # Detach RD/RA SHM if attached
        try:
            for shm in self._rd_blocks or []:
                try:
                    shm.close()
                except Exception:
                    pass
            for shm in self._ra_blocks or []:
                try:
                    shm.close()
                except Exception:
                    pass
        except Exception:
            pass
        # Cleanup radar HW
        _radar_hw_cleanup()
        self._running = False
        self._starting = False
        # Reset connection flags and latest buffers
        self._radar_connected = False
        self._camera_connected = False
        self._latest_frame_bgr = None
        self._latest_frame_ts = 0.0
        self._latest_point_cloud = None
        self._latest_radar_ts = 0.0
        with self._stats_lock:
            self._stats["last_camera_update"] = None
            self._stats["last_radar_update"] = None

    def _camera_reader_loop(self) -> None:
        # Read D455Analyser results: contains D455Frame and objects
        from queue import Empty

        while not self._stop_event.is_set():
            try:
                result = self._camera_results_queue.get(timeout=1)
                # Expect D455Results-like object
                try:
                    frame = getattr(result, "frame", None)
                    if frame is not None and hasattr(frame, "rgb_image"):
                        img = frame.rgb_image
                        if isinstance(img, np.ndarray) and img.size > 0:
                            self._latest_frame_bgr = img
                            self._latest_frame_ts = getattr(frame, "timestamp", 0.0)
                            self._camera_connected = True
                            with self._stats_lock:
                                self._stats["last_camera_update"] = time.time()
                            # Update camera recent updates window
                            self._camera_update_times.append(time.time())
                            # Update camera drops if available
                            try:
                                dt = int(getattr(frame, "drops_total", 0))
                                # Represent as windowed metric by duplicating last value with time
                                # We'll estimate drops over window from the delta (exposed separately if needed)
                                # For now, just record updates; the UI uses updates_60s for rate
                            except Exception:
                                pass
                            self._prune_windows(time.time())
                            # Update camera recent updates window
                            self._camera_update_times.append(time.time())
                            self._prune_windows(time.time())
                    self._latest_camera_objects = getattr(result, "objects", None)
                except Exception as e:
                    self.logger.error(f"Camera reader parse error: {e}")
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Camera reader error: {e}")
                time.sleep(0.05)

    def _radar_reader_loop(self) -> None:
        # Read radar analyser outputs; handle both SHM meta frames and in-band arrays
        from queue import Empty
        from multiprocessing import shared_memory

        while not self._stop_event.is_set():
            try:
                item = self._radar_results_queue.get(timeout=1)
                if isinstance(item, dict):
                    # SHM-based frame meta
                    if item.get("RADAR_RES_SHM_INIT"):
                        try:
                            rd_meta = item.get("rd")
                            ra_meta = item.get("ra")
                            # Attach RD blocks
                            self._rd_blocks = []
                            self._ra_blocks = []
                            if rd_meta and rd_meta.get("names"):
                                self._rd_blocks = [
                                    shared_memory.SharedMemory(name=n)
                                    for n in rd_meta["names"]
                                ]
                                self._rd_shape = tuple(rd_meta.get("shape", ()))
                                self._rd_dtype = str(rd_meta.get("dtype", "float32"))
                            if ra_meta and ra_meta.get("names"):
                                self._ra_blocks = [
                                    shared_memory.SharedMemory(name=n)
                                    for n in ra_meta["names"]
                                ]
                                self._ra_shape = tuple(ra_meta.get("shape", ()))
                                self._ra_dtype = str(ra_meta.get("dtype", "float32"))
                            self._last_res_slot = None
                            self._last_res_seq = 0
                            self.logger.info(
                                f"Attached RD/RA SHM: rd_blocks={len(self._rd_blocks)}, ra_blocks={len(self._ra_blocks)}"
                            )
                        except Exception as e:
                            self.logger.error(f"Attach RD/RA SHM failed: {e}")
                    elif item.get("RADAR_RES_SHM_FRAME"):
                        # Keep point cloud and timestamp; RD/RA will be read from SHM later
                        self._latest_point_cloud = item.get("point_cloud")
                        self._latest_radar_ts = float(item.get("frame_timestamp", 0.0))
                        try:
                            self._last_res_slot = int(item.get("slot", 0)) & 1
                            self._last_res_seq = int(item.get("seq", 0))
                        except Exception:
                            pass
                        self._radar_connected = True
                        with self._stats_lock:
                            self._stats["last_radar_update"] = time.time()
                        # Update rolling windows
                        nowt = time.time()
                        self._radar_update_times.append(nowt)
                        td = item.get("total_dropped_frames")
                        if isinstance(td, (int, float)):
                            self._radar_drop_hist.append((nowt, int(td)))
                        self._prune_windows(nowt)
                    else:
                        # In-band arrays present (no SHM path). Store if provided.
                        if isinstance(item.get("range_doppler"), np.ndarray):
                            self._latest_rd = item.get("range_doppler")
                        if isinstance(item.get("range_azimuth"), np.ndarray):
                            self._latest_ra = item.get("range_azimuth")
                        if isinstance(item.get("point_cloud"), dict):
                            self._latest_point_cloud = item.get("point_cloud")
                        if "frame_timestamp" in item:
                            self._latest_radar_ts = float(
                                item.get("frame_timestamp", 0.0)
                            )
                        self._radar_connected = True
                        with self._stats_lock:
                            self._stats["last_radar_update"] = time.time()
                        nowt = time.time()
                        self._radar_update_times.append(nowt)
                        try:
                            td = item.get("total_dropped_frames")
                            if isinstance(td, (int, float)):
                                self._radar_drop_hist.append((nowt, int(td)))
                        except Exception:
                            pass
                        self._prune_windows(nowt)
                # else: ignore non-dict (should not happen)
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Radar reader error: {e}")
                time.sleep(0.05)

    def _engine_monitor_loop(self) -> None:
        import time as _t

        while not self._stop_event.is_set():
            try:
                if self._engine_process is None:
                    _t.sleep(0.2)
                    continue
                if not self._engine_process.is_alive():
                    # Engine exited
                    try:
                        exit_code = self._engine_process.exitcode
                    except Exception:
                        exit_code = None
                    self._failed = True
                    self._failure_reason = f"engine exited (code={exit_code})"
                    self._running = False
                    break
            except Exception:
                pass
            _t.sleep(0.2)

    def send_control(self, command: str) -> None:
        try:
            self._control_queue.put_nowait(command)
        except Exception as e:
            self.logger.error(f"Failed to send control command '{command}': {e}")

    def get_status(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats_copy = dict(self._stats)
        # Time metrics (readable)
        now = time.time()
        start_time = stats_copy.get("start_time", now)
        uptime_s = max(0.0, now - (start_time or now))
        cam_last = stats_copy.get("last_camera_update")
        rad_last = stats_copy.get("last_radar_update")

        def _fmt_s(val: Optional[float]) -> Optional[str]:
            if val is None:
                return None
            return f"{val:.3f}s"

        camera_updates = self._count_in_window(self._camera_update_times, now, 60.0)
        radar_updates = self._count_in_window(self._radar_update_times, now, 60.0)
        stats_copy.update(
            {
                "camera_frame_ts": self._latest_frame_ts,
                "radar_frame_ts": self._latest_radar_ts,
                "engine_alive": (
                    bool(self._engine_process.is_alive())
                    if self._engine_process
                    else False
                ),
                "starting": self._starting,
                "running": self._running,
                "failed": self._failed,
                "failure_reason": self._failure_reason,
                "radar_connected": self._radar_connected,
                "camera_connected": self._camera_connected,
                "engine_pid": (
                    self._engine_process.pid if self._engine_process else None
                ),
                # Enhanced, readable times relative to start
                "uptime": _fmt_s(uptime_s),
                "start_time": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(start_time)
                ),
                # FPS over last 60s
                "camera_fps_60s": round(camera_updates / 60.0, 2),
                "radar_fps_60s": round(radar_updates / 60.0, 2),
            }
        )
        return stats_copy

    def _prune_windows(self, nowt: float) -> None:
        window = 60.0
        try:
            while (
                self._radar_update_times and nowt - self._radar_update_times[0] > window
            ):
                self._radar_update_times.popleft()
            while (
                self._camera_update_times
                and nowt - self._camera_update_times[0] > window
            ):
                self._camera_update_times.popleft()
            while self._radar_drop_hist and nowt - self._radar_drop_hist[0][0] > window:
                self._radar_drop_hist.popleft()
        except Exception:
            pass

    def _count_in_window(self, dq: deque, nowt: float, window: float) -> int:
        try:
            return sum(1 for t in dq if nowt - t <= window)
        except Exception:
            return 0

    def _drops_in_window(self, nowt: float, window: float) -> Optional[int]:
        try:
            if not self._radar_drop_hist:
                return 0
            first_val = None
            last_val = None
            for t, v in self._radar_drop_hist:
                if nowt - t <= window:
                    if first_val is None:
                        first_val = v
                    last_val = v
            if first_val is None or last_val is None:
                return 0
            return max(0, int(last_val - first_val))
        except Exception:
            return None

    def _get_process_statuses(self) -> List[Dict[str, Any]]:
        roles = [
            {"label": "WebServer", "pid": os.getpid(), "status": "running"},
            {
                "label": "FusionEngine",
                "pid": self._engine_process.pid if self._engine_process else None,
                "status": None,
            },
            {"label": "RadarFeed", "pid": None, "status": None},
            {"label": "RadarAnalyser", "pid": None, "status": None},
            {"label": "CameraFeed", "pid": None, "status": None},
            {"label": "CameraAnalyser", "pid": None, "status": None},
        ]
        # FusionEngine status
        if self._engine_process is None:
            roles[1]["status"] = "not_exists"
        else:
            roles[1]["status"] = (
                "running" if self._engine_process.is_alive() else "dead"
            )
        # Child processes mapping
        try:
            import psutil  # type: ignore

            if self._engine_process is not None and self._engine_process.is_alive():
                p = psutil.Process(self._engine_process.pid)
                children = p.children(recursive=False)
                # Sort by create_time to match creation order
                try:
                    children.sort(key=lambda c: c.create_time())
                except Exception:
                    children.sort(key=lambda c: c.pid)
                # Expected order: RadarFeed, RadarAnalyser, CameraFeed, CameraAnalyser
                for idx, child in enumerate(children[:4]):
                    status = child.status()
                    mapped = (
                        "running"
                        if status == psutil.STATUS_RUNNING
                        else (
                            "sleeping"
                            if status
                            in (
                                psutil.STATUS_SLEEPING,
                                getattr(psutil, "STATUS_DISK_SLEEP", "disk-sleep"),
                            )
                            else "dead"
                        )
                    )
                    roles[2 + idx]["pid"] = child.pid
                    roles[2 + idx]["status"] = mapped
            # If radar_only, mark camera roles explicitly as not_exists when no pid
            if self._radar_only:
                for i in (4, 5):
                    if roles[i]["pid"] is None:
                        roles[i]["status"] = "not_exists"
        except Exception:
            # Fallback: provide a clear unknown state when psutil missing
            for i in range(2, 6):
                if roles[i]["pid"] is None and roles[i]["status"] is None:
                    roles[i]["status"] = "unknown"
        # Map None to not_exists
        for r in roles:
            if r["status"] is None:
                r["status"] = "not_exists"
        return roles

    def _collect_process_tree(self) -> List[Dict[str, Any]]:
        # Deprecated in /status per UI request; retained for potential debugging
        return []

    def _encode_jpeg(self, bgr: np.ndarray, quality: int = 80) -> Optional[bytes]:
        try:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            ok, buf = cv2.imencode(".jpg", bgr, encode_params)
            if not ok:
                return None
            return buf.tobytes()
        except Exception:
            return None

    def _draw_detections(self, bgr: np.ndarray, objects: Optional[Any]) -> np.ndarray:
        try:
            if bgr is None or objects is None:
                return bgr
            img = bgr.copy()
            h, w = img.shape[:2]
            base_thickness = max(1, int(round(min(h, w) / 240)))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.3, min(1.0, base_thickness * 0.6))
            for obj in list(objects) if isinstance(objects, (list, tuple)) else []:
                # Robust attribute access
                x = int(getattr(obj, "x", getattr(obj, "left", 0)))
                y = int(getattr(obj, "y", getattr(obj, "top", 0)))
                w_box = int(getattr(obj, "width", getattr(obj, "w", 0)))
                h_box = int(getattr(obj, "height", getattr(obj, "h", 0)))
                class_id = getattr(obj, "class_id", None)
                confidence = getattr(obj, "confidence", None)
                label = getattr(obj, "object_type", None)
                if label is None:
                    label = f"{class_id}" if class_id is not None else "obj"
                if confidence is not None:
                    try:
                        label = f"{label}:{float(confidence):.2f}"
                    except Exception:
                        pass
                # Clamp coordinates
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                x2 = max(0, min(x + max(0, w_box), w - 1))
                y2 = max(0, min(y + max(0, h_box), h - 1))
                # Colors: class-dependent simple palette
                color = (0, 255, 0)
                try:
                    if class_id is not None:
                        palette = [
                            (0, 255, 0),
                            (0, 200, 255),
                            (255, 0, 0),
                            (255, 0, 255),
                            (0, 128, 255),
                            (128, 255, 0),
                            (255, 128, 0),
                        ]
                        color = palette[int(class_id) % len(palette)]
                except Exception:
                    pass
                cv2.rectangle(img, (x, y), (x2, y2), color, thickness=base_thickness)
                # Text background
                ((tw, th), _) = cv2.getTextSize(label, font, font_scale, base_thickness)
                tx2 = min(x + tw + 6, w - 1)
                ty2 = min(y + th + 6, h - 1)
                cv2.rectangle(img, (x, y), (tx2, ty2), color, thickness=-1)
                cv2.putText(
                    img,
                    label,
                    (x + 3, y + th + 2),
                    font,
                    font_scale,
                    (0, 0, 0),
                    base_thickness,
                    cv2.LINE_AA,
                )
            return img
        except Exception:
            return bgr

    def get_latest_frame_jpeg(self, quality: int = 80) -> Optional[bytes]:
        frame = self._latest_frame_bgr
        if frame is None:
            return None
        # Draw detections overlay if available
        frame_drawn = self._draw_detections(frame, self._latest_camera_objects)
        return self._encode_jpeg(frame_drawn, quality=quality)

    def mjpeg_generator(
        self, fps_limit: float = 10.0, quality: int = 70
    ) -> Generator[bytes, None, None]:
        boundary = "frame"
        min_interval = 1.0 / max(fps_limit, 0.1)
        last_sent = 0.0
        while True:
            now = time.time()
            if now - last_sent < min_interval:
                time.sleep(0.005)
                continue
            last_sent = now
            jpg = self.get_latest_frame_jpeg(quality=quality)
            if jpg is None:
                # Send a small blank JPEG to keep stream alive
                blank = np.zeros((2, 2, 3), dtype=np.uint8)
                jpg = self._encode_jpeg(blank, quality=50) or b""
            yield (
                b"--"
                + boundary.encode()
                + b"\r\n"
                + b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                + jpg
                + b"\r\n"
            )

    def _heatmap_to_png(
        self, array2d: np.ndarray, colormap: int = cv2.COLORMAP_JET
    ) -> Optional[bytes]:
        try:
            a = np.asarray(array2d)
            if a.ndim != 2 or a.size == 0:
                return None
            # Normalize to 0..255
            a = a.astype(np.float32, copy=False)
            finite = np.isfinite(a)
            if not np.any(finite):
                return None
            # Compute robust min/max over finite values only
            vmin = float(np.percentile(a[finite], 1.0))
            vmax = float(np.percentile(a[finite], 99.0))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                return None
            if vmax <= vmin:
                vmax = vmin + 1.0
            # Replace non-finite with bounds and clip to avoid overflow warnings
            af = np.nan_to_num(a, nan=vmin, posinf=vmax, neginf=vmin)
            af = np.clip(af, vmin, vmax, out=None)
            norm = (af - vmin) / (vmax - vmin)
            norm = np.clip(norm, 0.0, 1.0, out=None)
            img8 = (norm * 255.0).astype(np.uint8, copy=False)
            if colormap is not None:
                img_color = cv2.applyColorMap(img8, colormap)
            else:
                img_color = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
            # Fit into a 640x480 canvas (letterbox), do not exceed 640 on either axis
            try:
                target_w, target_h = 640, 480
                h, w = img_color.shape[:2]
                scale = min(target_w / max(1, w), target_h / max(1, h))
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                if new_w != w or new_h != h:
                    img_resized = cv2.resize(
                        img_color, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
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

    def get_latest_rd_png(self) -> Optional[bytes]:
        # Prefer SHM if available
        try:
            if self._rd_blocks and self._rd_shape and self._last_res_slot is not None:
                slot = self._last_res_slot & 1
                np_dtype = np.dtype(self._rd_dtype or "float32")
                view = np.ndarray(
                    self._rd_shape, dtype=np_dtype, buffer=self._rd_blocks[slot].buf
                )
                arr = np.array(view, copy=True)
                try:
                    arr = np.rot90(arr, 1)
                except Exception:
                    pass
                return self._heatmap_to_png(arr)
        except Exception:
            pass
        # Fallback to in-band
        if isinstance(self._latest_rd, np.ndarray):
            try:
                arr = np.rot90(self._latest_rd, 1)
            except Exception:
                arr = self._latest_rd
            return self._heatmap_to_png(arr)
        return None

    def get_latest_ra_png(self) -> Optional[bytes]:
        try:
            if self._ra_blocks and self._ra_shape and self._last_res_slot is not None:
                slot = self._last_res_slot & 1
                np_dtype = np.dtype(self._ra_dtype or "float32")
                view = np.ndarray(
                    self._ra_shape, dtype=np_dtype, buffer=self._ra_blocks[slot].buf
                )
                arr = np.array(view, copy=True)
                try:
                    arr = np.rot90(arr, 1)
                except Exception:
                    pass
                return self._heatmap_to_png(arr)
        except Exception:
            pass
        if isinstance(self._latest_ra, np.ndarray):
            try:
                arr = np.rot90(self._latest_ra, 1)
            except Exception:
                arr = self._latest_ra
            return self._heatmap_to_png(arr)
        return None

    def _render_point_cloud_png(
        self, width: int = 640, height: int = 480
    ) -> Optional[bytes]:
        try:
            pc = self._latest_point_cloud
            if not isinstance(pc, dict):
                return None
            x = pc.get("x")
            y = pc.get("y")
            if x is None or y is None:
                return None
            x = np.asarray(x).astype(np.float32)
            y = np.asarray(y).astype(np.float32)
            if x.size == 0 or y.size == 0:
                # blank image
                img = np.zeros((height, width, 3), dtype=np.uint8)
                ok, buf = cv2.imencode(".png", img)
                return buf.tobytes() if ok else None
            # Robust bounds
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
            # Avoid zero ranges
            if x_max <= x_min:
                x_max = x_min + 1e-3
            if y_max <= y_min:
                y_max = y_min + 1e-3
            # Map to image coords: x -> horizontal (left negative), y -> vertical (forward up)
            # Normalize to [0,1]
            nx = (x - x_min) / (x_max - x_min)
            ny = (y - y_min) / (y_max - y_min)
            # Pixel coords
            px = np.clip((nx * (width - 1)).astype(np.int32), 0, width - 1)
            py = np.clip(((1.0 - ny) * (height - 1)).astype(np.int32), 0, height - 1)
            img = np.zeros((height, width, 3), dtype=np.uint8)
            # Draw points
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


app = Flask(__name__)
runner: Optional[FusionRunner] = None


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fusion Live Server</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 12px; }
    .row { display: flex; gap: 12px; margin-bottom: 12px; }
    .col { flex: 1; }
    .panel { border: 1px solid #ccc; padding: 8px; border-radius: 6px; }
    button { padding: 8px 12px; margin-right: 8px; }
    img { max-width: 100%; height: auto; }
    pre { background: #f6f8fa; padding: 8px; overflow: auto; }
    .proc-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; }
    .proc { display: inline-block; min-height: 48px; border: 1px solid #bbb; border-radius: 6px; padding: 8px; text-align: center; font-weight: 600; }
    .proc.white { background: #fff; color: #333; }
    .proc.red { background: #ffebeb; border-color: #e55; color: #900; }
    .proc.yellow { background: #fff7db; border-color: #e5c100; color: #8a6d00; }
    .proc.green { background: #e8ffe8; border-color: #6c6; color: #060; }
    .meta { font-size: 12px; font-weight: normal; margin-top: 4px; color: #555; }
    /* rotation via server-side; keep class unused */
    .rotate-ccw { }
    .gap { display: inline-block; width: 16px; }
    #radar_view { display: block; width: 640px; height: 480px; }
  </style>
  <script>
    let streamActive = false;
    let isRunning = false;
    async function systemStart() {
      const ro = document.getElementById('radar_only').checked ? '?radar_only=1' : '';
      await fetch('/system/start' + ro, { method: 'POST' });
    }
    async function systemStop() {
      await fetch('/system/stop', { method: 'POST' });
    }
    async function systemRetry() {
      await fetch('/system/retry', { method: 'POST' });
    }
    async function refreshStatus() {
      try {
        const r = await fetch('/status');
        const j = await r.json();
        document.getElementById('status').textContent = JSON.stringify(j, null, 2);
        fetchProcesses();
        renderTimes(j);
        try {
          if (typeof j.running === 'boolean') {
            isRunning = j.running;
            if (!isRunning) { if (videoTimer) clearInterval(videoTimer); if (radarTimer) clearInterval(radarTimer); }
          }
        } catch(e) {}
      } catch(e) { console.error(e); }
    }
    function setupStatusSSE() {
      if (!!window.EventSource) {
        try {
          const es = new EventSource('/status/stream');
          es.onmessage = (ev) => {
            let j = null;
            try { j = JSON.parse(ev.data); document.getElementById('status').textContent = JSON.stringify(j, null, 2); renderTimes(j); } catch(e) {}
            // Gate polling based on running state
            try {
              if (j && typeof j.running === 'boolean') {
                if (j.running && !isRunning) {
                  isRunning = true;
                  scheduleVideoPoll();
                  scheduleRadarPoll();
                } else if (!j.running && isRunning) {
                  isRunning = false;
                  if (videoTimer) clearInterval(videoTimer);
                  if (radarTimer) clearInterval(radarTimer);
                }
              }
            } catch(e) {}
            // Refresh processes grid every tick
            fetchProcesses();
          };
          es.onerror = (e) => { console.warn('SSE error; falling back to polling'); es.close(); setInterval(refreshStatus, 1000); };
        } catch(e) { setInterval(refreshStatus, 1000); }
      } else {
        setInterval(refreshStatus, 1000);
      }
    }
    async function startRecord() {
      await fetch('/control/start_record', { method: 'POST' });
    }
    async function stopRecord() {
      await fetch('/control/stop_record', { method: 'POST' });
    }
    // Two-viewer polling logic
    let radarMode = 'rd';
    let videoTimer = null;
    let radarTimer = null;
    let procTimer = null;
    function scheduleVideoPoll() {
      if (videoTimer) clearInterval(videoTimer);
      if (!isRunning) return;
      const mode = document.querySelector('input[name="video_mode"]:checked').value;
      if (mode !== 'live') return;
      const fpsSel = document.getElementById('video_fps');
      let fps = parseInt(fpsSel.value || '1');
      fps = Math.max(1, fps);
      const period = Math.max(50, Math.floor(1000 / fps));
      videoTimer = setInterval(fetchVideoFrame, period);
    }
    function scheduleRadarPoll() {
      if (radarTimer) clearInterval(radarTimer);
      if (!isRunning) return;
      const mode = document.querySelector('input[name="radar_mode"]:checked').value;
      if (mode !== 'live') return;
      const fpsSel = document.getElementById('radar_fps');
      let fps = parseInt(fpsSel.value || '1');
      fps = Math.max(1, fps);
      const period = Math.max(50, Math.floor(1000 / fps));
      radarTimer = setInterval(fetchRadarFrame, period);
    }
    async function fetchVideoFrame() {
      if (!isRunning) return;
      const el = document.getElementById('video_view');
      el.src = '/video/frame.jpg?_=' + Date.now();
    }
    async function fetchRadarFrame() {
      if (!isRunning) return;
      const el = document.getElementById('radar_view');
      el.src = '/radar/frame.png?mode=' + encodeURIComponent(radarMode) + '&_=' + Date.now();
    }
    function bindViewerControls() {
      // Defaults to 1 fps
      const vSel = document.getElementById('video_fps');
      if (vSel) vSel.value = '1';
      const rSel = document.getElementById('radar_fps');
      if (rSel) rSel.value = '1';
      document.querySelectorAll('input[name="video_mode"]').forEach(r => r.addEventListener('change', () => { scheduleVideoPoll(); }));
      document.querySelectorAll('input[name="radar_mode"]').forEach(r => r.addEventListener('change', () => { scheduleRadarPoll(); }));
      document.getElementById('video_fps').addEventListener('change', () => { scheduleVideoPoll(); });
      document.getElementById('radar_fps').addEventListener('change', () => { scheduleRadarPoll(); });
      document.getElementById('video_next').addEventListener('click', () => { clearInterval(videoTimer); fetchVideoFrame(); });
      document.getElementById('radar_next').addEventListener('click', () => { clearInterval(radarTimer); fetchRadarFrame(); });
      document.querySelectorAll('[data-rmode]').forEach(b => b.addEventListener('click', (e) => {
        const m = e.target.getAttribute('data-rmode');
        if (m) { radarMode = m; fetchRadarFrame(); scheduleRadarPoll(); }
      }));
      // Do not schedule until running becomes true (via SSE/status)
    }
    async function fetchProcesses() {
      try {
        const r = await fetch('/status/processes');
        if (!r.ok) { renderProcesses([]); return; }
        const j = await r.json();
        renderProcesses(j.process_statuses || []);
      } catch(e) { renderProcesses([]); }
    }
    function renderProcesses(list) {
      const container = document.getElementById('proc_grid');
      if (!container) return;
      container.innerHTML = '';
      const colorMap = {
        'not_exists': 'white',
        'unknown': 'white',
        'dead': 'red',
        'sleeping': 'yellow',
        'running': 'green',
      };
      const expected = [
        'WebServer', 'FusionEngine', 'RadarFeed', 'RadarAnalyser', 'CameraFeed', 'CameraAnalyser'
      ];
      const byLabel = {}; (list || []).forEach(p => { if (p && p.label) byLabel[p.label] = p; });
      expected.forEach(name => {
        const p = byLabel[name] || { label: name, status: 'not_exists', pid: null };
        const div = document.createElement('div');
        const color = colorMap[p.status] || 'white';
        div.className = 'proc ' + color;
        div.innerHTML = name + '<div class="meta">' + (p.status || 'not_exists') + (p.pid ? (' · pid ' + p.pid) : '') + '</div>';
        container.appendChild(div);
      });
    }
    // Init binds
    function init() {
      refreshStatus();
      setupStatusSSE();
      bindViewerControls();
      // Ensure processes grid updates even if SSE is delayed
      procTimer = setInterval(fetchProcesses, 1000);
    }
    window.addEventListener('load', init);
  </script>
  </head>
  <body>
    <h2>Fusion Live Server</h2>
    <div class="row">
      <div class="col panel">
        <h3>Controls</h3>
        <label><input type="checkbox" id="radar_only"> Radar only</label>
        <button onclick="systemStart()">Start System</button>
        <button onclick="systemStop()">Stop System</button>
        <button onclick="systemRetry()">Retry</button>
        <button onclick="startRecord()">Start Recording</button>
        <button onclick="stopRecord()">Stop Recording</button>
      </div>
    </div>
    <div class="row">
      <div class="col panel">
        <h3>Processes</h3>
        <div id="proc_grid" class="proc-grid"></div>
      </div>
    </div>
    <div class="row">
      <div class="col panel">
        <h3>Video</h3>
        <div>
          <label>Mode: </label>
          <label><input type="radio" name="video_mode" value="live" checked> Live</label>
          <label>FPS:</label>
          <select id="video_fps">
            <option selected>1</option><option>3</option><option>5</option><option>10</option><option>15</option><option>30</option>
          </select>
          <span class="gap"></span>
          <label><input type="radio" name="video_mode" value="manual"> Manual</label>
          <button id="video_next">Next frame</button>
        </div>
        <img id="video_view" alt="Video" />
      </div>
      <div class="col panel">
        <h3>Radar</h3>
        <div>
          <label>Mode: </label>
          <button data-rmode="rd">RD</button>
          <button data-rmode="ra">RA</button>
          <button data-rmode="pc">PC</button>
          <label><input type="radio" name="radar_mode" value="live" checked> Live</label>
          <label>FPS:</label>
          <select id="radar_fps">
            <option selected>1</option><option>2</option><option>5</option><option>10</option>
          </select>
          <span class="gap"></span>
          <label><input type="radio" name="radar_mode" value="manual"> Manual</label>
          <button id="radar_next">Next frame</button>
        </div>
        <img id="radar_view" alt="Radar" />
      </div>
    </div>
    <div class="row">
      <div class="col panel">
        <h3>Status</h3>
        <div id="times" class="meta"></div>
        <pre id="status">{}</pre>
      </div>
    </div>
  </body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/status")
def status():
    global runner
    if runner is None:
        return jsonify({"running": False}), 503
    st = runner.get_status()
    return jsonify(st)


@app.route("/status/processes")
def status_processes():
    global runner
    if runner is None:
        return jsonify({"process_statuses": []}), 503
    return jsonify({"process_statuses": runner._get_process_statuses()})


@app.route("/system/start", methods=["POST"])
def system_start():
    global runner
    if runner is None:
        return ("runner not initialized", 503)
    radar_only = bool(request.args.get("radar_only", "0") in ("1", "true", "True"))
    ok = runner.start(radar_only=radar_only)
    if not ok and (runner._running or runner._starting):
        return ("already running or starting", 409)
    return (
        "OK" if ok else f"FAILED: {runner._failure_reason or 'unknown'}",
        200 if ok else 500,
    )


@app.route("/system/stop", methods=["POST"])
def system_stop():
    global runner
    if runner is None:
        return ("runner not initialized", 503)
    runner.stop()
    return ("OK", 200)


@app.route("/system/retry", methods=["POST"])
def system_retry():
    global runner
    if runner is None:
        return ("runner not initialized", 503)
    try:
        runner.stop()
    except Exception:
        pass
    ok = runner.start()
    return (
        "OK" if ok else f"FAILED: {runner._failure_reason or 'unknown'}",
        200 if ok else 500,
    )


@app.route("/control/start_record", methods=["POST"])
def start_record():
    global runner
    if runner is None or not runner._running:
        return ("runner not started", 503)
    # Build timestamped dir under DEST_STORAGE using current time (server process time)
    try:
        ts_dir = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        recording_dir = os.path.join(CFGS.DEST_STORAGE, ts_dir)
        os.makedirs(recording_dir, exist_ok=True)
        runner.send_control(f"start_recording:{recording_dir}")
    except Exception:
        # Fallback to simple start if path creation failed
        runner.send_control("start_recording")
    return ("OK", 200)


@app.route("/control/stop_record", methods=["POST"])
def stop_record():
    global runner
    if runner is None or not runner._running:
        return ("runner not started", 503)
    runner.send_control("stop_recording")
    return ("OK", 200)


@app.route("/video/frame.jpg")
def video_frame():
    global runner
    if runner is None:
        return ("runner not started", 503)
    quality = int(request.args.get("q", 80))
    jpg = runner.get_latest_frame_jpeg(quality=quality)
    if jpg is None:
        return ("no frame", 204)
    resp = make_response(jpg)
    resp.headers["Content-Type"] = "image/jpeg"
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/video/stream")
def video_stream():
    global runner
    if runner is None or not runner._running:
        return ("runner not started", 503)
    boundary = "frame"
    gen = runner.mjpeg_generator(fps_limit=10.0, quality=70)
    return Response(gen, mimetype=f"multipart/x-mixed-replace; boundary={boundary}")


@app.route("/radar/range_doppler.png")
def radar_range_doppler_png():
    global runner
    if runner is None or not runner._running:
        return ("runner not started", 503)
    png = runner.get_latest_rd_png()
    if png is None:
        return ("no rd", 204)
    resp = make_response(png)
    resp.headers["Content-Type"] = "image/png"
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/radar/range_azimuth.png")
def radar_range_azimuth_png():
    global runner
    if runner is None or not runner._running:
        return ("runner not started", 503)
    png = runner.get_latest_ra_png()
    if png is None:
        return ("no ra", 204)
    resp = make_response(png)
    resp.headers["Content-Type"] = "image/png"
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/results/detections")
def results_detections():
    # Deprecated per UI; keep for backward compatibility with 204
    return ("no content", 204)


@app.route("/radar/frame.png")
def radar_frame_png():
    global runner
    if runner is None or not runner._running:
        return ("runner not started", 503)
    mode = request.args.get("mode", "rd").lower()
    if mode == "rd":
        png = runner.get_latest_rd_png()
    elif mode == "ra":
        png = runner.get_latest_ra_png()
    elif mode == "pc":
        png = runner._render_point_cloud_png(width=640, height=480)
    else:
        return ("bad mode", 400)
    if png is None:
        return ("no data", 204)
    resp = make_response(png)
    resp.headers["Content-Type"] = "image/png"
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/status/stream")
def status_stream():
    global runner
    if runner is None:
        return ("runner not started", 503)

    def gen():
        while True:
            data = runner.get_status()
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(1.0)

    return Response(gen(), mimetype="text/event-stream")


def _radar_hw_init() -> bool:
    try:
        ret = fpga_udp.AWR2243_init(CFGS.AWR2243_CONFIG_FILE)
        if ret != 0:
            logger.error("Failed to initialize AWR2243 radar: %d", ret)
            return False
        fpga_udp.AWR2243_setFrameCfg(0)
        ret = fpga_udp.AWR2243_sensorStart()
        if ret != 0:
            logger.error("Failed to start AWR2243 sensor: %d", ret)
            return False
        dsp.precompile_kernels()
        return True
    except Exception as e:
        logger.error(f"Radar HW init error: {e}")
        return False


def _radar_hw_cleanup():
    try:
        fpga_udp.AWR2243_sensorStop()
    except Exception:
        pass
    try:
        fpga_udp.AWR2243_poweroff()
    except Exception:
        pass


def _signal_handler(signum, frame):
    logger.info(f"Signal {signum} received, shutting down...")
    try:
        if runner is not None:
            runner.stop()
    except Exception:
        pass
    _radar_hw_cleanup()
    # Allow Flask to exit
    os._exit(0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fusion Live Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument(
        "--radar-only",
        action="store_true",
        help="Run without camera (default: include camera)",
    )
    args = parser.parse_args()

    # Cleanup handlers
    atexit.register(_radar_hw_cleanup)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Disable SHM resource tracker noise in main
    try:
        disable_shm_resource_tracker(logger)
    except Exception:
        pass

    # Create runner; do not start until requested
    global runner
    runner = FusionRunner()
    logger.info("FusionRunner ready (idle). Use /system/start to begin.")

    # Run Flask app (threaded to serve MJPEG)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
