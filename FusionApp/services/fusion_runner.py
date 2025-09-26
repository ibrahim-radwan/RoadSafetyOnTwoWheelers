import os
import glob
import time
import threading
from collections import deque
from typing import Optional, Dict, Any, Generator

import numpy as np
import cv2

from utils import setup_logger
from engine.fusion_factory import FusionFactory
from engine.fusion_engine import FusionEngine

from render.encoders import encode_jpeg, heatmap_to_png
from services.radar_hw import radar_hw_init, radar_hw_cleanup


class FusionRunner:
    """
    Starts FusionEngine in background and exposes latest data to the web server.
    """

    def __init__(self):
        self.logger = setup_logger("FusionRunner")
        self._stop_event = threading.Event()

        from multiprocessing import Queue, Event, Process

        self._radar_results_queue = Queue(maxsize=3)
        self._camera_results_queue = Queue(maxsize=2)
        self._control_queue = Queue()
        self._engine_stop_event = Event()

        self._latest_frame_bgr: Optional[np.ndarray] = None
        self._latest_frame_ts: float = 0.0
        self._latest_camera_objects: Optional[Any] = None

        self._latest_point_cloud: Optional[Dict[str, Any]] = None
        self._latest_radar_ts: float = 0.0

        self._latest_rd: Optional[np.ndarray] = None
        self._latest_ra: Optional[np.ndarray] = None
        self._rd_blocks = []  # SHM blocks for RD
        self._ra_blocks = []  # SHM blocks for RA
        self._rd_shape: Optional[tuple] = None
        self._ra_shape: Optional[tuple] = None
        self._rd_dtype: str = "float32"
        self._ra_dtype: str = "float32"
        self._last_res_slot: Optional[int] = None
        self._last_res_seq: int = 0

        self._engine_process: Optional[Process] = None
        self._camera_reader_thread: Optional[threading.Thread] = None
        self._radar_reader_thread: Optional[threading.Thread] = None
        self._engine_monitor_thread: Optional[threading.Thread] = None

        self._stats_lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "start_time": time.time(),
            "last_camera_update": None,
            "last_radar_update": None,
            "camera_drops": 0,
            "radar_drops": 0,
        }

        self._record_dir: Optional[str] = None
        self._rec_history: deque = deque()
        self._rec_cam_total: int = 0
        self._rec_rad_total: int = 0
        self._rec_bytes_total: int = 0
        self._rec_is_recording: bool = False
        self._rec_start_ts: Optional[float] = None

        self._radar_drop_hist: deque = deque()
        self._radar_update_times: deque = deque()
        self._camera_update_times: deque = deque()

        self._starting: bool = False
        self._running: bool = False
        self._failed: bool = False
        self._failure_reason: Optional[str] = None
        self._radar_connected: bool = False
        self._camera_connected: bool = False
        self._radar_only: bool = False
        # Replay dir for matching frame filenames; last-saved guard
        self._replay_dir: Optional[str] = None
        self._last_saved_pc_ts: Optional[float] = None
        # Mode flags
        self._full_analysis: bool = False
        # Artefact toggles (effective only in full mode)
        self._enable_tesseract: bool = True
        self._enable_zyx_cube: bool = True

        # Runtime tuning (hot) applied by analyser via control queue
        self._tuning: Dict[str, Any] = {
            "cfar_2d": {
                "az": {"l_bound": 1.5, "guard_len": 2, "noise_len": 10},
                "range": {"l_bound": 3.0, "guard_len": 2, "noise_len": 10},
            },
            "cfar_3d": {
                "doppler": {"l_bound": 1.5, "guard_len": 4, "noise_len": 16},
                "range": {"l_bound": 2.5, "guard_len": 4, "noise_len": 16},
            },
            "thresholds_3d": {"snr_min": 8.0},
        }

    def start(
        self,
        radar_only: bool = False,
        radar_config_file: Optional[str] = None,
        mode: str = "live",
        replay_path: Optional[str] = None,
        enable_tesseract: Optional[bool] = None,
        enable_zyx_cube: Optional[bool] = None,
    ) -> bool:
        if self._running or self._starting:
            return False
        self._failed = False
        self._failure_reason = None
        self._starting = True
        self._radar_connected = False
        self._camera_connected = False
        self._mode = (mode or "live").lower()
        self._full_analysis = self._mode == "full"
        # Allow overrides from API parameters if provided
        if enable_tesseract is not None:
            self._enable_tesseract = bool(enable_tesseract)
        if enable_zyx_cube is not None:
            self._enable_zyx_cube = bool(enable_zyx_cube)

        try:
            self._stop_event = threading.Event()
            from multiprocessing import Event as _MPEvent

            self._engine_stop_event = _MPEvent()
            from multiprocessing import Queue as _MPQueue

            self._radar_results_queue = _MPQueue(maxsize=3)
            self._camera_results_queue = _MPQueue(maxsize=2)
        except Exception:
            pass

        fusion_engine: FusionEngine
        if self._mode in ("replay", "full"):
            # Create a shared sync_state and compute start timestamp from directory
            from multiprocessing import Manager
            from engine.sync_state import (
                create_sync_state,
                SyncStateUtils,
                TimestampScanner,
            )

            mgr = Manager()
            sync_state = create_sync_state(mgr)
            recording_dir = replay_path or ""
            self._replay_dir = recording_dir or None
            try:
                t0, _, _ = TimestampScanner.find_common_start_timestamp(recording_dir)
            except Exception:
                t0 = 0.0
            SyncStateUtils.set_start_timestamp(sync_state, t0)

            if radar_only:
                fusion_engine = FusionFactory.create_replay_radar_only(
                    recording_dir=recording_dir,
                    radar_config_file=radar_config_file,
                    sync_state=sync_state,
                )
            else:
                fusion_engine = FusionFactory.create_replay_fusion(
                    recording_dir=recording_dir,
                    radar_config_file=radar_config_file,
                    sync_state=sync_state,
                )
            # Store for status exposure
            self._sync_state = sync_state
            # Propagate full-analysis mode to engine
            try:
                setattr(fusion_engine, "full_analysis", bool(self._full_analysis))
                # Inject analyser artefact toggle flags into analyser config
                try:
                    if isinstance(fusion_engine.radar_analyser_config, dict):
                        fusion_engine.radar_analyser_config["enable_tesseract"] = bool(
                            self._enable_tesseract
                        )
                        fusion_engine.radar_analyser_config["enable_zyx_cube"] = bool(
                            self._enable_zyx_cube
                        )
                except Exception:
                    pass
            except Exception as e:
                # In full-analysis mode, this is fatal because downstream components rely on this hint
                if self._full_analysis:
                    self._failed = True
                    self._failure_reason = f"failed to set full_analysis on engine: {e}"
                    self._starting = False
                    self._running = False
                    return False
                else:
                    self.logger.warning("Failed to set full_analysis on engine: %s", e)

        else:
            fusion_engine = (
                FusionFactory.create_live_radar_only(
                    radar_config_file_override=radar_config_file
                )
                if radar_only
                else FusionFactory.create_live_fusion(
                    radar_config_file_override=radar_config_file
                )
            )
            # Inject toggles for live modes too (no effect unless full-analysis true later)
            try:
                if isinstance(fusion_engine.radar_analyser_config, dict):
                    fusion_engine.radar_analyser_config["enable_tesseract"] = bool(
                        self._enable_tesseract
                    )
                    fusion_engine.radar_analyser_config["enable_zyx_cube"] = bool(
                        self._enable_zyx_cube
                    )
            except Exception:
                pass

        from multiprocessing import Process

        # Propagate FULL_ANALYSIS hint to child processes via environment for mode-specific queue behavior
        try:
            if self._full_analysis:
                os.environ["FULL_ANALYSIS"] = "1"
            else:
                os.environ.pop("FULL_ANALYSIS", None)
        except Exception as e:
            if self._full_analysis:
                self._failed = True
                self._failure_reason = f"failed to set FULL_ANALYSIS env: {e}"
                self._starting = False
                self._running = False
                return False
            else:
                self.logger.warning("Failed to set FULL_ANALYSIS env: %s", e)

        self._engine_process = Process(
            target=fusion_engine.run,
            args=(
                self._radar_results_queue,
                None if radar_only else self._camera_results_queue,
                self._engine_stop_event,
                self._control_queue,
                None,
            ),
        )
        try:
            self.logger.info(
                "Starting engine: mode=%s full_analysis=%s enable_tesseract=%s enable_zyx_cube=%s",
                self._mode,
                str(self._full_analysis),
                str(self._enable_tesseract),
                str(self._enable_zyx_cube),
            )
        except Exception:
            pass
        try:
            # Live: initialize hardware; Replay/Full: skip HW init
            if self._mode not in ("replay", "full"):
                try:
                    radar_cfg_path = fusion_engine.radar_analyser_config.get(
                        "config_file"
                    )
                except Exception:
                    radar_cfg_path = radar_config_file
                ok_hw = radar_hw_init(radar_cfg_path or "")
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

        with self._stats_lock:
            self._stats["start_time"] = time.time()

        if not radar_only:
            self._camera_reader_thread = threading.Thread(
                target=self._camera_reader_loop, name="CameraReader", daemon=True
            )
            self._camera_reader_thread.start()

        self._radar_reader_thread = threading.Thread(
            target=self._radar_reader_loop, name="RadarReader", daemon=True
        )
        self._radar_reader_thread.start()

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
        if self._engine_process is not None:
            self._engine_process.join(timeout=3)
            if self._engine_process.is_alive():
                try:
                    self._engine_process.kill()
                except Exception:
                    pass
                self._engine_process.join(timeout=1)
            self._engine_process = None
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
        # Live mode uses HW; replay/full do not require cleanup
        try:
            if getattr(self, "_mode", "live") not in ("replay", "full"):
                radar_hw_cleanup()
        except Exception:
            pass
        self._running = False
        self._starting = False
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
        from queue import Empty

        while not self._stop_event.is_set():
            try:
                result = self._camera_results_queue.get(timeout=1)
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
        from queue import Empty
        from multiprocessing import shared_memory

        while not self._stop_event.is_set():
            try:
                item = self._radar_results_queue.get(timeout=1)
                if isinstance(item, dict):
                    # Capture replay status from radar feed
                    if item.get("RADAR_PLAYBACK_STATUS"):
                        try:
                            st = item.get("RADAR_PLAYBACK_STATUS") or {}
                            self._radar_playback_status = st
                        except Exception:
                            pass
                    if item.get("RADAR_RES_SHM_INIT"):
                        try:
                            rd_meta = item.get("rd")
                            ra_meta = item.get("ra")
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
                        self._latest_point_cloud = item.get("point_cloud")
                        self._latest_radar_ts = float(item.get("frame_timestamp", 0.0))
                        src_path = item.get("src_filepath")
                        try:
                            self._last_res_slot = int(item.get("slot", 0)) & 1
                            self._last_res_seq = int(item.get("seq", 0))
                        except Exception:
                            pass
                        self._radar_connected = True
                        with self._stats_lock:
                            self._stats["last_radar_update"] = time.time()
                        nowt = time.time()
                        self._radar_update_times.append(nowt)
                        td = item.get("total_dropped_frames")
                        if isinstance(td, (int, float)):
                            self._radar_drop_hist.append((nowt, int(td)))
                        self._prune_windows(nowt)
                    else:
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
                        src_path = item.get("src_filepath")
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
            if isinstance(command, str):
                if command.startswith("start_recording"):
                    try:
                        rec_dir = None
                        if ":" in command:
                            _, rec_dir = command.split(":", 1)
                            rec_dir = rec_dir.strip()
                        self._record_dir = rec_dir
                        self._rec_is_recording = True
                        self._rec_start_ts = time.time()
                        self._rec_history.clear()
                        self._rec_cam_total = 0
                        self._rec_rad_total = 0
                    except Exception:
                        pass
                elif command == "stop_recording":
                    self._rec_is_recording = False
                    self._rec_start_ts = None
            self._control_queue.put_nowait(command)
        except Exception as e:
            self.logger.error(f"Failed to send control command '{command}': {e}")

    # Tuning API
    def get_tuning(self) -> Dict[str, Any]:
        return dict(self._tuning)

    def set_tuning(self, tuning: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if isinstance(tuning, dict):
                # Shallow merge for known sections
                for k in ("cfar_2d", "cfar_3d", "thresholds_3d"):
                    if k in tuning and isinstance(tuning[k], dict):
                        cur = self._tuning.get(k, {})
                        cur.update(tuning[k])
                        self._tuning[k] = cur
                # Broadcast to analyser via control queue
                import json as _json

                self.send_control("TUNING:" + _json.dumps(self._tuning))
        except Exception as e:
            self.logger.error(f"set_tuning failed: {e}")
        return dict(self._tuning)

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
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                x2 = max(0, min(x + max(0, w_box), w - 1))
                y2 = max(0, min(y + max(0, h_box), h - 1))
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
        frame_drawn = self._draw_detections(
            frame, getattr(self, "_latest_camera_objects", None)
        )
        return encode_jpeg(frame_drawn, quality=quality)

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
                blank = np.zeros((2, 2, 3), dtype=np.uint8)
                jpg = encode_jpeg(blank, quality=50) or b""
            yield (
                b"--"
                + boundary.encode()
                + b"\r\n"
                + b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                + jpg
                + b"\r\n"
            )

    def get_latest_rd_png(self) -> Optional[bytes]:
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
                pc = self._latest_point_cloud or {}
                mr = pc.get("max_range") if isinstance(pc, dict) else None
                ms = pc.get("max_speed") if isinstance(pc, dict) else None
                if (
                    isinstance(mr, (int, float))
                    and isinstance(ms, (int, float))
                    and np.isfinite(mr)
                    and np.isfinite(ms)
                    and mr > 0
                    and ms > 0
                ):
                    extents = (-float(ms), float(ms), 0.0, float(mr))
                else:
                    extents = None
                return heatmap_to_png(
                    arr, extents=extents, force_square=True, target_size=(480, 480)
                )
        except Exception:
            pass
        if isinstance(self._latest_rd, np.ndarray):
            try:
                arr = np.rot90(self._latest_rd, 1)
            except Exception:
                arr = self._latest_rd
            pc = self._latest_point_cloud or {}
            mr = pc.get("max_range") if isinstance(pc, dict) else None
            ms = pc.get("max_speed") if isinstance(pc, dict) else None
            if (
                isinstance(mr, (int, float))
                and isinstance(ms, (int, float))
                and np.isfinite(mr)
                and np.isfinite(ms)
                and mr > 0
                and ms > 0
            ):
                extents = (-float(ms), float(ms), 0.0, float(mr))
            else:
                extents = None
            return heatmap_to_png(
                arr, extents=extents, force_square=True, target_size=(480, 480)
            )
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
                pc = self._latest_point_cloud or {}
                mr = pc.get("max_range") if isinstance(pc, dict) else None
                if isinstance(mr, (int, float)) and np.isfinite(mr) and mr > 0:
                    extents = (-90.0, 90.0, 0.0, float(mr))
                else:
                    extents = (-90.0, 90.0, 0.0, 10.0)
                return heatmap_to_png(
                    arr, extents=extents, force_square=True, target_size=(640, 480)
                )
        except Exception:
            pass
        if isinstance(self._latest_ra, np.ndarray):
            try:
                arr = np.rot90(self._latest_ra, 1)
            except Exception:
                arr = self._latest_ra
            pc = self._latest_point_cloud or {}
            mr = pc.get("max_range") if isinstance(pc, dict) else None
            if isinstance(mr, (int, float)) and np.isfinite(mr) and mr > 0:
                extents = (-90.0, 90.0, 0.0, float(mr))
            else:
                extents = (-90.0, 90.0, 0.0, 10.0)
            return heatmap_to_png(
                arr, extents=extents, force_square=True, target_size=(640, 480)
            )
        return None

    def get_latest_point_cloud_json(self) -> Optional[Dict[str, Any]]:
        try:
            pc = self._latest_point_cloud or None
            if not isinstance(pc, dict):
                return None
            x = pc.get("x")
            y = pc.get("y")
            if x is None or y is None:
                return None
            z = pc.get("z")
            inten = pc.get("intensity", pc.get("snr"))
            mr = pc.get("max_range")
            ms = pc.get("max_speed")
            # Convert to lists for JSON
            result = {
                "x": list(map(float, x)) if hasattr(x, "__len__") else [],
                "y": list(map(float, y)) if hasattr(y, "__len__") else [],
                "z": (
                    list(map(float, z))
                    if (z is not None and hasattr(z, "__len__"))
                    else None
                ),
                "intensity": (
                    list(map(float, inten))
                    if (inten is not None and hasattr(inten, "__len__"))
                    else None
                ),
                "max_range": float(mr) if isinstance(mr, (int, float)) else None,
                "max_speed": float(ms) if isinstance(ms, (int, float)) else None,
            }
            return result
        except Exception:
            return None

    def get_status(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats_copy = dict(self._stats)
        now = time.time()
        start_time = stats_copy.get("start_time", now)
        uptime_s = max(0.0, now - (start_time or now))

        def _fmt_s(val: Optional[float]) -> Optional[str]:
            if val is None:
                return None
            return f"{val:.3f}s"

        camera_updates = self._count_in_window(self._camera_update_times, now, 60.0)
        radar_updates = self._count_in_window(self._radar_update_times, now, 60.0)
        cam_total, rad_total, cam_fps5, rad_fps5, bytes_total, bw_bps5 = (
            self._compute_recording_stats(now)
        )

        if self._rec_is_recording and self._rec_start_ts:
            rec_dur_s = max(0.0, now - self._rec_start_ts)
        else:
            rec_dur_s = 0.0

        def _fmt_hms(sec: float) -> str:
            try:
                sec_i = int(sec)
                h = sec_i // 3600
                m = (sec_i % 3600) // 60
                s = sec_i % 60
                return f"{h:02d}:{m:02d}:{s:02d}"
            except Exception:
                return "00:00:00"

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
                "uptime": _fmt_s(uptime_s),
                "start_time": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(start_time)
                ),
                "camera_fps_60s": round(camera_updates / 60.0, 2),
                "radar_fps_60s": round(radar_updates / 60.0, 2),
                "recording_dir": self._record_dir if self._rec_is_recording else None,
                "camera_frames_total": cam_total if self._rec_is_recording else 0,
                "radar_frames_total": rad_total if self._rec_is_recording else 0,
                "camera_write_fps_5s": (
                    round(cam_fps5, 2) if self._rec_is_recording else 0.0
                ),
                "radar_write_fps_5s": (
                    round(rad_fps5, 2) if self._rec_is_recording else 0.0
                ),
                "total_bytes_written": (
                    int(bytes_total) if self._rec_is_recording else 0
                ),
                "avg_bandwidth_5s_bps": (
                    float(bw_bps5) if self._rec_is_recording else 0.0
                ),
                "recording_duration_hms": (
                    _fmt_hms(rec_dur_s) if self._rec_is_recording else "00:00:00"
                ),
                "tuning": self._tuning,
                "replay": {
                    "current_frame": (
                        int(
                            (getattr(self, "_radar_playback_status", {}) or {}).get(
                                "current_frame", 0
                            )
                        )
                    ),
                    "total_frames": (
                        int(
                            (getattr(self, "_radar_playback_status", {}) or {}).get(
                                "total_frames", 0
                            )
                        )
                    ),
                },
                "artefacts": {
                    "full_analysis": bool(self._full_analysis),
                    "enable_tesseract": bool(self._enable_tesseract),
                    "enable_zyx_cube": bool(self._enable_zyx_cube),
                },
            }
        )
        return stats_copy

    def _compute_recording_stats(self, now_ts: float) -> tuple:
        try:
            if not self._rec_is_recording or not self._record_dir:
                return 0, 0, 0.0, 0.0, 0, 0.0
            cam_total = 0
            rad_total = 0
            bytes_total = 0
            try:
                with os.scandir(self._record_dir) as it:
                    for entry in it:
                        if not entry.is_file():
                            continue
                        name = entry.name.lower()
                        try:
                            fsz = entry.stat().st_size
                        except Exception:
                            fsz = 0
                        if name.endswith(".png"):
                            cam_total += 1
                            bytes_total += fsz
                        elif name.endswith(".bin"):
                            rad_total += 1
                            bytes_total += fsz
            except Exception:
                pass
            self._rec_history.append((now_ts, cam_total, rad_total, bytes_total))
            window = 5.0
            while self._rec_history and now_ts - self._rec_history[0][0] > window:
                self._rec_history.popleft()
            if len(self._rec_history) >= 2:
                t0, c0, r0, b0 = self._rec_history[0]
                dt = max(1e-6, now_ts - t0)
                cam_fps = (cam_total - c0) / dt
                rad_fps = (rad_total - r0) / dt
                bw_bps = (bytes_total - b0) / dt
            else:
                cam_fps = 0.0
                rad_fps = 0.0
                bw_bps = 0.0
            self._rec_cam_total = cam_total
            self._rec_rad_total = rad_total
            self._rec_bytes_total = bytes_total
            return cam_total, rad_total, cam_fps, rad_fps, bytes_total, bw_bps
        except Exception:
            return 0, 0, 0.0, 0.0, 0, 0.0

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
