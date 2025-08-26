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
from typing import Optional, Dict, Any, Generator

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

        # Stats
        self._stats_lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "start_time": time.time(),
            "last_camera_update": None,
            "last_radar_update": None,
            "camera_drops": 0,
            "radar_drops": 0,
        }

    def start(self, radar_only: bool = False) -> None:
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
        self._engine_process.start()
        self.logger.info(
            f"FusionEngine process started: pid={self._engine_process.pid}"
        )

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
                            with self._stats_lock:
                                self._stats["last_camera_update"] = time.time()
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
                        with self._stats_lock:
                            self._stats["last_radar_update"] = time.time()
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
                        with self._stats_lock:
                            self._stats["last_radar_update"] = time.time()
                # else: ignore non-dict (should not happen)
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Radar reader error: {e}")
                time.sleep(0.05)

    def send_control(self, command: str) -> None:
        try:
            self._control_queue.put_nowait(command)
        except Exception as e:
            self.logger.error(f"Failed to send control command '{command}': {e}")

    def get_status(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats_copy = dict(self._stats)
        stats_copy.update(
            {
                "camera_frame_ts": self._latest_frame_ts,
                "radar_frame_ts": self._latest_radar_ts,
            }
        )
        return stats_copy

    def _encode_jpeg(self, bgr: np.ndarray, quality: int = 80) -> Optional[bytes]:
        try:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            ok, buf = cv2.imencode(".jpg", bgr, encode_params)
            if not ok:
                return None
            return buf.tobytes()
        except Exception:
            return None

    def get_latest_frame_jpeg(self, quality: int = 80) -> Optional[bytes]:
        frame = self._latest_frame_bgr
        if frame is None:
            return None
        return self._encode_jpeg(frame, quality=quality)

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
            a = a.astype(np.float32)
            finite = np.isfinite(a)
            if not np.any(finite):
                return None
            vmin = float(np.percentile(a[finite], 1.0))
            vmax = float(np.percentile(a[finite], 99.0))
            if vmax <= vmin:
                vmax = vmin + 1.0
            norm = np.clip((a - vmin) / (vmax - vmin), 0.0, 1.0)
            img8 = (norm * 255.0).astype(np.uint8)
            if colormap is not None:
                img_color = cv2.applyColorMap(img8, colormap)
            else:
                img_color = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
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
                return self._heatmap_to_png(arr)
        except Exception:
            pass
        # Fallback to in-band
        if isinstance(self._latest_rd, np.ndarray):
            return self._heatmap_to_png(self._latest_rd)
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
                return self._heatmap_to_png(arr)
        except Exception:
            pass
        if isinstance(self._latest_ra, np.ndarray):
            return self._heatmap_to_png(self._latest_ra)
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
  </style>
  <script>
    async function refreshStatus() {
      try {
        const r = await fetch('/status');
        const j = await r.json();
        document.getElementById('status').textContent = JSON.stringify(j, null, 2);
      } catch(e) { console.error(e); }
    }
    function setupStatusSSE() {
      if (!!window.EventSource) {
        try {
          const es = new EventSource('/status/stream');
          es.onmessage = (ev) => {
            try { const j = JSON.parse(ev.data); document.getElementById('status').textContent = JSON.stringify(j, null, 2); } catch(e) {}
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
    async function grabFrame() {
      const el = document.getElementById('frame');
      el.src = '/video/frame.jpg?_=' + Date.now();
    }
    async function grabRD() {
      const el = document.getElementById('rd');
      el.src = '/radar/range_doppler.png?_=' + Date.now();
    }
    async function grabRA() {
      const el = document.getElementById('ra');
      el.src = '/radar/range_azimuth.png?_=' + Date.now();
    }
    async function grabDetections() {
      try {
        const r = await fetch('/results/detections');
        const j = await r.json();
        document.getElementById('detections').textContent = JSON.stringify(j, null, 2);
      } catch(e) { console.error(e); }
    }
    function init() {
      refreshStatus();
      setupStatusSSE();
      // MJPEG auto loads in <img src="/video/stream">
    }
    window.addEventListener('load', init);
  </script>
  </head>
  <body>
    <h2>Fusion Live Server</h2>
    <div class="row">
      <div class="col panel">
        <h3>Controls</h3>
        <button onclick="startRecord()">Start Recording</button>
        <button onclick="stopRecord()">Stop Recording</button>
      </div>
      <div class="col panel">
        <h3>Status</h3>
        <pre id="status">{}</pre>
      </div>
    </div>
    <div class="row">
      <div class="col panel">
        <h3>Live Video (MJPEG)</h3>
        <img id="mjpeg" src="/video/stream" alt="MJPEG stream" />
      </div>
      <div class="col panel">
        <h3>Last Frame (on demand)</h3>
        <button onclick="grabFrame()">Grab Current Frame</button>
        <img id="frame" alt="Single frame" />
      </div>
    </div>
    <div class="row">
      <div class="col panel">
        <h3>Radar Range-Doppler</h3>
        <button onclick="grabRD()">Refresh RD</button>
        <img id="rd" alt="Range-Doppler" />
      </div>
      <div class="col panel">
        <h3>Radar Range-Azimuth</h3>
        <button onclick="grabRA()">Refresh RA</button>
        <img id="ra" alt="Range-Azimuth" />
      </div>
    </div>
    <div class="row">
      <div class="col panel">
        <h3>Detections (point cloud)</h3>
        <button onclick="grabDetections()">Fetch Detections</button>
        <pre id="detections">{}</pre>
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
    st.update({"running": True})
    return jsonify(st)


@app.route("/control/start_record", methods=["POST"])
def start_record():
    global runner
    if runner is None:
        return ("runner not started", 503)
    runner.send_control("start_recording")
    return ("OK", 200)


@app.route("/control/stop_record", methods=["POST"])
def stop_record():
    global runner
    if runner is None:
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
    if runner is None:
        return ("runner not started", 503)
    boundary = "frame"
    gen = runner.mjpeg_generator(fps_limit=10.0, quality=70)
    return Response(gen, mimetype=f"multipart/x-mixed-replace; boundary={boundary}")


@app.route("/radar/range_doppler.png")
def radar_range_doppler_png():
    global runner
    if runner is None:
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
    if runner is None:
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
    global runner
    if runner is None:
        return jsonify({"ok": False, "error": "runner not started"}), 503
    # Pass-through latest point cloud
    pc = runner._latest_point_cloud or {"x": [], "y": [], "z": [], "intensity": []}
    return jsonify(
        {"ok": True, "point_cloud": pc, "timestamp": runner._latest_radar_ts}
    )


@app.route("/status/stream")
def status_stream():
    global runner
    if runner is None:
        return ("runner not started", 503)

    def gen():
        while True:
            data = runner.get_status()
            try:
                yield f"data: {jsonify(data).get_data(as_text=True)}\n\n"
            except Exception:
                # Fallback JSON serialization if Flask jsonify not usable here
                import json as _json

                yield f"data: {_json.dumps(data)}\n\n"
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
    parser.add_argument("--radar-only", action="store_true", help="Run without camera")
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

    # Initialize radar HW like the desktop GUI app
    ok = _radar_hw_init()
    if not ok:
        logger.error("Radar HW init failed; continuing to start server anyway.")

    # Start fusion runner
    global runner
    runner = FusionRunner()
    runner.start(radar_only=args.radar_only)
    logger.info("FusionRunner started.")

    # Run Flask app (threaded to serve MJPEG)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
