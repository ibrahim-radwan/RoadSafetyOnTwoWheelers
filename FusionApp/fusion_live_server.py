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
import time
import atexit
import signal
import threading
import json
from typing import Optional, Dict, Any, Generator, List

from flask import Flask, Response, jsonify, request, make_response
from flask import render_template

# Local imports
from utils import setup_logger, disable_shm_resource_tracker
from config_params import CFGS

from services.radar_hw import radar_hw_cleanup
from services.fusion_runner import FusionRunner as FusionRunnerService
from services.process_inspector import get_process_statuses


logger = setup_logger("fusion_live_server")


class FusionRunner:
    """
    Starts FusionEngine in background and exposes latest data to the web server.
    """

    def __init__(self):
        # Delegate to service implementation
        self._impl = FusionRunnerService()

        # Provide attributes accessed by routes via properties
        pass

    def start(
        self, radar_only: bool = False, radar_config_file: Optional[str] = None
    ) -> bool:
        return self._impl.start(
            radar_only=radar_only, radar_config_file=radar_config_file
        )

    def stop(self) -> None:
        self._impl.stop()

    def send_control(self, command: str) -> None:
        self._impl.send_control(command)

    def get_status(self) -> Dict[str, Any]:
        return self._impl.get_status()

    # Removed dead stats helpers; status is provided by the service implementation

    def _get_process_statuses(self) -> List[Dict[str, Any]]:
        engine = getattr(self._impl, "_engine_process", None)
        engine_pid = engine.pid if engine else None
        radar_only = bool(getattr(self._impl, "_radar_only", False))
        return get_process_statuses(engine_pid, radar_only)

    # Removed unused collect_process_tree

    def _draw_detections(self, bgr, objects):
        return self._impl._draw_detections(bgr, objects)

    def get_latest_frame_jpeg(self, quality: int = 80) -> Optional[bytes]:
        return self._impl.get_latest_frame_jpeg(quality)

    def mjpeg_generator(
        self, fps_limit: float = 10.0, quality: int = 70
    ) -> Generator[bytes, None, None]:
        return self._impl.mjpeg_generator(fps_limit=fps_limit, quality=quality)

    def get_latest_rd_png(self) -> Optional[bytes]:
        return self._impl.get_latest_rd_png()

    def get_latest_ra_png(self) -> Optional[bytes]:
        return self._impl.get_latest_ra_png()

    def get_latest_point_cloud_json(self) -> Optional[Dict[str, Any]]:
        return self._impl.get_latest_point_cloud_json()

    # Tuning passthrough
    def get_tuning(self) -> Dict[str, Any]:
        return self._impl.get_tuning()

    def set_tuning(self, tuning: Dict[str, Any]) -> Dict[str, Any]:
        return self._impl.set_tuning(tuning)

    # Expose minimal state needed by routes (back-compat with existing checks)
    @property
    def _running(self) -> bool:
        return bool(getattr(self._impl, "_running", False))

    @property
    def _starting(self) -> bool:
        return bool(getattr(self._impl, "_starting", False))

    @property
    def _failure_reason(self) -> Optional[str]:
        return getattr(self._impl, "_failure_reason", None)


app = Flask(__name__, template_folder="templates")
runner: Optional[FusionRunner] = None


@app.route("/")
def index():
    # Build config options for radar configs
    try:
        from config_params import CFGS

        base_dir = getattr(CFGS, "AWR2243_CONFIG_DIR", "")
        names = getattr(CFGS, "AWR2243_CONFIG_FILE_NAMES", {})
        options = []
        for mode in ("2D", "3D"):
            for fname in names.get(mode, []) or []:
                # Extract distance like '10m' from filename parts
                parts = fname.split("_")
                dist = None
                for p in parts:
                    if p.endswith("m"):
                        dist = p
                        break
                label = f"{mode} - {dist}" if dist else f"{mode} - {fname}"
                options.append(
                    {
                        "mode": mode,
                        "name": fname,
                        "path": base_dir + fname,
                        "label": label,
                    }
                )
    except Exception:
        options = []
    # Pass options as JSON for client-side population
    return render_template("index.html", radar_config_options=options)


@app.route("/status")
def status():
    global runner
    if runner is None:
        return jsonify({"running": False}), 503
    st = runner.get_status()
    return jsonify(st)


@app.route("/tuning", methods=["GET", "POST"])
def tuning():
    global runner
    if runner is None:
        return ("runner not initialized", 503)
    if request.method == "GET":
        return jsonify(runner.get_tuning())
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        body = {}
    return jsonify(runner.set_tuning(body))


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
    # Optional radar config file selection (only when not running)
    radar_cfg = request.args.get("radar_cfg")
    ok = runner.start(radar_only=radar_only, radar_config_file=radar_cfg)
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


@app.route("/radar/pc.json")
def radar_point_cloud_json():
    global runner
    if runner is None or not runner._running:
        return ("runner not started", 503)
    data = runner.get_latest_point_cloud_json()
    if data is None:
        return jsonify({})
    # Ensure z semantics: missing -> None; present but empty -> []
    return jsonify(data)


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
    elif mode == "pc" or mode == "pc2d" or mode == "pc3d":
        # Deprecated server-side PC rendering removed; serve JSON/Plotly on the client.
        return ("use /radar/pc.json", 410)
    else:
        return ("bad mode", 400)
    if png is None:
        return ("no data", 204)
    resp = make_response(png)
    resp.headers["Content-Type"] = "image/png"
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    try:
        logger.info(f"PC reply bytes={len(png)}")
    except Exception:
        pass
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


def _signal_handler(signum, frame):
    logger.info(f"Signal {signum} received, shutting down...")
    try:
        if runner is not None:
            runner.stop()
    except Exception:
        pass
    radar_hw_cleanup()
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
    atexit.register(radar_hw_cleanup)
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
