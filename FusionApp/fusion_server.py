"""
Unified Fusion web server with live and replay modes.

- /fs/list for simple home-scoped file browsing
- /replay/control for playback controls in replay mode (play/pause/stop/seek)
"""

import os
import time
import atexit
import signal
import json
from typing import Optional, Dict, Any, Generator, List

from flask import Flask, Response, jsonify, request, make_response
from flask import render_template

# Local imports
from utils import setup_logger, disable_shm_resource_tracker
from config_params import CFGS

from services.radar_hw import radar_hw_cleanup  # Hardware driver cleanup hook.
from services.fusion_runner import (
    FusionRunner as FusionRunnerService,
)  # Core fusion pipeline controller.
from services.process_inspector import (
    get_process_statuses,
)  # Helper describing child process state.


logger = setup_logger("fusion_server")


class FusionRunner:
    """
    Starts FusionEngine in background and exposes latest data to the web server.
    """

    def __init__(self):
        """Instantiate the process-aware FusionRunner service used by the CLI code path."""
        self._impl = FusionRunnerService()

    def start(
        self,
        *,
        mode: str = "live",
        radar_only: bool = False,
        radar_config_file: Optional[str] = None,
        replay_path: Optional[str] = None,
    ) -> bool:
        """Launch the fusion pipeline subprocesses for live sensors or replayed logs."""
        return self._impl.start(
            radar_only=radar_only,
            radar_config_file=radar_config_file,
            mode=mode,
            replay_path=replay_path,
        )

    def stop(self) -> None:
        """Signal the underlying FusionRunnerService to tear down spawned processes."""
        self._impl.stop()

    def send_control(self, command: str) -> None:
        """Forward player-style commands (play, pause, seek) to the fusion controller."""
        self._impl.send_control(command)

    def get_status(self) -> Dict[str, Any]:
        """Expose health metrics, timestamps, and runtime mode gathered by FusionRunnerService."""
        return self._impl.get_status()

    def _get_process_statuses(self) -> List[Dict[str, Any]]:
        # Surface subprocess health so the UI can warn when radar or engine processes crash.
        engine = getattr(self._impl, "_engine_process", None)
        engine_pid = engine.pid if engine else None
        radar_only = bool(getattr(self._impl, "_radar_only", False))
        return get_process_statuses(engine_pid, radar_only)

    def _draw_detections(self, bgr, objects):
        """Overlay radar/camera detections on camera frames using the shared drawing util."""
        # Delegate to the underlying runner for consistent overlay logic.
        return self._impl._draw_detections(bgr, objects)

    def get_latest_frame_jpeg(self, quality: int = 80) -> Optional[bytes]:
        """Retrieve the newest fused camera frame the engine encoded as JPEG bytes."""
        return self._impl.get_latest_frame_jpeg(quality)

    def mjpeg_generator(
        self, fps_limit: float = 10.0, quality: int = 70
    ) -> Generator[bytes, None, None]:
        """Yield multipart MJPEG chunks for streaming endpoints.

        Flask wraps this generator inside a multipart response for live video in browsers.
        """
        return self._impl.mjpeg_generator(fps_limit=fps_limit, quality=quality)

    def get_latest_rd_png(self) -> Optional[bytes]:
        """Return the range-doppler heatmap rendered by the radar post-processing pipeline."""
        return self._impl.get_latest_rd_png()

    def get_latest_ra_png(self) -> Optional[bytes]:
        """Return the range-azimuth heatmap rendered by the radar post-processing pipeline."""
        return self._impl.get_latest_ra_png()

    def get_latest_point_cloud_json(self) -> Optional[Dict[str, Any]]:
        """Expose the most recent radar point-cloud as a JSON-serializable structure."""
        return self._impl.get_latest_point_cloud_json()

    def get_tuning(self) -> Dict[str, Any]:
        """Fetch the current sensor-tuning parameters tracked by FusionRunnerService."""
        return self._impl.get_tuning()

    def set_tuning(self, tuning: Dict[str, Any]) -> Dict[str, Any]:
        """Apply tuning parameters (e.g., radar thresholds) and return sanitized values."""
        return self._impl.set_tuning(tuning)

    @property
    def _running(self) -> bool:
        """True while the fusion subprocess is actively streaming sensor data."""
        return bool(getattr(self._impl, "_running", False))

    @property
    def _starting(self) -> bool:
        """True while FusionRunnerService is still spinning up child processes."""
        return bool(getattr(self._impl, "_starting", False))

    @property
    def _failure_reason(self) -> Optional[str]:
        """Provide a short error string captured during the most recent failed start."""
        return getattr(self._impl, "_failure_reason", None)


# Flask wires functions to HTTP endpoints via the @app.route decorators defined below.
app = Flask(__name__, template_folder="templates")
runner: Optional[FusionRunner] = None
# Runner stays module-global so the HTTP handlers can manipulate a single engine instance.


@app.route("/")
def index():
    # Flask calls this when a browser hits GET /; render_template injects data into HTML.
    # Build config options for radar configs (for live mode quick select)
    try:
        base_dir = getattr(CFGS, "AWR2243_CONFIG_DIR", "")
        names = getattr(CFGS, "AWR2243_CONFIG_FILE_NAMES", {})
        options = []
        for mode in ("2D", "3D"):
            for fname in names.get(mode, []) or []:
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
        # Missing config entries should not block page load; fall back gracefully.
        options = []
    # Defaults for replay inputs from server config
    default_replay_path = app.config.get("DEFAULT_REPLAY_PATH", "")
    default_replay_cfg = app.config.get("DEFAULT_REPLAY_CFG", "")
    return render_template(
        "index.html",
        radar_config_options=options,
        default_replay_path=default_replay_path,
        default_replay_cfg=default_replay_cfg,
    )


@app.route("/status")
def status():
    global runner
    if runner is None:
        # Surface a 503 so callers know the service has not been initialized yet.
        return jsonify({"running": False}), 503
    # jsonify converts dictionaries to JSON HTTP responses with the correct headers.
    st = runner.get_status()
    return jsonify(st)


@app.route("/tuning", methods=["GET", "POST"])
def tuning():
    global runner
    if runner is None:
        return ("runner not initialized", 503)
    if request.method == "GET":
        # GET requests have no body; just reflect current tuning as JSON.
        return jsonify(runner.get_tuning())
    try:
        # For POST, Flask makes query/body data available through request.get_json().
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        body = {}
    # Forward sanitized tuning payloads to the engine; FusionRunner validates values.
    return jsonify(runner.set_tuning(body))


@app.route("/status/processes")
def status_processes():
    global runner
    if runner is None:
        return jsonify({"process_statuses": []}), 503
    # Provide UI-friendly metadata for watchdog dashboards.
    return jsonify({"process_statuses": runner._get_process_statuses()})


@app.route("/system/start", methods=["POST"])
def system_start():
    global runner
    if runner is None:
        return ("runner not initialized", 503)
    # request.args reads query-string values like /system/start?mode=replay&radar_only=1.
    mode = request.args.get("mode", "live").lower()
    radar_only = bool(request.args.get("radar_only", "0") in ("1", "true", "True"))
    replay_path = request.args.get("replay_path")
    radar_cfg = request.args.get("radar_cfg")
    ok = runner.start(
        mode=mode,
        radar_only=radar_only,
        radar_config_file=radar_cfg,
        replay_path=replay_path,
    )
    if not ok and (runner._running or runner._starting):
        # Expose a conflict status when a start request races with an active session.
        return ("already running or starting", 409)
    # HTTP 200/500 tell the UI whether boot succeeded; the body includes a short message.
    return (
        "OK" if ok else f"FAILED: {runner._failure_reason or 'unknown'}",
        200 if ok else 500,
    )


@app.route("/system/stop", methods=["POST"])
def system_stop():
    global runner
    if runner is None:
        return ("runner not initialized", 503)
    # stop() is idempotent; swallow repeated stop requests from the UI.
    runner.stop()
    return ("OK", 200)


@app.route("/replay/control", methods=["POST"])
def replay_control():
    global runner
    if runner is None:
        return ("runner not initialized", 503)
    try:
        cmd = request.args.get("cmd") or (request.get_json(silent=True) or {}).get(
            "cmd"
        )
        if isinstance(cmd, str) and cmd:
            # Commands are thin wrappers around FusionRunner IPC (play, pause, stop, seek).
            runner.send_control(cmd)
            return ("OK", 200)
    except Exception:
        pass
    # Returning 400 signals the client supplied an invalid payload.
    return ("bad command", 400)


@app.route("/replay/info")
def replay_info():
    """Return simple info about a recording directory: counts of BIN and PNG files.
    Query: path=<dir>
    Response: { radar_frames: int, camera_frames: int }
    """
    try:
        path = request.args.get("path") or ""
        if not path:
            return jsonify({"radar_frames": 0, "camera_frames": 0})
        # Normalize and ensure directory exists
        d = os.path.normpath(path)
        if not os.path.isdir(d):
            return jsonify({"radar_frames": 0, "camera_frames": 0})
        rad = 0
        cam = 0
        try:
            for root, dirs, files in os.walk(d):
                for fn in files:
                    l = fn.lower()
                    if l.endswith(".bin"):
                        rad += 1
                    elif l.endswith(".png"):
                        # Only count PNG files whose names contain only numbers and underscores
                        # and end with a number (e.g., 0000000412_06917_000000012314.png)
                        # Exclude files like _xz.png, frame_001.png, heatmap_xz.png, etc.
                        name_without_ext = l[:-4]  # Remove .png extension
                        if name_without_ext and name_without_ext[-1].isdigit():
                            # Check that name only contains digits and underscores
                            if all(c.isdigit() or c == "_" for c in name_without_ext):
                                # Guard against counting thumbnails or debug exports.
                                cam += 1
        except Exception:
            pass
        return jsonify({"radar_frames": int(rad), "camera_frames": int(cam)})
    except Exception:
        return jsonify({"radar_frames": 0, "camera_frames": 0})


@app.route("/fs/list")
def fs_list():
    """List files/dirs under user home. Query param 'path' is absolute or relative under home.
    Returns JSON: { cwd, entries: [ {name, path, type} ] }
    """
    home = os.path.expanduser("~")
    req_path = request.args.get("path")
    if not req_path:
        target = home
    else:
        # Allow absolute or relative-to-home; then normalize and constrain
        if os.path.isabs(req_path):
            target = os.path.normpath(req_path)
        else:
            target = os.path.normpath(os.path.join(home, req_path))
    try:
        # Constrain to home
        if os.path.commonpath([home, target]) != os.path.commonpath([home]):
            # Reject attempts to traverse outside the user home directory.
            return jsonify({"cwd": home, "entries": []})
        entries = []
        with os.scandir(target) as it:
            for e in it:
                try:
                    etype = (
                        "dir" if e.is_dir() else ("file" if e.is_file() else "other")
                    )
                    entries.append({"name": e.name, "path": e.path, "type": etype})
                except Exception:
                    pass
        # Present directories first, then files sorted alphabetically for predictable UX.
        entries.sort(key=lambda x: (x["type"] != "dir", x["name"].lower()))
        return jsonify({"cwd": target, "entries": entries})
    except Exception:
        return jsonify({"cwd": home, "entries": []})


@app.route("/control/start_record", methods=["POST"])
def start_record():
    global runner
    if runner is None or not runner._running:
        return ("runner not started", 503)
    try:
        ts_dir = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        recording_dir = os.path.join(CFGS.DEST_STORAGE, ts_dir)
        os.makedirs(recording_dir, exist_ok=True)
        # Pass an absolute recording path so the engine writes into a unique session folder.
        runner.send_control(f"start_recording:{recording_dir}")
    except Exception:
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
    # Disable caching so the browser always fetches the freshest frame.
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
    # Serve a multipart MJPEG stream that browsers can consume as a video element.
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
    # JSON is lightweight enough for polling dashboards that render 2D/3D views.
    return jsonify(data)


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
    elif mode in ("pc", "pc2d", "pc3d"):
        return ("use /radar/pc.json", 410)
    else:
        return ("bad mode", 400)
    if png is None:
        return ("no data", 204)
    resp = make_response(png)
    resp.headers["Content-Type"] = "image/png"
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    try:
        # Track payload sizes to help diagnose network bottlenecks.
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
        # Runner is captured from outer scope and guaranteed non-None here
        rr = runner
        assert rr is not None
        while True:
            data = rr.get_status()
            # Use Server-Sent Events format so the UI can react without polling.
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(1.0)

    return Response(gen(), mimetype="text/event-stream")


def _signal_handler(signum, frame):
    logger.info(f"Signal {signum} received, shutting down...")
    try:
        if runner is not None:
            # Ensure the engine terminates cleanly before tearing down shared resources.
            runner.stop()
    except Exception:
        pass
    radar_hw_cleanup()
    os._exit(0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fusion Web Server (Live/Replay)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument(
        "--default-replay-path",
        dest="default_replay_path",
        default="",
        help="Default recording directory to prefill in the UI",
    )
    parser.add_argument(
        "--default-replay-cfg",
        dest="default_replay_cfg",
        default="",
        help="Default radar config file to prefill in the UI",
    )
    args = parser.parse_args()

    atexit.register(radar_hw_cleanup)
    # Mirror signal handling with the CLI usage so Ctrl+C cleans up hardware.
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        # Multiprocessing on Linux can leak shared memory trackers; disable where possible.
        disable_shm_resource_tracker(logger)
    except Exception:
        pass

    global runner
    runner = FusionRunner()
    logger.info(
        "FusionRunner ready (idle). Use /system/start to begin (mode=live|replay)."
    )

    # Store defaults for template rendering
    try:
        app.config["DEFAULT_REPLAY_PATH"] = args.default_replay_path or ""
        app.config["DEFAULT_REPLAY_CFG"] = args.default_replay_cfg or ""
    except Exception:
        pass

    # Enable threaded mode so camera and radar endpoints can respond concurrently.
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
