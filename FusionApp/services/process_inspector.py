import os
from typing import List, Dict, Any, Optional

from utils import setup_logger


logger = setup_logger("process_inspector")


def get_process_statuses(
    engine_pid: Optional[int], radar_only: bool
) -> List[Dict[str, Any]]:
    roles = [
        {"label": "WebServer", "pid": os.getpid(), "status": "running"},
        {"label": "FusionEngine", "pid": engine_pid, "status": None},
        {"label": "RadarFeed", "pid": None, "status": None},
        {"label": "RadarAnalyser", "pid": None, "status": None},
        {"label": "CameraFeed", "pid": None, "status": None},
        {"label": "CameraAnalyser", "pid": None, "status": None},
    ]

    try:
        import psutil  # type: ignore

        if engine_pid is None:
            roles[1]["status"] = "not_exists"
        else:
            try:
                ep = psutil.Process(engine_pid)
                roles[1]["status"] = "running" if ep.is_running() else "dead"
            except Exception:
                roles[1]["status"] = "dead"

        if engine_pid is not None:
            try:
                p = psutil.Process(engine_pid)
                children = p.children(recursive=False)
                try:
                    children.sort(key=lambda c: c.create_time())
                except Exception:
                    children.sort(key=lambda c: c.pid)
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
            except Exception as e:
                logger.debug(f"process children fetch failed: {e}")

        if radar_only:
            for i in (4, 5):
                if roles[i]["pid"] is None:
                    roles[i]["status"] = "not_exists"
    except Exception:
        for i in range(2, 6):
            if roles[i]["pid"] is None and roles[i]["status"] is None:
                roles[i]["status"] = "unknown"

    for r in roles:
        if r["status"] is None:
            r["status"] = "not_exists"
    return roles
