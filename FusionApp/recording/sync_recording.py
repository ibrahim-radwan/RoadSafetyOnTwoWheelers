"""
Cross-process synchronized recording for live camera + radar capture.

Radar is the pacing sensor: each saved radar frame requests one paired camera
save that uses the closest recent camera frame to the latched capture time.
Both files share the same pair sequence number in the filename.
"""

from __future__ import annotations

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment,misc]
import json
import os
import struct
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple

from recording.clock import CLOCK_DOMAIN, capture_clock_ns, wall_clock_ns

# Shared-memory layout (little-endian, 64 bytes).
_SHM_FORMAT = "<qqqqqqd"
_SHM_SIZE = struct.calcsize(_SHM_FORMAT)
_OFF_RECORDING_EPOCH_NS = 0
_OFF_PAIR_SEQ = 1
_OFF_PAIR_REQUEST_GEN = 2
_OFF_PAIR_TARGET_CAMERA_MONO_NS = 3
_OFF_PAIR_RADAR_CAPTURE_MONO_NS = 4
_OFF_LATEST_CAMERA_MONO_NS = 5
_OFF_LATEST_CAMERA_RS_MS = 6

MANIFEST_FILENAME = "recording_manifest.jsonl"
SESSION_FILENAME = "recording_session.json"


@dataclass(frozen=True)
class StartRecordingCommand:
    directory: Optional[str]
    epoch_ns: int


@dataclass(frozen=True)
class PairSaveRequest:
    generation: int
    pair_seq: int
    target_camera_mono_ns: int
    radar_capture_mono_ns: int


def _directory_from_command_parts(path_parts: List[str]) -> Optional[str]:
    if not path_parts:
        return None
    if (
        len(path_parts) == 2
        and len(path_parts[0]) == 1
        and path_parts[1][:1] in ("\\", "/")
    ):
        directory = f"{path_parts[0]}:{path_parts[1]}"
    else:
        directory = ":".join(path_parts)
    directory = os.path.normpath(directory.strip())
    return directory or None


def parse_start_recording_command(command: str) -> StartRecordingCommand:
    """Parse ``start_recording[:dir[:epoch_ns]]`` control commands."""
    directory: Optional[str] = None
    epoch_ns = capture_clock_ns()
    if not isinstance(command, str) or not command.startswith("start_recording"):
        return StartRecordingCommand(directory=directory, epoch_ns=epoch_ns)

    payload = command[len("start_recording") :]
    if payload.startswith(":"):
        payload = payload[1:]
    if not payload:
        return StartRecordingCommand(directory=directory, epoch_ns=epoch_ns)

    parts = payload.split(":")
    if len(parts) == 1:
        directory = _directory_from_command_parts(parts)
        return StartRecordingCommand(directory=directory, epoch_ns=epoch_ns)

    try:
        epoch_ns = int(parts[-1])
        path_parts = parts[:-1]
    except ValueError:
        path_parts = parts
    directory = _directory_from_command_parts(path_parts)
    return StartRecordingCommand(directory=directory, epoch_ns=epoch_ns)


def format_start_recording_command(
    directory: Optional[str], epoch_ns: Optional[int] = None
) -> str:
    """Serialize a start-recording command with a Windows-safe path encoding."""
    if epoch_ns is None:
        epoch_ns = capture_clock_ns()
    if directory:
        dir_fwd = os.path.normpath(directory).replace("\\", "/")
        return f"start_recording:{dir_fwd}:{int(epoch_ns)}"
    return f"start_recording::{int(epoch_ns)}"


def relative_timestamp_s(capture_mono_ns: int, recording_epoch_ns: int) -> float:
    if recording_epoch_ns <= 0:
        return 0.0
    return max(0.0, (capture_mono_ns - recording_epoch_ns) / 1_000_000_000.0)


def create_recording_pair_shm() -> Tuple[shared_memory.SharedMemory, Dict[str, Any]]:
    """Create shared memory used to coordinate paired recording.

    Returns the owner handle plus metadata for child processes. The caller
    must keep the returned ``SharedMemory`` object alive for the session;
    otherwise Windows immediately destroys the segment and attach fails.
    """
    shm = shared_memory.SharedMemory(create=True, size=_SHM_SIZE)
    state = RecordingPairState(shm)
    state.reset()
    return shm, {"name": shm.name, "size": _SHM_SIZE}


class RecordingPairState:
    """Attach to the engine-owned recording coordination block."""

    def __init__(self, shm: shared_memory.SharedMemory):
        self._shm = shm

    @classmethod
    def attach(cls, meta: Dict[str, Any]) -> "RecordingPairState":
        shm = shared_memory.SharedMemory(name=str(meta["name"]))
        return cls(shm)

    def close(self) -> None:
        try:
            self._shm.close()
        except Exception:
            pass

    def _read(self) -> Tuple[int, ...]:
        return struct.unpack(_SHM_FORMAT, self._shm.buf[:_SHM_SIZE])

    def _write(self, values: Tuple[int, ...]) -> None:
        self._shm.buf[:_SHM_SIZE] = struct.pack(_SHM_FORMAT, *values)

    def reset(self) -> None:
        self._write((0, 0, 0, 0, 0, 0, 0.0))

    def begin_recording(self, epoch_ns: int) -> bool:
        """Arm paired recording. Idempotent: ignores duplicate start calls."""
        values = list(self._read())
        if int(values[_OFF_RECORDING_EPOCH_NS]) > 0:
            return False
        values[_OFF_RECORDING_EPOCH_NS] = int(epoch_ns)
        values[_OFF_PAIR_SEQ] = 0
        values[_OFF_PAIR_REQUEST_GEN] = 0
        values[_OFF_PAIR_TARGET_CAMERA_MONO_NS] = 0
        values[_OFF_PAIR_RADAR_CAPTURE_MONO_NS] = 0
        self._write(tuple(values))
        return True

    def end_recording(self) -> None:
        values = list(self._read())
        if int(values[_OFF_RECORDING_EPOCH_NS]) <= 0:
            return
        values[_OFF_RECORDING_EPOCH_NS] = 0
        self._write(tuple(values))

    def recording_epoch_ns(self) -> int:
        return int(self._read()[_OFF_RECORDING_EPOCH_NS])

    def publish_camera_frame(self, capture_mono_ns: int, rs_ms: float) -> None:
        values = list(self._read())
        values[_OFF_LATEST_CAMERA_MONO_NS] = int(capture_mono_ns)
        values[_OFF_LATEST_CAMERA_RS_MS] = float(rs_ms)
        self._write(tuple(values))

    def latest_camera_mono_ns(self) -> int:
        return int(self._read()[_OFF_LATEST_CAMERA_MONO_NS])

    def request_pair_save(self, radar_capture_mono_ns: int) -> Tuple[int, int]:
        """Reserve a pair sequence and notify the camera process."""
        values = list(self._read())
        pair_seq = int(values[_OFF_PAIR_SEQ]) + 1
        target_camera_mono_ns = int(values[_OFF_LATEST_CAMERA_MONO_NS])
        values[_OFF_PAIR_SEQ] = pair_seq
        values[_OFF_PAIR_TARGET_CAMERA_MONO_NS] = target_camera_mono_ns
        values[_OFF_PAIR_RADAR_CAPTURE_MONO_NS] = int(radar_capture_mono_ns)
        values[_OFF_PAIR_REQUEST_GEN] = int(values[_OFF_PAIR_REQUEST_GEN]) + 1
        self._write(tuple(values))
        return pair_seq, target_camera_mono_ns

    def pair_request_generation(self) -> int:
        return int(self._read()[_OFF_PAIR_REQUEST_GEN])

    def read_pair_request(self) -> PairSaveRequest:
        values = self._read()
        return PairSaveRequest(
            generation=int(values[_OFF_PAIR_REQUEST_GEN]),
            pair_seq=int(values[_OFF_PAIR_SEQ]),
            target_camera_mono_ns=int(values[_OFF_PAIR_TARGET_CAMERA_MONO_NS]),
            radar_capture_mono_ns=int(values[_OFF_PAIR_RADAR_CAPTURE_MONO_NS]),
        )


class RecordingManifest:
    """Append-only JSONL manifest for paired recording sessions."""

    def __init__(self, directory: str):
        self._path = os.path.join(directory, MANIFEST_FILENAME)

    @property
    def path(self) -> str:
        return self._path

    def append(self, entry: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        line = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        with open(self._path, "a", encoding="utf-8") as handle:
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.write(line + "\n")
                handle.flush()
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def load_pairs(directory: str) -> List[Dict[str, Any]]:
        path = os.path.join(directory, MANIFEST_FILENAME)
        if not os.path.isfile(path):
            return []
        pairs: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return pairs


def write_recording_session(
    directory: str,
    *,
    recording_epoch_mono_ns: int,
    recording_epoch_wall_ns: int,
    paired_recording: bool = True,
    radar_config_file: Optional[str] = None,
) -> str:
    """Write session metadata describing the unified capture clock."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, SESSION_FILENAME)
    payload = {
        "clock_domain": CLOCK_DOMAIN,
        "recording_epoch_mono_ns": int(recording_epoch_mono_ns),
        "recording_epoch_wall_ns": int(recording_epoch_wall_ns),
        "paired_recording": bool(paired_recording),
        "pairing": "radar_master",
        "filename_time_base": "recording_epoch_mono_ns",
        "pair_id_field": "pair_seq",
    }
    if radar_config_file:
        payload["radar_config_file"] = os.path.abspath(
            os.path.normpath(str(radar_config_file))
        )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
