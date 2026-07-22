"""
Unified capture clock for live recording.

All sensors map capture instants into CLOCK_MONOTONIC nanoseconds via
``time.perf_counter_ns()``. On Linux this clock is system-wide and stable
across processes, which is the common time domain for radar + host software.

RealSense hardware timestamps (device milliseconds) are calibrated once and
converted into this same domain. Radar has no separate host-visible hardware
timestamp, so its capture instant is sampled with the same monotonic clock.
"""

from __future__ import annotations

import time

# Documented domain written into recording_session.json / manifest entries.
CLOCK_DOMAIN = "CLOCK_MONOTONIC"


def capture_clock_ns() -> int:
    """Current capture clock reading in nanoseconds."""
    return int(time.perf_counter_ns())


def wall_clock_ns() -> int:
    """Wall-clock reading for human-readable session metadata only."""
    return int(time.time_ns())


def realsense_ms_to_capture_ns(rs_ms: float, rs_mono_offset_ns: int) -> int:
    """Map a RealSense device timestamp (ms) into the capture clock domain."""
    return int(rs_mono_offset_ns) + int(float(rs_ms) * 1_000_000)


def calibrate_realsense_offset(rs_ms: float, mono_ns: int | None = None) -> int:
    """Compute offset so ``realsense_ms_to_capture_ns`` aligns with *mono_ns*."""
    if mono_ns is None:
        mono_ns = capture_clock_ns()
    return int(mono_ns) - int(float(rs_ms) * 1_000_000)
