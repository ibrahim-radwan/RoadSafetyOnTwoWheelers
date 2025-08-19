"""
Radar Heatmap Analyser for radar_app.py
Returns dictionary with heatmap and point cloud data for visualization
Based on dca1000_analyser_awr2243_pd.py
"""

import multiprocessing
import numpy as np
from queue import Empty, Full
from radar.dca1000_awr2243 import DCA1000Frame
from typing import Dict, Any, Optional
import time
import logging
import os
from config_params import CFGS
from sample_processing.radar_params import ADCParams
from sample_processing.radar_proc import (
    openradar_pd_process_frame,
    openradar_pd_process_frame_optimised,
    pyradar_process_frame,
    openradar_rt_process_frame,
    custom_process_frame,
)
from utils import setup_logger
from multiprocessing import shared_memory

from engine.interfaces import RadarAnalyser


class RadarHeatmapAnalyser(RadarAnalyser):
    def __init__(
        self,
        config_file: Optional[str] = None,
        *,
        prealloc_shm_meta: Optional[dict] = None,
        prealloc_res_shm_meta: Optional[dict] = None,
    ):
        # Only store serializable configuration
        self.config_file = config_file
        # Initialize these in run() method
        self.logger: Optional[logging.Logger] = None
        self.adc_params: Optional[ADCParams] = None
        # Shared memory state (initialized in run)
        self._shm_blocks = []
        self._shm_nbytes: int = 0
        self._shm_dtype: Optional[str] = None
        self._shm_shape: Optional[tuple] = None
        self._last_shm_seq: Optional[int] = None
        self._prealloc_shm_meta = prealloc_shm_meta
        # Results SHM (range_doppler, range_azimuth)
        self._prealloc_res_shm_meta = prealloc_res_shm_meta
        self._rd_blocks = []
        self._rd_shape = None
        self._rd_dtype = None
        self._ra_blocks = []
        self._ra_shape = None
        self._ra_dtype = None
        self._res_seq = 0

    def _preprocess_frame_from_raw_data(self, dca_frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame data from raw DCA1000 data format to complex radar frame
        Based on preprocess_frame_from_awr2243 from sample_processing.radar_preproc but adapted for raw data input

        This method uses the same preprocessing logic as the existing preprocess_frame_from_awr2243
        function but works with raw data arrays instead of file paths.

        Args:
            dca_frame: Raw data array from DCA1000 format

        Returns:
            Complex-valued radar frame with shape (chirps, tx, rx, samples)
        """
        # The data is organized as:
        # [chirps, tx, adc_samples, IQ, rx]

        if self.adc_params is None:
            raise RuntimeError(
                "ADC parameters not initialized. Call run() method first."
            )

        frame = np.reshape(
            dca_frame,
            (
                self.adc_params.chirps,
                self.adc_params.tx,
                self.adc_params.samples,
                self.adc_params.IQ,
                self.adc_params.rx,
            ),
        )

        frame = np.transpose(frame, (0, 1, 4, 2, 3))
        # Shape is now (chirps, tx, rx, samples, IQ)

        complex_frame = (1j * frame[..., 1] + frame[..., 0]).astype(
            np.complex64
        )  # I first

        assert complex_frame.shape == (
            self.adc_params.chirps,
            self.adc_params.tx,
            self.adc_params.rx,
            self.adc_params.samples,
        ), f"complex_frame shape mismatch! Expected: {(self.adc_params.chirps, self.adc_params.tx, self.adc_params.rx, self.adc_params.samples)}, Actual: {complex_frame.shape}"

        return complex_frame

    def _analyse_frame(self, dca_frame: DCA1000Frame) -> Dict[str, Any]:
        """
        Analyse frame and return dictionary with heatmap and point cloud data

        Returns:
            Dictionary with keys:
            - 'range_doppler': 2D array for range-doppler heatmap
            - 'range_azimuth': 2D array for range-azimuth heatmap
            - 'point_cloud': dict with 'x', 'y', 'z', 'intensity' arrays
            - 'processing_time': float
        """
        start_time = time.perf_counter()

        # Preprocess frame using the unified preprocessing method
        frame = self._preprocess_frame_from_raw_data(dca_frame.data)

        # Use openradar_pd_process_frame for processing
        # IS_INDOOR=True is a reasonable default for most indoor radar applications

        if self.adc_params is None:
            raise RuntimeError(
                "ADC parameters not initialized. Call run() method first."
            )

        # result = openradar_pd_process_frame(frame, self.adc_params, IS_INDOOR=True)
        result = openradar_pd_process_frame_optimised(
            frame, self.adc_params, IS_INDOOR=True
        )
        # result = openradar_rt_process_frame(frame, self.adc_params)

        # frame = frame.reshape(frame.shape[0], frame.shape[1] * frame.shape[2], -1)
        # result = pyradar_process_frame(frame, self.adc_params, doa_method="MUSIC", IS_INDOOR=False)
        # result = custom_process_frame(frame, self.adc_params)

        # Extract results
        range_doppler_matrix = result[
            "range_doppler"
        ]  # This will be None for openradar method
        range_azimuth_matrix = result["range_azimuth"]
        x_pos = result["x_pos"]
        y_pos = result["y_pos"]
        z_pos = result["z_pos"]
        velocities = result["velocities"]
        snrs = result["snrs"]
        cluster_labels = result["cluster_labels"]

        # Create point cloud data from the results
        point_cloud_data = {
            "x": x_pos,
            "y": y_pos,
            "z": z_pos,
            "intensity": snrs,  # Use SNR as intensity
        }

        processing_time = time.perf_counter() - start_time

        return {
            "range_doppler": range_doppler_matrix,
            "range_azimuth": range_azimuth_matrix,
            "point_cloud": point_cloud_data,
            "processing_time": processing_time,
            "frame_timestamp": dca_frame.timestamp,
        }

    def run(
        self,
        input_queue: multiprocessing.Queue,
        output_queue: multiprocessing.Queue,
        stop_event,
    ):
        """Main processing loop"""
        # Initialize logger and ADC parameters in the target process
        self.logger = setup_logger("RadarHeatmapAnalyser")

        # Initialize ADC parameters from provided config file or default
        config_to_use = (
            self.config_file if self.config_file else CFGS.AWR2243_CONFIG_FILE
        )
        self.adc_params = ADCParams(config_to_use)
        self.logger.info(
            f"ADC parameters initialized from config file: {config_to_use}"
        )

        self.logger.info("RadarHeatmapAnalyser starting...")

        # Check if ADC parameters are available
        if self.adc_params is None:
            self.logger.error("ADC parameters not initialized")
            return

        # Attach to preallocated radar SHM if provided by engine
        if self._prealloc_shm_meta:
            try:
                names = self._prealloc_shm_meta["names"]
                self._shm_blocks = [
                    shared_memory.SharedMemory(name=name) for name in names
                ]
                self._shm_nbytes = int(self._prealloc_shm_meta["nbytes"])
                self._shm_dtype = str(self._prealloc_shm_meta["dtype"])
                self._shm_shape = (
                    tuple(self._prealloc_shm_meta["shape"])
                    if isinstance(self._prealloc_shm_meta["shape"], (list, tuple))
                    else (int(self._prealloc_shm_meta["shape"]),)
                )
                self.logger.info(
                    f"Attached preallocated radar SHM: names={names}, nbytes={self._shm_nbytes}, dtype={self._shm_dtype}, shape={self._shm_shape}"
                )
            except Exception as e:
                self.logger.error(f"Failed to attach preallocated radar SHM: {e}")

        # Attach to preallocated results SHM (rd/ra)
        if self._prealloc_res_shm_meta:
            try:
                rd_meta = self._prealloc_res_shm_meta.get("rd")
                ra_meta = self._prealloc_res_shm_meta.get("ra")
                if rd_meta:
                    self._rd_blocks = [
                        shared_memory.SharedMemory(name=n) for n in rd_meta["names"]
                    ]
                    self._rd_shape = (
                        tuple(rd_meta["shape"])
                        if isinstance(rd_meta["shape"], (list, tuple))
                        else (int(rd_meta["shape"]),)
                    )
                    self._rd_dtype = str(rd_meta["dtype"])  # 'float32'
                if ra_meta:
                    self._ra_blocks = [
                        shared_memory.SharedMemory(name=n) for n in ra_meta["names"]
                    ]
                    self._ra_shape = (
                        tuple(ra_meta["shape"])
                        if isinstance(ra_meta["shape"], (list, tuple))
                        else (int(ra_meta["shape"]),)
                    )
                    self._ra_dtype = str(ra_meta["dtype"])  # 'float32'
                self.logger.info(
                    f"Attached preallocated results SHM: rd={rd_meta['names'] if rd_meta else None}, ra={ra_meta['names'] if ra_meta else None}"
                )
                # Send one-time init meta so GUI can attach without env vars
                try:
                    init_msg = {
                        "RADAR_RES_SHM_INIT": True,
                        "rd": (
                            {
                                "names": rd_meta["names"],
                                "shape": self._rd_shape,
                                "dtype": self._rd_dtype,
                            }
                            if rd_meta
                            else None
                        ),
                        "ra": (
                            {
                                "names": ra_meta["names"],
                                "shape": self._ra_shape,
                                "dtype": self._ra_dtype,
                            }
                            if ra_meta
                            else None
                        ),
                    }
                    # output_queue available later; stash for first loop send
                    self._pending_res_init = init_msg
                except Exception:
                    self._pending_res_init = None
            except Exception as e:
                self.logger.error(f"Failed to attach preallocated results SHM: {e}")
                self._pending_res_init = None
        else:
            self._pending_res_init = None

        # Optionally consume ADC_PARAMS if present without losing first data item
        preloaded_item: Optional[Any] = None
        try:
            first_item = input_queue.get(timeout=10)
            if isinstance(first_item, dict) and "ADC_PARAMS" in first_item:
                self.logger.info(
                    "Skipping ADC_PARAMS from queue (using config file parameters)"
                )
            else:
                preloaded_item = first_item
        except Exception as e:
            self.logger.warning(f"No ADC_PARAMS received from queue: {e}")

        self.logger.info(
            f"Initialized with {self.adc_params.tx} TX, {self.adc_params.rx} RX antennas"
        )
        self.logger.info(f"Range Resolution: {self.adc_params.range_resolution:.4f} m")
        self.logger.info(
            f"Doppler Resolution: {self.adc_params.doppler_resolution:.4f} m/s"
        )

        # Process frames
        self._total_dropped_frames = 0
        while not stop_event.is_set():
            try:
                wait_start_ns = time.perf_counter_ns()
                # If we have a pending SHM init for results, send it now
                if getattr(self, "_pending_res_init", None) is not None:
                    try:
                        output_queue.put_nowait(self._pending_res_init)
                        self._pending_res_init = None
                    except Full:
                        pass
                # Use preloaded first item if available; otherwise read from queue
                if preloaded_item is not None:
                    item = preloaded_item
                    preloaded_item = None
                else:
                    item = input_queue.get(timeout=1)
                # Support STOP sentinel for immediate shutdown
                if isinstance(item, dict) and item.get("STOP"):
                    break
                # SHM is engine-owned; SHM_INIT should not be received
                if isinstance(item, dict) and item.get("RADAR_SHM_INIT"):
                    self.logger.error(
                        "Unexpected RADAR_SHM_INIT received; engine should preallocate SHM"
                    )
                    continue
                # Handle SHM frame metadata
                if isinstance(item, dict) and item.get("RADAR_SHM_FRAME"):
                    if not self._shm_blocks:
                        self.logger.warning(
                            "Received SHM frame before SHM init, dropping."
                        )
                        continue

                    recv_ns = time.perf_counter_ns()
                    slot = int(item.get("slot", 0))
                    seq = int(item.get("seq", 0))
                    # Report missed sequence numbers as drops
                    if self._last_shm_seq is not None and seq != self._last_shm_seq + 1:
                        missed = seq - (self._last_shm_seq + 1)
                        if missed > 0:
                            self.logger.warning(
                                f"Radar frame drop detected: missed={missed} (last_seq={self._last_shm_seq}, seq={seq})"
                            )
                            self._total_dropped_frames += missed
                    self._last_shm_seq = seq

                    # Build numpy view into SHM and copy out
                    try:
                        np_dtype = (
                            np.dtype(self._shm_dtype)
                            if self._shm_dtype is not None
                            else np.int16
                        )
                        shm_view = np.ndarray(
                            self._shm_shape,
                            dtype=np_dtype,
                            buffer=self._shm_blocks[slot].buf,
                        )
                        dca_frame_data = shm_view.copy()
                    except Exception as e:
                        self.logger.warning(f"Failed to read SHM slot {slot}: {e}")
                        continue

                    # Construct frame object with propagated timestamps
                    latest_frame = DCA1000Frame(
                        timestamp=item.get("frame_timestamp", 0.0),
                        data=dca_frame_data,
                        capture_monotonic_ns=int(item.get("capture_monotonic_ns", 0)),
                        enqueue_monotonic_ns=int(item.get("enqueue_monotonic_ns", 0)),
                    )

                    # Analyse
                    results = self._analyse_frame(latest_frame)
                    end_ns = time.perf_counter_ns()

                    # Write results to SHM if available
                    shm_written = False
                    try:
                        slot_res = self._res_seq & 1
                        # Range-Doppler
                        if (
                            self._rd_blocks
                            and isinstance(results.get("range_doppler"), np.ndarray)
                            and results["range_doppler"].size > 0
                        ):
                            rd_out = np.asarray(
                                results["range_doppler"], dtype=np.float32
                            )
                            if self._rd_shape and tuple(rd_out.shape) != tuple(
                                self._rd_shape
                            ):
                                try:
                                    rd_out = rd_out.reshape(self._rd_shape)
                                except Exception:
                                    pass
                            mv_rd = memoryview(self._rd_blocks[slot_res].buf)
                            mv_rd[: rd_out.nbytes] = rd_out.tobytes()
                            shm_written = True
                            results.pop("range_doppler", None)
                        # Range-Azimuth
                        if (
                            self._ra_blocks
                            and isinstance(results.get("range_azimuth"), np.ndarray)
                            and results["range_azimuth"].size > 0
                        ):
                            ra_out = np.asarray(
                                results["range_azimuth"], dtype=np.float32
                            )
                            if self._ra_shape and tuple(ra_out.shape) != tuple(
                                self._ra_shape
                            ):
                                try:
                                    ra_out = ra_out.reshape(self._ra_shape)
                                except Exception:
                                    pass
                            mv_ra = memoryview(self._ra_blocks[slot_res].buf)
                            mv_ra[: ra_out.nbytes] = ra_out.tobytes()
                            shm_written = True
                            results.pop("range_azimuth", None)
                        if shm_written:
                            self._res_seq += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to write results to SHM: {e}")

                    # Best-effort queue size hint
                    try:
                        qsize_hint = input_queue.qsize()
                    except Exception:
                        qsize_hint = -1

                    results.update(
                        {
                            "capture_monotonic_ns": getattr(
                                latest_frame, "capture_monotonic_ns", 0
                            ),
                            "capture_wall_ns": getattr(
                                latest_frame, "capture_wall_ns", 0
                            ),
                            "enqueue_monotonic_ns": getattr(
                                latest_frame, "enqueue_monotonic_ns", 0
                            ),
                            "analyser_receive_ns": recv_ns,
                            "analyser_end_ns": end_ns,
                            "first_dequeue_wait_ns": recv_ns - wait_start_ns,
                            "drain_ns": 0,
                            "drained_count": 0,
                            "total_dropped_frames": self._total_dropped_frames,
                            "input_queue_size_hint": qsize_hint,
                        }
                    )
                    if shm_written:
                        meta = {
                            "RADAR_RES_SHM_FRAME": True,
                            "slot": (self._res_seq - 1) & 1,
                            "seq": self._res_seq - 1,
                            "capture_monotonic_ns": getattr(
                                latest_frame, "capture_monotonic_ns", 0
                            ),
                            "capture_wall_ns": getattr(
                                latest_frame, "capture_wall_ns", 0
                            ),
                            "enqueue_monotonic_ns": getattr(
                                latest_frame, "enqueue_monotonic_ns", 0
                            ),
                            "analyser_receive_ns": recv_ns,
                            "analyser_end_ns": end_ns,
                            "first_dequeue_wait_ns": recv_ns - wait_start_ns,
                            "drain_ns": 0,
                            "drained_count": 0,
                            "total_dropped_frames": self._total_dropped_frames,
                            "input_queue_size_hint": qsize_hint,
                            "frame_timestamp": latest_frame.timestamp,
                            # Keep small payloads (point cloud) in-band
                            "point_cloud": results.get("point_cloud"),
                        }
                        try:
                            output_queue.put_nowait(meta)
                        except Full:
                            self.logger.warning("Output queue full, skipping frame")
                        continue
                    else:
                        try:
                            output_queue.put_nowait(results)
                        except Full:
                            self.logger.warning("Output queue full, skipping frame")
                        continue
                if isinstance(item, DCA1000Frame):
                    # Do not drop further in analyser; process the first frame we dequeued
                    latest_frame = item
                    drained_count = 0
                    drain_start_ns = time.perf_counter_ns()
                    drain_end_ns = drain_start_ns
                    recv_ns = time.perf_counter_ns()
                    # Process the most recent frame
                    results = self._analyse_frame(latest_frame)
                    end_ns = time.perf_counter_ns()
                    # Attach timing metadata (monotonic ns) and propagate capture times
                    self._total_dropped_frames += drained_count
                    # Best-effort queue size hint (may not be implemented on some platforms)
                    try:
                        qsize_hint = input_queue.qsize()
                    except Exception:
                        qsize_hint = -1
                    results.update(
                        {
                            "capture_monotonic_ns": getattr(
                                latest_frame, "capture_monotonic_ns", 0
                            ),
                            "capture_wall_ns": getattr(
                                latest_frame, "capture_wall_ns", 0
                            ),
                            "enqueue_monotonic_ns": getattr(
                                latest_frame, "enqueue_monotonic_ns", 0
                            ),
                            "analyser_receive_ns": recv_ns,
                            "analyser_end_ns": end_ns,
                            "first_dequeue_wait_ns": recv_ns - wait_start_ns,
                            "drain_ns": drain_end_ns - drain_start_ns,
                            "drained_count": drained_count,
                            "total_dropped_frames": self._total_dropped_frames,
                            "input_queue_size_hint": qsize_hint,
                        }
                    )

                    # Send results
                    try:
                        output_queue.put_nowait(results)
                    except Full:
                        # Queue might be full, skip this frame
                        self.logger.warning("Output queue full, skipping frame")
                        pass
                else:
                    # Ignore unrelated items to avoid log spam
                    self.logger.debug(f"Ignoring non-frame item: {type(item)}")
                    continue

            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing frame: {e}")
                import traceback

                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                stop_event.set()  # Stop on critical error

        self.logger.info("RadarHeatmapAnalyser stopped")

        # Detach from SHM blocks if attached
        if self._shm_blocks:
            for shm in self._shm_blocks:
                try:
                    shm.close()
                except Exception:
                    pass
            self._shm_blocks = []

        return
