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

from engine.interfaces import RadarAnalyser


class RadarHeatmapAnalyser(RadarAnalyser):
    def __init__(self, config_file: Optional[str] = None):
        # Only store serializable configuration
        self.config_file = config_file
        # Initialize these in run() method
        self.logger: Optional[logging.Logger] = None
        self.adc_params: Optional[ADCParams] = None

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

        # Skip the ADC_PARAMS dictionary since we already have them
        try:
            item = input_queue.get(timeout=10)
            if isinstance(item, dict) and "ADC_PARAMS" in item:
                self.logger.info(
                    "Skipping ADC_PARAMS from queue (using config file parameters)"
                )
                # Continue to process frames after this
            else:
                # Put the item back in the queue if it's not ADC_PARAMS
                input_queue.put(item)
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
        while not stop_event.is_set():
            try:
                item = input_queue.get(timeout=1)
                # Support STOP sentinel for immediate shutdown
                if isinstance(item, dict) and item.get("STOP"):
                    break
                if isinstance(item, DCA1000Frame):
                    # Process frame
                    results = self._analyse_frame(item)

                    # Send results
                    try:
                        output_queue.put_nowait(results)
                    except Full:
                        # Queue might be full, skip this frame
                        self.logger.warning("Output queue full, skipping frame")
                        pass

            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing frame: {e}")
                import traceback

                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                stop_event.set()  # Stop on critical error

        self.logger.info("RadarHeatmapAnalyser stopped")
        # Forcefully exit to avoid any background threads (e.g., BLAS/OpenMP) keeping process alive
        os._exit(0)
