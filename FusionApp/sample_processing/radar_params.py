"""
Radar parameter processing functions
"""

import numpy as np
import logging
from mmwave.dataloader.adc import DCA1000
from config_params import CFGS
from utils import setup_logger

INDOOR_CONFIG_FILE = "sample_data/AWR2243_indoor_2d.txt"
OUTDOOR_CONFIG_FILE = "sample_data/AWR2243_outdoor_2d.txt"
INDOOR_SAMPLE_1 = "sample_data/indoor_f1.bin"
INDOOR_SAMPLE_2 = "sample_data/indoor_f2.bin"
OUTDOOR_SAMPLE_1 = "sample_data/outdoor_f1.bin"
OUTDOOR_SAMPLE_2 = "sample_data/outdoor_f2.bin"


class ADCParams:
    def __init__(self, config_file):
        from mmwave import dsp

        # Set up logger
        self.logger = setup_logger("ADCParams")

        _, _, adc_params, _ = DCA1000.AWR2243_read_config(config_file)
        self.logger.info("Extracted ADC parameters:")
        for key, value in adc_params.items():
            self.logger.info(f"  {key}: {value}")
        self.chirps = adc_params["chirps"]
        self.rx = adc_params["rx"]
        self.tx = adc_params["tx"]
        self.samples = adc_params["samples"]
        self.IQ = adc_params["IQ"]
        self.bytes = adc_params["bytes"]
        self.startFreq = adc_params["startFreq"]
        self.idleTime = adc_params["idleTime"]
        self.adc_valid_start_time = adc_params["adc_valid_start_time"]
        self.rampEndTime = adc_params["rampEndTime"]
        self.freq_slope = adc_params["freq_slope"]
        self.txStartTime = adc_params["txStartTime"]
        self.sample_rate = adc_params["sample_rate"]
        self.frame_periodicity = adc_params["frame_periodicity"]
        self.range_resolution, self.chirp_bandwidth = dsp.range_resolution(
            self.samples, self.sample_rate, self.freq_slope
        )
        self.doppler_resolution = dsp.doppler_resolution(
            self.chirp_bandwidth,
            self.startFreq,
            self.rampEndTime,
            self.idleTime,
            self.chirps,
            self.tx,
        )
        self.angle_bins = np.linspace(
            -90, 90, self.chirps
        )  # angle resolution from ADC params
        self.range_bins = (
            np.arange(self.samples) * 0.0485
        )  # range resolution from ADC params

        # Calculate extent properties for plotting
        self.max_range = self.samples * self.range_resolution
        self.max_doppler = (self.chirps * self.doppler_resolution) / 2

        self.logger.info(
            f"Range resolution: {self.range_resolution:.2f} m, "
            f"Doppler resolution: {self.doppler_resolution:.2f} m/s"
        )

        self.logger.info(
            f"Max range: {self.max_range:.2f} m, "
            f"Max Doppler: {self.max_doppler:.2f} m/s"
        )

        self.max_azimuth = 60  # degrees, typical for radar systems

        # Pre-calculated extents for imshow/plotting
        self.range_doppler_extents = [
            -self.max_doppler,
            self.max_doppler,
            0,
            self.max_range,
        ]
        self.range_azimuth_extents = [
            -self.max_azimuth,
            self.max_azimuth,
            0,
            self.max_range,
        ]
