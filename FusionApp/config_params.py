import os
import time
from typing import Optional
from mmwave.dsp import utils
import logging


class CFGS:
    @staticmethod
    def _find_awr_ports():
        if os.name == "nt":
            return "COM4", "COM3"

        # Find all available ttyACM devices
        acm_devices = [
            f"/dev/ttyACM{i}" for i in range(10) if os.path.exists(f"/dev/ttyACM{i}")
        ]

        if len(acm_devices) >= 2:
            # Sort the devices - CLI port (smaller number) comes first
            acm_devices.sort()
            return acm_devices[0], acm_devices[1]
        else:
            # Honor suppression flag (e.g., replay mode sets this)
            if os.environ.get("FUSION_SUPPRESS_TTYACM_WARNING", "0") != "1":
                logging.getLogger("ConfigParams").warning(
                    "Could not find enough ttyACM devices. Using defaults."
                )
            return "/dev/ttyACM0", "/dev/ttyACM1"

    # Default logging level
    # Change to logging.DEBUG for more verbose output
    # available levels in the order of verbosity:
    # logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG
    LOG_LEVEL = logging.INFO

    AWR_CLI_PORT, AWR_DATA_PORT = _find_awr_ports.__func__()
    AWR_CLI_BR = 115200
    AWR_DATA_BR = 921600

    AWR1843_CONFIG_FILE = os.path.join(os.getcwd(), "config_files/profile_3d_5m.cfg")
    AWR2243_CONFIG_FILE = os.path.join(
        os.getcwd(), "config_files/AWR2243_180m_35cm_64_2_256.txt"
    )
    # AWR_CONFIG_FILE = os.path.join(os.getcwd(), "config_files/matlab_indoor_sample.cfg")
    DCA_CONFIG_FILE = os.path.join(os.getcwd(), "config_files/cf.json")

    DEST_DIR = os.path.join(
        os.getcwd(),
        "../../recordings/",
        time.strftime("%Y_%m_%d_%H_%M_00", time.localtime()),
    )

    RADAR_TO_CAMERA_OFFSET = {"x": -0.1, "y": 0.1, "z": 0.05}
    RADAR_RANGE_LIMIT = 10000.0

    COLOR_CAMERA_MATRIX = [
        [385.8772583, 0.0, 320.08758545],
        [0.0, 385.4173584, 242.15074158],
        [0.0, 0.0, 1.0],
    ]
    DEPTH_CAMERA_MATRIX = [
        [397.41882324, 0.0, 326.93692017],
        [0.0, 397.41882324, 241.10147095],
        [
            0.0,
            0.0,
            1.0,
        ],
    ]

    RADAR_ANGLE_RES = 1.0  # degrees
    RADAR_ANGLE_RANGE = 90.0  # degrees
    RADAR_SKIP_SIZE = 4
    RADAR_ANGLE_BINS = (RADAR_ANGLE_RANGE * 2) // RADAR_ANGLE_RES + 1
    RADAR_BINS_PROCESSED = 128

    RADAR_1D_FFT_WINDOW_TYPE = (
        utils.Window.HAMMING
    )  # Options: None or  utils.Window.[BARTLETT, BLACKMAN, HAMMING, HANNING]
    RADAR_SKIP_SIZE = 4
