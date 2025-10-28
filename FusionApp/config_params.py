import os
import time
from typing import Optional
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

    AWR2243_CONFIG_FILE_NAMES = {
        "2D": [
            "AWR2243_10m_4cm_64_2_256.txt",
            "AWR2243_24m_5cm_128_2_256.txt",
            "AWR2243_87m_17cm_64_2_256.txt",
            "AWR2243_180m_70cm_64_2_512.txt",
        ],
        "3D": [
            "AWR2243_10m_4cm_64_3_256.txt",
            "AWR2243_24m_5cm_64_3_512.txt",
            "AWR2243_87m_17cm_64_3_256.txt",
            "AWR2243_180m_70cm_64_3_512.txt",
        ],
    }

    AWR2243_CONFIG_DIR = os.path.join(os.getcwd(), "config_files/")

    AWR1843_CONFIG_FILE = os.path.join(os.getcwd(), "config_files/profile_3d_5m.cfg")
    AWR2243_CONFIG_FILE = os.path.join(
        os.getcwd(), "config_files/AWR2243_180m_35cm_64_2_256.txt"
    )
    # AWR_CONFIG_FILE = os.path.join(os.getcwd(), "config_files/matlab_indoor_sample.cfg")
    DCA_CONFIG_FILE = os.path.join(os.getcwd(), "config_files/cf.json")

    DEST_STORAGE = os.path.join(
        os.getcwd(),
        "../../recordings/",
    )

    DEST_DIR = os.path.join(
        os.getcwd(),
        "../../recordings/",
        time.strftime("%Y_%m_%d_%H_%M_00", time.localtime()),
    )
