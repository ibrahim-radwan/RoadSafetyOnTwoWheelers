from utils import setup_logger
from config_params import CFGS


logger = setup_logger("radar_hw")
_initialized = False


def radar_hw_init() -> bool:
    """Initialize AWR2243 radar hardware and precompile DSP kernels."""
    try:
        import fpga_udp  # Local import to avoid hard dependency at import time
        from mmwave import dsp

        ret = fpga_udp.AWR2243_init(CFGS.AWR2243_CONFIG_FILE)
        if ret != 0:
            logger.error("Failed to initialize AWR2243 radar: %d", ret)
            return False
        fpga_udp.AWR2243_setFrameCfg(0)
        ret = fpga_udp.AWR2243_sensorStart()
        if ret != 0:
            logger.error("Failed to start AWR2243 sensor: %d", ret)
            return False
        dsp.precompile_kernels()
        global _initialized
        _initialized = True
        return True
    except Exception as e:
        logger.error(f"Radar HW init error: {e}")
        return False


def radar_hw_cleanup() -> None:
    """Power down AWR2243 radar hardware safely."""
    try:
        import fpga_udp  # Local import to avoid hard dependency at import time

        global _initialized
        if not _initialized:
            return
        try:
            fpga_udp.AWR2243_sensorStop()
        except Exception:
            pass
        try:
            fpga_udp.AWR2243_poweroff()
        except Exception:
            pass
        _initialized = False
    except Exception:
        # Module may be unavailable in some environments; ignore.
        pass
