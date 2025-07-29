"""
Radar initialization module that handles all radar-related setup in the main process
to avoid multiprocessing issues with DCA1000 and fpga_udp.
"""

import time
from typing import Tuple, Dict, Any
from mmwave.dataloader import DCA1000
import fpga_udp as radar
from utils import setup_logger
from config_params import CFGS


class RadarInitializer:
    """
    Handles radar initialization in the main process to avoid multiprocessing issues.
    This class performs all the heavy initialization work that cannot be serialized
    across process boundaries.
    """
    
    def __init__(self, radar_config_file: str = CFGS.AWR2243_CONFIG_FILE, 
                 dca_config_file: str = CFGS.DCA_CONFIG_FILE):
        self.radar_config_file = radar_config_file
        self.dca_config_file = dca_config_file
        self.logger = setup_logger("RadarInitializer")
        
        # Initialization results
        self.adc_params: Dict[str, Any] = {}
        self.cfg_params: Dict[str, Any] = {}
        self.lvds_data_size_per_chirp: Any = 0
        self.max_send_bytes_per_chirp: Any = 0
        self.is_initialized = False
        self._dca_instance = None  # Store the initialized DCA instance
        
    def initialize(self) -> Dict[str, Any]:
        """
        Perform all radar initialization in the main process.
        
        Returns:
            Dictionary containing all initialization results needed by the worker process,
            including the initialized DCA1000 instance.
        """
        self.logger.info("Starting radar initialization in main process...")
        
        try:
            # Create DCA1000 instance for configuration reading
            dca = DCA1000()
            
            # Reset hardware
            dca.reset_radar()
            dca.reset_fpga()
            self.logger.info("Waiting 1s for radar and FPGA reset...")
            time.sleep(1)
            
            # Initialize radar firmware
            radar.AWR2243_init(self.radar_config_file)
            radar.AWR2243_setFrameCfg(0)
            
            # Read configuration parameters
            (
                self.lvds_data_size_per_chirp,
                self.max_send_bytes_per_chirp,
                self.adc_params,
                self.cfg_params,
            ) = dca.AWR2243_read_config(self.radar_config_file)
            
            # Refresh DCA parameters
            dca.refresh_parameter()

            dca.stream_start()
            dca.fastRead_in_Cpp_thread_start()

            radar.AWR2243_sensorStart()
            
            self.logger.info(
                f"LVDSDataSizePerChirp: {self.lvds_data_size_per_chirp} "
                f"must <= maxSendBytesPerChirp: {self.max_send_bytes_per_chirp}"
            )
            
            # Test system connectivity
            self.logger.info(f"System connection check: {dca.sys_alive_check()}")
            self.logger.info(dca.read_fpga_version())
            
            # Configure FPGA
            fpga_config_result = dca.config_fpga(self.dca_config_file)
            self.logger.info(f"Config fpga: {fpga_config_result}")
            
            # Configure record packet delay
            record_config_result = dca.config_record(self.dca_config_file)
            self.logger.info(f"Config record packet delay: {record_config_result}")
            
            # Store the DCA instance for reuse (don't close it)
            self._dca_instance = dca
            
            self.is_initialized = True
            
            # Return all the initialization data needed by the worker process
            # Including the initialized DCA instance
            init_data = {
                "adc_params": self.adc_params,
                "cfg_params": self.cfg_params,
                "lvds_data_size_per_chirp": self.lvds_data_size_per_chirp,
                "max_send_bytes_per_chirp": self.max_send_bytes_per_chirp,
                "radar_config_file": self.radar_config_file,
                "dca_config_file": self.dca_config_file,
                "dca_instance": dca,  # Pass the initialized DCA instance
            }
            
            self.logger.info("Radar initialization completed successfully")
            return init_data
            
        except Exception as e:
            self.logger.error(f"Failed to initialize radar: {e}")
            raise
    
    def cleanup(self):
        """
        Clean up radar resources. Should be called when shutting down.
        """
        if self.is_initialized:
            try:
                # Clean up the DCA instance if we have one
                if self._dca_instance is not None:
                    self.logger.info("Closing DCA1000 instance...")
                    self._dca_instance.close()
                    self._dca_instance = None
                
                self.logger.info("Powering off radar...")
                radar.AWR2243_poweroff()
                self.logger.info("Radar powered off successfully")
            except Exception as e:
                self.logger.error(f"Error during radar cleanup: {e}")
            finally:
                self.is_initialized = False
