"""
Factory class for creating FusionEngine configurations for different scenarios.
This demonstrates how to use the polymorphic architecture.
"""

from config_params import CFGS
from engine.fusion_engine import FusionEngine

# Live mode imports
from camera.d455 import D455Config
from radar.dca1000_awr2243 import DCA1000Config

from typing import Optional


class FusionFactory:
    """Factory for creating different FusionEngine configurations"""

    @staticmethod
    def create_live_fusion(
        camera_config: Optional[D455Config] = None,
        radar_config: Optional[DCA1000Config] = None,
    ) -> FusionEngine:
        """
        Create a FusionEngine for live camera + radar mode.

        Args:
            camera_config: Configuration for camera
            radar_config: Configuration for radar

        Returns:
            Configured FusionEngine with configuration dictionaries
        """
        # Create minimal configuration dictionaries
        radar_feed_config = {
            "type": "DCA1000EVM",
            "dest_dir": radar_config.dest_dir if radar_config else CFGS.DEST_DIR,
            "config_file": (
                radar_config.radar_config_file
                if radar_config
                else CFGS.AWR2243_CONFIG_FILE
            ),
        }

        radar_analyser_config = {
            "type": "RadarHeatmapAnalyser",
            "config_file": (
                radar_config.radar_config_file
                if radar_config
                else CFGS.AWR2243_CONFIG_FILE
            ),
        }

        camera_feed_config = {
            "type": "D455",
            "dest_dir": camera_config.dest_dir if camera_config else CFGS.DEST_DIR,
        }

        camera_analyser_config = {"type": "D455Analyser"}

        return FusionEngine(
            radar_feed_config=radar_feed_config,
            radar_analyser_config=radar_analyser_config,
            camera_feed_config=camera_feed_config,
            camera_analyser_config=camera_analyser_config,
        )

    @staticmethod
    def create_live_radar_only(
        radar_config: Optional[DCA1000Config] = None,
    ) -> FusionEngine:
        """
        Create a FusionEngine for live radar-only mode.

        Args:
            radar_config: Configuration for radar

        Returns:
            Configured FusionEngine with configuration dictionaries for radar only
        """
        # Create minimal configuration dictionaries for radar components only
        radar_feed_config = {
            "type": "DCA1000EVM",
            "dest_dir": radar_config.dest_dir if radar_config else CFGS.DEST_DIR,
            "config_file": (
                radar_config.radar_config_file
                if radar_config
                else CFGS.AWR2243_CONFIG_FILE
            ),
        }

        radar_analyser_config = {
            "type": "RadarHeatmapAnalyser",
            "config_file": (
                radar_config.radar_config_file
                if radar_config
                else CFGS.AWR2243_CONFIG_FILE
            ),
        }

        return FusionEngine(
            radar_feed_config=radar_feed_config,
            radar_analyser_config=radar_analyser_config,
            camera_feed_config=None,
            camera_analyser_config=None,
        )

    @staticmethod
    def create_replay_fusion(
        recording_dir: str,
        radar_config_file: Optional[str] = None,
        sync_state: Optional[object] = None,
    ) -> FusionEngine:
        """
        Create a FusionEngine for replay camera + radar mode.
        Both camera and radar will use recorded data from the same directory.

        Args:
            recording_dir: Directory containing both camera and radar recording files
            radar_config_file: Path to radar configuration file for analyzer
            sync_state: Shared synchronization state (Manager.Namespace) for coordinated playback

        Returns:
            Configured FusionEngine with configuration dictionaries for replay
        """
        # Create minimal configuration dictionaries for replay
        radar_feed_config = {
            "type": "DCA1000Recording",
            "dest_dir": recording_dir,
            "sync_state": sync_state,
            "config_file": radar_config_file,
        }

        radar_analyser_config = {
            "type": "RadarHeatmapAnalyser",
            "config_file": radar_config_file,
        }

        camera_feed_config = {
            "type": "PNGCamera",
            "dest_dir": recording_dir,
            "sync_state": sync_state,
        }

        camera_analyser_config = {"type": "D455Analyser"}

        return FusionEngine(
            radar_feed_config=radar_feed_config,
            radar_analyser_config=radar_analyser_config,
            camera_feed_config=camera_feed_config,
            camera_analyser_config=camera_analyser_config,
        )

    @staticmethod
    def create_replay_radar_only(
        recording_dir: str,
        radar_config_file: Optional[str] = None,
    ) -> FusionEngine:
        """
        Create a FusionEngine for replay radar-only mode.

        Args:
            recording_dir: Directory containing radar recording files
            radar_config_file: Path to radar configuration file for analyzer

        Returns:
            Configured FusionEngine with configuration dictionaries for radar replay only
        """
        # Create minimal configuration dictionaries for radar replay only
        radar_feed_config = {"type": "DCA1000Recording", "dest_dir": recording_dir}

        radar_analyser_config = {
            "type": "RadarHeatmapAnalyser",
            "config_file": radar_config_file,
        }

        return FusionEngine(
            radar_feed_config=radar_feed_config,
            radar_analyser_config=radar_analyser_config,
            camera_feed_config=None,
            camera_analyser_config=None,
        )
