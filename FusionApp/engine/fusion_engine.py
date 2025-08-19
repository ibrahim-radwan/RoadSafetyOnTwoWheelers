import time
from config_params import CFGS
from utils import setup_logger

from typing import Optional
from multiprocessing import Process, Queue, Event
import os
from multiprocessing import shared_memory
from sample_processing.radar_params import ADCParams
import queue


class FusionEngine:
    def __init__(
        self,
        radar_feed_config: dict,
        radar_analyser_config: dict,
        camera_feed_config: Optional[dict] = None,
        camera_analyser_config: Optional[dict] = None,
    ):
        """
        Initialize the FusionEngine with configuration dictionaries.

        Feed and analyzer instances will be created in the run() method
        within the target process to avoid serialization issues.

        Args:
            radar_feed_config: Configuration dict for radar data source
            radar_analyser_config: Configuration dict for radar analysis component
            camera_feed_config: Optional configuration dict for camera data source
            camera_analyser_config: Optional configuration dict for camera analysis component
        """
        self.running = True
        self.logger = None

        # Store configuration dictionaries
        self.radar_feed_config = radar_feed_config
        self.radar_analyser_config = radar_analyser_config
        self.camera_feed_config = camera_feed_config
        self.camera_analyser_config = camera_analyser_config

        # Initialize queues (keep radar queue at 1 to avoid backlog; source drops to latest)
        self._radar_stream_queue: Queue = Queue(maxsize=1)
        self._camera_stream_queue: Optional[Queue] = None

        # Initialize camera queue only if camera config is available
        if self.camera_feed_config is not None:
            self._camera_stream_queue = Queue(maxsize=2)

        # Validate that if camera feed config is provided, analyser config must also be provided
        if (self.camera_feed_config is None) != (self.camera_analyser_config is None):
            raise ValueError(
                "Camera feed config and camera analyser config must both be provided or both be None"
            )

    def _create_radar_feed(self, config: dict):
        """Create radar feed instance from configuration."""
        feed_type = config["type"]
        sync_state = config.get("sync_state")

        if feed_type == "DCA1000EVM":
            from radar.dca1000_awr2243 import DCA1000EVM, DCA1000Config

            # Pass radar_config_file like the old working version
            radar_config_file = config.get("config_file", CFGS.AWR2243_CONFIG_FILE)
            radar_config = DCA1000Config(radar_config_file=radar_config_file)
            if "dest_dir" in config:
                radar_config.dest_dir = config["dest_dir"]
            return DCA1000EVM(
                radar_config, prealloc_shm_meta=config.get("prealloc_shm_meta")
            )
        elif feed_type == "DCA1000Recording":
            from radar.dca1000_awr2243 import DCA1000Recording, DCA1000Config

            # Pass radar_config_file for recording mode too
            radar_config_file = config.get("config_file", CFGS.AWR2243_CONFIG_FILE)
            radar_config = DCA1000Config(radar_config_file=radar_config_file)
            radar_config.dest_dir = config["dest_dir"]
            return DCA1000Recording(radar_config, sync_state=sync_state)
        else:
            raise ValueError(f"Unknown radar feed type: {feed_type}")

    def _create_radar_analyser(self, config: dict):
        """Create radar analyser instance from configuration."""
        analyser_type = config["type"]
        if analyser_type == "RadarHeatmapAnalyser":
            from analysis.radar_heatmap_analyser import RadarHeatmapAnalyser

            return RadarHeatmapAnalyser(
                config.get("config_file"),
                prealloc_shm_meta=config.get("prealloc_shm_meta"),
                prealloc_res_shm_meta=config.get("prealloc_res_shm_meta"),
            )
        else:
            raise ValueError(f"Unknown radar analyser type: {analyser_type}")

    def _create_camera_feed(self, config: dict):
        """Create camera feed instance from configuration."""
        feed_type = config["type"]
        sync_state = config.get("sync_state")

        if feed_type == "D455":
            from camera.d455 import D455, D455Config

            camera_config = D455Config()
            if "dest_dir" in config:
                camera_config.dest_dir = config["dest_dir"]
            return D455(camera_config)
        elif feed_type == "PNGCamera":
            from camera.png_camera import PNGCamera, PNGCameraConfig

            camera_config = PNGCameraConfig()
            camera_config.dest_dir = config["dest_dir"]
            return PNGCamera(camera_config, sync_state=sync_state)
        else:
            raise ValueError(f"Unknown camera feed type: {feed_type}")

    def _create_camera_analyser(self, config: dict):
        """Create camera analyser instance from configuration."""
        analyser_type = config["type"]
        if analyser_type == "D455Analyser":
            from analysis.d455_analyser import D455Analyser

            return D455Analyser()
        else:
            raise ValueError(f"Unknown camera analyser type: {analyser_type}")

    def run(
        self,
        radar_results_queue: Queue,
        camera_results_queue: Optional[Queue] = None,
        stop_event=None,
        control_queue: Optional[Queue] = None,
        status_queue: Optional[Queue] = None,
    ):
        """
        Run the fusion engine with polymorphic feeds and analyzers.

        Feed and analyzer instances are created here in the target process.

        Args:
            radar_results_queue: Queue for radar analysis results
            camera_results_queue: Optional queue for camera analysis results
            stop_event: Event to signal when to stop processing
            control_queue: Optional queue for playback control commands (replay mode)
            status_queue: Optional queue for status updates (replay mode)
        """
        self.logger = setup_logger("FusionEngine")

        if stop_event is None:
            stop_event = Event()

        self.logger.info("FusionEngine starting...")

        # Validate camera requirements
        if self.camera_feed_config is not None and camera_results_queue is None:
            raise ValueError(
                "camera_results_queue must be provided when camera_feed_config is available"
            )

        # Create separate control queues for camera and radar
        radar_control_queue = Queue() if control_queue is not None else None
        camera_control_queue = Queue() if control_queue is not None else None

        # Preallocate shared memory for radar raw frames (engine-owned)
        prealloc_shm_meta = None
        prealloc_res_shm_meta = None
        radar_cfg_file = self.radar_analyser_config.get(
            "config_file", CFGS.AWR2243_CONFIG_FILE
        )
        try:
            adc = ADCParams(radar_cfg_file)
            num_elements = adc.chirps * adc.tx * adc.samples * adc.IQ * adc.rx
            nbytes = num_elements * 2  # int16
            shm0 = shared_memory.SharedMemory(create=True, size=nbytes)
            shm1 = shared_memory.SharedMemory(create=True, size=nbytes)
            prealloc_shm_meta = {
                "names": [shm0.name, shm1.name],
                "nbytes": nbytes,
                "dtype": "int16",
                "shape": (num_elements,),
            }
            # Store for cleanup
            self._radar_shm_blocks = [shm0, shm1]
            self.logger.info(
                f"Preallocated radar SHM in engine: names={prealloc_shm_meta['names']}, bytes={nbytes}, dtype=int16, shape={(num_elements,)}"
            )

            # Preallocate SHM for radar analyser outputs (range_doppler, range_azimuth)
            # RD shape ~ (chirps, samples), RA shape ~ (angle_bins, samples) with angle_bins=181 (±90 deg, 1 deg step)
            angle_bins = (90 * 2) // 1 + 1
            rd_shape = (adc.chirps, adc.samples)
            ra_shape = (angle_bins, adc.samples)
            rd_nbytes = adc.chirps * adc.samples * 4  # float32
            ra_nbytes = angle_bins * adc.samples * 4  # float32
            rd0 = shared_memory.SharedMemory(create=True, size=rd_nbytes)
            rd1 = shared_memory.SharedMemory(create=True, size=rd_nbytes)
            ra0 = shared_memory.SharedMemory(create=True, size=ra_nbytes)
            ra1 = shared_memory.SharedMemory(create=True, size=ra_nbytes)
            prealloc_res_shm_meta = {
                "rd": {
                    "names": [rd0.name, rd1.name],
                    "nbytes": rd_nbytes,
                    "dtype": "float32",
                    "shape": rd_shape,
                },
                "ra": {
                    "names": [ra0.name, ra1.name],
                    "nbytes": ra_nbytes,
                    "dtype": "float32",
                    "shape": ra_shape,
                },
            }
            self._radar_res_shm_blocks = {"rd": [rd0, rd1], "ra": [ra0, ra1]}
            self.logger.info(
                f"Preallocated radar results SHM: rd={prealloc_res_shm_meta['rd']['names']}, ra={prealloc_res_shm_meta['ra']['names']}"
            )

            # Pass meta to feed and analyser configs
            self.radar_feed_config["prealloc_shm_meta"] = prealloc_shm_meta
            self.radar_analyser_config["prealloc_shm_meta"] = prealloc_shm_meta
            self.radar_analyser_config["prealloc_res_shm_meta"] = prealloc_res_shm_meta

            # Expose results SHM identifiers to the GUI process via environment variables
            try:
                rd_names = ",".join(
                    [blk.name for blk in self._radar_res_shm_blocks.get("rd", [])]
                )
                ra_names = ",".join(
                    [blk.name for blk in self._radar_res_shm_blocks.get("ra", [])]
                )
                if rd_names:
                    os.environ["RADAR_RD_SHM_NAMES"] = rd_names
                if ra_names:
                    os.environ["RADAR_RA_SHM_NAMES"] = ra_names
                angle_bins = (90 * 2) // 1 + 1
                os.environ["RADAR_RD_SHAPE"] = f"{adc.chirps},{adc.samples}"
                os.environ["RADAR_RA_SHAPE"] = f"{angle_bins},{adc.samples}"
            except Exception:
                pass
        except Exception as e:
            self.logger.warning(
                f"Failed to preallocate radar SHM in engine (falling back to feed-created): {e}"
            )
            self._radar_shm_blocks = []
            self._radar_res_shm_blocks = {}

        # Create instances using configuration in the target process
        self.logger.info("Creating feed and analyzer instances...")
        try:
            radar_feed = self._create_radar_feed(self.radar_feed_config)
            radar_analyser = self._create_radar_analyser(self.radar_analyser_config)

            camera_feed = None
            camera_analyser = None
            if (
                self.camera_feed_config is not None
                and self.camera_analyser_config is not None
            ):
                camera_feed = self._create_camera_feed(self.camera_feed_config)
                camera_analyser = self._create_camera_analyser(
                    self.camera_analyser_config
                )

            self.logger.info("Feed and analyzer instances created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create feed/analyzer instances: {e}")
            raise

        processes = []

        # Start radar feed process
        self.logger.info("Creating radar feed process...")
        radar_process = Process(
            target=radar_feed.run,
            args=(
                self._radar_stream_queue,
                stop_event,
                radar_control_queue,
                status_queue,
            ),
        )
        processes.append(radar_process)

        # Start radar analyzer process
        self.logger.info("Creating radar analyzer process...")
        radar_analyser_process = Process(
            target=radar_analyser.run,
            args=(
                self._radar_stream_queue,
                radar_results_queue,
                stop_event,
            ),
        )
        processes.append(radar_analyser_process)

        # Start camera processes if available
        camera_process = None
        camera_analyser_process = None

        if camera_feed is not None and camera_analyser is not None:
            self.logger.info("Creating camera feed process...")
            camera_process = Process(
                target=camera_feed.run,
                args=(
                    self._camera_stream_queue,
                    stop_event,
                    camera_control_queue,
                ),
            )
            processes.append(camera_process)

            self.logger.info("Creating camera analyzer process...")
            camera_analyser_process = Process(
                target=camera_analyser.run,
                args=(
                    self._camera_stream_queue,
                    camera_results_queue,
                    stop_event,
                ),
            )
            processes.append(camera_analyser_process)

        # Start all processes
        self.logger.info(f"Starting {len(processes)} processes...")
        for i, process in enumerate(processes):
            self.logger.info(f"Starting process {i}: {process.name}")
            process.start()
            self.logger.info(
                f"Process {i} started, PID: {process.pid}, alive: {process.is_alive()}"
            )

        # Monitor processes briefly
        time.sleep(2)  # Give processes time to start
        self.logger.info("Process status after 2 seconds:")
        for i, process in enumerate(processes):
            self.logger.info(
                f"Process {i}: PID: {process.pid}, alive: {process.is_alive()}, exit_code: {process.exitcode}"
            )

        # Control command forwarding loop
        if control_queue is not None:
            self.logger.info("Starting control command forwarding...")

        # Wait for stop signal and handle control commands
        while not stop_event.is_set():
            # Check for control commands and forward them to both camera and radar
            if control_queue is not None:
                try:
                    command = control_queue.get_nowait()
                    self.logger.info(f"Received control command: {command}")

                    # Forward command to radar
                    if radar_control_queue is not None:
                        try:
                            radar_control_queue.put(command)
                            self.logger.debug(f"Forwarded command to radar: {command}")
                        except Exception as e:
                            self.logger.error(f"Error forwarding command to radar: {e}")

                    # Forward command to camera
                    if camera_control_queue is not None:
                        try:
                            camera_control_queue.put(command)
                            self.logger.debug(f"Forwarded command to camera: {command}")
                        except Exception as e:
                            self.logger.error(
                                f"Error forwarding command to camera: {e}"
                            )

                except queue.Empty:
                    pass
                except Exception as e:
                    self.logger.error(f"Error processing control command: {e}")

            time.sleep(0.01)  # Small sleep to prevent busy waiting

        self.logger.info("FusionEngine stopping...")

        # Proactively signal analysers/feeds to exit immediately
        stop_sentinel = {"STOP": True}
        try:
            if self._radar_stream_queue is not None:
                self._radar_stream_queue.put_nowait(stop_sentinel)
        except Exception:
            pass
        try:
            if self._camera_stream_queue is not None:
                self._camera_stream_queue.put_nowait(stop_sentinel)
        except Exception:
            pass
        # Also notify control queues so feeds that watch control break promptly
        try:
            if radar_control_queue is not None:
                radar_control_queue.put_nowait("stop")
                radar_control_queue.put_nowait(stop_sentinel)
        except Exception:
            pass
        try:
            if camera_control_queue is not None:
                camera_control_queue.put_nowait("stop")
                camera_control_queue.put_nowait(stop_sentinel)
        except Exception:
            pass

        # Final process status check
        self.logger.info("Final process status:")
        for i, process in enumerate(processes):
            self.logger.info(
                f"Process {i}: PID: {process.pid}, alive: {process.is_alive()}, exit_code: {process.exitcode}"
            )

        # Give children a grace period to exit on their own
        time.sleep(0.5)

        # Attempt to join/terminate/kill all child processes robustly (join analysers first)
        for i in reversed(range(len(processes))):
            process = processes[i]
            try:
                self.logger.info(f"Joining process {i}...")
                process.join(timeout=6)
                if process.is_alive():
                    self.logger.warning(
                        f"Process {i} still alive after join, terminating..."
                    )
                    process.terminate()
                    process.join(timeout=4)
                if process.is_alive():
                    self.logger.error(f"Process {i} did not terminate, killing...")
                    process.kill()
                    process.join()
                self.logger.info(f"Process {i} joined, exit_code: {process.exitcode}")
            except Exception as e:
                self.logger.error(f"Error while shutting down process {i}: {e}")

        # Close internal queues owned by the engine to help GC
        try:
            if self._radar_stream_queue is not None:
                self._radar_stream_queue.close()
                self._radar_stream_queue.join_thread()
        except Exception:
            pass
        try:
            if self._camera_stream_queue is not None:
                self._camera_stream_queue.close()
                self._camera_stream_queue.join_thread()
        except Exception:
            pass
        try:
            if control_queue is not None and radar_control_queue is not None:
                radar_control_queue.close()
                radar_control_queue.join_thread()
        except Exception:
            pass
        try:
            if control_queue is not None and camera_control_queue is not None:
                camera_control_queue.close()
                camera_control_queue.join_thread()
        except Exception:
            pass

        # Cleanup engine-owned shared memory blocks
        try:
            for shm in getattr(self, "_radar_shm_blocks", []) or []:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass
            for group in getattr(self, "_radar_res_shm_blocks", {}).values():
                for shm in group:
                    try:
                        shm.close()
                    except Exception:
                        pass
                    try:
                        shm.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

        self.logger.info("FusionEngine stopped successfully.")
