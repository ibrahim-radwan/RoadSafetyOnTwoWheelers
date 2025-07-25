"""
Interface definitions for camera and radar feeds and analyzers.
These abstract classes define the common interface for both live and replay implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional
import multiprocessing


class CameraFeed(ABC):
    """Abstract base class for camera data sources (live or recorded)"""

    @abstractmethod
    def run(self, stream_queue: multiprocessing.Queue, stop_event) -> None:
        """
        Stream camera data to the output queue.

        Args:
            stream_queue: Queue to send camera frames to
            stop_event: Event to signal when to stop streaming
        """
        pass


class RadarFeed(ABC):
    """Abstract base class for radar data sources (live or recorded)"""

    @abstractmethod
    def run(
        self,
        stream_queue: multiprocessing.Queue,
        stop_event,
        control_queue: Optional[multiprocessing.Queue] = None,
        status_queue: Optional[multiprocessing.Queue] = None,
    ) -> None:
        """
        Stream radar data to the output queue.

        Args:
            stream_queue: Queue to send radar frames to
            stop_event: Event to signal when to stop streaming
            control_queue: Optional queue for playback control commands (replay mode)
            status_queue: Optional queue for status updates (replay mode)
        """
        pass


class CameraAnalyser(ABC):
    """Abstract base class for camera data analysis"""

    @abstractmethod
    def run(
        self,
        input_queue: multiprocessing.Queue,
        output_queue: multiprocessing.Queue,
        stop_event,
    ) -> None:
        """
        Analyze camera frames and output results.

        Args:
            input_queue: Queue to receive camera frames from
            output_queue: Queue to send analysis results to
            stop_event: Event to signal when to stop processing
        """
        pass


class RadarAnalyser(ABC):
    """Abstract base class for radar data analysis"""

    @abstractmethod
    def run(
        self,
        input_queue: multiprocessing.Queue,
        output_queue: multiprocessing.Queue,
        stop_event,
    ) -> None:
        """
        Analyze radar frames and output results.

        Args:
            input_queue: Queue to receive radar frames from
            output_queue: Queue to send analysis results to
            stop_event: Event to signal when to stop processing
        """
        pass
